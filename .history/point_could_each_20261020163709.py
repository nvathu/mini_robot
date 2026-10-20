# point_could_each.py
#
# Per-frame point clouds (camera + world) and one fused global model.
# OFFLINE only: MoCap pose is used to place clouds in the world frame and for checks.
#
# Camera optical frame : X right, Y down, Z forward (m)
# World frame (stored) : [x_world, height_up, y_world] (m), floor at height 0
#                        when camera_height.json is available.
#
# Usage (from project root):
#   python point_could_each.py                        # validated frames only (labels.csv)
#   python point_could_each.py --frames all --every 5 # every 5th image of the whole session

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from geometric_projection import (
    world_to_camera_frame,
    relative_marker_position_to_camera_optical_frame,
    project_to_pixel,
)

SESSION = "2025_0813_1543-S3"

RGB_ROOT = Path("dataset/rgb")
DEPTH_ROOT = Path("dataset/depth_dino") / SESSION
LABELS_CSV = Path("dataset/valid_frames/labels.csv")
POSE_CSV = Path("dataset/poses/records_20250813-1535.csv")
INTRINSICS_PATH = Path("dataset/calib/intrinsics.json")
SCALE_SHIFT_PATH = Path("dataset/calib/scale_shift.json")
CAM_HEIGHT_PATH = Path("dataset/calib/camera_height.json")

OUTPUT_ROOT = Path("dataset/pointclouds") / SESSION
MERGED_PATH = Path("dataset/pointclouds") / f"{SESSION}_merged.npz"

ROBOTS = ["S1", "S2"]
FOV_HALF_DEG = 35.0  # 70 deg horizontal FOV

# Marker -> camera offset taken from geometric_projection.py (single source of truth).
# Marker position seen from the camera is [0, 0, -offset] in optical coordinates.
CAM_FWD_OFFSET_M = -float(
    relative_marker_position_to_camera_optical_frame(np.zeros(3))[2]
) / 1000.0

_warned_aspect = False


# --------------------------------------------------
# Geometry
# --------------------------------------------------

def camera_to_world(points_cam, x_mm, y_mm, yaw_deg, cam_height_m=0.0):
    """Optical camera points (Nx3, m) -> [x_world, height_up, y_world] (m).

    Exact inverse of world_to_camera_frame() in geometric_projection.py
    (pitch = 0, camera CAM_FWD_OFFSET_M in front of the marker).
    """
    X, Y, Z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]
    forward = Z + CAM_FWD_OFFSET_M
    left = -X

    yaw = np.radians(yaw_deg)
    c, s = np.cos(yaw), np.sin(yaw)

    world_x = x_mm / 1000.0 + c * forward - s * left
    world_y = y_mm / 1000.0 + s * forward + c * left
    height = cam_height_m - Y
    return np.column_stack([world_x, height, world_y])


def scale_intrinsics(K, out_w, out_h):
    ref_w = float(K.get("width", 640))
    ref_h = float(K.get("height", 480))
    sx, sy = out_w / ref_w, out_h / ref_h
    return {
        "fx": float(K["fx"]) * sx,
        "fy": float(K["fy"]) * sy,
        "cx": float(K["cx"]) * sx,
        "cy": float(K["cy"]) * sy,
    }


def load_cam_height(path):
    if not path.exists():
        print(f"[WARN] {path} not found -> camera height = 0 (heights relative to camera)")
        return 0.0
    with open(path) as f:
        d = json.load(f)
    for key in ("h_c_m", "camera_height_m", "h_c", "height_m", "h"):
        if key in d:
            return float(d[key])
    print(f"[WARN] No known height key in {path} (keys: {list(d)}) -> camera height = 0")
    return 0.0


# --------------------------------------------------
# Depth -> points
# --------------------------------------------------

def load_depth_aligned(depth_path, size_wh):
    """Load depth and resize to RGB size (assumes full-frame resize, no crop/pad)."""
    global _warned_aspect
    depth = np.load(depth_path).astype(np.float32)
    depth = np.squeeze(depth)
    if depth.ndim != 2:
        return None
    w, h = size_wh
    if depth.shape != (h, w):
        if not _warned_aspect and abs(depth.shape[1] / depth.shape[0] - w / h) > 1e-3:
            print(
                f"[INFO] depth {depth.shape[::-1]} vs RGB {(w, h)}: aspect differs. "
                "Assuming depth = full RGB frame squashed (same as the ResNet 180x180 "
                "pipeline). If DINOv3 cropped/padded, the cloud will be distorted."
            )
            _warned_aspect = True
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
    return depth


def depth_to_points(metric_depth, rgb, K, min_d, max_d, stride):
    vs, us = np.mgrid[0:metric_depth.shape[0]:stride, 0:metric_depth.shape[1]:stride]
    z = metric_depth[::stride, ::stride]
    valid = np.isfinite(z) & (z > min_d) & (z < max_d)

    z = z[valid].astype(np.float32)
    u = us[valid].astype(np.float32)
    v = vs[valid].astype(np.float32)

    x = (u - K["cx"]) * z / K["fx"]
    y = (v - K["cy"]) * z / K["fy"]
    colors = rgb[::stride, ::stride][valid].astype(np.float32) / 255.0
    return np.column_stack([x, y, z]), colors


def voxel_downsample(points, colors, voxel):
    """Vectorised voxel grid; each voxel gets the mean position and mean color."""
    if voxel <= 0 or len(points) == 0:
        return points, colors
    keys = np.floor(points / voxel).astype(np.int64)
    keys -= keys.min(axis=0)
    dims = keys.max(axis=0) + 1
    flat = (keys[:, 0] * dims[1] + keys[:, 1]) * dims[2] + keys[:, 2]
    uniq, inv, cnt = np.unique(flat, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    n = len(uniq)
    pts = np.stack([np.bincount(inv, weights=points[:, k], minlength=n) for k in range(3)], 1)
    col = np.stack([np.bincount(inv, weights=colors[:, k], minlength=n) for k in range(3)], 1)
    return (pts / cnt[:, None]).astype(np.float32), (col / cnt[:, None]).astype(np.float32)


class Fusion:
    """Accumulates world points of all frames; compacts with voxel grid to bound memory."""

    def __init__(self, voxel, limit=15_000_000):
        self.voxel, self.limit = voxel, limit
        self.p, self.c, self.n = [], [], 0
        self.cam_path = []
        self.robot_pts, self.robot_names = [], []

    def add(self, pts, cols):
        self.p.append(pts)
        self.c.append(cols)
        self.n += len(pts)
        if self.n > self.limit:
            self.compact()

    def compact(self):
        if not self.p:
            return
        p, c = voxel_downsample(np.concatenate(self.p), np.concatenate(self.c), self.voxel)
        self.p, self.c, self.n = [p], [c], len(p)

    def save(self, path, cam_height):
        self.compact()
        p = self.p[0] if self.p else np.zeros((0, 3), np.float32)
        c = self.c[0] if self.c else np.zeros((0, 3), np.float32)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            points=p, colors=c,
            cam_path=np.asarray(self.cam_path, np.float32).reshape(-1, 3),
            robot_points=np.asarray(self.robot_pts, np.float32).reshape(-1, 3),
            robot_names=np.asarray(self.robot_names, dtype="U2"),
            cam_height=np.array(cam_height),
            voxel=np.array(self.voxel),
        )
        return len(p)


# --------------------------------------------------
# Frame selection
# --------------------------------------------------

def yaw_rate_deg_s(poses, k=3):
    """Yaw rate over a +-k sample window (deg/s); inf where undefined."""
    ts = poses["pose_timestamp"].to_numpy(np.float64) / 1e9
    yaw = poses["self_pose_angle"].interpolate().bfill().ffill().to_numpy(np.float64)
    yaw = np.degrees(np.unwrap(np.radians(yaw)))
    n = len(yaw)
    idx = np.arange(n)
    lo, hi = np.clip(idx - k, 0, n - 1), np.clip(idx + k, 0, n - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.abs(yaw[hi] - yaw[lo]) / (ts[hi] - ts[lo])
    return np.nan_to_num(rate, nan=np.inf, posinf=np.inf)


def list_frames(mode, poses):
    frames = []
    if mode == "valid":
        labels = pd.read_csv(LABELS_CSV, dtype={"img_id": str})
        labels = labels[(labels["S1_valid"] == 1) | (labels["S2_valid"] == 1)]
        for _, lab in labels.iterrows():
            valid = [r for r in ROBOTS
                     if f"{r}_valid" in lab.index and int(lab[f"{r}_valid"]) == 1]
            frames.append(dict(
                img_id=str(lab["img_id"]),
                rgb_path=Path(str(lab["img_path"])),
                csv_idx=int(lab["csv_idx"]),
                valid_robots=valid,
            ))
    else:
        seen = set()
        for csv_idx, name in poses["image"].items():
            if pd.isna(name) or str(name) == "none":
                continue
            img_id = Path(str(name)).stem.split("_")[-1]
            if img_id in seen:
                continue
            seen.add(img_id)
            frames.append(dict(
                img_id=img_id,
                rgb_path=RGB_ROOT / f"image_{img_id}.png",
                csv_idx=int(csv_idx),
                valid_robots=[],
            ))
    return frames


def get_pose(poses, csv_idx):
    if csv_idx < 0 or csv_idx >= len(poses):
        return None
    row = poses.iloc[csv_idx]
    if row[["self_pose_x", "self_pose_y", "self_pose_angle"]].isna().any():
        return None
    return row


# --------------------------------------------------
# One frame
# --------------------------------------------------

def process_frame(fr, poses, K_ref, scale, shift, cam_h, calib_range, args, fusion, stats):
    img_id = fr["img_id"]
    depth_path = DEPTH_ROOT / f"image_{img_id}_depth.npy"
    if not depth_path.exists():
        return "missing_depth"
    if not fr["rgb_path"].exists():
        return "missing_rgb"

    bgr = cv2.imread(str(fr["rgb_path"]))
    if bgr is None:
        return "unreadable_rgb"
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    depth = load_depth_aligned(depth_path, (w, h))
    if depth is None:
        return "invalid_depth_shape"

    pose = get_pose(poses, fr["csv_idx"])
    if pose is None:
        return "no_pose"

    K = scale_intrinsics(K_ref, w, h)
    sx, sy, yaw = float(pose["self_pose_x"]), float(pose["self_pose_y"]), float(pose["self_pose_angle"])

    if calib_range:
        lo, hi = calib_range
        stats["outside_range"].append(float(np.mean((depth < lo) | (depth > hi))))

    metric = scale * depth + shift
    pts_cam, cols = depth_to_points(metric, rgb, K, args.min_depth, args.max_depth, args.stride)
    if len(pts_cam) == 0:
        return "no_valid_points"
    pts_cam, cols = voxel_downsample(pts_cam, cols, args.voxel)
    pts_world = camera_to_world(pts_cam, sx, sy, yaw, cam_h).astype(np.float32)

    cam_pos = camera_to_world(np.zeros((1, 3)), sx, sy, yaw, cam_h)[0]

    # MoCap markers of the other robots (offline reference + checks)
    r_cam, r_world, r_names = [], [], []
    for name in ROBOTS:
        cx_col, cy_col = f"car_{name}_pose_x", f"car_{name}_pose_y"
        if cx_col not in pose.index or pd.isna(pose[cx_col]) or pd.isna(pose[cy_col]):
            continue
        ox, oy = float(pose[cx_col]), float(pose[cy_col])
        pcam = np.asarray(world_to_camera_frame(sx, sy, 0.0, yaw, ox, oy, 0.0), float)
        if pcam[2] <= 0.1 or abs(np.degrees(np.arctan2(pcam[0], pcam[2]))) > FOV_HALF_DEG:
            continue
        pw = camera_to_world(pcam[None], sx, sy, yaw, cam_h)[0]
        stats["roundtrip"].append(float(np.hypot(pw[0] - ox / 1000.0, pw[2] - oy / 1000.0)))

        u, v = project_to_pixel(pcam, K["fx"], K["fy"], K["cx"], K["cy"])
        ui, vi = int(round(u)), int(round(v))
        if 2 <= ui < w - 2 and 2 <= vi < h - 2:
            d_est = float(np.median(metric[vi - 2:vi + 3, ui - 2:ui + 3]))
            stats["marker_err"].append(d_est - float(pcam[2]))
            stats["marker_true"].append(float(pcam[2]))

        r_cam.append(pcam)
        r_world.append(pw)
        r_names.append(name)
        if fusion is not None:
            fusion.robot_pts.append(pw)
            fusion.robot_names.append(name)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_ROOT / f"image_{img_id}.npz",
        points_camera=pts_cam.astype(np.float32),
        points_world=pts_world,
        colors=cols.astype(np.float32),
        cam_pos_world=cam_pos.astype(np.float32),
        yaw_deg=np.array(yaw),
        robot_cam=np.asarray(r_cam, np.float32).reshape(-1, 3),
        robot_world=np.asarray(r_world, np.float32).reshape(-1, 3),
        robot_names=np.asarray(r_names, dtype="U2"),
        valid_robots=np.asarray(fr["valid_robots"], dtype="U2"),
        img_id=img_id,
        img_path=str(fr["rgb_path"]),
        scale=np.array(scale), shift=np.array(shift),
        cam_height=np.array(cam_h),
        K_used=np.array([K["fx"], K["fy"], K["cx"], K["cy"]]),
    )

    if fusion is not None:
        fusion.add(pts_world, cols.astype(np.float32))
        fusion.cam_path.append(cam_pos)
    return "saved"


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Per-frame point clouds + fused global model.")
    ap.add_argument("--frames", choices=["valid", "all"], default="valid",
                    help="valid = labels.csv frames; all = every image in the session")
    ap.add_argument("--every", type=int, default=1, help="take every N-th frame")
    ap.add_argument("--limit", type=int, default=0, help="stop after N frames (0 = no limit)")
    ap.add_argument("--max-yaw-rate", type=float, default=15.0,
                    help="deg/s, applied only to --frames all (pose/image desync while rotating); 0 = off")
    ap.add_argument("--min-depth", type=float, default=0.1)
    ap.add_argument("--max-depth", type=float, default=5.0)
    ap.add_argument("--stride", type=int, default=1, help="pixel stride when unprojecting")
    ap.add_argument("--voxel", type=float, default=0.005, help="per-frame voxel (m), 0 = off")
    ap.add_argument("--merge-voxel", type=float, default=0.02, help="fused model voxel (m)")
    ap.add_argument("--no-merge", action="store_true")
    ap.add_argument("--scale", type=float, default=None)
    ap.add_argument("--shift", type=float, default=None)
    ap.add_argument("--cam-height", type=float, default=None, help="camera height above floor (m)")
    args = ap.parse_args()

    with open(INTRINSICS_PATH) as f:
        K_ref = json.load(f)
    with open(SCALE_SHIFT_PATH) as f:
        calib = json.load(f)
    scale = args.scale if args.scale is not None else float(calib["scale"])
    shift = args.shift if args.shift is not None else float(calib["shift"])
    calib_range = calib.get("d_pred_range")
    if calib.get("usable") is False:
        print("[WARN] scale_shift.json is flagged UNUSABLE by fit_scale.py: "
              + "; ".join(calib.get("unusable_reasons", [])))
        print("       Metric depth below is not trustworthy.")
    cam_h = args.cam_height if args.cam_height is not None else load_cam_height(CAM_HEIGHT_PATH)

    print("=" * 60)
    print(f"Session {SESSION} | frames={args.frames} | scale={scale:.4f} shift={shift:.4f}")
    print(f"K_ref={K_ref} | cam_height={cam_h:.3f} m | marker->cam offset={CAM_FWD_OFFSET_M:.3f} m")
    print(f"calibrated d_pred_range={calib_range}")
    print("=" * 60)

    poses = pd.read_csv(POSE_CSV)
    frames = list_frames(args.frames, poses)
    n_all = len(frames)

    if args.frames == "all" and args.max_yaw_rate > 0:
        rate = yaw_rate_deg_s(poses)
        frames = [fr for fr in frames if rate[fr["csv_idx"]] <= args.max_yaw_rate]
        print(f"Yaw-rate filter (<= {args.max_yaw_rate} deg/s): {len(frames)}/{n_all} frames")

    frames = frames[::max(1, args.every)]
    if args.limit > 0:
        frames = frames[:args.limit]
    print(f"Processing {len(frames)} frames")

    fusion = None if args.no_merge else Fusion(args.merge_voxel)
    stats = {"outside_range": [], "roundtrip": [], "marker_err": [], "marker_true": []}
    counts = {}

    for i, fr in enumerate(frames, start=1):
        res = process_frame(fr, poses, K_ref, scale, shift, cam_h, calib_range, args, fusion, stats)
        counts[res] = counts.get(res, 0) + 1
        if i % 50 == 0 or i == len(frames):
            print(f"  {i}/{len(frames)} | last: {res}")

    print("\nResults:", dict(sorted(counts.items())))

    if fusion is not None and counts.get("saved", 0) > 0:
        n = fusion.save(MERGED_PATH, cam_h)
        print(f"Fused model: {n} points -> {MERGED_PATH}")

    # ---- built-in sanity checks ----
    print("\nChecks:")
    if stats["roundtrip"]:
        print(f"  [geometry] camera_to_world vs MoCap marker round-trip, max error: "
              f"{max(stats['roundtrip']):.2e} m (must be ~0)")
    if stats["outside_range"]:
        print(f"  [calibration] mean fraction of pixels outside calibrated d_pred_range: "
              f"{100 * np.mean(stats['outside_range']):.1f}% "
              "(high => scale-shift is extrapolating)")
    if stats["marker_err"]:
        e = np.asarray(stats["marker_err"])
        t = np.asarray(stats["marker_true"])
        print(f"  [depth@marker] n={len(e)}  MAE={np.mean(np.abs(e)):.3f} m  "
              f"median={np.median(np.abs(e)):.3f} m  bias={np.mean(e):+.3f} m  "
              f"(true range {t.min():.2f}-{t.max():.2f} m; indicative only)")
        if len(e) > 2 and np.std(t) > 0 and np.std(e + t) > 0:
            print(f"  [depth@marker] corr(estimated, MoCap) = {np.corrcoef(e + t, t)[0, 1]:.3f}")
    print(f"\nOutput: {OUTPUT_ROOT.resolve()}")


if __name__ == "__main__":
    main()