
import argparse
import glob
import os
import shutil
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from geometric_projection import world_to_camera_frame, project_to_pixel

IMG_DIR = "dataset/rgb"
POSE_CSV = "dataset/poses/records_20250813-1535.csv"
INTRINSICS_PATH = "dataset/calib/intrinsics.json"
OUT_DIR = "dataset/valid_frames"
IMG_OUT_DIR = os.path.join(OUT_DIR, "images")
LABELS_CSV = os.path.join(OUT_DIR, "labels.csv")

ROBOTS = ["S1", "S2"]
COLORS = {"S1": (0, 255, 0), "S2": (0, 0, 255)}  # BGR: green / red
IMG_W, IMG_H = 640, 480
MIN_FORWARD_M = 0.1
FONT = cv2.FONT_HERSHEY_SIMPLEX


def img_id_of(name):
    return Path(str(name)).stem.split("_")[-1]


def project_robot(row, tgt, K):
    """Return projection info for target robot, or None if missing / behind / outside image."""
    cols = [f"car_{tgt}_pose_x", f"car_{tgt}_pose_y"]
    if any(c not in row.index or pd.isna(row[c]) for c in cols):
        return None
    pcam = world_to_camera_frame(
        row["self_pose_x"], row["self_pose_y"], 0.0, row["self_pose_angle"],
        row[cols[0]], row[cols[1]], 0.0,
    )
    forward, lateral = float(pcam[2]), float(pcam[0])
    if forward <= MIN_FORWARD_M:
        return None
    u, v = project_to_pixel(pcam, K["fx"], K["fy"], K["cx"], K["cy"])
    if not (0 <= u < IMG_W and 0 <= v < IMG_H):
        return None
    return {"forward": forward, "lateral": lateral, "u": float(u), "v": float(v)}


def build_candidates(df, K, img_by_id, max_yaw_rate, stride):
    t = df["pose_timestamp"].to_numpy(float) / 1e9
    yaw_series = pd.Series(df["self_pose_angle"].to_numpy(float)).interpolate(limit_direction="both")
    yaw = np.unwrap(np.radians(yaw_series.to_numpy()))
    with np.errstate(divide="ignore", invalid="ignore"):
        yaw_rate = np.abs(np.degrees(np.gradient(yaw, t)))  # deg/s

    candidates, seen = [], set()
    n_rot_skipped = 0
    for idx, row in df.iterrows():
        if any(pd.isna(row[c]) for c in ("self_pose_x", "self_pose_y", "self_pose_angle")):
            continue
        if pd.isna(row["image"]) or str(row["image"]) == "none":
            continue
        img_id = img_id_of(row["image"])
        img_path = img_by_id.get(img_id)
        if img_path is None or img_id in seen:
            continue
        proj = {r: p for r in ROBOTS if (p := project_robot(row, r, K)) is not None}
        if not proj:
            continue
        if max_yaw_rate > 0 and not (yaw_rate[idx] <= max_yaw_rate):  # NaN -> skipped too
            n_rot_skipped += 1
            continue
        seen.add(img_id)
        candidates.append({"idx": int(idx), "img_id": img_id, "img_path": img_path, "proj": proj})

    print(f"Candidates: {len(candidates)} (skipped {n_rot_skipped} frames: yaw rate > {max_yaw_rate} deg/s)")
    return candidates[::stride]


def make_row(cand, valid):
    row = {"img_id": cand["img_id"], "csv_idx": cand["idx"], "img_path": cand["img_path"]}
    for r in ROBOTS:
        p = cand["proj"].get(r)
        row[f"{r}_valid"] = int(valid[r])
        row[f"{r}_forward_m"] = p["forward"] if p else np.nan
        row[f"{r}_lateral_m"] = p["lateral"] if p else np.nan
        row[f"{r}_u"] = p["u"] if p else np.nan
        row[f"{r}_v"] = p["v"] if p else np.nan
    return row


def load_labels():
    if not os.path.exists(LABELS_CSV):
        return {}
    df = pd.read_csv(LABELS_CSV, dtype={"img_id": str})
    return {r["img_id"]: r for r in df.to_dict("records")}


def save_labels(labels):
    os.makedirs(OUT_DIR, exist_ok=True)
    if labels:
        rows = sorted(labels.values(), key=lambda r: r["csv_idx"])
        pd.DataFrame(rows).to_csv(LABELS_CSV, index=False)


def sync_valid_folder(labels):
    """Copy frames with >=1 valid robot into IMG_OUT_DIR; remove frames no longer valid."""
    os.makedirs(IMG_OUT_DIR, exist_ok=True)
    keep = {}
    for r in labels.values():
        if any(int(r[f"{t}_valid"]) for t in ROBOTS):
            keep[os.path.basename(r["img_path"])] = r["img_path"]
    for name, src in keep.items():
        dst = os.path.join(IMG_OUT_DIR, name)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
    for name in os.listdir(IMG_OUT_DIR):
        if name not in keep:
            os.remove(os.path.join(IMG_OUT_DIR, name))
    return len(keep)


def render(cand, labels, show_overlay, pos, total, scale):
    img = cv2.imread(cand["img_path"])
    if img is None:
        img = np.zeros((IMG_H, IMG_W, 3), np.uint8)
    if show_overlay:
        for r, p in cand["proj"].items():
            col, u, v = COLORS[r], int(p["u"]), int(p["v"])
            cv2.circle(img, (u, v), 10, col, 2)
            cv2.line(img, (u - 15, v), (u + 15, v), col, 1)
            cv2.line(img, (u, v - 15), (u, v + 15), col, 1)
            cv2.putText(img, r, (u + 12, v - 12), FONT, 0.6, col, 2)
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    lab = labels.get(cand["img_id"])
    status = "unlabeled" if lab is None else " ".join(
        f"{t}={'OK' if int(lab[f'{t}_valid']) else 'NO'}" for t in ROBOTS)
    lines = [(f"[{pos + 1}/{total}] id={cand['img_id']} overlay={'ON' if show_overlay else 'OFF'} | {status}", (255, 255, 255))]
    for r in ROBOTS:
        p = cand["proj"].get(r)
        txt = f"{r}: fwd={p['forward']:.2f}m lat={p['lateral']:+.2f}m u={p['u']:.0f}" if p else f"{r}: not in view"
        lines.append((txt, COLORS[r]))

    bar_h = 24 * len(lines) + 6
    cv2.rectangle(img, (0, 0), (img.shape[1], bar_h), (0, 0, 0), -1)
    for k, (txt, col) in enumerate(lines):
        cv2.putText(img, txt, (8, 20 + 24 * k), FONT, 0.55, col, 1, cv2.LINE_AA)
    h = img.shape[0]
    cv2.rectangle(img, (0, h - 28), (img.shape[1], h), (0, 0, 0), -1)
    cv2.putText(img, "1:S1 ok  2:S2 ok  3:both ok  n/space:none  b:back  o:overlay  q:save+quit",
                (8, h - 9), FONT, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=5, help="keep every N-th candidate frame")
    ap.add_argument("--max-yaw-rate", type=float, default=15.0, help="deg/s; 0 disables rotation filter")
    ap.add_argument("--scale", type=float, default=1.5, help="display scale")
    args = ap.parse_args()

    with open(INTRINSICS_PATH) as f:
        K = json.load(f)
    df = pd.read_csv(POSE_CSV)
    img_by_id = {img_id_of(f): f for f in glob.glob(f"{IMG_DIR}/*.png")}

    cands = build_candidates(df, K, img_by_id, args.max_yaw_rate, args.stride)
    if not cands:
        print("No candidate frames found.")
        return

    labels = load_labels()
    pos = next((i for i, c in enumerate(cands) if c["img_id"] not in labels), 0)
    print(f"{len(cands)} frames to review, resuming at {pos + 1}. Existing labels: {len(labels)}")

    show_overlay = True
    win = "manual_filter"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    try:
        while 0 <= pos < len(cands):
            cand = cands[pos]
            cv2.imshow(win, render(cand, labels, show_overlay, pos, len(cands), args.scale))
            key = cv2.waitKey(0) & 0xFF

            if key in (ord("q"), 27):
                break
            if key == ord("o"):
                show_overlay = not show_overlay
                continue
            if key == ord("b"):
                pos = max(0, pos - 1)
                continue

            if key in (ord("1"), ord("2"), ord("3"), ord("n"), ord(" ")):
                want = {"S1": key in (ord("1"), ord("3")), "S2": key in (ord("2"), ord("3"))}
                if any(want[r] and r not in cand["proj"] for r in ROBOTS):
                    print("  Robot not in view for this frame -> key ignored")
                    continue
                labels[cand["img_id"]] = make_row(cand, want)
                save_labels(labels)
                pos += 1
        else:
            print("Reached the end of the candidate list.")
    finally:
        cv2.destroyAllWindows()
        save_labels(labels)
        n_valid = sync_valid_folder(labels)
        n_s1 = sum(int(r["S1_valid"]) for r in labels.values())
        n_s2 = sum(int(r["S2_valid"]) for r in labels.values())
        print(f"Labeled: {len(labels)} | valid frames: {n_valid} | S1 valid: {n_s1} | S2 valid: {n_s2}")
        print(f"Labels : {LABELS_CSV}\nImages : {IMG_OUT_DIR}")


if __name__ == "__main__":
    main()