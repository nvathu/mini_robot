# visual_each.py
#
# Viewer for the clouds written by point_could_each.py.
#   python visual_each.py                 # browse per-frame clouds
#   python visual_each.py --merged        # start with the fused global model
#   python visual_each.py --start 10 --frame world
#
# Keys:
#   Right / Left : next / previous frame        M : toggle fused model <-> per-frame
#   C / W        : camera frame / world frame   1 : front view, 2 : top-down, 3 : oblique
#   Esc          : quit                         +/- (Open3D built-in) : point size
#
# Display conventions (proper rotations, no mirroring):
#   camera : optical [X right, Y down, Z fwd]  -> [X, -Y, -Z]  (looks like the camera image)
#   world  : stored [x, up, y]                  -> [x, up, -y]  (top-down = standard map, y up)
# Markers: S1 = green, S2 = red (MoCap, offline reference only).
# Camera: camera frame -> camera = origin of the RGB axes (looks along -Z of the screen = into the scene);
#         world frame / fused model -> blue frustum = camera pose (apex = camera, opens toward the viewing direction).

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

SESSION = "2025_0813_1543-S3"
POINTCLOUD_DIR = Path("dataset/pointclouds") / SESSION
MERGED_PATH = Path("dataset/pointclouds") / f"{SESSION}_merged.npz"

WINDOW_W, WINDOW_H = 1280, 800
POINT_SIZE = 2.0
BACKGROUND = [1.0, 1.0, 1.0]
MARKER_RADIUS = 0.03
MAX_MERGED_MARKERS = 300
MARKER_COLORS = {"S1": [0.0, 0.7, 0.0], "S2": [0.9, 0.0, 0.0]}

KEY_RIGHT, KEY_LEFT, KEY_ESC = 262, 263, 256


# --------------------------------------------------
# Coordinate helpers
# --------------------------------------------------

def to_display(points, frame):
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if frame == "camera":
        return p * np.array([1.0, -1.0, -1.0])
    return p * np.array([1.0, 1.0, -1.0])


def make_cloud(points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None and len(colors) == len(points):
        pcd.colors = o3d.utility.Vector3dVector(np.clip(colors.astype(np.float64), 0, 1))
    return pcd


def make_markers(points_disp, names, max_n=None):
    meshes = []
    idx = np.arange(len(points_disp))
    if max_n is not None and len(idx) > max_n:
        idx = idx[np.linspace(0, len(idx) - 1, max_n).astype(int)]
    for i in idx:
        m = o3d.geometry.TriangleMesh.create_sphere(radius=MARKER_RADIUS, resolution=8)
        m.translate(points_disp[i])
        m.paint_uniform_color(MARKER_COLORS.get(str(names[i]), [0.0, 0.0, 0.0]))
        meshes.append(m)
    return meshes


def make_frustum(cam, yaw_deg, K, length=0.4, color=(0.0, 0.0, 1.0)):
    """Camera frustum in stored world coords [x, up, y]; returns a LineSet in display coords."""
    fx, fy, cx, cy = [float(v) for v in K]
    if fx <= 0 or fy <= 0:
        return None
    th, tv = cx / fx, cy / fy
    yaw = np.radians(yaw_deg)
    fwd = np.array([np.cos(yaw), 0.0, np.sin(yaw)])
    left = np.array([-np.sin(yaw), 0.0, np.cos(yaw)])
    up = np.array([0.0, 1.0, 0.0])
    cam = np.asarray(cam, dtype=np.float64).reshape(3)
    corners = [cam + length * (fwd - sx * th * left + sy * tv * up)
               for sx, sy in ((-1, -1), (-1, 1), (1, 1), (1, -1))]
    pts = to_display(np.vstack([cam] + corners), "world")
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines = o3d.utility.Vector2iVector(
        [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]])
    ls.colors = o3d.utility.Vector3dVector(np.tile(color, (8, 1)))
    return ls


def make_polyline(points_disp, color):
    if len(points_disp) < 2:
        return None
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(points_disp)
    ls.lines = o3d.utility.Vector2iVector(
        np.column_stack([np.arange(len(points_disp) - 1), np.arange(1, len(points_disp))]))
    ls.colors = o3d.utility.Vector3dVector(np.tile(color, (len(points_disp) - 1, 1)))
    return ls


# --------------------------------------------------
# Loading
# --------------------------------------------------

def load_frame(path, frame):
    with np.load(path, allow_pickle=False) as d:
        key = "points_camera" if frame == "camera" else "points_world"
        pts, cols = d[key], d["colors"]
        if len(pts) == 0 or len(pts) != len(cols):
            return None
        info = {
            "img_id": str(d["img_id"]),
            "valid": [str(x) for x in d["valid_robots"]],
            "names": [str(x) for x in d["robot_names"]],
            "robots": d["robot_cam" if frame == "camera" else "robot_world"],
            "cam_pos": d["cam_pos_world"],
            "yaw": float(d["yaw_deg"]),
            "K": d["K_used"],
        }
    ok = np.isfinite(pts).all(axis=1)
    return pts[ok], cols[ok], info


def load_merged(path):
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


# --------------------------------------------------
# Scene builders
# --------------------------------------------------

def build_frame_scene(path, frame):
    loaded = load_frame(path, frame)
    if loaded is None:
        return None, None
    pts, cols, info = loaded
    geoms = [make_cloud(to_display(pts, frame), cols)]

    if len(info["robots"]):
        geoms += make_markers(to_display(info["robots"], frame), info["names"])

    if frame == "camera":
        geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15))
    else:
        frustum = make_frustum(info["cam_pos"][0] if info["cam_pos"].ndim > 1 else info["cam_pos"],
                               info["yaw"], info["K"])
        if frustum is not None:
            geoms.append(frustum)
        geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3))

    summary = (f"{info['img_id']} | {len(pts)} pts | valid robots: {info['valid'] or '-'} | "
               + ", ".join(f"{n} fwd={p[2]:.2f} m" if frame == "camera" else n
                           for n, p in zip(info["names"], info["robots"])))
    return geoms, summary


def build_merged_scene(merged):
    pts = to_display(merged["points"], "world")
    geoms = [make_cloud(pts, merged["colors"])]
    path = make_polyline(to_display(merged["cam_path"], "world"), [0.0, 0.0, 1.0])
    if path is not None:
        geoms.append(path)
    if len(merged["robot_points"]):
        geoms += make_markers(to_display(merged["robot_points"], "world"),
                              merged["robot_names"], MAX_MERGED_MARKERS)
    if "cam_yaw" in merged and "K_used" in merged and len(merged["cam_yaw"]):
        n = len(merged["cam_yaw"])
        for i in np.linspace(0, n - 1, min(n, 25)).astype(int):
            fr = make_frustum(merged["cam_path"][i], merged["cam_yaw"][i], merged["K_used"], length=0.25)
            if fr is not None:
                geoms.append(fr)
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3))
    return geoms, f"fused model | {len(pts)} pts | {len(merged['cam_path'])} camera poses"


# --------------------------------------------------
# Viewer
# --------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", action="store_true", help="start with the fused model")
    ap.add_argument("--frame", choices=["camera", "world"], default="camera")
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()

    files = sorted(POINTCLOUD_DIR.glob("image_*.npz"))
    merged = load_merged(MERGED_PATH) if MERGED_PATH.exists() else None
    if not files and merged is None:
        print(f"[ERROR] Nothing to show in {POINTCLOUD_DIR} / {MERGED_PATH}")
        return
    if args.merged and merged is None:
        print(f"[WARN] {MERGED_PATH} not found, starting in per-frame mode")

    state = {
        "mode": "merged" if (args.merged and merged is not None) or not files else "frame",
        "frame": args.frame,
        "index": int(np.clip(args.start, 0, max(len(files) - 1, 0))),
        "view": "front" if args.frame == "camera" else "top",
    }

    vis = o3d.visualization.VisualizerWithKeyCallback()
    if not vis.create_window("Point Cloud Viewer", WINDOW_W, WINDOW_H):
        print("[ERROR] Failed to create the Open3D window.")
        return
    opt = vis.get_render_option()
    opt.background_color = np.asarray(BACKGROUND)
    opt.point_size = POINT_SIZE
    opt.show_coordinate_frame = False

    def set_view(center):
        vc = vis.get_view_control()
        front, up = {
            "front": ([0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
            "top": ([0.0, 1.0, 0.0], [0.0, 0.0, -1.0]),
            "oblique": ([-0.35, 0.65, 0.65], [0.0, 1.0, 0.0]),
        }[state["view"]]
        vc.set_lookat(center)
        vc.set_front(front)
        vc.set_up(up)
        vc.set_zoom(0.5)

    def show(reset_view):
        if state["mode"] == "merged":
            geoms, summary = build_merged_scene(merged)
        else:
            geoms, summary = None, None
            # skip unreadable frames
            for _ in range(len(files)):
                geoms, summary = build_frame_scene(files[state["index"]], state["frame"])
                if geoms is not None:
                    break
                state["index"] = (state["index"] + 1) % len(files)
            if geoms is None:
                print("[ERROR] No readable frame.")
                return
            summary = f"[{state['index'] + 1}/{len(files)}] {state['frame']} | {summary}"
        vis.clear_geometries()
        for g in geoms:
            vis.add_geometry(g, reset_bounding_box=reset_view)
        if reset_view:
            set_view(geoms[0].get_center())
        print(summary)

    def step(delta):
        def cb(_):
            if state["mode"] == "frame" and files:
                state["index"] = (state["index"] + delta) % len(files)
                show(reset_view=False)
            return True
        return cb

    def set_frame(frame):
        def cb(_):
            state["frame"] = frame
            state["mode"] = "frame" if files else state["mode"]
            state["view"] = "front" if frame == "camera" else "top"
            show(reset_view=True)
            return True
        return cb

    def toggle_merged(_):
        if merged is None:
            print("[WARN] fused model missing - run point_could_each.py without --no-merge")
            return False
        state["mode"] = "frame" if state["mode"] == "merged" and files else "merged"
        if state["mode"] == "merged":
            state["view"] = "top"
        show(reset_view=True)
        return True

    def set_view_key(name):
        def cb(_):
            state["view"] = name
            show(reset_view=True)
            return True
        return cb

    vis.register_key_callback(KEY_RIGHT, step(1))
    vis.register_key_callback(KEY_LEFT, step(-1))
    vis.register_key_callback(KEY_ESC, lambda v: v.close() or False)
    vis.register_key_callback(ord("C"), set_frame("camera"))
    vis.register_key_callback(ord("W"), set_frame("world"))
    vis.register_key_callback(ord("M"), toggle_merged)
    vis.register_key_callback(ord("1"), set_view_key("front"))
    vis.register_key_callback(ord("2"), set_view_key("top"))
    vis.register_key_callback(ord("3"), set_view_key("oblique"))

    print(__doc__)
    show(reset_view=True)
    try:
        vis.run()
    finally:
        vis.destroy_window()


if __name__ == "__main__":
    main()