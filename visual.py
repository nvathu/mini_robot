import argparse

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--input", default="dataset/pointclouds/room_pointcloud.npz")
ap.add_argument("--output", default="room_pointcloud_result.png")
ap.add_argument("--max-points", type=int, default=200000)
ap.add_argument("--color", choices=["rgb", "height"], default="rgb")
ap.add_argument("--show", action="store_true", help="open an interactive window")
args = ap.parse_args()

import matplotlib
if not args.show:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

data = np.load(args.input)
pts, cols = data["scene_points"], data["scene_colors"]
robots = data["robot_points"].reshape(-1, 3)
selfp = data["self_points"].reshape(-1, 3) if "self_points" in data.files else np.zeros((0, 3))

x, h, y = pts[:, 0], pts[:, 1], pts[:, 2]
lo = np.percentile(pts, 1, axis=0)
hi = np.percentile(pts, 99, axis=0)
print(f"Points: {len(pts)} | 1-99% extent: X {hi[0] - lo[0]:.2f} m, Y {hi[2] - lo[2]:.2f} m, height {hi[1] - lo[1]:.2f} m")
print(f"X [{lo[0]:.2f}, {hi[0]:.2f}]  Y [{lo[2]:.2f}, {hi[2]:.2f}]  height [{lo[1]:.2f}, {hi[1]:.2f}]")
print(f"Robots (validated MoCap): {len(robots)} | self poses: {len(selfp)}")

inside = np.all((pts >= lo) & (pts <= hi), axis=1)
idx = np.flatnonzero(inside)
if len(idx) > args.max_points:
    idx = np.random.default_rng(0).choice(idx, args.max_points, replace=False)
p = pts[idx]
c = np.clip(cols[idx] * 2.2 + 0.1, 0, 1) if args.color == "rgb" else p[:, 1]
kw = {"c": c} if args.color == "rgb" else {"c": c, "cmap": "viridis"}

fig = plt.figure(figsize=(21, 7))
ax1 = fig.add_subplot(1, 3, 1, projection="3d")
ax1.scatter(p[:, 0], p[:, 2], p[:, 1], s=1, alpha=0.5, **kw)
ax1.scatter(robots[:, 0], robots[:, 2], robots[:, 1], c="red", s=12)
ax1.view_init(elev=25, azim=-60)
ax1.set_xlabel("X (m)"); ax1.set_ylabel("Y (m)"); ax1.set_zlabel("height (m)")
ax1.set_box_aspect([hi[0] - lo[0], hi[2] - lo[2], max(hi[1] - lo[1], 0.5)])
ax1.set_title("3D (oblique)")

ax2 = fig.add_subplot(1, 3, 2)
ax2.scatter(p[:, 0], p[:, 2], s=1, alpha=0.5, **kw)
ax2.scatter(selfp[:, 0], selfp[:, 2], c="blue", marker="^", s=18, label="camera (self) positions")
ax2.scatter(robots[:, 0], robots[:, 2], c="red", s=14, label="target robots (MoCap)")
ax2.set_aspect("equal"); ax2.set_xlabel("X (m)"); ax2.set_ylabel("Y (m)")
ax2.legend(fontsize=8, loc="upper right"); ax2.set_title("Top-down (should look like a ~3x4 m rectangle)")
ax2.grid(alpha=0.3)

ax3 = fig.add_subplot(1, 3, 3)
ax3.scatter(p[:, 0], p[:, 1], s=1, alpha=0.4, **kw)
ax3.set_aspect("equal"); ax3.set_xlabel("X (m)"); ax3.set_ylabel("height rel. camera (m)")
ax3.set_title("Side view (X-height)"); ax3.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(args.output, dpi=150)
print(f"Saved {args.output}")
if args.show:
    plt.show()