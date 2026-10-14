# 3d_generation.py
#
# Generate one 3D point cloud for each valid RGB frame.
# RGB source: dataset/rgb/
# DINOv3 depth: dataset/depth_dino/2025_0813_1543-S3/
# Labels: dataset/valid_frames/labels.csv
#
# Output:
# dataset/pointclouds/2025_0813_1543-S3/
#     image_<img_id>.npz
#
# Coordinate conventions:
#   Camera: [right, down, forward], metres
#   World:  [x_world, height_up, y_world], metres

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

SESSION = "2025_0813_1543-S3"

RGB_ROOT = Path("dataset/rgb")
DEPTH_ROOT = Path("dataset/depth_dino") / SESSION

LABELS_CSV = Path("dataset/valid_frames/labels.csv")
POSE_CSV = Path("dataset/poses/records_20250813-1535.csv")
INTRINSICS_PATH = Path("dataset/calib/intrinsics.json")
SCALE_SHIFT_PATH = Path("dataset/calib/scale_shift.json")

OUTPUT_ROOT = Path("dataset/pointclouds") / SESSION

MIN_DEPTH_M = 0.1
MAX_DEPTH_M = 5.0

# Camera is 10 cm in front of the robot marker.
CAM_FWD_OFFSET_M = 0.10
CAM_HEIGHT_M = 0.0

# Save only one point per occupied voxel.
# Smaller voxels preserve much more geometric detail. Set to 0 to disable downsampling.
VOXEL_SIZE = 0.002


# --------------------------------------------------
# 2. Camera-to-world transformation
# --------------------------------------------------

def camera_to_world(points_camera, x_mm, y_mm, yaw_deg):
    """
    Transform camera points into world coordinates.

    points_camera:
        Nx3 array in camera optical coordinates:
        X = right, Y = down, Z = forward (metres).

    x_mm, y_mm:
        Robot marker position in world coordinates (mm).

    yaw_deg:
        Robot heading in degrees.

    Returns:
        Nx3 array [x_world, height_up, y_world], metres.

    Assumptions:
        - Camera pitch is 0 degrees.
        - Camera is 0.10 m in front of the robot marker.
        - The pose and camera coordinate conventions match
          geometric_projection.py.
    """

    X = points_camera[:, 0]
    Y = points_camera[:, 1]
    Z = points_camera[:, 2]

    # Camera origin -> robot marker origin.
    forward = Z + CAM_FWD_OFFSET_M
    left = -X

    yaw = np.radians(yaw_deg)
    c = np.cos(yaw)
    s = np.sin(yaw)

    dx = c * forward - s * left
    dy = s * forward + c * left

    world_x = x_mm / 1000.0 + dx
    world_height = CAM_HEIGHT_M - Y
    world_y = y_mm / 1000.0 + dy

    return np.column_stack([
        world_x,
        world_height,
        world_y
    ])


# --------------------------------------------------
# 3. Convert depth map into camera-coordinate points
# --------------------------------------------------

def depth_to_pointcloud(depth, rgb, K, scale, shift):
    """
    Convert one depth map and its corresponding RGB image
    into a colored point cloud.

    IMPORTANT:
        The caller must ensure RGB and depth are aligned.

    depth:
        HxW DINOv3 depth prediction.

    rgb:
        HxWx3 RGB image aligned with depth.

    K:
        Camera intrinsics for the RGB resolution.

    scale, shift:
        Calibration parameters: metric_depth = scale * raw + shift.

    Returns:
        points_camera: Nx3, metres
        colors: Nx3, values from 0 to 1
        metric_depth: HxW calibrated depth
    """

    h, w = depth.shape

    if rgb.shape[:2] != (h, w):
        raise ValueError(
            f"RGB/depth size mismatch: RGB={rgb.shape[:2]}, "
            f"depth={depth.shape}. Align the images before projection."
        )

    # Apply the depth calibration.
    metric_depth = scale * depth + shift

    # Camera intrinsics correspond to the RGB image resolution.
    fx = float(K["fx"])
    fy = float(K["fy"])
    cx = float(K["cx"])
    cy = float(K["cy"])

    u, v = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32)
    )

    # Keep only valid depth values.
    valid = (
        np.isfinite(metric_depth)
        & (metric_depth > MIN_DEPTH_M)
        & (metric_depth < MAX_DEPTH_M)
    )

    z = metric_depth[valid]
    u_valid = u[valid]
    v_valid = v[valid]

    # Pinhole camera model.
    x = (u_valid - cx) * z / fx
    y = (v_valid - cy) * z / fy

    points_camera = np.column_stack([x, y, z])

    colors = rgb[valid].astype(np.float32) / 255.0

    return points_camera, colors, metric_depth


# --------------------------------------------------
# 4. Voxel downsampling
# --------------------------------------------------

def voxel_downsample(points, colors, voxel_size):
    """
    Keep one representative point per voxel.

    This reduces file size and makes visualization faster.
    """

    if len(points) == 0 or voxel_size <= 0:
        return points, colors

    voxel_indices = np.floor(
        points / voxel_size
    ).astype(np.int64)

    _, indices = np.unique(
        voxel_indices,
        axis=0,
        return_index=True
    )

    indices.sort()

    return points[indices], colors[indices]


# --------------------------------------------------
# 5. Load pose corresponding to a frame
# --------------------------------------------------

def get_pose(poses, csv_idx):
    """
    Retrieve the robot pose using csv_idx from labels.csv.
    """

    idx = int(csv_idx)

    if idx < 0 or idx >= len(poses):
        return None

    row = poses.iloc[idx]

    required = [
        "self_pose_x",
        "self_pose_y",
        "self_pose_angle"
    ]

    if row[required].isna().any():
        return None

    return row


# --------------------------------------------------
# 6. Process one image
# --------------------------------------------------

def process_frame(lab, poses, K, scale, shift, args):
    """
    Generate and save one point cloud for one valid frame.
    """

    img_id = str(lab["img_id"])

    rgb_path = Path(str(lab["img_path"]))

    depth_path = (
        DEPTH_ROOT / f"image_{img_id}_depth.npy"
    )

    output_path = (
        OUTPUT_ROOT / f"image_{img_id}.npz"
    )

    # Skip frames without a matching depth map.
    if not depth_path.exists():
        return "missing_depth"

    if not rgb_path.exists():
        return "missing_rgb"

    # Load RGB.
    bgr = cv2.imread(str(rgb_path))

    if bgr is None:
        return "unreadable_rgb"

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Load DINOv3 depth.
    depth = np.load(depth_path)

    if depth.ndim != 2:
        print(
            f"[WARNING] Unexpected depth shape for {img_id}: "
            f"{depth.shape}"
        )
        return "invalid_depth_shape"
    print(
    f"Image ID: {img_id} | "
    f"RGB resolution: {rgb.shape[1]}x{rgb.shape[0]} | "
    f"Depth resolution: {depth.shape[1]}x{depth.shape[0]}"
    )
    # DINOv3 depth and RGB currently have different resolutions.
    # Do not silently resize them: pixel correspondence must be
    # verified before point-cloud generation.
    if depth.shape != rgb.shape[:2]:
        if not args.resize_depth:
            print(
                f"[SKIP] {img_id}: RGB={rgb.shape[:2]}, "
                f"depth={depth.shape}. Alignment required."
            )
            return "unaligned"

        print(
            f"[WARNING] Resizing depth for {img_id}: "
            f"{depth.shape} -> {rgb.shape[:2]}. "
            "This is provisional because aspect ratios differ."
        )

        depth = cv2.resize(
            depth.astype(np.float32),
            (rgb.shape[1], rgb.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )

    # Generate colored point cloud in camera coordinates.
    points_camera, colors, metric_depth = depth_to_pointcloud(
        depth, rgb, K, scale, shift
    )

    if len(points_camera) == 0:
        return "no_valid_points"

    # Retrieve the pose associated with this frame.
    pose = get_pose(poses, lab["csv_idx"])

    # Downsample only when requested. A small voxel preserves detail;
    # --voxel 0 disables downsampling and keeps all valid depth samples.
    if args.voxel > 0:
        points_camera, colors = voxel_downsample(
            points_camera,
            colors,
            args.voxel
        )

    # Default: world coordinates unavailable.
    points_world = np.empty((0, 3), dtype=np.float32)

    if pose is not None:
        # The transform uses the robot's marker pose.
        points_world = camera_to_world(
            points_camera,
            pose["self_pose_x"],
            pose["self_pose_y"],
            pose["self_pose_angle"]
        ).astype(np.float32)

    # Store valid target labels for reference.
    robot_labels = []

    for robot in ["S1", "S2"]:
        valid_col = f"{robot}_valid"

        if valid_col in lab.index and int(lab[valid_col]) == 1:
            robot_labels.append(robot)

    # Save one file per image.
    np.savez_compressed(
        output_path,
        points_camera=points_camera.astype(np.float32),
        points_world=points_world,
        colors=colors.astype(np.float32),
        img_id=img_id,
        img_path=str(rgb_path),
        depth_path=str(depth_path),
        pose_available=np.array(pose is not None),
        valid_robots=np.array(robot_labels),
        scale=np.array(scale),
        shift=np.array(shift),
        voxel_size=np.array(args.voxel)
    )

    return "saved"


# --------------------------------------------------
# 7. Main
# --------------------------------------------------

def main():
    global MAX_DEPTH_M
    parser = argparse.ArgumentParser(
        description="Generate one DINOv3 point cloud per valid image."
    )

    parser.add_argument(
        "--resize-depth",
        action="store_true",
        help=(
            "Temporarily resize depth to RGB resolution. "
            "Use only if you accept the unverified alignment assumption."
        )
    )

    parser.add_argument(
        "--max-depth",
        type=float,
        default=MAX_DEPTH_M
    )

    parser.add_argument(
        "--voxel",
        type=float,
        default=VOXEL_SIZE
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=None
    )

    parser.add_argument(
        "--shift",
        type=float,
        default=None
    )

    args = parser.parse_args()

    
    MAX_DEPTH_M = args.max_depth

    # Load camera intrinsics.
    with open(INTRINSICS_PATH) as f:
        K = json.load(f)

    # Load depth calibration.
    with open(SCALE_SHIFT_PATH) as f:
        calibration = json.load(f)

    scale = (
        args.scale
        if args.scale is not None
        else float(calibration["scale"])
    )

    shift = (
        args.shift
        if args.shift is not None
        else float(calibration["shift"])
    )

    print("=" * 60)
    print("DINOv3 per-frame point cloud generation")
    print("=" * 60)
    print("Session:", SESSION)
    print("Depth directory:", DEPTH_ROOT)
    print("Output directory:", OUTPUT_ROOT)
    print("Scale:", scale)
    print("Shift:", shift)
    print("Voxel size:", args.voxel)
    print("Resize depth:", args.resize_depth)

    calibrated_range = calibration.get("d_pred_range")

    if calibrated_range:
        print("Calibration prediction range:", calibrated_range)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Load labels.
    labels = pd.read_csv(
        LABELS_CSV,
        dtype={"img_id": str}
    )

    # Keep frames where at least one target robot is valid.
    valid_mask = (
        (labels["S1_valid"] == 1)
        | (labels["S2_valid"] == 1)
    )

    labels = labels.loc[valid_mask].copy()

    # Load robot poses.
    poses = pd.read_csv(POSE_CSV)

    print("Valid frames in labels.csv:", len(labels))

    counts = {}

    for i, (_, lab) in enumerate(labels.iterrows(), start=1):
        result = process_frame(
            lab, poses, K, scale, shift, args
        )

        counts[result] = counts.get(result, 0) + 1

        if i % 100 == 0 or i == len(labels):
            print(
                f"Progress: {i}/{len(labels)} | "
                f"Result: {result}"
            )

    print("\n" + "=" * 60)
    print("Finished")
    print("=" * 60)

    for key, value in sorted(counts.items()):
        print(f"{key}: {value}")

    print("Output directory:", OUTPUT_ROOT.resolve())


if __name__ == "__main__":
    main()