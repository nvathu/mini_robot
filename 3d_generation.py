import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from geometric_projection import world_to_camera_frame


# ============================================================
# CONFIGURATION
# ============================================================

LABELS_CSV = Path("dataset/valid_frames/labels.csv")
POSE_CSV = Path("dataset/poses/records_20250813-1535.csv")
INTRINSICS_PATH = Path("dataset/calib/intrinsics.json")
SCALE_SHIFT_PATH = Path("dataset/calib/scale_shift.json")
WEIGHTS_PATH = Path("weights/best_model.pth")

OUTPUT_DIR = Path("dataset/pointclouds/2025_0813_1543-S3")
OUTPUT_PATH = Path("dataset/pointclouds/room_pointcloud.npz")

ROBOTS = ["S1", "S2"]

# Resolution used as the input to the depth model.
NET_SIZE = 180

# Approximately five times as many depth samples as NET_SIZE x NET_SIZE.
# 402 x 402 is approximately five times 180 x 180.
UPSAMPLED_SIZE = 402

MIN_DEPTH_M = 0.1
MAX_DEPTH_M = 4.5

# Voxel size for the merged room point cloud.
VOXEL_SIZE = 0.002

CAM_FWD_OFFSET_M = 0.10
CAM_HEIGHT_M = 0.0


# ============================================================
# COORDINATE TRANSFORMATION
# ============================================================

def camera_to_world(
    p_cam,
    x_mm,
    y_mm,
    yaw_deg,
    cam_height_m=CAM_HEIGHT_M
):
    """
    Transform camera optical coordinates into world coordinates.

    Camera coordinates:
        X = right
        Y = down
        Z = forward

    Output coordinates:
        X = world X
        Y = height upward
        Z = world Y

    All output coordinates are in meters.
    """

    X = p_cam[:, 0]
    Y = p_cam[:, 1]
    Z = p_cam[:, 2]

    forward = Z + CAM_FWD_OFFSET_M
    left = -X

    yaw = np.radians(yaw_deg)
    c = np.cos(yaw)
    s = np.sin(yaw)

    dx = c * forward - s * left
    dy = s * forward + c * left

    return np.stack(
        [
            x_mm / 1000.0 + dx,
            cam_height_m - Y,
            y_mm / 1000.0 + dy
        ],
        axis=1
    )


def self_check():
    """Check whether the camera-to-world transform is consistent."""

    rng = np.random.default_rng(0)

    for i in range(100):
        sx, sy = rng.uniform(-1500, 1500, 2)
        yaw = rng.uniform(-180, 180)

        ox = rng.uniform(-1500, 1500)
        oy = rng.uniform(-1500, 1500)
        oz = rng.uniform(-300, 300)

        p_cam = np.asarray(
            world_to_camera_frame(
                sx, sy, 0.0, yaw, ox, oy, oz
            )
        )[None, :]

        world = camera_to_world(
            p_cam,
            sx,
            sy,
            yaw,
            cam_height_m=0.0
        )[0]

        expected = np.array(
            [ox / 1000.0, oz / 1000.0, oy / 1000.0]
        )

        if not np.allclose(world, expected, atol=1e-6):
            print(f"Test failed at iteration {i}")
            print("Expected:", expected)
            print("Recovered:", world)
            print("Difference:", world - expected)

            raise AssertionError(
                "camera_to_world is inconsistent with "
                "world_to_camera_frame"
            )

    print("self_check OK")


# ============================================================
# CAMERA RAYS
# ============================================================

def make_rays(K, image_width, image_height, size):
    """
    Create normalized camera rays at the specified output resolution.

    Intrinsics are scaled from the original image dimensions.
    """

    fx = K["fx"] * size / image_width
    fy = K["fy"] * size / image_height

    cx = K["cx"] * size / image_width
    cy = K["cy"] * size / image_height

    u, v = np.meshgrid(
        np.arange(size, dtype=np.float32),
        np.arange(size, dtype=np.float32)
    )

    ray_x = (u - cx) / fx
    ray_y = (v - cy) / fy

    return ray_x, ray_y


# ============================================================
# VOXEL DOWNSAMPLING
# ============================================================

def voxel_downsample(points, colors, voxel):
    """
    Keep one point per occupied voxel.

    The color of each retained point is taken from the same original
    sample, preserving point-color correspondence.
    """

    if len(points) == 0 or voxel <= 0:
        return points, colors

    voxel_indices = np.floor(
        points / voxel
    ).astype(np.int64)

    _, indices = np.unique(
        voxel_indices,
        axis=0,
        return_index=True
    )

    indices.sort()

    return points[indices], colors[indices]


# ============================================================
# DEPTH TO POINT CLOUD
# ============================================================

def depth_to_pointcloud(depth, rgb, ray_x, ray_y, min_depth, max_depth):
    """
    Convert a depth map into a colored camera-frame point cloud.

    Parameters
    ----------
    depth : ndarray, shape (H, W)
        Metric depth in meters.
    rgb : ndarray, shape (H, W, 3)
        RGB image aligned with the depth map.
    ray_x, ray_y : ndarray
        Normalized camera ray coordinates.
    min_depth, max_depth : float
        Valid depth range in meters.

    Returns
    -------
    points_camera : ndarray, shape (N, 3)
    colors : ndarray, shape (N, 3), values in [0, 1]
    """

    valid = (
        np.isfinite(depth)
        & (depth > min_depth)
        & (depth < max_depth)
    )

    z = depth[valid]

    points_camera = np.stack(
        [
            ray_x[valid] * z,
            ray_y[valid] * z,
            z
        ],
        axis=1
    ).astype(np.float32)

    # Sample colors from the matching RGB pixels.
    colors = rgb[valid].astype(np.float32) / 255.0

    return points_camera, colors


# ============================================================
# LOAD MODEL
# ============================================================

def load_depth_model(device):
    """Load the trained ResNetDepth model."""

    import torch
    from model import ResNetDepth

    model = ResNetDepth().to(device)

    checkpoint = torch.load(
        WEIGHTS_PATH,
        map_location=device
    )

    model.load_state_dict(checkpoint)
    model.eval()

    return model


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    import torch

    parser = argparse.ArgumentParser(
        description="Generate dense RGB-colored point clouds."
    )

    parser.add_argument(
        "--max-depth",
        type=float,
        default=MAX_DEPTH_M,
        help="Maximum accepted depth in meters."
    )

    parser.add_argument(
        "--voxel",
        type=float,
        default=VOXEL_SIZE,
        help="Voxel size for the merged point cloud in meters."
    )

    parser.add_argument(
        "--upsampled-size",
        type=int,
        default=UPSAMPLED_SIZE,
        help="Depth sampling resolution after interpolation."
    )

    args = parser.parse_args()

    if args.upsampled_size < NET_SIZE:
        raise ValueError(
            f"--upsampled-size must be at least {NET_SIZE}"
        )

    if args.voxel < 0:
        raise ValueError("--voxel cannot be negative.")

    self_check()

    # Load camera calibration.
    with open(INTRINSICS_PATH, encoding="utf-8") as file:
        K = json.load(file)

    with open(SCALE_SHIFT_PATH, encoding="utf-8") as file:
        scale_shift = json.load(file)

    scale = float(scale_shift["scale"])
    shift = float(scale_shift["shift"])

    calibrated_low, calibrated_high = scale_shift["d_true_range"]

    print(
        f"Calibration: {scale_shift['selected']} | "
        f"scale={scale:.6f}, shift={shift:.6f}"
    )

    print(
        f"Calibration depth range: "
        f"{calibrated_low:.2f}-{calibrated_high:.2f} m"
    )

    print(
        f"Original sampling: {NET_SIZE} x {NET_SIZE} "
        f"= {NET_SIZE ** 2:,} pixels"
    )

    print(
        f"Upsampled sampling: "
        f"{args.upsampled_size} x {args.upsampled_size} "
        f"= {args.upsampled_size ** 2:,} pixels"
    )

    print(
        f"Sampling factor: "
        f"{args.upsampled_size ** 2 / NET_SIZE ** 2:.2f}x"
    )

    # Initialize the model.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")

    model = load_depth_model(device)

    # Load valid frame labels and poses.
    labels = pd.read_csv(
        LABELS_CSV,
        dtype={"img_id": str}
    )

    labels = labels[
        (labels["S1_valid"] == 1)
        | (labels["S2_valid"] == 1)
    ]

    poses = pd.read_csv(POSE_CSV)

    print(f"Validated frames: {len(labels)}")

    # Prepare output directories.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Store sampled points for the merged room point cloud.
    scene_points = []
    scene_colors = []

    robot_points = []
    self_points = []

    processed_frames = 0
    skipped_frames = 0
    total_frame_points = 0
    depth_values = []

    ray_x = None
    ray_y = None

    output_size = args.upsampled_size

    with torch.no_grad():

        for row_number, (_, label) in enumerate(
            labels.iterrows(),
            start=1
        ):
            img_path = Path(label["img_path"])

            # Resolve image paths relative to the project directory.
            if not img_path.is_absolute():
                img_path = Path.cwd() / img_path

            if not img_path.exists():
                print(f"[SKIP] Image not found: {img_path}")
                skipped_frames += 1
                continue

            # Read the original image.
            image_bgr = cv2.imread(str(img_path))

            if image_bgr is None:
                print(f"[SKIP] Cannot read: {img_path}")
                skipped_frames += 1
                continue

            image_height, image_width = image_bgr.shape[:2]

            # Match the preprocessing used during model training.
            model_rgb = cv2.cvtColor(
                cv2.resize(
                    image_bgr,
                    (NET_SIZE, NET_SIZE),
                    interpolation=cv2.INTER_AREA
                ),
                cv2.COLOR_BGR2RGB
            )

            model_input = (
                torch.from_numpy(
                    model_rgb.astype(np.float32) / 255.0
                )
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(device)
            )

            # Predict depth at the model's native output resolution.
            raw_depth = model(model_input).squeeze().cpu().numpy()

            if raw_depth.ndim != 2:
                print(
                    f"[SKIP] Unexpected depth shape: "
                    f"{raw_depth.shape}"
                )
                skipped_frames += 1
                continue

            # Upsample depth to obtain a denser sampling grid.
            dense_raw_depth = cv2.resize(
                raw_depth.astype(np.float32),
                (output_size, output_size),
                interpolation=cv2.INTER_LINEAR
            )

            # Resize the original RGB image to the exact same grid.
            # These RGB pixels supply the colors for the 3D points.
            dense_rgb = cv2.cvtColor(
                cv2.resize(
                    image_bgr,
                    (output_size, output_size),
                    interpolation=cv2.INTER_LINEAR
                ),
                cv2.COLOR_BGR2RGB
            )

            # Convert predicted depth into metric units.
            dense_depth = scale * dense_raw_depth + shift

            # Generate camera rays for the dense sampling grid.
            if ray_x is None:
                ray_x, ray_y = make_rays(
                    K,
                    image_width,
                    image_height,
                    output_size
                )

            # Project depth pixels into 3D camera coordinates.
            points_camera, colors = depth_to_pointcloud(
                dense_depth,
                dense_rgb,
                ray_x,
                ray_y,
                MIN_DEPTH_M,
                args.max_depth
            )

            if len(points_camera) == 0:
                print(f"[SKIP] No valid depth: {img_path.name}")
                skipped_frames += 1
                continue

            # Retrieve the corresponding robot pose.
            csv_idx = int(label["csv_idx"])

            if csv_idx < 0 or csv_idx >= len(poses):
                print(f"[SKIP] Invalid pose index: {csv_idx}")
                skipped_frames += 1
                continue

            pose = poses.iloc[csv_idx]

            pose_columns = [
                "self_pose_x",
                "self_pose_y",
                "self_pose_angle"
            ]

            if pose[pose_columns].isna().any():
                print(f"[SKIP] Invalid robot pose: {img_path.name}")
                skipped_frames += 1
                continue

            sx = float(pose["self_pose_x"])
            sy = float(pose["self_pose_y"])
            yaw = float(pose["self_pose_angle"])

            # Transform points into the world coordinate system.
            points_world = camera_to_world(
                points_camera,
                sx,
                sy,
                yaw
            ).astype(np.float32)

            # Keep the original per-pixel colors for every point.
            colors = colors.astype(np.float32)

            # Use the image ID to identify each individual point cloud.
            img_id = str(label["img_id"])

            # Save a separate point cloud for this image.
            # No voxel downsampling is applied to individual frames.
            frame_output = OUTPUT_DIR / f"image_{img_id}_pointcloud.npz"

            np.savez_compressed(
                frame_output,
                points_camera=points_camera,
                points_world=points_world,
                colors=colors,
                img_id=np.asarray(img_id),
                img_path=np.asarray(str(img_path)),
                pose_available=np.asarray(True),
                scale=np.asarray(scale),
                shift=np.asarray(shift),
                sampling_size=np.asarray(output_size)
            )

            # Downsample this frame before adding it to the merged cloud.
            # This helps reduce peak memory consumption.
            if args.voxel > 0:
                merged_points, merged_colors = voxel_downsample(
                    points_world,
                    colors,
                    args.voxel
                )
            else:
                merged_points, merged_colors = points_world, colors

            scene_points.append(merged_points)
            scene_colors.append(merged_colors)

            total_frame_points += len(points_camera)
            depth_values.append(dense_depth[
                np.isfinite(dense_depth)
                & (dense_depth > MIN_DEPTH_M)
                & (dense_depth < args.max_depth)
            ])

            self_points.append([
                sx / 1000.0,
                0.0,
                sy / 1000.0
            ])

            # Add valid target robot positions for visual reference.
            for robot in ROBOTS:

                valid_column = f"{robot}_valid"

                if (
                    valid_column in label.index
                    and int(label[valid_column]) == 1
                ):
                    robot_x = f"car_{robot}_pose_x"
                    robot_y = f"car_{robot}_pose_y"

                    if (
                        robot_x in pose.index
                        and robot_y in pose.index
                        and pd.notna(pose[robot_x])
                        and pd.notna(pose[robot_y])
                    ):
                        robot_points.append([
                            float(pose[robot_x]) / 1000.0,
                            0.0,
                            float(pose[robot_y]) / 1000.0
                        ])

            processed_frames += 1

            if processed_frames % 25 == 0:
                print(
                    f"Processed {processed_frames} frames; "
                    f"latest cloud: {len(points_camera):,} points"
                )

    # Stop if no frames were successfully processed.
    if not scene_points:
        raise SystemExit(
            "No frames processed. Check image paths, labels, poses, "
            "and depth calibration."
        )

    # Merge the frame-level point clouds.
    points = np.concatenate(scene_points, axis=0)
    colors = np.concatenate(scene_colors, axis=0)

    # Apply a final voxel filter to remove duplicate spatial samples.
    before_downsample = len(points)

    if args.voxel > 0:
        points, colors = voxel_downsample(
            points,
            colors,
            args.voxel
        )

    print(f"\nProcessed frames: {processed_frames}")
    print(f"Skipped frames: {skipped_frames}")
    print(f"Total individual-frame points: {total_frame_points:,}")
    print(f"Merged points before final downsampling: {before_downsample:,}")
    print(f"Merged points after downsampling: {len(points):,}")

    if depth_values:
        valid_depths = np.concatenate(depth_values)

        if len(valid_depths) > 0:
            print(
                "Depth percentiles p50/p95/max: "
                f"{np.percentile(valid_depths, 50):.2f} / "
                f"{np.percentile(valid_depths, 95):.2f} / "
                f"{valid_depths.max():.2f} m"
            )

            extrapolated_fraction = np.mean(
                valid_depths > calibrated_high
            )

            print(
                f"{100 * extrapolated_fraction:.1f}% of valid depth "
                "samples exceed the upper calibration range."
            )

    # Save the merged room point cloud.
    robots = np.asarray(
        robot_points,
        dtype=np.float32
    ).reshape(-1, 3)

    np.savez_compressed(
        OUTPUT_PATH,
        scene_points=points.astype(np.float32),
        scene_colors=colors.astype(np.float32),
        robot_points=robots,
        robot_colors=np.tile(
            np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
            (len(robots), 1)
        ),
        self_points=np.asarray(
            self_points,
            dtype=np.float32
        ).reshape(-1, 3)
    )

    print(f"\nMerged point cloud saved to: {OUTPUT_PATH}")
    print(f"Individual point clouds saved to: {OUTPUT_DIR}")
    print("Coordinate layout: [x_world, height_up, y_world], in meters.")
    print("Point colors are stored as RGB values in the [0, 1] range.")


if __name__ == "__main__":
    main()