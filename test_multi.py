import cv2
import json
import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt

from geometric_projection import world_to_camera_frame, project_to_pixel
from pathlib import Path


IMG_DIR = "dataset/rgb"
POSE_CSV = "dataset/poses/records_20250813-1535.csv"
INTRINSICS_PATH = "dataset/calib/intrinsics.json"

OUT_DIR = "test_projection_output"
os.makedirs(OUT_DIR, exist_ok=True)


with open(INTRINSICS_PATH) as f:
    K = json.load(f)


def get_timestamp(img_path):
    path = Path(img_path)
    return path.stem.split("_")[-1]


df = pd.read_csv(POSE_CSV)
img_files = sorted(glob.glob(f"{IMG_DIR}/*.png"))


def find_img_with_id(idz):
    idz = str(idz)

    for filez in img_files:
        if idz in os.path.basename(filez):
            return filez

    return None


# ============================================================
# Find frames where S1 should be visible.
#
# We use S1 for candidate selection, but also store S2's
# position for the same frame.
# ============================================================

candidates = []

for idx in range(len(df)):

    row = df.iloc[idx]

    # Need our own pose
    if (
        pd.isna(row["self_pose_x"])
        or pd.isna(row["self_pose_y"])
        or pd.isna(row["self_pose_angle"])
    ):
        continue

    # Need S1 because S1 is what determines candidate frames
    s1_x_col = "car_S1_pose_x"
    s1_y_col = "car_S1_pose_y"

    if (
        s1_x_col not in row
        or s1_y_col not in row
        or pd.isna(row[s1_x_col])
        or pd.isna(row[s1_y_col])
    ):
        continue

    # Need image
    if pd.isna(row["image"]) or row["image"] == "none":
        continue

    img_id = get_timestamp(row["image"])
    img_path = find_img_with_id(img_id)

    if img_path is None:
        continue

    print(f"\n{img_id} matches {img_path}")

    # --------------------------------------------------------
    # Self pose
    # --------------------------------------------------------

    self_pose = {
        "x": row["self_pose_x"],
        "y": row["self_pose_y"],
        "angle": row["self_pose_angle"],
    }

    # --------------------------------------------------------
    # S1
    # --------------------------------------------------------

    s1_pose = {
        "x": row[s1_x_col],
        "y": row[s1_y_col],
    }

    pcam_s1 = world_to_camera_frame(
        self_pose["x"],
        self_pose["y"],
        0.0,
        self_pose["angle"],
        s1_pose["x"],
        s1_pose["y"],
        0.0,
    )

    forward_s1 = pcam_s1[2]
    lateral_s1 = pcam_s1[0]

    angle_s1 = np.degrees(
        np.arctan2(lateral_s1, forward_s1)
    )

    # --------------------------------------------------------
    # S2
    #
    # S2 is optional. If there is no valid S2 pose for this
    # frame, just store None.
    # --------------------------------------------------------

    pcam_s2 = None
    s2_pose = None

    s2_x_col = "car_S2_pose_x"
    s2_y_col = "car_S2_pose_y"

    if (
        s2_x_col in row
        and s2_y_col in row
        and not pd.isna(row[s2_x_col])
        and not pd.isna(row[s2_y_col])
    ):

        s2_pose = {
            "x": row[s2_x_col],
            "y": row[s2_y_col],
        }

        pcam_s2 = world_to_camera_frame(
            self_pose["x"],
            self_pose["y"],
            0.0,
            self_pose["angle"],
            s2_pose["x"],
            s2_pose["y"],
            0.0,
        )

    # --------------------------------------------------------
    # Print BOTH
    # --------------------------------------------------------

    print(f"[{img_id}] Self: {self_pose}")

    print(
        f"    S1 world pose: {s1_pose}\n"
        f"    S1 pcam:       {pcam_s1}\n"
        f"    S1 forward/lateral: "
        f"{forward_s1:.4f}, {lateral_s1:.4f}"
    )

    if pcam_s2 is not None:

        forward_s2 = pcam_s2[2]
        lateral_s2 = pcam_s2[0]

        print(
            f"    S2 world pose: {s2_pose}\n"
            f"    S2 pcam:       {pcam_s2}\n"
            f"    S2 forward/lateral: "
            f"{forward_s2:.4f}, {lateral_s2:.4f}"
        )

    else:
        print("    S2: no valid pose")

    # --------------------------------------------------------
    # Candidate selection ONLY based on S1
    # --------------------------------------------------------

    if forward_s1 > 0.1 and abs(angle_s1) <= 25:

        candidates.append(
            (
                idx,
                img_path,
                pcam_s1,
                pcam_s2,
            )
        )

    print(f"Found {len(candidates)} frames with S1 in front of camera")


# ============================================================
# Plot all candidate frames, 6 per grid
# ============================================================

candidates.sort(key=lambda c: c[0])

GRID_SIZE = 6


for grid_idx, start_idx in enumerate(
    range(0, len(candidates), GRID_SIZE)
):

    samples = candidates[
        start_idx:start_idx + GRID_SIZE
    ]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 10),
    )

    for ax, (
        csv_idx,
        img_path,
        pcam_s1,
        pcam_s2,
    ) in zip(axes.flat, samples):

        img = cv2.imread(img_path)

        if img is None:
            ax.axis("off")
            continue

        timestamp = get_timestamp(img_path)

        # ====================================================
        # S1 projection -- GREEN
        # ====================================================

        result_s1 = project_to_pixel(
            pcam_s1,
            K["fx"],
            K["fy"],
            K["cx"],
            K["cy"],
        )

        forward_s1 = pcam_s1[2]
        lateral_s1 = pcam_s1[0]

        # ====================================================
        # S2 projection -- RED
        # ====================================================

        result_s2 = None

        if pcam_s2 is not None:

            result_s2 = project_to_pixel(
                pcam_s2,
                K["fx"],
                K["fy"],
                K["cx"],
                K["cy"],
            )

            forward_s2 = pcam_s2[2]
            lateral_s2 = pcam_s2[0]

        # ====================================================
        # PRINT BOTH
        # ====================================================

        print(f"\n[{timestamp}]")

        print(
            f"    S1 GREEN:"
            f" forward={forward_s1:.4f},"
            f" lateral={lateral_s1:.4f},"
            f" pixel={result_s1}"
        )

        if pcam_s2 is not None:

            print(
                f"    S2 RED:  "
                f" forward={forward_s2:.4f},"
                f" lateral={lateral_s2:.4f},"
                f" pixel={result_s2}"
            )

        else:
            print("    S2 RED: no pose")

        # ====================================================
        # Draw S1 -- GREEN
        # ====================================================

        if result_s1 is not None:

            u, v = result_s1

            cv2.circle(
                img,
                (int(u), int(v)),
                10,
                (0, 255, 0),
                2,
            )

            cv2.line(
                img,
                (int(u) - 15, int(v)),
                (int(u) + 15, int(v)),
                (0, 255, 0),
                1,
            )

            cv2.line(
                img,
                (int(u), int(v) - 15),
                (int(u), int(v) + 15),
                (0, 255, 0),
                1,
            )

            cv2.putText(
                img,
                "S1",
                (int(u) + 12, int(v) - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        # ====================================================
        # Draw S2 -- RED
        #
        # OpenCV uses BGR, so red = (0, 0, 255)
        # ====================================================

        if result_s2 is not None:

            u, v = result_s2

            cv2.circle(
                img,
                (int(u), int(v)),
                10,
                (0, 0, 255),
                2,
            )

            cv2.line(
                img,
                (int(u) - 15, int(v)),
                (int(u) + 15, int(v)),
                (0, 0, 255),
                1,
            )

            cv2.line(
                img,
                (int(u), int(v) - 15),
                (int(u), int(v) + 15),
                (0, 0, 255),
                1,
            )

            cv2.putText(
                img,
                "S2",
                (int(u) + 12, int(v) - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

        # ====================================================
        # Plot
        # ====================================================

        ax.imshow(
            cv2.cvtColor(
                img,
                cv2.COLOR_BGR2RGB,
            )
        )

        if pcam_s2 is not None:

            ax.set_title(
                f"{timestamp}\n"
                f"S1 green: lat={lateral_s1:.3f}, "
                f"fwd={forward_s1:.3f}\n"
                f"S2 red: lat={lateral_s2:.3f}, "
                f"fwd={forward_s2:.3f}"
            )

        else:

            ax.set_title(
                f"{timestamp}\n"
                f"S1 green: lat={lateral_s1:.3f}, "
                f"fwd={forward_s1:.3f}\n"
                f"S2: no pose"
            )

        ax.axis("off")

    # Hide unused axes in final grid
    for ax in axes.flat[len(samples):]:
        ax.axis("off")

    plt.suptitle(
        "S1 vs S2 projection check\n"
        "GREEN = predicted S1 marker, RED = predicted S2 marker",
        fontsize=13,
    )

    plt.tight_layout()

    # Different filename from S1-only experiment
    out_path = (
        f"{OUT_DIR}/"
        f"projection_check_s1_s2_grid_{grid_idx}.png"
    )

    plt.savefig(
        out_path,
        dpi=140,
    )

    plt.close(fig)

    print(f"Saved {out_path}")