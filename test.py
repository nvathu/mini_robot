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
TARGET_ROBOT = "S1" # I see just working with S1
OUT_DIR = "test_projection_output"
os.makedirs(OUT_DIR, exist_ok=True)

with open(INTRINSICS_PATH) as f:
    K = json.load(f)


def get_timestamp(img_path):
    path = Path(img_path)
    filename = path.name
    return path.stem.split("_")[-1]



df = pd.read_csv(POSE_CSV)
img_files = sorted(glob.glob(f"{IMG_DIR}/*.png"))


def find_img_with_id(idz):
    idz = str(idz)

    for filez in img_files:
        if f"{idz}" in os.path.basename(filez):
            return filez

    return None

candidates = []
for idx in range(len(df)):
    row = df.iloc[idx]
    if pd.isna(row['self_pose_x']) or pd.isna(row['self_pose_y']) or pd.isna(row['self_pose_angle']):
        continue
    tx = f'car_{TARGET_ROBOT}_pose_x'
    if tx not in row or pd.isna(row[tx]):
        continue
    if pd.isna(row["image"]) or row["image"] == "none": # Image skip condition
        continue
    img_id = get_timestamp(row["image"]) # Get the image id!
    img_path = find_img_with_id(img_id)    
    if img_path is None:
        continue
    print(f"{img_id} matches {img_path}")
    self_pose = {'x': row['self_pose_x'], 'y': row['self_pose_y'], 'angle': row['self_pose_angle']}
    target_pose = {'x': row[tx], 'y': row[f'car_{TARGET_ROBOT}_pose_y']}
    #forward, lateral = world_to_camera_frame(self_pose['x'], self_pose['y'], self_pose['angle'],
    #                                           target_pose['x'], target_pose['y'])
    pcam = world_to_camera_frame(self_pose['x'], self_pose['y'], 0.0, self_pose['angle'],
                                            target_pose['x'], target_pose['y'], 0.0)
    print(f'PCAM PCAM {img_id}: {pcam}')
    forward = pcam[2]
    lateral = pcam[0] 

    print(f"[{img_id}] Self {self_pose}, Target: {target_pose} \n {(forward, lateral)}")
    angle = np.degrees(np.arctan2(lateral, forward))
    if forward > 0.1 and abs(angle) <= 25: # 10 cm's ahead..
        candidates.append((idx, img_path,pcam))
    print(f"Found {len(candidates)} frame has {TARGET_ROBOT} in front of camera")

candidates.sort(key=lambda c: c[0])

GRID_SIZE = 6

for grid_idx, start_idx in enumerate(range(0, len(candidates), GRID_SIZE)):

    samples = candidates[start_idx:start_idx + GRID_SIZE]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for ax, (csv_idx, img_path, pcam) in zip(axes.flat, samples):

        img = cv2.imread(img_path)

        if img is None:
            ax.axis('off')
            continue

        forward = pcam[2]
        lateral = pcam[0]

        result = project_to_pixel(
            pcam,
            K['fx'],
            K['fy'],
            K['cx'],
            K['cy']
        )

        print(
            f"{get_timestamp(img_path)}: "
            f"{forward, lateral} / {result}"
        )

        if result is not None:
            u, v = result

            cv2.circle(
                img,
                (int(u), int(v)),
                10,
                (0, 255, 0),
                2
            )

            cv2.line(
                img,
                (int(u) - 15, int(v)),
                (int(u) + 15, int(v)),
                (0, 255, 0),
                1
            )

            cv2.line(
                img,
                (int(u), int(v) - 15),
                (int(u), int(v) + 15),
                (0, 255, 0),
                1
            )

        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        ax.set_title(
            f"{get_timestamp(img_path)}\n"
            f"lateral={lateral:.4f}m  "
            f"forward={forward:.4f}m\n"
            f"green dot = predicted S1 position"
        )

        ax.axis('off')

    # Hide unused axes in the final grid
    for ax in axes.flat[len(samples):]:
        ax.axis('off')

    plt.suptitle(
        "Projection sanity check: does the green dot land on the real robot?\n"
        "Check left/right (mirror) errors especially on frames with large |lateral|",
        fontsize=13
    )

    plt.tight_layout()

    out_path = f"{OUT_DIR}/projection_check_grid_{grid_idx}.png"

    plt.savefig(out_path, dpi=140)
    plt.close(fig)

    print(f"Saved {out_path}")


# candidates.sort(key=lambda c: c[0])
# pick_idx = np.linspace(0, len(candidates) - 1, 6).astype(int)
# samples = [candidates[i] for i in pick_idx]

# fig, axes = plt.subplots(2, 3, figsize=(15, 10))
# for ax, (csv_idx, img_path, pcam) in zip(axes.flat, samples):
#     img = cv2.imread(img_path)
#     if img is None:
#         ax.axis('off')
#         continue
#     forward = pcam[2]
#     lateral = pcam[0] 
#     result = project_to_pixel(pcam, K['fx'], K['fy'], K['cx'], K['cy'])
 
#     print(f"{get_timestamp(img_path)}:  {forward, lateral} / {result}")
#     if result is not None:
#         u, v = result
#         cv2.circle(img, (int(u), int(v)), 10, (0, 255, 0), 2)
#         cv2.line(img, (int(u) - 15, int(v)), (int(u) + 15, int(v)), (0, 255, 0), 1)
#         cv2.line(img, (int(u), int(v) - 15), (int(u), int(v) + 15), (0, 255, 0), 1)

#     ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     ax.set_title(f"{get_timestamp(img_path)} \n"
#                  f"lateral={lateral:.4f}m  forward={forward:.4f}m\n"
#                  f"green dot = predicted S1 position")
#     ax.axis('off')

# plt.suptitle("Projection sanity check: does the green dot land on the real robot?\n"
#              "Check left/right (mirror) errors especially on frames with large |lateral|",
#              fontsize=13)
# plt.tight_layout()
# out_path = f"{OUT_DIR}/projection_check_grid.png"
# plt.savefig(out_path, dpi=140)
# print(f"Saved {out_path}")
