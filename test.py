import cv2
import json
import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt

IMG_DIR = "dataset/rgb"
POSE_CSV = "dataset/poses/records_20250813-1535.csv"
INTRINSICS_PATH = "dataset/calib/intrinsics.json"
TARGET_ROBOT = "S1"
OUT_DIR = "test_projection_output"
os.makedirs(OUT_DIR, exist_ok=True)

with open(INTRINSICS_PATH) as f:
    K = json.load(f)

def world_to_camera_frame(self_x, self_y, self_angle, target_x, target_y, angle_sign=1):
    self_angle = angle_sign * np.radians(self_angle)
    dx = target_x - self_x
    dy = target_y - self_y
    forward = np.cos(-self_angle) * dx - np.sin(-self_angle) * dy
    lateral = np.sin(-self_angle) * dx + np.cos(-self_angle) * dy
    return forward, lateral

def project_to_pixel(forward_mm, lateral_mm, fx, fy, cx, cy, height_mm=0.0, pitch_deg=0.0):
    forward_m = forward_mm / 1000.0
    lateral_m = lateral_mm / 1000.0
    height_m = height_mm / 1000.0
    if forward_m <= 0.02:
        return None
    u = cx + (lateral_m * fx / forward_m)
    if pitch_deg == 0.0:
        v = cy + (height_m * fy / forward_m)
    else:
        theta = np.radians(pitch_deg)
        Y_rot = height_m * np.cos(theta) - forward_m * np.sin(theta)
        Z_rot = height_m * np.sin(theta) + forward_m * np.cos(theta)
        if Z_rot <= 0.02:
            return None
        v = cy + (Y_rot * fy / Z_rot)
    return u, v

df = pd.read_csv(POSE_CSV)
img_files = sorted(glob.glob(f"{IMG_DIR}/*.png"))

def get_ts(f):
    return int(os.path.splitext(os.path.basename(f))[0].replace("image_", "")) / 1e9
img_ts = np.array([get_ts(f) for f in img_files])

def find_img(ts, max_gap=0.03):
    i = np.argmin(np.abs(img_ts - ts))
    return img_files[i] if abs(img_ts[i] - ts) <= max_gap else None

candidates = []
for idx in range(len(df)):
    row = df.iloc[idx]
    if pd.isna(row['self_pose_x']) or pd.isna(row['self_pose_y']) or pd.isna(row['self_pose_angle']):
        continue
    tx = f'car_{TARGET_ROBOT}_pose_x'
    if tx not in row or pd.isna(row[tx]):
        continue
    ts = row['pose_timestamp'] / 1e9
    img_path = find_img(ts)
    if img_path is None:
        continue
    self_pose = {'x': row['self_pose_x'], 'y': row['self_pose_y'], 'angle': row['self_pose_angle']}
    target_pose = {'x': row[tx], 'y': row[f'car_{TARGET_ROBOT}_pose_y']}
    forward, lateral = world_to_camera_frame(self_pose['x'], self_pose['y'], self_pose['angle'],
                                                target_pose['x'], target_pose['y'])
    if forward > 100:
        candidates.append((idx, img_path, forward, lateral))

print(f"Found {len(candidates)} frame has {TARGET_ROBOT} in fornt of camera")

candidates.sort(key=lambda c: c[3])
pick_idx = np.linspace(0, len(candidates) - 1, 6).astype(int)
samples = [candidates[i] for i in pick_idx]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for ax, (csv_idx, img_path, forward, lateral) in zip(axes.flat, samples):
    img = cv2.imread(img_path)
    if img is None:
        ax.axis('off')
        continue

    result = project_to_pixel(forward, lateral, K['fx'], K['fy'], K['cx'], K['cy'])
    if result is not None:
        u, v = result
        cv2.circle(img, (int(u), int(v)), 10, (0, 255, 0), 2)
        cv2.line(img, (int(u) - 15, int(v)), (int(u) + 15, int(v)), (0, 255, 0), 1)
        cv2.line(img, (int(u), int(v) - 15), (int(u), int(v) + 15), (0, 255, 0), 1)

    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title(f"lateral={lateral:.0f}mm  forward={forward:.0f}mm\n"
                 f"green dot = predicted S1 position")
    ax.axis('off')

plt.suptitle("Projection sanity check: does the green dot land on the real robot?\n"
             "Check left/right (mirror) errors especially on frames with large |lateral|",
             fontsize=13)
plt.tight_layout()
out_path = f"{OUT_DIR}/projection_check_grid.png"
plt.savefig(out_path, dpi=140)
print(f"Saved {out_path}")
