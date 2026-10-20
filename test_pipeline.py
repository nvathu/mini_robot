import torch
import cv2
import json
import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
from model import ResNetDepth
from geometric_projection import world_to_camera_frame, project_to_pixel

IMG_DIR = "dataset/rgb"
POSE_CSV = "dataset/poses/records_20250813-1535.csv"
INTRINSICS_PATH = "dataset/calib/intrinsics.json"
GLOBAL_SCALE_PATH = "dataset/calib/global_scale.json"
WEIGHTS_PATH = "weights/best_model.pth"
TARGET_ROBOTS = ["S1", "S2"]
MARGIN_RATIO = 0.15
ROBOT_SIZE_M = 0.15
N_FRAMES = 6
OUT_DIR = "test_pipeline_output"
os.makedirs(OUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

with open(INTRINSICS_PATH) as f:
    K = json.load(f)
with open(GLOBAL_SCALE_PATH) as f:
    GS = json.load(f)
GLOBAL_SCALE = GS["global_scale"]

model = ResNetDepth().to(device)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
model.eval()

df = pd.read_csv(POSE_CSV)
img_files = sorted(glob.glob(f"{IMG_DIR}/*.png"))

def get_ts(f):
    return int(os.path.splitext(os.path.basename(f))[0].replace("image_", "")) / 1e9
img_ts = np.array([get_ts(f) for f in img_files])

def find_img(ts, max_gap=0.03):
    i = np.argmin(np.abs(img_ts - ts))
    return img_files[i] if abs(img_ts[i] - ts) <= max_gap else None

@torch.no_grad()
def predict_depth(img_path):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_small = cv2.resize(img_rgb, (180, 180))
    tensor = torch.tensor(img_small / 255.).permute(2, 0, 1).unsqueeze(0).float().to(device)
    pred = model(tensor).squeeze().cpu().numpy()
    return pred, img.shape[:2], img_small

def mask_dynamic_robots(depth_map, row, self_pose, w0, h0):
    masked = depth_map.copy()
    for tgt in TARGET_ROBOTS:
        tx = f'car_{tgt}_pose_x'
        if tx not in row or pd.isna(row[tx]):
            continue
        target_pose = {'x': row[tx], 'y': row[f'car_{tgt}_pose_y']}
        
        p_cam = world_to_camera_frame(
            self_pose['x'],    
            self_pose['y'],     
            0,                 
            self_pose['angle'], 
            target_pose['x'],   
            target_pose['y'],   
            0                   
        )
        

        forward_mm = p_cam[2] * 1000.0 
        if forward_mm <= 50:
            continue
            

        result = project_to_pixel(p_cam, K['fx'], K['fy'], K['cx'], K['cy'])
        if result is None:
            continue
            
        u, v = result
        u_s, v_s = u * 180 / w0, v * 180 / h0
        margin = MARGIN_RATIO * 180
        if not (margin <= u_s < 180 - margin):
            continue
            

        r_px = (K['fx'] * ROBOT_SIZE_M / p_cam[2]) * 180 / w0
        cv2.rectangle(masked, (int(u_s - r_px), 0), (int(u_s + r_px), 180), 0, thickness=-1)
    return masked

def unproject(depth_map, fx, fy, cx, cy, rgb_image, max_depth=3.0):
    H, W = depth_map.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    valid = (depth_map > 0) & (depth_map < max_depth)
    z = depth_map[valid]
    x = (u[valid] - cx) * z / fx
    y = (v[valid] - cy) * z / fy
    points = np.stack([x, y, z], axis=1)
    colors = rgb_image[valid] / 255.0
    return points, colors

def transform_to_world(points, self_pose):
    if len(points) == 0:
        return points
    angle = np.radians(self_pose['angle'])
    c, s = np.cos(angle), np.sin(angle)
    X, Z = points[:, 0], points[:, 2]
    X_w = c * X + s * Z
    Z_w = -s * X + c * Z
    x_f = X_w + self_pose['x'] / 1000.0
    z_f = Z_w + self_pose['y'] / 1000.0
    return np.stack([x_f, points[:, 1], z_f], axis=1)


valid_rows = []
for idx in range(len(df)):
    row = df.iloc[idx]
    if pd.isna(row['self_pose_x']) or pd.isna(row['self_pose_y']) or pd.isna(row['self_pose_angle']):
        continue
    ts = row['pose_timestamp'] / 1e9
    if find_img(ts) is not None:
        valid_rows.append(idx)

step = max(1, len(valid_rows) // N_FRAMES)
selected_rows = valid_rows[::step][:N_FRAMES]
print(f"Testing {len(selected_rows)} frames")

all_points, all_colors = [], []
for csv_idx in selected_rows:
    row = df.iloc[csv_idx]
    ts = row['pose_timestamp'] / 1e9
    img_path = find_img(ts)
    self_pose = {'x': row['self_pose_x'], 'y': row['self_pose_y'], 'angle': row['self_pose_angle']}

    raw_depth, orig_shape, img_small = predict_depth(img_path)
    h0, w0 = orig_shape
    raw_depth = mask_dynamic_robots(raw_depth, row, self_pose, w0, h0)

    depth_metric = raw_depth * GLOBAL_SCALE
    points, colors = unproject(depth_metric, K['fx'], K['fy'],
                                 K['cx'] * 180 / w0, K['cy'] * 180 / h0, img_small)
    points_world = transform_to_world(points, self_pose)
    all_points.append(points_world)
    all_colors.append(colors)
    print(f"csv_idx={csv_idx}  {len(points_world)} points")

points = np.concatenate(all_points, axis=0)
colors = np.concatenate(all_colors, axis=0)
np.savez(f"{OUT_DIR}/pipeline_test_pointcloud.npz", points=points, colors=colors)

extent_x = points[:, 0].max() - points[:, 0].min()
extent_z = points[:, 2].max() - points[:, 2].min()
print(f"Point cloud extent: X={extent_x:.2f}m, Z={extent_z:.2f}m")
print(f"Actual room size: 3m x 4m -> expect extent roughly in this range if geometry is correct")

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(points[:, 0], points[:, 2], c=np.clip(colors * 2.2, 0, 1), s=2, alpha=0.5)

# reference rectangle = known physical room size (3m x 4m), centered on point cloud centroid,
# so it's directly visually comparable to the actual room shape/size
cx_ref, cz_ref = points[:, 0].mean(), points[:, 2].mean()
room_w, room_h = 3.0, 4.0
ax.add_patch(plt.Rectangle((cx_ref - room_w / 2, cz_ref - room_h / 2), room_w, room_h,
                            fill=False, edgecolor='red', linewidth=2, linestyle='--',
                            label='reference 3x4m room outline'))

ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)")
ax.set_aspect('equal')
ax.legend()
ax.set_title(f"Pipeline test output, {len(selected_rows)} frames\n"
             f"Extent: {extent_x:.2f}m x {extent_z:.2f}m (dashed red = actual room size for reference)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/pipeline_test_topdown.png", dpi=150)
print(f"Saved {OUT_DIR}/pipeline_test_pointcloud.npz and {OUT_DIR}/pipeline_test_topdown.png")
