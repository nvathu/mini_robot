import numpy as np


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

    u = cx - (lateral_m * fx / forward_m)

    if pitch_deg == 0.0:
        v = cy - (height_m * fy / forward_m)
    else:
        theta = np.radians(pitch_deg)
        Y_rot = height_m * np.cos(theta) - forward_m * np.sin(theta)
        Z_rot = height_m * np.sin(theta) + forward_m * np.cos(theta)
        if Z_rot <= 0.02:
            return None
        v = cy + (Y_rot * fy / Z_rot)

    return u, v


def get_target_pixel(self_pose, target_pose, intrinsics, camera_height_offset_mm=0.0,
                      camera_pitch_deg=0.0, margin_ratio=0.15):
    forward, lateral = world_to_camera_frame(
        self_pose['x'], self_pose['y'], self_pose['angle'],
        target_pose['x'], target_pose['y']
    )
    result = project_to_pixel(
        forward, lateral,
        intrinsics['fx'], intrinsics['fy'], intrinsics['cx'], intrinsics['cy'],
        height_mm=camera_height_offset_mm, pitch_deg=camera_pitch_deg
    )
    if result is None:
        return None
    u, v = result
    W, H = 2 * intrinsics['cx'], 2 * intrinsics['cy']
    margin = margin_ratio * W
    if not (margin <= u < W - margin and 0 <= v < H):
        return None
    return u, v