import numpy as np


#def world_to_camera_frame(self_x, self_y, self_angle_deg, target_x, target_y):
#    theta = np.radians(self_angle_deg)
#    dx = target_x - self_x
#    dy = target_y - self_y
#
#    forward = np.cos(theta) * dx + np.sin(theta) * dy
#    lateral = -np.sin(theta) * dx + np.cos(theta) * dy
#
#    return forward, lateral



def world_to_camera_frame(self_x_mm,
        self_y_mm,
        self_z_mm,
        self_yaw_deg,
        other_x_mm,
        other_y_mm,
        other_z_mm):
    relative_marker_position = world_markers_to_relative_marker_position(self_x_mm,
        self_y_mm,
        self_z_mm,
        self_yaw_deg,
        other_x_mm,
        other_y_mm,
        other_z_mm)
    camera_optical_position = relative_marker_position_to_camera_optical_frame(relative_marker_position)
    camera_optical_position = camera_optical_position / 1000 # to m
    return camera_optical_position

def world_markers_to_relative_marker_position(
    self_x_mm,
    self_y_mm,
    self_z_mm,
    self_yaw_deg,
    other_x_mm,
    other_y_mm,
    other_z_mm,
):
    """
    Other robot's marker position relative to OUR marker,
    expressed in our robot/base axes:

        X = forward
        Y = left
        Z = up
    """

    # World-space marker-to-marker vector
    dx = other_x_mm - self_x_mm
    dy = other_y_mm - self_y_mm
    dz = other_z_mm - self_z_mm

    # World -> our robot orientation
    yaw = np.radians(self_yaw_deg)

    forward =  np.cos(yaw) * dx + np.sin(yaw) * dy
    lateral = -np.sin(yaw) * dx + np.cos(yaw) * dy

    return np.array([forward, lateral, dz], dtype=float)



def relative_marker_position_to_camera_optical_frame(
    p_marker_relative,
    camera_forward_offset_mm=100.0,
    camera_vertical_offset_mm=0.0,
    pitch_deg=0.0,
):
    """
    Input:
        Position of OTHER robot's marker relative to OUR marker,
        expressed in our robot/base axes.

        base axes:
            X = forward
            Y = left
            Z = up

    Output:
        Position of OTHER robot's marker relative to OUR CAMERA,
        expressed in camera optical coordinates:

            X = right
            Y = down
            Z = forward
    """

    p_marker_relative = np.asarray(p_marker_relative, dtype=float)

    # --------------------------------------------------
    # 1. Move origin: self marker -> camera
    #
    # marker is ~5 cm behind wheels
    # camera is ~5 cm ahead of wheels
    #
    # Therefore camera is ~10 cm in front of marker.
    # --------------------------------------------------

    marker_to_camera = np.array([
        camera_forward_offset_mm,
        0.0,
        camera_vertical_offset_mm
    ])

    p_base_from_camera = p_marker_relative - marker_to_camera

    # --------------------------------------------------
    # 2. base_link axes -> camera optical axes
    #
    # base:
    #   X forward
    #   Y left
    #   Z up
    #
    # optical:
    #   X right
    #   Y down
    #   Z forward
    # --------------------------------------------------

    R_base_to_optical = np.array([
        [0.0, -1.0,  0.0],
        [0.0,  0.0, -1.0],
        [1.0,  0.0,  0.0]
    ])

    p_cam = R_base_to_optical @ p_base_from_camera

    # --------------------------------------------------
    # 3. Physical camera pitch
    #
    # Positive = camera looking downward
    # --------------------------------------------------

    theta = np.radians(pitch_deg)

    R_pitch = np.array([
        [1.0, 0.0,             0.0],
        [0.0, np.cos(theta), -np.sin(theta)],
        [0.0, np.sin(theta),  np.cos(theta)]
    ])

    p_cam = R_pitch @ p_cam

    return p_cam




def project_to_pixel(
    p_cam,
    fx, fy, cx, cy
):
    """
    Pure pinhole projection.

    p_cam must already be in camera optical coordinates:
        X = right
        Y = down
        Z = forward
    """

    Xc, Yc, Zc = p_cam

    # mm here, so 20 mm minimum distance
    #print(f'Zc was {Zc}')
    #if Zc <= 20.0:
    #    return None

    u = fx * Xc / Zc + cx
    v = fy * Yc / Zc + cy

    return u, v

def get_target_pixel(self_pose, target_pose, intrinsics, camera_height_offset_mm=0.0,
                      camera_pitch_deg=0.0, margin_ratio=0.15):
    forward, lateral = world_to_camera_frame(
        self_pose['x'], self_pose['y'], self_pose['angle'],
        target_pose['x'], target_pose['y']
    )
    result = project_to_pixel(
        forward_mm=forward, lateral_mm=lateral,
        fx=intrinsics['fx'], fy=intrinsics['fy'], cx=intrinsics['cx'], cy=intrinsics['cy'],
        height_mm=camera_height_offset_mm, pitch_deg=camera_pitch_deg
    )
    if result is None:
        return None
    u, v = result
    W, H = 640, 480 #2 * intrinsics['cx'], 2 * intrinsics['cy'] # why? does it have to be this way? no I guess.
    margin = margin_ratio * W
    if not (margin <= u < W - margin and 0 <= v < H):
        return None
    return u, v