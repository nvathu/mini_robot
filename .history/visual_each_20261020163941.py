from pathlib import Path
import glfw
import numpy as np
import open3d as o3d


# ============================================================
# CONFIGURATION
# ============================================================

POINTCLOUD_DIR = Path(
    "dataset/pointclouds/2025_0813_1543-S3"
)

# Select the coordinate system: "camera" or "world".
COORDINATE_FRAME = "camera"
FLIP_CAMERA_Y_FOR_DISPLAY = True

# Maximum number of points displayed per point cloud.
# Set to None to display all points.
MAX_POINTS = None  # Display every point saved in the NPZ file.

# Point cloud visualization settings.
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 800
POINT_SIZE = 4.5  # Larger points are easier to see.

# Background color in RGB format, with values in [0, 1].
BACKGROUND_COLOR = [1.0, 1.0, 1.0]
VIEW_FRONT = [0.0, 0.0, -1.0]
VIEW_UP = [0.0, -1.0, 0.0]

INITIAL_ZOOM = 0.55

# ============================================================
# LOAD POINT CLOUD
# ============================================================

def load_pointcloud(npz_path, coordinate_frame="camera"):
    """
    Load a point cloud from an NPZ file.

    Parameters
    ----------
    npz_path : Path
        Path to the NPZ file.
    coordinate_frame : str
        Coordinate system: "camera" or "world".

    Returns
    -------
    pcd : open3d.geometry.PointCloud or None
        Loaded point cloud, or None if invalid.
    img_id : str
        Corresponding image ID.
    """

    with np.load(npz_path, allow_pickle=False) as data:

        if coordinate_frame == "camera":
            key = "points_camera"
        elif coordinate_frame == "world":
            key = "points_world"
        else:
            raise ValueError(
                "coordinate_frame must be 'camera' or 'world'."
            )

        if key not in data:
            print(f"[SKIP] {npz_path.name}: missing {key}")
            return None, npz_path.stem

        points = np.asarray(data[key], dtype=np.float64)

        if points.ndim != 2 or points.shape[1] != 3:
            print(
                f"[SKIP] {npz_path.name}: "
                f"invalid point array shape {points.shape}"
            )
            return None, npz_path.stem
        if "colors" in data:
            colors = np.asarray(data["colors"], dtype=np.float64)
        else:
            colors = None

        if "img_id" in data:
            img_id = str(data["img_id"].item())
        else:
            img_id = npz_path.stem

        # Remove points with invalid coordinates.
        valid = np.isfinite(points).all(axis=1)
        points = points[valid]

        # Keep colors aligned with the remaining points.
        if colors is not None and len(colors) == len(valid):
            colors = colors[valid]
        else:
            colors = None

        if len(points) == 0:
            print(f"[SKIP] {npz_path.name}: empty point cloud")
            return None, img_id

        # Randomly subsample large point clouds for faster rendering.
        if MAX_POINTS is not None and len(points) > MAX_POINTS:
            rng = np.random.default_rng(42)

            indices = rng.choice(
                len(points),
                size=MAX_POINTS,
                replace=False
            )

            points = points[indices]

            if colors is not None:
                colors = colors[indices]

        # Create the Open3D point cloud.
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        if colors is not None and len(colors) > 0:

            # Convert colors to the [0, 1] range if necessary.
            if colors.max() > 1.0:
                colors = colors / 255.0

            colors = np.clip(colors, 0.0, 1.0)

            # Keep the original RGB color associated with each point.
            # Do not call colorize_by_depth(), because that would overwrite
            # the RGB colors sampled from the corresponding image pixels.
            pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd, img_id




def colorize_by_depth(pcd):
    """
    Colorize points using different shades of blue based on depth.
    Near points are dark blue, while far points are light blue.
    """

    points = np.asarray(pcd.points)

    if len(points) == 0:
        return pcd

    # Use camera Z as depth.
    if COORDINATE_FRAME == "camera":
        depth = points[:, 2]
    else:
        depth = np.linalg.norm(points, axis=1)

    valid = np.isfinite(depth)

    if not np.any(valid):
        return pcd

    # Use percentiles to reduce the influence of outliers.
    d_min, d_max = np.percentile(depth[valid], [2, 98])

    if d_max <= d_min:
        d_max = d_min + 1e-6

    # Normalize depth to [0, 1].
    normalized = np.clip(
        (depth - d_min) / (d_max - d_min),
        0.0,
        1.0
    )

    # Keep one hue: dark blue for near points, light blue for far points.
    near_color = np.array([0.02, 0.12, 0.45])
    far_color = np.array([0.65, 0.82, 1.00])

    colors = (
        (1.0 - normalized[:, None]) * near_color
        + normalized[:, None] * far_color
    )

    # Assign neutral gray to invalid points.
    colors[~valid] = [0.5, 0.5, 0.5]

    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd
# ============================================================
# LOAD ALL VALID POINT CLOUDS
# ============================================================

def load_all_pointclouds():
    """
    Load all valid NPZ point clouds from the configured directory.

    Returns
    -------
    list
        A list of tuples: (file_path, image_id, point_cloud).
    """

    if not POINTCLOUD_DIR.exists():
        print(f"[ERROR] Directory not found: {POINTCLOUD_DIR}")
        return []

    files = sorted(POINTCLOUD_DIR.glob("*.npz"))

    if not files:
        print(f"[ERROR] No NPZ files found in {POINTCLOUD_DIR}")
        return []

    print(f"Found {len(files)} NPZ files.")
    print("Loading point clouds...")

    valid_clouds = []

    for path in files:
        try:
            pcd, img_id = load_pointcloud(
                path,
                coordinate_frame=COORDINATE_FRAME
            )

            if pcd is not None and len(pcd.points) > 0:
                valid_clouds.append((path, img_id, pcd))

        except Exception as error:
            print(f"[SKIP] {path.name}: {error}")

    print(f"Successfully loaded {len(valid_clouds)} valid point clouds.")

    return valid_clouds


# ============================================================
# VISUALIZE POINT CLOUDS
# ============================================================

def visualize_all_pointclouds():
    """
    Visualize point clouds one at a time.

    Right Arrow : Next point cloud.
    Left Arrow  : Previous point cloud.
    Q           : Exit the viewer.
    """

    valid_clouds = load_all_pointclouds()

    if not valid_clouds:
        print("[ERROR] No valid point clouds to visualize.")
        return

    # Create the Open3D visualizer.
    vis = o3d.visualization.VisualizerWithKeyCallback()

    created = vis.create_window(
        window_name="Point Cloud Viewer",
        width=WINDOW_WIDTH,
        height=WINDOW_HEIGHT
    )

    if not created:
        print("[ERROR] Failed to create the Open3D window.")
        return

    state = {
        "index": 0,
        "geometry": valid_clouds[0][2]
    }

    # Display the first point cloud.
    vis.add_geometry(state["geometry"])

    # Configure the rendering options.
    render_option = vis.get_render_option()
    render_option.background_color = np.asarray(BACKGROUND_COLOR)
    render_option.point_size = POINT_SIZE
    render_option.show_coordinate_frame = True

    view_control = vis.get_view_control()
    view_control.set_lookat(
        state["geometry"].get_center()
    )
    view_control.set_front(VIEW_FRONT)
    view_control.set_up(VIEW_UP)
    view_control.set_zoom(INITIAL_ZOOM)
    vis.poll_events()
    vis.update_renderer()

    def print_current_info():
        """Print information about the current point cloud."""

        path, img_id, pcd = valid_clouds[state["index"]]

        print(
            f"[{state['index'] + 1}/{len(valid_clouds)}] "
            f"Image ID: {img_id} | "
            f"File: {path.name} | "
            f"Points: {len(pcd.points)}"
        )

    def change_pointcloud(step):
        """Create a keyboard callback to navigate between point clouds."""

        def callback(viewer):
            old_geometry = state["geometry"]

            # Remove the currently displayed point cloud.
            viewer.remove_geometry(
                old_geometry,
                reset_bounding_box=False
            )

            # Move forward or backward, wrapping around at the ends.
            state["index"] = (
                state["index"] + step
            ) % len(valid_clouds)

            path, img_id, new_geometry = valid_clouds[state["index"]]

            state["geometry"] = new_geometry

            # Display the new point cloud.
            viewer.add_geometry(
                new_geometry,
                reset_bounding_box=False
            )

            print_current_info()

            return False

        return callback

    def quit_viewer(viewer):
        """Close the visualization window."""

        viewer.close()
        return False

    # Register keyboard callbacks using GLFW key codes.
    vis.register_key_callback(
        glfw.KEY_RIGHT,
        change_pointcloud(1)
    )

    vis.register_key_callback(
        glfw.KEY_LEFT,
        change_pointcloud(-1)
    )

    vis.register_key_callback(
        glfw.KEY_ESCAPE,
        quit_viewer
    )

    print("\nKeyboard controls:")
    print("  Right Arrow : Next point cloud")
    print("  Left Arrow  : Previous point cloud")
    print("  Esc       : Exit")
    print("  Mouse       : Rotate, zoom, and move the view")

    print("\nCoordinate frame:")
    print("  X (Red)     : Horizontal axis")
    print("  Y (Green)   : Vertical axis")
    print("  Z (Blue)    : Camera viewing direction")
    print("  Note        : The exact convention depends on how")
    print("                the camera point cloud was generated.")
    print("  Point color : Original RGB color from the corresponding image")
    print_current_info()

    # Run the visualization event loop.
    try:
        vis.run()
    finally:
        vis.destroy_window()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    visualize_all_pointclouds()