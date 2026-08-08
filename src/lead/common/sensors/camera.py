"""Camera projection helpers."""

import jaxtyping as jt
import numpy as np
import numpy.typing as npt

from lead.common.geometry import euler_degrees_to_rotation_matrix


def project_points_to_image(
    camera_rot: list[float],
    camera_pos: list[float],
    camera_fov: int | float,
    camera_width: int,
    camera_height: int,
    points: jt.Float[npt.NDArray, "n 2"],
) -> tuple[list[tuple[float, float]], list[bool]]:
    """
    Project 2D points (with z=0) to 2D image coordinates using camera parameters.

    Args:
        camera_rot: list of (roll, pitch, yaw) in degrees
        camera_pos: list of (x, y, z) camera position
        camera_fov: field of view in degrees (vertical FOV)
        camera_width: image width in pixels
        camera_height: image height in pixels
        points: numpy array of shape (N, 2) containing 2D points (z=0 assumed)

    Returns:
        list of (x, y) tuples for each point,
        list of booleans indicating if point is inside image bounds
    """

    # Convert inputs to numpy arrays
    camera_pos_arr = np.array(camera_pos)
    points_2d = np.array(points)

    # Make points 3D
    points_3d = np.column_stack([points_2d, np.zeros(len(points_2d))])

    # Get rotation matrix using the provided function
    roll, pitch, yaw = camera_rot
    R = euler_degrees_to_rotation_matrix(roll, pitch, yaw)

    # Transform points to camera coordinate system
    # Translate points relative to camera position
    points_translated = points_3d - camera_pos_arr

    # Rotate points to camera coordinate system
    points_camera: jt.Float[npt.NDArray, "n 3"] = (R @ points_translated.T).T

    # Convert from world coordinates (x=forward, y=right, z=up)
    # to camera coordinates (x=right, y=down, z=forward)
    points_cam_remapped = np.zeros_like(points_camera)
    points_cam_remapped[:, 0] = points_camera[:, 1]  # x_cam = y_world (right)
    points_cam_remapped[:, 1] = -points_camera[:, 2]  # y_cam = -z_world (down)
    points_cam_remapped[:, 2] = points_camera[:, 0]  # z_cam = x_world (forward)

    # Calculate camera intrinsic parameters
    fov_rad = np.radians(camera_fov)
    focal_length_y = camera_height / (2 * np.tan(fov_rad / 2))
    aspect_ratio = camera_width / camera_height
    focal_length_x = focal_length_y * aspect_ratio

    # Principal point (center of image)
    cx = camera_width / 2
    cy = camera_height / 2

    # Project points to image plane
    # Avoid division by zero for points behind camera
    z = points_cam_remapped[:, 2]
    valid_z = z > 1e-6  # Points must be in front of camera

    projected_points = []
    inside_image = []

    for i in range(len(points_cam_remapped)):
        if valid_z[i]:
            # Perspective projection
            x_img = (focal_length_x * points_cam_remapped[i, 0] / z[i]) + cx
            y_img = (focal_length_y * points_cam_remapped[i, 1] / z[i]) + cy

            # Check if point is inside image bounds
            inside = bool((0 <= x_img < camera_width) and (0 <= y_img < camera_height))

            projected_points.append((x_img, y_img))
            inside_image.append(inside)
        else:
            # Point is behind camera
            projected_points.append((0, 0))  # Default coordinate
            inside_image.append(False)

    return projected_points, inside_image
