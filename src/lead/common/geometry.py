"""Angle normalization and rigid transforms between world, ego and 2D frames."""

from __future__ import annotations

import math
import typing

import jaxtyping as jt
import numpy as np
import numpy.typing as npt

if typing.TYPE_CHECKING:
    import carla


def angle_to_yaw_class(angle: float, num_yaw_bins: int) -> tuple[int, float]:
    """Convert continuous angle to discrete class and residual.

    Encodes a continuous angle into a discrete class and a small regression
    residual from the class center to the actual angle.

    Args:
        angle: Continuous angle in radians (0-2π or -π~π).
        num_yaw_bins: Number of discrete direction bins for encoding.

    Returns:
        A tuple containing:
            - Discrete angle class as integer
            - Angle residual as float (difference from class center)
    """
    angle = angle % (2 * np.pi)
    angle_per_class = 2 * np.pi / float(num_yaw_bins)
    shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
    angle_cls = shifted_angle // angle_per_class
    angle_res = shifted_angle - (angle_cls * angle_per_class + angle_per_class / 2)
    return int(angle_cls), angle_res


def normalize_angle_rad(angle_rad: float) -> float:
    """
    Normalize an angle to the range [-π, π).

    This function takes an angle in radians and converts it to the standard
    range of [-π, π), which is commonly used in robotics, navigation, and
    other applications where angle wrapping is needed.

    Args:
        angle_rad: Angle in radians (can be any real number)

    Returns:
        float: Normalized angle in the range [-π, π)

    Examples:
        >>> normalize_angle_rad(3 * np.pi)
        -3.141592653589793
        >>> normalize_angle_rad(-3 * np.pi)
        3.141592653589793
        >>> normalize_angle_rad(np.pi / 4)
        0.7853981633974483
        >>> normalize_angle_rad(0)
        0.0
    """
    angle_rad = angle_rad % (2 * np.pi)
    if angle_rad > np.pi:
        angle_rad -= 2 * np.pi
    return angle_rad


def normalize_angle_deg(angle_deg: float) -> float:
    """
    Normalize an angle to the range [-180°, 180°).

    Args:
        angle_deg: Angle in degrees (can be any real number)

    Returns:
        float: Normalized angle in the range [-180°, 180°)

    Examples:
        >>> normalize_angle_deg(450)
        90.0
        >>> normalize_angle_deg(-270)
        90.0
        >>> normalize_angle_deg(180)
        -180.0
        >>> normalize_angle_deg(90)
        90.0
        >>> normalize_angle_deg(0)
        0.0
    """
    angle_deg = angle_deg % 360.0
    if angle_deg > 180.0:
        angle_deg -= 360.0
    return angle_deg


def euler_degrees_to_rotation_matrix(
    roll: float,
    pitch: float,
    yaw: float,
) -> jt.Float[npt.NDArray, "3 3"]:
    """Convert Euler angles in degrees to rotation matrix.

    Computes the 3D rotation matrix from roll, pitch, and yaw angles
    using the standard aerospace sequence (roll-pitch-yaw).

    Args:
        roll: Rotation around x-axis in degrees.
        pitch: Rotation around y-axis in degrees.
        yaw: Rotation around z-axis in degrees.

    Returns:
        3x3 rotation matrix combining all three rotations.
    """
    r = np.deg2rad(roll)
    p = np.deg2rad(pitch)
    y = np.deg2rad(yaw)
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    # Compose roll about x, pitch about y, yaw about z.
    return Rz @ Ry @ Rx


def to_local_frame_2d(
    point: jt.Float[npt.NDArray, " 2"],
    translation: jt.Float[npt.NDArray, " 2"],
    yaw: float,
) -> jt.Float[npt.NDArray, " 2"]:
    """Express a global 2D point in the local frame at ``(translation, yaw)``.

    Computes ``Rᵀ (point - translation)`` — the inverse of
    :func:`to_global_frame_2d`.

    Args:
        point: Point in the global frame.
        translation: Global 2D origin of the local frame.
        yaw: Yaw of the local frame in radians.

    Returns:
        The point expressed in the local frame.
    """
    rotation_matrix = np.array(
        [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]],
    )
    return rotation_matrix.T @ (point - translation)


def to_global_frame_2d(
    point: jt.Float[npt.NDArray, " 2"],
    translation: jt.Float[npt.NDArray, " 2"],
    yaw: float,
) -> jt.Float[npt.NDArray, " 2"]:
    """Express a local 2D point in the global frame; inverse of :func:`to_local_frame_2d`.

    Computes ``R point + translation``.

    Args:
        point: Point in the local frame at ``(translation, yaw)``.
        translation: Global 2D origin of the local frame.
        yaw: Yaw of the local frame in radians.

    Returns:
        The point expressed in the global frame.
    """
    rotation_matrix = np.array(
        [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]],
    )

    return rotation_matrix @ point + translation


def world_coordinate_2d(
    ego_transform: carla.Transform,
    local_location: carla.Location,
) -> carla.Location:
    """
    Ignore pitch and roll of car to get only 2D position with only yaw.
    """
    yaw = math.radians(ego_transform.rotation.yaw)
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)

    x = local_location.x
    y = local_location.y
    z = local_location.z

    world_x = ego_transform.location.x + cos_yaw * x - sin_yaw * y
    world_y = ego_transform.location.y + sin_yaw * x + cos_yaw * y
    world_z = ego_transform.location.z + z  # preserve vertical offset

    import carla

    return carla.Location(x=world_x, y=world_y, z=world_z)


def get_relative_transform(
    ego_matrix: jt.Float[npt.NDArray, "4 4"],
    vehicle_matrix: jt.Float[npt.NDArray, "4 4"],
) -> jt.Float[npt.NDArray, " 3"]:
    """Returns the position of the vehicle matrix in the ego coordinate system.

    Args:
        ego_matrix: 4x4 matrix of the ego vehicle in global coordinates.
        vehicle_matrix: 4x4 Matrix of another actor in global coordinates.
    Returns:
        3D position of the other vehicle in the ego coordinate system
    """
    relative_pos = vehicle_matrix[:3, 3] - ego_matrix[:3, 3]
    rot = ego_matrix[:3, :3].T
    return rot @ relative_pos


def get_horizontal_distance(actor_a: carla.Actor, actor_b: carla.Actor) -> float:
    """Get horizontal distance between two actors.

    Args:
        actor_a: First actor.
        actor_b: Second actor.

    Returns:
        float: Horizontal distance between the two actors.
    """
    location1, location2 = actor_a.get_location(), actor_b.get_location()

    # Compute the distance vector (ignoring the z-coordinate)
    import carla

    diff_vector = carla.Vector3D(
        location1.x - location2.x,
        location1.y - location2.y,
        0,
    )

    norm: float = diff_vector.length()

    return norm


def _obb_axes(obb: carla.BoundingBox) -> jt.Float[npt.NDArray, "3 3"]:
    """Return the three local axes of an oriented bounding box as matrix rows.

    Args:
        obb: The oriented bounding box.

    Returns:
        Matrix whose rows are the forward, right and up unit vectors.
    """
    forward = obb.rotation.get_forward_vector()
    right = obb.rotation.get_right_vector()
    up = obb.rotation.get_up_vector()
    return np.array(
        [
            [forward.x, forward.y, forward.z],
            [right.x, right.y, right.z],
            [up.x, up.y, up.z],
        ],
    )


def check_obb_intersection(box_a: carla.BoundingBox, box_b: carla.BoundingBox) -> bool:
    """
    Check if two 3D oriented bounding boxes (OBBs) intersect.

    Uses the separating axis theorem over the 15 candidate axes (3 + 3 box axes
    and their 9 cross products), preceded by a bounding-sphere early-out.

    Args:
        box_a: The first oriented bounding box.
        box_b: The second oriented bounding box.

    Returns:
        True if the two OBBs intersect, False otherwise.
    """
    relative_position = np.array(
        [
            box_b.location.x - box_a.location.x,
            box_b.location.y - box_a.location.y,
            box_b.location.z - box_a.location.z,
        ],
    )
    extent1 = np.array([box_a.extent.x, box_a.extent.y, box_a.extent.z])
    extent2 = np.array([box_b.extent.x, box_b.extent.y, box_b.extent.z])

    # Boxes further apart than the sum of their bounding spheres cannot intersect
    max_reach = np.linalg.norm(extent1) + np.linalg.norm(extent2)
    if relative_position @ relative_position > max_reach * max_reach:
        return False

    axes1 = _obb_axes(box_a)
    axes2 = _obb_axes(box_b)
    cross_axes = np.cross(axes1[:, None, :], axes2[None, :, :]).reshape(9, 3)
    candidate_axes = np.concatenate([axes1, axes2, cross_axes], axis=0)

    center_projections = np.abs(candidate_axes @ relative_position)
    reach_projections = (
        np.abs(candidate_axes @ axes1.T) @ extent1
        + np.abs(
            candidate_axes @ axes2.T,
        )
        @ extent2
    )

    # A separating axis exists iff the centers project further apart than the
    # combined box reach; degenerate cross products (parallel axes) project to
    # zero on both sides and never separate.
    return not np.any(center_projections > reach_projections)


def rotate_point(point: carla.Vector3D, angle: float) -> carla.Vector3D:
    """Rotate a given point by a given angle.

    Args:
        point: The point to be rotated.
        angle: The angle in degrees.
    Returns:
        The rotated point.
    """
    x_ = (
        math.cos(math.radians(angle)) * point.x
        - math.sin(math.radians(angle)) * point.y
    )
    y_ = (
        math.sin(math.radians(angle)) * point.x
        + math.cos(math.radians(angle)) * point.y
    )
    import carla

    return carla.Vector3D(x_, y_, point.z)
