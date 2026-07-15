"""Pixel-based visibility and occlusion checks from instance segmentation and LiDAR."""

import jaxtyping as jt
import numpy as np
import numpy.typing as npt

from lead.common.constants import INSTANCE_ID_MASK


def truncate_actor_id(actor_id: int) -> int:
    """Truncate a CARLA actor ID to the 16 bits stored in instance segmentation pixels.

    Args:
        actor_id: Full CARLA actor ID.

    Returns:
        The lower 16 bits of the actor ID as encoded in the camera's G/B channels.
    """
    return actor_id & INSTANCE_ID_MASK


def instance_pixel_counts(
    instances: list[jt.Int32[npt.NDArray, "H W 2"]],
) -> jt.Int64[npt.NDArray, " num_ids"]:
    """Count visible pixels per instance ID across all cameras.

    Args:
        instances: Converted instance segmentation images with channels [semantic_id, instance_id].

    Returns:
        Array of length 2**16 mapping each truncated actor ID to its total pixel count.
    """
    instance_ids = np.concatenate([instance[..., 1].ravel() for instance in instances])
    return np.bincount(instance_ids, minlength=INSTANCE_ID_MASK + 1)


def visible_pixels_of_actor(
    pixel_counts: jt.Int64[npt.NDArray, " num_ids"],
    actor_id: int,
) -> int:
    """Return how many camera pixels belong to a CARLA actor.

    Args:
        pixel_counts: Per-instance-ID pixel counts from :func:`instance_pixel_counts`.
        actor_id: Full CARLA actor ID.

    Returns:
        Number of pixels showing the actor across all cameras.
    """
    return int(pixel_counts[truncate_actor_id(actor_id)])


def transform_points_to_actor_frame(
    ego_matrix: npt.NDArray,
    actor_matrix: npt.NDArray,
    ego_point_cloud: npt.NDArray,
) -> npt.NDArray:
    if ego_point_cloud.shape[0] == 0:
        return np.array([])

    ego_point_cloud = ego_point_cloud[:, :3]  # Only xyz

    # Ego pose (rotation + translation)
    R_ego = ego_matrix[:3, :3]
    ego_pos = ego_matrix[:3, 3]

    # Actor pose (rotation + translation)
    R_actor = actor_matrix[:3, :3]
    actor_pos = actor_matrix[:3, 3]

    # Actor center in ego frame
    vehicle_pos = R_ego.T @ (actor_pos - ego_pos)

    # Rotation: world -> ego -> actor
    R_rel = R_ego.T @ R_actor  # ego -> actor frame

    # Transform LiDAR points into actor's local frame
    return (R_rel.T @ (ego_point_cloud - vehicle_pos).T).T


def get_points_in_actor_frame_and_in_bbox(
    ego_matrix: npt.NDArray,
    actor_matrix: npt.NDArray,
    actor_extent: npt.NDArray,
    ego_point_cloud: npt.NDArray,
    pad: bool = False,
) -> npt.NDArray:
    """Return the LiDAR points, in the actor's frame, that lie inside its bounding box.

    Args:
        ego_matrix: 4x4 world transform of the ego vehicle.
        actor_matrix: 4x4 world transform of the actor whose bounding box is checked.
        actor_extent: Half extents of the actor's bounding box.
        ego_point_cloud: 3D point cloud in the ego frame.
        pad: Whether to widen the bounding box by a 0.25 m margin.

    Returns:
        The points within the bounding box of the actor.
    """
    margin = 0.25 if pad else 0.0
    if ego_point_cloud.shape[0] > 0:
        # Points beyond the box's bounding sphere cannot be inside it; only
        # transform the nearby ones into the actor frame
        actor_pos_in_ego = ego_matrix[:3, :3].T @ (
            actor_matrix[:3, 3] - ego_matrix[:3, 3]
        )
        reach = np.linalg.norm(actor_extent + margin)
        offsets = ego_point_cloud[:, :3] - actor_pos_in_ego
        near = np.einsum("ij,ij->i", offsets, offsets) <= reach * reach
        ego_point_cloud = ego_point_cloud[near]

    vehicle_lidar = transform_points_to_actor_frame(
        ego_matrix,
        actor_matrix,
        ego_point_cloud,
    )
    if len(vehicle_lidar) == 0:
        return np.array([])

    # Count points inside bounding box
    mask = (
        (np.abs(vehicle_lidar[:, 0]) < actor_extent[0] + margin)
        & (np.abs(vehicle_lidar[:, 1]) < actor_extent[1] + margin)
        & (np.abs(vehicle_lidar[:, 2]) < actor_extent[2] + margin)
    )
    return vehicle_lidar[mask]
