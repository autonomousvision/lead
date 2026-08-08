"""The TransFuser LiDAR input: per-sweep processing and ego-motion accumulation."""

import jaxtyping as jt
import numpy as np
import numpy.typing as npt

from lead.common.sensors import point_clouds, ransac
from lead.config import LeadConfig


def _duplicate_near_ego_radar_points(
    radar_points_ego_frame: jt.Float[npt.NDArray, "n 3"],
    lead_config: LeadConfig,
) -> jt.Float[npt.NDArray, "m 3"]:
    """Duplicate radar returns near the ego to weight them like LiDAR points."""
    transfuser_config = lead_config.policy.transfuser
    if not transfuser_config.duplicate_radar_near_ego:
        return radar_points_ego_frame
    radar_near_ego = radar_points_ego_frame[
        np.linalg.norm(radar_points_ego_frame[:, :2], axis=1)
        < transfuser_config.duplicate_radar_radius_meter
    ]
    radar_near_ego = np.concatenate(
        [radar_near_ego] * transfuser_config.duplicate_radar_repeat_count,
        axis=0,
    )
    return np.concatenate([radar_points_ego_frame, radar_near_ego], axis=0)


def _filter_sweep_points(
    lidar_points_ego_frame: jt.Float[npt.NDArray, "n 3"],
    radar_points_ego_frame: jt.Float[npt.NDArray, "m 3"],
    lead_config: LeadConfig,
) -> tuple[jt.Float[npt.NDArray, "k 3"], jt.Float[npt.NDArray, "l 3"]]:
    """Process one tick's raw sweep: radar merging, densification, ground removal."""
    transfuser_config = lead_config.policy.transfuser
    merge_radar_into_lidar = (
        lead_config.expert.sensor_rig.use_radars
        and transfuser_config.merge_radar_into_lidar
    )
    num_lidar = lidar_points_ego_frame.shape[0]
    num_radar = radar_points_ego_frame.shape[0]

    if merge_radar_into_lidar:
        lidar_points_ego_frame = np.concatenate(
            [
                lidar_points_ego_frame,
                _duplicate_near_ego_radar_points(radar_points_ego_frame, lead_config),
            ],
            axis=0,
        )

    if transfuser_config.remove_lidar_ground_points:
        ground_mask = ransac.remove_ground(lidar_points_ego_frame, lead_config.expert)
        lidar_points_ego_frame = lidar_points_ego_frame[~ground_mask]
        if merge_radar_into_lidar:
            radar_ground_mask = ground_mask[num_lidar:]
            radar_points_ego_frame = radar_points_ego_frame[
                ~radar_ground_mask[:num_radar]
            ]

    return lidar_points_ego_frame, radar_points_ego_frame


def accumulate_lidar_points(
    # Every sweep holds a different number of points and the result is their
    # concatenation, so no two of these shapes share an axis.
    lidar_sweeps: dict[int, jt.Float[npt.NDArray, "_ 3"]],
    radar_sweeps: dict[int, jt.Float[npt.NDArray, "_ 3"]] | None,
    past_ego_positions: jt.Float[npt.NDArray, "t 2"],
    past_ego_yaws: jt.Float[npt.NDArray, " t"],
    lead_config: LeadConfig,
) -> jt.Float[npt.NDArray, "n 3"]:
    """Process the raw sweep history into the model's ego-frame point cloud.

    Per-sweep radar merging, densification and ground removal, then ego-motion
    alignment of every sweep into the anchor frame.

    Args:
        lidar_sweeps: Raw lidar sweeps keyed by tick age (0 = anchor), each in
            the CARLA ego frame of its own tick.
        radar_sweeps: Merged radar returns keyed by tick age, or None.
        past_ego_positions: Filtered ego positions in the anchor frame, index
            ``i`` is ``i`` ticks ago.
        past_ego_yaws: Filtered ego yaws relative to the anchor, same indexing.
        lead_config: Root config tree.

    Returns:
        Point cloud of shape (N, 3) in the CARLA ego frame of the anchor tick.
    """
    transfuser_config = lead_config.policy.transfuser
    if radar_sweeps is None:
        radar_sweeps = {}

    past_ego_positions = np.asarray(past_ego_positions, dtype=np.float64)
    past_ego_yaws = np.asarray(past_ego_yaws, dtype=np.float64)
    # Alignment needs the pose of every age; sweeps beyond the pose history drop.
    num_history_ticks = min(len(past_ego_positions), len(past_ego_yaws))

    # Per-sweep processing: radar merging, densification, ground removal.
    lidar_entries: dict[int, jt.Float[npt.NDArray, "_ 3"]] = {}
    radar_entries: dict[int, jt.Float[npt.NDArray, "_ 3"]] = {}
    for age in range(num_history_ticks):
        lidar_points_ego_frame = lidar_sweeps.get(age)
        if lidar_points_ego_frame is None:
            continue
        radar_points_ego_frame = radar_sweeps.get(
            age,
            np.zeros((0, 3), dtype=np.float64),
        )
        lidar_points_ego_frame, radar_points_ego_frame = _filter_sweep_points(
            lidar_points_ego_frame,
            radar_points_ego_frame,
            lead_config,
        )
        lidar_entries[age] = lidar_points_ego_frame
        radar_entries[age] = radar_points_ego_frame

    aligned_lidar_sweeps = []
    for age in sorted(lidar_entries):
        if age > 0 and not transfuser_config.accumulate_lidar_sweeps:
            break
        ego_dx_anchor_frame, ego_dy_anchor_frame = past_ego_positions[age]
        ego_dyaw_anchor_frame = past_ego_yaws[age]
        lidar_points_ego_frame = point_clouds.align_lidar(
            lidar_entries[age],
            np.array([-ego_dx_anchor_frame, -ego_dy_anchor_frame, 0.0]),
            -ego_dyaw_anchor_frame,
        )
        aligned_lidar_sweeps.append(lidar_points_ego_frame)
    accumulated_lidar_points = (
        np.concatenate(aligned_lidar_sweeps, axis=0)
        if len(aligned_lidar_sweeps) > 0
        else np.zeros((0, 3), dtype=np.float32)
    )

    aligned_radar_sweeps = []
    for age in sorted(radar_entries):
        if age > 0 and not transfuser_config.accumulate_lidar_sweeps:
            break
        ego_dx_anchor_frame, ego_dy_anchor_frame = past_ego_positions[age]
        ego_dyaw_anchor_frame = past_ego_yaws[age]
        radar_points_ego_frame = radar_entries[age]

        if transfuser_config.merge_radar_into_lidar:
            radar_points_ego_frame = point_clouds.align_lidar(
                radar_points_ego_frame,
                np.array([-ego_dx_anchor_frame, -ego_dy_anchor_frame, 0.0]),
                -ego_dyaw_anchor_frame,
            )
            radar_points_ego_frame = _duplicate_near_ego_radar_points(
                radar_points_ego_frame,
                lead_config,
            )

        aligned_radar_sweeps.append(radar_points_ego_frame)
    accumulated_radar_points = (
        np.concatenate(aligned_radar_sweeps, axis=0)
        if len(aligned_radar_sweeps) > 0
        else np.zeros((0, 3), dtype=np.float32)
    )

    return np.concatenate(
        [accumulated_lidar_points, accumulated_radar_points],
        axis=0,
    )
