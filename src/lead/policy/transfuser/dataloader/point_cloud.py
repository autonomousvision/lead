"""The TransFuser LiDAR input: per-sweep processing and ego-motion accumulation."""

import jaxtyping as jt
import numpy as np
import numpy.typing as npt

from lead.common.sensors import point_clouds, ransac
from lead.config import LeadConfig


def densify_radar(
    radar_pc: jt.Float[npt.NDArray, "n 3"],
    config: LeadConfig,
) -> jt.Float[npt.NDArray, "m 3"]:
    """Duplicate radar returns near the ego to weight them like LiDAR points.

    Args:
        radar_pc: Radar returns of one tick in the CARLA ego frame.
        config: Root config tree with the densification knobs.

    Returns:
        The returns with the near-ego points repeated, or unchanged when
        densification is disabled.
    """
    if not config.policy.transfuser.duplicate_radar_near_ego:
        return radar_pc
    radar_near_ego = radar_pc[
        np.linalg.norm(radar_pc[:, :2], axis=1)
        < config.policy.transfuser.duplicate_radar_radius
    ]
    radar_near_ego = np.concatenate(
        [radar_near_ego] * config.policy.transfuser.duplicate_radar_factor,
        axis=0,
    )
    return np.concatenate([radar_pc, radar_near_ego], axis=0)


def process_sweep(
    lidar_pc: jt.Float[npt.NDArray, "n 3"],
    radar_pc: jt.Float[npt.NDArray, "m 3"],
    config: LeadConfig,
) -> tuple[jt.Float[npt.NDArray, "k 3"], jt.Float[npt.NDArray, "l 3"]]:
    """Process one tick's raw sweep: radar merging, densification, ground removal.

    Args:
        lidar_pc: Raw lidar sweep in the CARLA ego frame of its tick.
        radar_pc: Merged radar returns of the same tick.
        config: Root config tree.

    Returns:
        The processed lidar sweep (radar merged in when configured) and the
        ground-filtered radar returns.
    """
    transfuser = config.policy.transfuser
    radar_as_lidar = config.expert.sensor_rig.use_radars and transfuser.radar_as_lidar
    num_lidar = lidar_pc.shape[0]
    num_radar = radar_pc.shape[0]

    if radar_as_lidar:
        lidar_pc = np.concatenate([lidar_pc, densify_radar(radar_pc, config)], axis=0)

    if transfuser.remove_lidar_ground_points:
        ground_mask = ransac.remove_ground(lidar_pc, config.expert)
        lidar_pc = lidar_pc[~ground_mask]
        if radar_as_lidar:
            radar_ground_mask = ground_mask[num_lidar:]
            radar_pc = radar_pc[~radar_ground_mask[:num_radar]]

    return lidar_pc, radar_pc


def accumulate_lidar_points(
    lidar_sweeps: dict[int, jt.Float[npt.NDArray, "n 3"]],
    radar_sweeps: dict[int, jt.Float[npt.NDArray, "m 3"]] | None,
    past_positions: jt.Float[npt.NDArray, "t 2"],
    past_yaws: jt.Float[npt.NDArray, " t"],
    config: LeadConfig,
) -> jt.Float[npt.NDArray, "n 4"]:
    """Process the raw sweep history into the model's ego-frame point cloud.

    Per-sweep radar merging, densification and ground removal, then ego-motion
    alignment of every sweep into the anchor frame.

    Args:
        lidar_sweeps: Raw lidar sweeps keyed by frame age (0 = anchor), each in
            the CARLA ego frame of its own tick.
        radar_sweeps: Merged radar returns keyed by frame age, or None.
        past_positions: Filtered ego positions in the anchor frame, index
            ``i`` is ``i`` ticks ago.
        past_yaws: Filtered ego yaws relative to the anchor, same indexing.
        config: Root config tree.

    Returns:
        Point cloud of shape (N, 4) with [x, y, z, frame_age] in the CARLA
        ego frame of the anchor tick.
    """
    transfuser = config.policy.transfuser
    stack_size = config.expert.data_collection.lidar_stack_size
    if radar_sweeps is None:
        radar_sweeps = {}

    past_positions = np.asarray(past_positions, dtype=np.float64)
    past_yaws = np.asarray(past_yaws, dtype=np.float64)
    max_age = min(2 * stack_size, len(past_positions), len(past_yaws))

    # Per-sweep processing: radar merging, densification, ground removal.
    lidar_entries: dict[int, jt.Float[npt.NDArray, "n 3"]] = {}
    radar_entries: dict[int, jt.Float[npt.NDArray, "n 3"]] = {}
    for age in range(max_age):
        lidar_pc = lidar_sweeps.get(age)
        if lidar_pc is None:
            continue
        radar_pc = radar_sweeps.get(age, np.zeros((0, 3), dtype=np.float64))
        lidar_pc, radar_pc = process_sweep(lidar_pc, radar_pc, config)
        if age < stack_size:
            lidar_entries[age] = lidar_pc
        radar_entries[age] = radar_pc

    lidar_accumulated = []
    for age in sorted(lidar_entries):
        if age > 0 and not transfuser.lidar_accumulation:
            break
        dx, dy = past_positions[age]
        dyaw = past_yaws[age]
        lidar_pc = point_clouds.align_lidar(
            lidar_entries[age],
            np.array([-dx, -dy, 0.0]),
            -dyaw,
        )
        lidar_pc = np.concatenate(
            [lidar_pc, np.full((lidar_pc.shape[0], 1), age, dtype=lidar_pc.dtype)],
            axis=1,
        )
        lidar_accumulated.append(lidar_pc)
    lidar_accumulated = (
        np.concatenate(lidar_accumulated, axis=0)
        if len(lidar_accumulated) > 0
        else np.zeros((0, 4), dtype=np.float32)
    )

    radar_accumulated = []
    for age in sorted(radar_entries):
        if age > 0 and not transfuser.lidar_accumulation:
            break
        dx, dy = past_positions[age]
        dyaw = past_yaws[age]
        radar_pc = radar_entries[age]

        if transfuser.radar_as_lidar:
            radar_pc = point_clouds.align_lidar(
                radar_pc,
                np.array([-dx, -dy, 0.0]),
                -dyaw,
            )
            radar_pc = densify_radar(radar_pc, config)

        radar_pc = np.concatenate(
            [radar_pc, np.full((radar_pc.shape[0], 1), age, dtype=radar_pc.dtype)],
            axis=1,
        )
        radar_accumulated.append(radar_pc)
    radar_accumulated = (
        np.concatenate(radar_accumulated, axis=0)
        if len(radar_accumulated) > 0
        else np.zeros((0, 4), dtype=np.float32)
    )

    return np.concatenate([lidar_accumulated, radar_accumulated], axis=0)
