"""Convert 123D point-cloud sensors between 123D's IMU frame and the CARLA ego frame.

Recorders write sweeps in the IMU frame; TransFuser featurization consumes the CARLA ego frame.
"""

import jaxtyping as jt
import numpy as np
import numpy.typing as npt
from py123d.datatypes import Lidar
from py123d.datatypes.sensors.lidar import LidarMetadata
from py123d.datatypes.sensors.radar import Radar

from lead.api.py123d_log_api import (
    CARLA_LINCOLN_MKZ_2020_METADATA,
    RADAR_ID_BY_LEAD_INDEX,
    RADIAL_VELOCITY_FEATURE,
)
from lead.config import LeadConfig


def lidar_sweep_from_carla_ego_frame(
    points: jt.Float[npt.NDArray, "n 3"],
    lead_config: LeadConfig,
) -> jt.Float[npt.NDArray, "n 3"]:
    """Convert a CARLA-ego-frame lidar sweep into the 123D IMU frame.

    The transform ``LidarRecorder.record`` applies before storing a sweep, and
    the exact inverse of :func:`lidar_sweep_to_carla_ego_frame`.

    Args:
        points: Raw sweep of shape (N, 3) in the CARLA ego frame.
        lead_config: Root config tree with the LiDAR mounting position.

    Returns:
        The sweep in the 123D IMU frame.
    """
    ego_metadata = CARLA_LINCOLN_MKZ_2020_METADATA
    imu_frame_points = points.astype(np.float64).copy()
    imu_frame_points[:, 1] = -imu_frame_points[:, 1]
    imu_frame_points[:, 0] += ego_metadata.rear_axle_to_center_longitudinal
    imu_frame_points[:, 2] += (
        lead_config.expert.sensor_rig.lidar_pos_1[-1] / 2
        - ego_metadata.rear_axle_to_center_vertical
    )
    return imu_frame_points


def radar_returns_from_carla_ego_frame(
    points: jt.Float[npt.NDArray, "n 3"],
) -> jt.Float[npt.NDArray, "n 3"]:
    """Convert CARLA-ego-frame radar returns into the 123D IMU frame.

    The transform ``RadarRecorder.record`` applies before storing the returns.

    Args:
        points: Radar returns of shape (N, 3) in the CARLA ego frame.

    Returns:
        The returns in the 123D IMU frame.
    """
    ego_metadata = CARLA_LINCOLN_MKZ_2020_METADATA
    imu_frame_points = points.astype(np.float64).copy()
    imu_frame_points[:, 1] = -imu_frame_points[:, 1]
    imu_frame_points[:, 0] += ego_metadata.rear_axle_to_center_longitudinal
    return imu_frame_points


def lidar_sweep_to_carla_ego_frame(
    lidar: Lidar,
) -> jt.Float[npt.NDArray, "n 3"]:
    """Convert a 123D lidar sweep back into the CARLA ego frame.

    Exact inverse of ``LidarRecorder.record``: undo the rear-axle shift, the
    Y-axis flip, and the vertical adjustment; the mounting height comes from
    the sweep's own calibration.

    Args:
        lidar: 123D lidar modality holding one raw sweep.

    Returns:
        Point cloud of shape (N, 3) in the CARLA ego frame.
    """
    ego_metadata = CARLA_LINCOLN_MKZ_2020_METADATA
    assert isinstance(lidar.metadata, LidarMetadata)
    points = lidar.point_cloud_3d.astype(np.float64).copy()
    points[:, 0] -= ego_metadata.rear_axle_to_center_longitudinal
    points[:, 1] = -points[:, 1]
    points[:, 2] -= (
        lidar.metadata.lidar_to_imu_se3.z / 2
        - ego_metadata.rear_axle_to_center_vertical
    )
    return points


def _per_point_feature(
    feature: jt.Shaped[npt.NDArray, "*shape"],
    num_points: int,
) -> jt.Shaped[npt.NDArray, " n"]:
    """A radar point-cloud feature as a flat per-point array.

    The draco decoder collapses a single-point feature column to a scalar, so
    the stored shape is only guaranteed to hold ``num_points`` values.
    """
    flat = np.reshape(feature, -1)
    assert len(flat) == num_points, (
        f"Radar feature holds {len(flat)} values for {num_points} points."
    )
    return flat


def split_radar_by_sensor(
    radar: Radar,
) -> list[jt.Float16[npt.NDArray, "n 4"]]:
    """Split merged returns into per-sensor CARLA-frame clouds, in LEAD radar order.

    Args:
        radar: 123D radar modality of the merged stream, with the per-point
            sensor ids.

    Returns:
        One [x, y, z, radial_velocity] cloud per LEAD radar index (1..n).
    """
    points = radar_returns_to_carla_ego_frame(radar)
    assert radar.ids is not None
    per_point_sensor_ids = _per_point_feature(radar.ids, len(points))
    return [
        points[per_point_sensor_ids == int(radar_id.value)]
        for _, radar_id in sorted(RADAR_ID_BY_LEAD_INDEX.items())
    ]


def radar_returns_to_carla_ego_frame(
    radar: Radar,
) -> jt.Float16[npt.NDArray, "n 4"]:
    """Convert a 123D radar back into the CARLA ego-frame layout.

    Exact inverse of ``RadarRecorder.record``.

    Args:
        radar: 123D radar modality with the radial-velocity feature.

    Returns:
        Point cloud of shape (N, 4) with [x, y, z, radial_velocity] in the
        CARLA ego frame.
    """
    assert radar.point_cloud_features is not None
    ego_metadata = CARLA_LINCOLN_MKZ_2020_METADATA
    points = radar.point_cloud_3d.astype(np.float32).copy()
    points[:, 0] -= ego_metadata.rear_axle_to_center_longitudinal
    points[:, 1] = -points[:, 1]
    radial_velocities = _per_point_feature(
        radar.point_cloud_features[RADIAL_VELOCITY_FEATURE],
        len(points),
    )
    points_with_radial_velocity = np.concatenate(
        [points, radial_velocities[:, None].astype(np.float32)],
        axis=1,
    )
    return points_with_radial_velocity.astype(np.float16)
