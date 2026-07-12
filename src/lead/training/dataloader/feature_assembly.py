"""Decoding of 123D sensor modalities back into the legacy CARLA ego frame.

Hosts the exact inverses of the ISO 8855 IMU-frame transforms applied by the
recorders in ``src/lead/lead/recorders``.
"""

import jaxtyping as jt
import numpy as np
import numpy.typing as npt
from py123d.datatypes import Lidar
from py123d.datatypes.sensors.lidar import LidarFeature
from py123d.datatypes.sensors.radar import Radar

from lead.config import LeadConfig
from lead.lead.utils import carla_to_123d


def lidar_to_carla_ego_frame(
    lidar: Lidar,
    config: LeadConfig,
) -> jt.Float[npt.NDArray, "n 4"]:
    """Convert a 123D lidar back into the legacy CARLA ego-frame layout.

    Exact inverse of ``LidarRecorder.record``: undo the rear-axle shift, the
    Y-axis flip, and the vertical adjustment, and re-attach the frame-age
    column the legacy loader filters on.

    Args:
        lidar: 123D lidar modality with the frame-age feature.
        config: Training configuration with the LiDAR mounting position.

    Returns:
        Point cloud of shape (N, 4) with [x, y, z, frame_age] in the CARLA
        ego frame.
    """
    assert lidar.point_cloud_features is not None
    ego_metadata = carla_to_123d.get_carla_lincoln_mkz_2020_metadata()
    points = lidar.point_cloud_3d.astype(np.float64).copy()
    points[:, 0] -= ego_metadata.rear_axle_to_center_longitudinal
    points[:, 1] = -points[:, 1]
    points[:, 2] -= (
        config.expert.sensor_rig.lidar_pos_1[-1] / 2
        - ego_metadata.rear_axle_to_center_vertical
    )
    frame_age = lidar.point_cloud_features[LidarFeature.TIMESTAMPS.serialize()]
    return np.concatenate([points, frame_age[:, None].astype(np.float64)], axis=1)


def radar_to_carla_ego_frame(
    radar: Radar,
) -> jt.Float16[npt.NDArray, "n 4"]:
    """Convert a 123D radar back into the legacy CARLA ego-frame layout.

    Exact inverse of ``RadarRecorder.record``. Returns float16 like the
    legacy npz files.

    Args:
        radar: 123D radar modality with the radial-velocity feature.

    Returns:
        Point cloud of shape (N, 4) with [x, y, z, radial_velocity] in the
        CARLA ego frame.
    """
    from lead.lead.recorders.radar_recorder import RADIAL_VELOCITY_FEATURE

    assert radar.point_cloud_features is not None
    ego_metadata = carla_to_123d.get_carla_lincoln_mkz_2020_metadata()
    points = radar.point_cloud_3d.astype(np.float32).copy()
    points[:, 0] -= ego_metadata.rear_axle_to_center_longitudinal
    points[:, 1] = -points[:, 1]
    velocity = radar.point_cloud_features[RADIAL_VELOCITY_FEATURE]
    combined = np.concatenate(
        [points, velocity[:, None].astype(np.float32)],
        axis=1,
    )
    return combined.astype(np.float16)
