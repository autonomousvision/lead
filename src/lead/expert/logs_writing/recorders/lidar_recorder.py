"""Recorder for the LiDAR modality stream."""

import logging
import typing

import numpy as np
from py123d.datatypes import (
    BaseModality,
    EgoStateSE3,
    Lidar,
    LidarFeature,
    Timestamp,
)

from lead.api import py123d_log_api
from lead.common import carla_to_123d
from lead.expert.logs_writing.recorders.base_recorder import BaseRecorder

if typing.TYPE_CHECKING:
    from lead.expert.logs_writing import ExpertData

logger = logging.getLogger(__name__)


class LidarRecorder(BaseRecorder):
    """Records the current tick's raw LiDAR sweep in the IMU frame."""

    def __init__(self, expert: "ExpertData", store_freq: int = 1) -> None:
        """Initialize recorder and build the LiDAR metadata.

        Args:
            expert: The expert agent owning the CARLA state to record.
            store_freq: Storage period of the stream in simulator steps.
        """
        super().__init__(expert, store_freq=store_freq)
        self.ego_metadata = py123d_log_api.get_carla_lincoln_mkz_2020_metadata()
        self.lidar_metadata = carla_to_123d.build_lidar_metadata(expert.config_expert)

    def record(
        self,
        input_data: dict,
        timestamp: Timestamp,
        ego_state: EgoStateSE3,
    ) -> list[BaseModality]:
        """Convert the current tick's LiDAR sweep into a py123d Lidar.

        Args:
            input_data: Post-tick sensor data with the unprocessed sweep.
            timestamp: Current simulation timestamp.
            ego_state: Ego state of the current tick (unused; points are
                stored in the IMU frame).

        Returns:
            A single-element list with the sweep, or an empty list if the
            sweep has no points.
        """
        lidar_pc = input_data["lidar_sweep"].copy()
        if lidar_pc.shape[0] == 0:
            logger.warning(
                "Empty LiDAR sweep at timestamp %s; a 360-degree sweep should "
                "always return at least ground-plane points, so this likely "
                "indicates a sensor or timing bug rather than an empty scene.",
                timestamp,
            )
            return []

        # Convert to ISO 8855: invert Y, shift X by rear axle offset, adjust Z
        lidar_pc[:, 1] = -lidar_pc[:, 1]  # Y
        lidar_pc[:, 0] += self.ego_metadata.rear_axle_to_center_longitudinal  # X
        lidar_pc[:, 2] += (
            self.expert.config_expert.sensor_rig.lidar_pos_1[-1] / 2
            - self.ego_metadata.rear_axle_to_center_vertical
        )  # Z

        # Readers filter a single-sensor cloud by the per-point IDS feature and
        # drop the sweep entirely when it is missing.
        point_cloud_features = {
            LidarFeature.IDS.serialize(): np.full(
                lidar_pc.shape[0],
                int(self.lidar_metadata.lidar_id.value),
                dtype=np.uint8,
            ),
        }

        return [
            Lidar(
                timestamp=timestamp,
                timestamp_end=timestamp,
                metadata=self.lidar_metadata,
                point_cloud_3d=lidar_pc.astype(np.float32),
                point_cloud_features=point_cloud_features,
            ),
        ]
