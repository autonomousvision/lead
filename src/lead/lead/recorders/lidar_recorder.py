"""Recorder for the LiDAR modality stream."""

import typing

import numpy as np
from py123d.datatypes import (
    BaseModality,
    EgoStateSE3,
    Lidar,
    LidarFeature,
    Timestamp,
)
from py123d.datatypes.sensors.lidar import LidarID

from lead.lead.recorders.base_recorder import BaseRecorder
from lead.lead.utils import carla_to_123d

if typing.TYPE_CHECKING:
    from lead.lead.data_collection import ExpertData


class LidarRecorder(BaseRecorder):
    """Records the accumulated top-LiDAR point cloud in the IMU frame."""

    def __init__(self, expert: "ExpertData") -> None:
        """Initialize recorder and build the LiDAR metadata.

        Args:
            expert: The expert agent owning the CARLA state to record.
        """
        super().__init__(expert)
        self.ego_metadata = carla_to_123d.get_carla_lincoln_mkz_2020_metadata()
        self.lidar_metadata = carla_to_123d.build_lidar_metadata(expert.config_expert)

    def record(
        self,
        input_data: dict,
        timestamp: Timestamp,
        ego_state: EgoStateSE3,
    ) -> list[BaseModality]:
        """Convert the accumulated LiDAR point cloud into a py123d Lidar.

        Args:
            input_data: Post-tick sensor data (unused; points come from the
                expert's LiDAR accumulation buffer).
            timestamp: Current simulation timestamp.
            ego_state: Ego state of the current tick (unused; points are
                stored in the IMU frame).

        Returns:
            A single-element list with the accumulated point cloud, or an
            empty list if no points were accumulated.
        """
        lidar_pc = self.expert.accumulate_lidar().copy()
        if lidar_pc.shape[0] == 0:
            return []

        # Convert to ISO 8855: invert Y, shift X by rear axle offset, adjust Z
        lidar_pc[:, 1] = -lidar_pc[:, 1]  # Y
        lidar_pc[:, 0] += self.ego_metadata.rear_axle_to_center_longitudinal  # X
        lidar_pc[:, 2] += (
            self.expert.config_expert.sensor_rig.lidar_pos_1[-1] / 2
            - self.ego_metadata.rear_axle_to_center_vertical
        )  # Z

        num_points = lidar_pc.shape[0]
        point_cloud_features = {
            LidarFeature.IDS.serialize(): np.full(
                num_points,
                int(LidarID.LIDAR_TOP.value),
                dtype=np.uint8,
            ),
            LidarFeature.INTENSITY.serialize(): np.ones(num_points, dtype=np.uint8),
            # Frame age of each accumulated point (0 = current tick, 1 = one
            # save tick older, ...). The loader stacks past frames by
            # filtering this column against ``training_used_lidar_steps``.
            LidarFeature.TIMESTAMPS.serialize(): lidar_pc[:, 3].astype(np.uint8),
        }

        return [
            Lidar(
                timestamp=timestamp,
                timestamp_end=timestamp,
                metadata=self.lidar_metadata,
                point_cloud_3d=lidar_pc[:, :3].astype(np.float32),
                point_cloud_features=point_cloud_features,
            ),
        ]
