"""Recorder for the merged radar modality stream."""

import typing

import numpy as np
from py123d.datatypes import BaseModality, EgoStateSE3, Timestamp
from py123d.datatypes.sensors.radar import (
    Radar,
    RadarFeature,
    RadarID,
)

from lead.api import py123d_log_api
from lead.common import carla_to_123d
from lead.expert.logs_writing.recorders.base_recorder import BaseRecorder

if typing.TYPE_CHECKING:
    from lead.expert.logs_writing import ExpertData

# LEAD radar index (1-based; radar ``i`` is ``sensor_rig.radars[i - 1]``) → 123D
# radar ID, derived from each radar's mounting position and yaw in the calibration.
RADAR_ID_BY_LEAD_INDEX = py123d_log_api.RADAR_ID_BY_LEAD_INDEX
RADIAL_VELOCITY_FEATURE = py123d_log_api.RADIAL_VELOCITY_FEATURE


class RadarRecorder(BaseRecorder):
    """Records the returns of all LEAD radar sensors as one merged stream.

    Points are stored as cartesian positions in the ISO 8855 IMU frame plus
    the radial velocity CARLA reports; each carries its originating
    ``RadarID`` in the ``RadarFeature.IDS`` column, which py123d's per-sensor
    reads split the cloud on.
    """

    def __init__(
        self,
        expert: "ExpertData",
        perturbated: bool = False,
        store_freq: int = 1,
    ) -> None:
        """Initialize recorder and build per-radar metadata.

        Args:
            expert: The expert agent owning the CARLA state to record.
            perturbated: If true record the perturbated radar views.
            store_freq: Storage period of the stream in simulator steps.
        """
        super().__init__(expert, perturbated, store_freq=store_freq)
        self.ego_metadata = py123d_log_api.CARLA_LINCOLN_MKZ_2020_METADATA
        # LEAD radar index of the input-data key → 123D radar ID of its points.
        self.recorded_radar_ids: dict[int, RadarID] = {
            sensor_index: RADAR_ID_BY_LEAD_INDEX[sensor_index]
            for sensor_index in range(
                1,
                len(expert.config_expert.sensor_rig.radars) + 1,
            )
        }
        _, self.merged_metadata = carla_to_123d.build_radar_metadatas(
            expert.config_expert,
            perturbation_translation=self.perturbation_translation,
            perturbation_rotation=self.perturbation_rotation,
        )

    def record(
        self,
        input_data: dict,
        timestamp: Timestamp,
        ego_state: EgoStateSE3,
    ) -> list[BaseModality]:
        """Convert the current radar returns into one merged py123d radar.

        Args:
            input_data: Post-tick sensor data with per-radar arrays of shape
                (N, 4) holding [x, y, z, radial_velocity] in the CARLA ego frame.
            timestamp: Current simulation timestamp.
            ego_state: Ego state of the current tick (unused; points are
                stored in the IMU frame).

        Returns:
            A single-element list with the merged returns of all sensors, or
            an empty list if none of the sensors returned any points.
        """
        points_per_sensor = []
        velocities_per_sensor = []
        ids_per_sensor = []
        for sensor_index, radar_id in self.recorded_radar_ids.items():
            radar_points = input_data[f"radar{sensor_index}{self.key_suffix}"].astype(
                np.float32,
            )

            # Convert to ISO 8855: invert Y, shift X by rear axle offset
            points_3d = radar_points[:, :3].copy()
            points_3d[:, 1] = -points_3d[:, 1]  # Y
            points_3d[:, 0] += self.ego_metadata.rear_axle_to_center_longitudinal  # X

            points_per_sensor.append(points_3d)
            velocities_per_sensor.append(radar_points[:, 3])
            ids_per_sensor.append(
                np.full(points_3d.shape[0], int(radar_id.value), dtype=np.uint8),
            )

        if sum(points.shape[0] for points in points_per_sensor) == 0:
            return []

        return [
            Radar(
                timestamp=timestamp,
                metadata=self.merged_metadata,
                point_cloud_3d=np.concatenate(points_per_sensor, axis=0),
                point_cloud_features={
                    RadarFeature.IDS.serialize(): np.concatenate(ids_per_sensor),
                    RADIAL_VELOCITY_FEATURE: np.concatenate(velocities_per_sensor),
                },
            ),
        ]
