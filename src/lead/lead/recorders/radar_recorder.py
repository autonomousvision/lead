"""Recorder for the radar modality streams."""

import typing

import carla
import numpy as np
from py123d.datatypes import BaseModality, EgoStateSE3, Timestamp
from py123d.datatypes.sensors.radar import Radar, RadarID, RadarMetadata
from py123d.geometry import PoseSE3

from lead.lead.recorders.base_recorder import BaseRecorder
from lead.lead.sensor_rig.sensor_setup import perturbated_sensor_cfg
from lead.lead.utils import carla_to_123d

if typing.TYPE_CHECKING:
    from lead.lead.data_collection import ExpertData

# LEAD radar index (1-based; radar ``i`` is ``sensor_rig.radars[i - 1]``) → 123D radar ID,
# derived from the mounting position/yaw in the calibration (not its comments):
# 1: front bumper, yaw -45° (front-left), 2: front bumper, yaw +45° (front-right),
# 3: rear bumper, yaw 135° (back-right), 4: rear bumper, yaw 225° (back-left).
RADAR_ID_MAPPING: dict[int, RadarID] = {
    1: RadarID.RADAR_FRONT_LEFT,
    2: RadarID.RADAR_FRONT_RIGHT,
    3: RadarID.RADAR_BACK_RIGHT,
    4: RadarID.RADAR_BACK_LEFT,
}

# Feature key for CARLA's per-point radial (Doppler) velocity in m/s.
RADIAL_VELOCITY_FEATURE = "radial_velocity"


class RadarRecorder(BaseRecorder):
    """Records one radar point-cloud stream per LEAD radar sensor.

    The per-point content matches the legacy npz files: cartesian points plus
    the radial velocity CARLA reports, only transformed from the CARLA ego
    frame into the ISO 8855 IMU frame.
    """

    def __init__(self, expert: "ExpertData", perturbated: bool = False) -> None:
        """Initialize recorder and build per-radar metadata.

        Args:
            expert: The expert agent owning the CARLA state to record.
            perturbated: If true record the perturbated radar views.
        """
        super().__init__(expert, perturbated)
        self.ego_metadata = carla_to_123d.get_carla_lincoln_mkz_2020_metadata()
        self.radar_metadatas: dict[int, RadarMetadata] = {}
        for sensor_index, calibration in enumerate(
            expert.config_expert.sensor_rig.radars,
            start=1,
        ):
            radar_id = RADAR_ID_MAPPING[sensor_index]
            pos = calibration["pos"]
            rot = calibration["rot"]
            if perturbated:
                perturbated_pose = perturbated_sensor_cfg(
                    {
                        "x": pos[0],
                        "y": pos[1],
                        "z": pos[2],
                        "roll": rot[0],
                        "pitch": rot[1],
                        "yaw": rot[2],
                    },
                    self.perturbation_translation,
                    self.perturbation_rotation,
                )
                pos = [
                    perturbated_pose["x"],
                    perturbated_pose["y"],
                    perturbated_pose["z"],
                ]
                rot = [
                    perturbated_pose["roll"],
                    perturbated_pose["pitch"],
                    perturbated_pose["yaw"],
                ]
            quaternion = carla_to_123d.quaternion_from_carla_rotation(
                carla.Rotation(roll=rot[0], pitch=rot[1], yaw=rot[2]),
            )
            self.radar_metadatas[sensor_index] = RadarMetadata(
                radar_name=str(radar_id),
                radar_id=radar_id,
                radar_to_imu_se3=PoseSE3(
                    x=pos[0] + self.ego_metadata.rear_axle_to_center_longitudinal,
                    y=-pos[1],  # Invert Y for ISO 8855
                    z=pos[2],
                    qw=quaternion.qw,
                    qx=quaternion.qx,
                    qy=quaternion.qy,
                    qz=quaternion.qz,
                ),
            )

    def record(
        self,
        input_data: dict,
        timestamp: Timestamp,
        ego_state: EgoStateSE3,
    ) -> list[BaseModality]:
        """Convert the current radar returns into py123d radars.

        Args:
            input_data: Post-tick sensor data with per-radar arrays of shape
                (N, 4) holding [x, y, z, radial_velocity] in the CARLA ego frame.
            timestamp: Current simulation timestamp.
            ego_state: Ego state of the current tick (unused; points are
                stored in the IMU frame).

        Returns:
            One Radar per LEAD radar sensor.
        """
        radars: list[BaseModality] = []
        for sensor_index, metadata in self.radar_metadatas.items():
            radar_points = input_data[f"radar{sensor_index}{self.key_suffix}"].astype(
                np.float32,
            )

            # Convert to ISO 8855: invert Y, shift X by rear axle offset
            points_3d = radar_points[:, :3].copy()
            points_3d[:, 1] = -points_3d[:, 1]  # Y
            points_3d[:, 0] += self.ego_metadata.rear_axle_to_center_longitudinal  # X

            radars.append(
                Radar(
                    timestamp=timestamp,
                    metadata=metadata,
                    point_cloud_3d=points_3d,
                    point_cloud_features={
                        RADIAL_VELOCITY_FEATURE: radar_points[:, 3],
                    },
                ),
            )
        return radars
