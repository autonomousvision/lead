"""Recorder for the per-camera metric depth modality streams."""

import typing

from py123d.datatypes import BaseModality, EgoStateSE3, Timestamp
from py123d.datatypes.sensors.base_camera import Camera, CameraID
from py123d.datatypes.sensors.depth_camera import DepthCameraMetadata
from py123d.geometry.transform import rel_to_abs_se3

from lead.lead.recorders.base_recorder import BaseRecorder
from lead.lead.utils import carla_to_123d

if typing.TYPE_CHECKING:
    from lead.lead.data_collection import ExpertData

# Far planes matching lead's legacy PNG encoders (``encode_depth_8bit``/``_16bit``).
_MAX_DEPTH_METERS = {8: 50.0, 16: 96.0}


class DepthRecorder(BaseRecorder):
    """Records one metric depth stream per LEAD camera.

    The depth quantization mirrors the legacy per-frame PNG encoding: linear,
    8 bit clipped at 50 m (or 16 bit at 96 m, per ``save_depth_bits``).
    """

    def __init__(self, expert: "ExpertData", perturbated: bool = False) -> None:
        """Initialize recorder and build per-camera depth metadata.

        Args:
            expert: The expert agent owning the CARLA state to record.
            perturbated: If true record the perturbated camera views.
        """
        super().__init__(expert, perturbated)
        ego_metadata = carla_to_123d.get_carla_lincoln_mkz_2020_metadata()
        pinhole_metadatas = carla_to_123d.build_pinhole_camera_metadatas(
            expert.config_expert,
            ego_metadata,
            perturbation_translation=self.perturbation_translation,
            perturbation_rotation=self.perturbation_rotation,
        )
        depth_bits = expert.config_expert.storage.save_depth_bits
        self.depth_metadatas: dict[CameraID, DepthCameraMetadata] = {
            camera_id: DepthCameraMetadata(
                camera_metadata=pinhole_metadata,
                max_depth=_MAX_DEPTH_METERS[depth_bits],
                depth_bits=depth_bits,
            )
            for camera_id, pinhole_metadata in pinhole_metadatas.items()
        }

    def record(
        self,
        input_data: dict,
        timestamp: Timestamp,
        ego_state: EgoStateSE3,
    ) -> list[BaseModality]:
        """Convert the current metric depth maps into py123d depth cameras.

        Args:
            input_data: Post-tick sensor data with per-camera metric depth in
                meters (already converted and enhanced by the expert tick).
            timestamp: Current simulation timestamp.
            ego_state: Ego state for composing camera-to-global poses.

        Returns:
            One depth Camera per LEAD camera.
        """
        depth_cameras: list[BaseModality] = []
        for cam_key in range(1, self.expert.config_expert.sensor_rig.num_cameras + 1):
            camera_id = carla_to_123d.CAMERA_ID_MAPPING[cam_key]
            metadata = self.depth_metadatas[camera_id]
            depth_cameras.append(
                Camera(
                    metadata=metadata,
                    image=metadata.encode_depth(
                        input_data[f"depth_{cam_key}{self.key_suffix}"],
                    ),
                    camera_to_global_se3=rel_to_abs_se3(
                        origin=ego_state.imu_se3,
                        pose_se3=metadata.camera_to_imu_se3,
                    ),
                    timestamp=timestamp,
                ),
            )
        return depth_cameras
