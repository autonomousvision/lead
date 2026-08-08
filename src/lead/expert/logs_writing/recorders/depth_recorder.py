"""Recorder for the per-camera metric depth modality streams."""

import typing

from py123d.datatypes import BaseModality, EgoStateSE3, Timestamp
from py123d.datatypes.sensors.base_camera import Camera, CameraID
from py123d.datatypes.sensors.depth_camera import DepthCameraMetadata
from py123d.geometry.transform import rel_to_abs_se3

from lead.api import py123d_log_api
from lead.common import carla_to_123d
from lead.expert.logs_writing.recorders.base_recorder import BaseRecorder

if typing.TYPE_CHECKING:
    from lead.expert.logs_writing import ExpertData


class DepthRecorder(BaseRecorder):
    """Records one metric depth stream per LEAD camera.

    Depth is quantized linearly, clipped at ``storage.save_depth_max_meters``.
    """

    def __init__(
        self,
        expert: "ExpertData",
        perturbated: bool = False,
        store_freq: int = 1,
    ) -> None:
        """Initialize recorder and build per-camera depth metadata.

        Args:
            expert: The expert agent owning the CARLA state to record.
            perturbated: If true record the perturbated camera views.
            store_freq: Storage period of the stream in simulator steps.
        """
        super().__init__(expert, perturbated, store_freq=store_freq)
        ego_metadata = py123d_log_api.CARLA_LINCOLN_MKZ_2020_METADATA
        pinhole_metadatas = carla_to_123d.build_pinhole_camera_metadatas(
            expert.config_expert,
            ego_metadata,
            perturbation_translation=self.perturbation_translation,
            perturbation_rotation=self.perturbation_rotation,
        )
        storage = expert.config_expert.storage
        self.depth_metadatas: dict[CameraID, DepthCameraMetadata] = {
            camera_id: DepthCameraMetadata(
                camera_metadata=pinhole_metadata,
                max_depth=storage.save_depth_max_meters,
                depth_bits=storage.save_depth_bits,
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
            camera_id = py123d_log_api.CAMERA_ID_BY_LEAD_INDEX[cam_key]
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
