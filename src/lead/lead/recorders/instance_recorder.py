"""Recorder for the per-camera instance segmentation modality streams."""

import typing

import numpy as np
from py123d.datatypes import BaseModality, EgoStateSE3, Timestamp
from py123d.datatypes.sensors.base_camera import Camera, CameraChannelType, CameraID
from py123d.datatypes.sensors.segmentation_camera import SegmentationCameraMetadata
from py123d.geometry.transform import rel_to_abs_se3

from lead.lead.recorders.base_recorder import BaseRecorder
from lead.lead.utils import carla_to_123d

if typing.TYPE_CHECKING:
    from lead.lead.data_collection import ExpertData

# CARLA instance ids are uint16, so semantic and instance ids are packed as
# ``semantic_id * 2**16 + instance_id`` (recover with ``// 2**16`` / ``% 2**16``).
INSTANCE_ID_FACTOR = 2**16


class InstanceRecorder(BaseRecorder):
    """Records one panoptic/instance segmentation stream per LEAD camera."""

    def __init__(self, expert: "ExpertData", perturbated: bool = False) -> None:
        """Initialize recorder and build per-camera instance metadata.

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
        self.instance_metadatas: dict[CameraID, SegmentationCameraMetadata] = {
            camera_id: SegmentationCameraMetadata(
                camera_metadata=pinhole_metadata,
                segmentation_label_class=carla_to_123d.CarlaCameraSegmentationLabel,
                channel_type=CameraChannelType.INSTANCE,
            )
            for camera_id, pinhole_metadata in pinhole_metadatas.items()
        }

    def record(
        self,
        input_data: dict,
        timestamp: Timestamp,
        ego_state: EgoStateSE3,
    ) -> list[BaseModality]:
        """Convert the current instance segmentation maps into py123d cameras.

        Args:
            input_data: Post-tick sensor data with per-camera instance maps of
                shape (H, W, 2) holding [semantic_id, instance_id].
            timestamp: Current simulation timestamp.
            ego_state: Ego state for composing camera-to-global poses.

        Returns:
            One instance Camera per LEAD camera.
        """
        instance_cameras: list[BaseModality] = []
        for cam_key in range(1, self.expert.config_expert.sensor_rig.num_cameras + 1):
            camera_id = carla_to_123d.CAMERA_ID_MAPPING[cam_key]
            metadata = self.instance_metadatas[camera_id]
            semantic_and_instance = input_data[f"instance_{cam_key}{self.key_suffix}"]
            packed = semantic_and_instance[:, :, 0].astype(
                np.int32,
            ) * INSTANCE_ID_FACTOR + semantic_and_instance[:, :, 1].astype(np.int32)
            instance_cameras.append(
                Camera(
                    metadata=metadata,
                    image=packed,
                    camera_to_global_se3=rel_to_abs_se3(
                        origin=ego_state.imu_se3,
                        pose_se3=metadata.camera_to_imu_se3,
                    ),
                    timestamp=timestamp,
                ),
            )
        return instance_cameras
