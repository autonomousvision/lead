"""Recorder for the per-camera segmentation modality streams."""

import typing

import numpy as np
from py123d.datatypes import BaseModality, EgoStateSE3, Timestamp
from py123d.datatypes.sensors.base_camera import Camera, CameraChannelType, CameraID
from py123d.datatypes.sensors.segmentation_camera import SegmentationCameraMetadata
from py123d.geometry.transform import rel_to_abs_se3

from lead.expert.logs_writing import carla_to_123d
from lead.expert.logs_writing.recorders.base_recorder import BaseRecorder

if typing.TYPE_CHECKING:
    from lead.expert.logs_writing import ExpertData


class InstanceRecorder(BaseRecorder):
    """Records the semantic and instance segmentation streams of every camera.

    123D stores a label map as an 8- or 16-bit lossless PNG, so the two
    channels travel as separate streams: the raw CARLA class ids (29 classes,
    uint8) and the CARLA actor ids (uint16).
    """

    def __init__(
        self,
        expert: "ExpertData",
        perturbated: bool = False,
        store_freq: int = 1,
    ) -> None:
        """Initialize recorder and build per-camera segmentation metadata.

        Args:
            expert: The expert agent owning the CARLA state to record.
            perturbated: If true record the perturbated camera views.
            store_freq: Storage period of the stream in simulator steps.
        """
        super().__init__(expert, perturbated, store_freq=store_freq)
        ego_metadata = carla_to_123d.get_carla_lincoln_mkz_2020_metadata()
        pinhole_metadatas = carla_to_123d.build_pinhole_camera_metadatas(
            expert.config_expert,
            ego_metadata,
            perturbation_translation=self.perturbation_translation,
            perturbation_rotation=self.perturbation_rotation,
        )

        def metadatas(
            channel_type: CameraChannelType,
        ) -> dict[CameraID, SegmentationCameraMetadata]:
            return {
                camera_id: SegmentationCameraMetadata(
                    camera_metadata=pinhole_metadata,
                    segmentation_label_class=carla_to_123d.CarlaCameraSegmentationLabel,
                    channel_type=channel_type,
                )
                for camera_id, pinhole_metadata in pinhole_metadatas.items()
            }

        self.semantic_metadatas = metadatas(CameraChannelType.SEMANTIC)
        self.instance_metadatas = metadatas(CameraChannelType.INSTANCE)

    def record(
        self,
        input_data: dict,
        timestamp: Timestamp,
        ego_state: EgoStateSE3,
    ) -> list[BaseModality]:
        """Convert the current segmentation maps into py123d cameras.

        Args:
            input_data: Post-tick sensor data with per-camera instance maps of
                shape (H, W, 2) holding [semantic_id, instance_id].
            timestamp: Current simulation timestamp.
            ego_state: Ego state for composing camera-to-global poses.

        Returns:
            The semantic and the instance camera of every LEAD camera.
        """
        cameras: list[BaseModality] = []
        for cam_key in range(1, self.expert.config_expert.sensor_rig.num_cameras + 1):
            camera_id = carla_to_123d.CAMERA_ID_MAPPING[cam_key]
            semantic_and_instance = input_data[
                f"converted_instance_{cam_key}{self.key_suffix}"
            ]
            channels = [
                (
                    self.semantic_metadatas[camera_id],
                    semantic_and_instance[:, :, 0].astype(np.uint8),
                ),
                (
                    self.instance_metadatas[camera_id],
                    semantic_and_instance[:, :, 1].astype(np.uint16),
                ),
            ]
            for metadata, image in channels:
                cameras.append(
                    Camera(
                        metadata=metadata,
                        image=image,
                        camera_to_global_se3=rel_to_abs_se3(
                            origin=ego_state.imu_se3,
                            pose_se3=metadata.camera_to_imu_se3,
                        ),
                        timestamp=timestamp,
                    ),
                )
        return cameras
