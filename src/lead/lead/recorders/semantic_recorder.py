"""Recorder for the per-camera semantic segmentation modality streams."""

import typing

from py123d.datatypes import BaseModality, EgoStateSE3, Timestamp
from py123d.datatypes.sensors.base_camera import Camera, CameraID
from py123d.datatypes.sensors.segmentation_camera import SegmentationCameraMetadata
from py123d.geometry.transform import rel_to_abs_se3

from lead.lead.recorders.base_recorder import BaseRecorder
from lead.lead.utils import carla_to_123d

if typing.TYPE_CHECKING:
    from lead.lead.data_collection import ExpertData


class SemanticRecorder(BaseRecorder):
    """Records one semantic segmentation stream per LEAD camera.

    Stores the reduced label space (``CarlaCameraSegmentationLabel``, mirroring
    ``TransfuserSemanticSegmentationClass``) that the legacy pipeline saved,
    i.e. raw CARLA class ids passed through ``SEMANTIC_SEGMENTATION_CONVERTER``.
    """

    def __init__(self, expert: "ExpertData", perturbated: bool = False) -> None:
        """Initialize recorder and build per-camera segmentation metadata.

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
        self.segmentation_metadatas: dict[CameraID, SegmentationCameraMetadata] = {
            camera_id: SegmentationCameraMetadata(
                camera_metadata=pinhole_metadata,
                segmentation_label_class=carla_to_123d.CarlaCameraSegmentationLabel,
            )
            for camera_id, pinhole_metadata in pinhole_metadatas.items()
        }

    def record(
        self,
        input_data: dict,
        timestamp: Timestamp,
        ego_state: EgoStateSE3,
    ) -> list[BaseModality]:
        """Convert the current semantic segmentation maps into py123d cameras.

        Args:
            input_data: Post-tick sensor data with per-camera raw CARLA class
                ids (single channel).
            timestamp: Current simulation timestamp.
            ego_state: Ego state for composing camera-to-global poses.

        Returns:
            One segmentation Camera per LEAD camera.
        """
        segmentation_cameras: list[BaseModality] = []
        for cam_key in range(1, self.expert.config_expert.sensor_rig.num_cameras + 1):
            camera_id = carla_to_123d.CAMERA_ID_MAPPING[cam_key]
            metadata = self.segmentation_metadatas[camera_id]
            label_image = self.expert.semantics_converter[
                input_data[f"semantics_{cam_key}{self.key_suffix}"]
            ]
            segmentation_cameras.append(
                Camera(
                    metadata=metadata,
                    image=label_image,
                    camera_to_global_se3=rel_to_abs_se3(
                        origin=ego_state.imu_se3,
                        pose_se3=metadata.camera_to_imu_se3,
                    ),
                    timestamp=timestamp,
                ),
            )
        return segmentation_cameras
