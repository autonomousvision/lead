"""Recorder for the RGB camera modality streams."""

import typing

import cv2
from py123d.datatypes import BaseModality, EgoStateSE3, Timestamp
from py123d.geometry.transform import rel_to_abs_se3
from py123d.parser.base_dataset_parser import ParsedCamera

from lead.expert.logs_writing import carla_to_123d
from lead.expert.logs_writing.recorders.base_recorder import BaseRecorder

if typing.TYPE_CHECKING:
    from lead.expert.logs_writing import ExpertData


class CameraRecorder(BaseRecorder):
    """Records one RGB camera stream per LEAD camera.

    Frames are JPEG-encoded here (not by the log writer) so that LEAD's
    weather-conditioned ``jpeg_storage_quality`` is preserved; the writer
    passes pre-encoded JPEG byte strings through unchanged.
    """

    def __init__(
        self,
        expert: "ExpertData",
        perturbated: bool = False,
        store_freq: int = 1,
    ) -> None:
        """Initialize recorder and build per-camera pinhole metadata.

        Args:
            expert: The expert agent owning the CARLA state to record.
            perturbated: If true record the perturbated camera views.
            store_freq: Storage period of the stream in simulator steps.
        """
        super().__init__(expert, perturbated, store_freq=store_freq)
        ego_metadata = carla_to_123d.get_carla_lincoln_mkz_2020_metadata()
        self.camera_metadatas = carla_to_123d.build_pinhole_camera_metadatas(
            expert.config_expert,
            ego_metadata,
            perturbation_translation=self.perturbation_translation,
            perturbation_rotation=self.perturbation_rotation,
        )

    def record(
        self,
        input_data: dict,
        timestamp: Timestamp,
        ego_state: EgoStateSE3,
    ) -> list[BaseModality]:
        """Convert the current CARLA RGB images into py123d cameras.

        Args:
            input_data: Post-tick sensor data with per-camera BGRA images.
            timestamp: Current simulation timestamp.
            ego_state: Ego state for composing camera-to-global poses.

        Returns:
            One camera frame per LEAD camera, JPEG-encoded at the expert's
            storage quality.
        """
        cameras: list[BaseModality] = []
        for cam_key in range(1, self.expert.config_expert.sensor_rig.num_cameras + 1):
            camera_id = carla_to_123d.CAMERA_ID_MAPPING[cam_key]

            # CARLA images are already in opencv's BGR(A) format
            rgb_key = f"rgb_{cam_key}{self.key_suffix}"
            bgra_image = (
                input_data[rgb_key][1]
                if isinstance(input_data[rgb_key], tuple)
                else input_data[rgb_key]
            )
            _, encoded_image = cv2.imencode(
                ".jpg",
                bgra_image[:, :, :3],
                [int(cv2.IMWRITE_JPEG_QUALITY), self.expert.jpeg_storage_quality],
            )

            camera_metadata = self.camera_metadatas[camera_id]
            cameras.append(
                ParsedCamera(
                    metadata=camera_metadata,
                    timestamp=timestamp,
                    camera_to_global_se3=rel_to_abs_se3(
                        origin=ego_state.imu_se3,
                        pose_se3=camera_metadata.camera_to_imu_se3,
                    ),
                    byte_string=encoded_image.tobytes(),
                ),
            )
        return cameras
