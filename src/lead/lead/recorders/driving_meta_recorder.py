"""Recorder for the driving-meta custom modality stream."""

import logging
import typing

import numpy as np
from py123d.datatypes import (
    CustomModality,
    CustomModalityMetadata,
    Timestamp,
)

if typing.TYPE_CHECKING:
    from lead.lead.data_collection import ExpertData

LOG = logging.getLogger(__name__)

# Name of the custom modality stream holding LEAD's per-tick driving state.
DRIVING_META_MODALITY_ID = "driving_meta"

# Meta keys that are static per log; they move into the stream's metadata
# instead of being repeated every tick.
_STATIC_META_KEYS = ("dataset_information", "sensor_information")


def _contains_numpy_bool(value: object) -> bool:
    """Whether a value or its nested dict/list entries hold a ``np.bool_``.

    Args:
        value: Any value from the driving-meta subtree.

    Returns:
        True if a ``np.bool_`` is present anywhere in the subtree.
    """
    if isinstance(value, np.bool_):
        return True
    if isinstance(value, dict):
        return any(_contains_numpy_bool(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_numpy_bool(item) for item in value)
    return False


def _to_python_scalars(value: object) -> object:
    """Convert numpy scalars nested in dicts/lists to their Python equivalents.

    py123d serializes custom modalities with msgpack, whose numpy hook only
    handles ``np.ndarray``. Numpy floats/ints slip through because they
    subclass the Python types, but ``np.bool_`` does not and aborts the write.
    Arrays are left untouched for the writer's own hook to encode.

    Args:
        value: Any value from the driving-meta subtree.

    Returns:
        The value with every numpy scalar replaced by a Python scalar.
    """
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _to_python_scalars(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_python_scalars(item) for item in value]
    return value


class DrivingMetaRecorder:
    """Records LEAD's per-tick driving state (route, hazards, control, scenario).

    Wraps the meta dict built by ``Expert.save_meta`` into a 123D custom
    modality. Static per-log blocks (dataset/sensor information) are stored
    once in the modality metadata; temporal enrichment (future/past fields)
    is no longer stored — it is derived at read time from the ego-state and
    box-detection streams.
    """

    def __init__(self, expert: "ExpertData") -> None:
        """Initialize recorder and build the custom modality metadata.

        Args:
            expert: The expert agent owning the CARLA state to record.
        """
        self.expert = expert
        config = expert.config_expert
        self.metadata = CustomModalityMetadata(
            modality_id=DRIVING_META_MODALITY_ID,
            metadata={
                "dataset_information": {
                    "save_depth_bits": config.storage.save_depth_bits,
                    "save_only_non_ground_lidar": config.storage.save_only_non_ground_lidar,
                    "target_dataset": int(config.target_dataset),
                    "data_save_freq": config.data_collection.data_save_freq,
                },
                "sensor_information": {
                    "lidar_pos_1": config.sensor_rig.lidar_pos_1,
                    "lidar_rot_1": config.sensor_rig.lidar_rot_1,
                    "lidar_pos_2": config.sensor_rig.lidar_pos_2,
                    "lidar_rot_2": config.sensor_rig.lidar_rot_2,
                    "lidar_accumulation": config.sensor_rig.lidar_accumulation,
                    "num_cameras": config.sensor_rig.num_cameras,
                    "camera_calibration": config.sensor_rig.cameras,
                    "num_radar_sensors": config.sensor_rig.num_radar_sensors,
                    "radar_calibration": config.sensor_rig.radars,
                },
            },
        )

    def record_meta(
        self,
        meta: dict,
        input_data: dict,
        timestamp: Timestamp,
    ) -> CustomModality:
        """Wrap the expert's per-tick meta dict into a custom modality.

        Args:
            meta: Driving meta dict built by ``Expert.save_meta``.
            input_data: Post-tick sensor data with LEAD's ``bounding_boxes``.
            timestamp: Current simulation timestamp.

        Returns:
            CustomModality with the per-tick driving state, including the
            expert's full bounding-box dicts (the training-label fields like
            occlusion counts, ``affects_ego``, or mesh paths have no slot in
            the box detections stream; storing the dicts verbatim keeps the
            legacy label builders working unchanged).
        """
        data = {
            key: value for key, value in meta.items() if key not in _STATIC_META_KEYS
        }
        data["bounding_boxes"] = input_data["bounding_boxes"]
        # Log which fields carried a numpy scalar so the upstream assignment can
        # be fixed at the source, then strip them so msgpack can serialize.
        for key, value in data.items():
            if _contains_numpy_bool(value):
                LOG.warning(f"driving_meta['{key}'] contains a numpy.bool_")
        data = _to_python_scalars(data)
        return CustomModality(
            data=data,
            metadata=self.metadata,
            timestamp=timestamp,
        )
