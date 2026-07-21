"""Recorder for the driving-meta custom modality stream."""

import logging
import typing

import numpy as np
from py123d.datatypes import (
    CustomModality,
    CustomModalityMetadata,
    Timestamp,
)

from lead.api import py123d_log_api

if typing.TYPE_CHECKING:
    from lead.expert.logs_writing import ExpertData

LOG = logging.getLogger(__name__)

DRIVING_META_MODALITY_ID = py123d_log_api.DRIVING_META_MODALITY_ID
TARGET_POINTS_KEY = py123d_log_api.TARGET_POINTS_KEY
BOX_ATTRIBUTES_KEY = py123d_log_api.BOX_ATTRIBUTES_KEY


def _box_attributes(bounding_boxes: list[dict]) -> dict[str, dict]:
    """Split the expert's box dicts into the fields the box stream cannot hold.

    Args:
        bounding_boxes: The expert's bounding-box dicts of one tick.

    Returns:
        The non-native fields of every stored box, keyed by its track token.
    """
    return {
        str(bb["id"]): {
            key: value
            for key, value in bb.items()
            if key not in py123d_log_api.NATIVE_BOX_KEYS
        }
        for bb in bounding_boxes
        if bb["class"] != "ego_car"
    }


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

    py123d serializes custom modalities with msgpack, whose numpy hook handles
    ``np.ndarray`` (left untouched here) but not ``np.bool_``, which aborts
    the write.

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
    modality. The target point route is static per log and stored once in the
    modality metadata; temporal enrichment (future/past fields) is derived at
    read time from the ego-state and box-detection streams.
    """

    def __init__(self, expert: "ExpertData") -> None:
        """Initialize recorder and build the custom modality metadata.

        Args:
            expert: The expert agent owning the CARLA state to record.
        """
        self.expert = expert
        # Every planner walks the same target points, they differ only in how
        # far ahead of the ego they place the current one.
        route_planner = next(iter(expert.gps_waypoint_planners_dict.values()))
        self.metadata = CustomModalityMetadata(
            modality_id=DRIVING_META_MODALITY_ID,
            metadata={
                TARGET_POINTS_KEY: route_planner.target_points.tolist(),
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
            per-box fields the box detections stream has no slot for (occlusion
            counts, ``affects_ego``, mesh paths), keyed by track token.
        """
        data = dict(meta)
        data[BOX_ATTRIBUTES_KEY] = _box_attributes(input_data["bounding_boxes"])
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
