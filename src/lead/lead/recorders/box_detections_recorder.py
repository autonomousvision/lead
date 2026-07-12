"""Recorder for the box detections modality stream."""

import typing

from py123d.datatypes import (
    BaseModality,
    BoxDetectionAttributes,
    BoxDetectionSE3,
    BoxDetectionsSE3,
    BoxDetectionsSE3Metadata,
    EgoStateSE3,
    Timestamp,
)
from py123d.geometry import Vector3D

from lead.lead.recorders.base_recorder import BaseRecorder
from lead.lead.utils import carla_to_123d
from lead.lead.utils.carla_to_123d import CarlaBoxDetectionLabel

if typing.TYPE_CHECKING:
    from lead.lead.data_collection import ExpertData

# Legacy ``bb["class"]`` string → native 123D label.
_CLASS_LABELS: dict[str, CarlaBoxDetectionLabel] = {
    "car": CarlaBoxDetectionLabel.CAR,
    "walker": CarlaBoxDetectionLabel.WALKER,
    "traffic_light": CarlaBoxDetectionLabel.TRAFFIC_LIGHT,
    "stop_sign": CarlaBoxDetectionLabel.STOP_SIGN,
    "static": CarlaBoxDetectionLabel.STATIC,
    "static_prop_car": CarlaBoxDetectionLabel.STATIC_PROP_CAR,
}


class BoxDetectionsRecorder(BaseRecorder):
    """Records the expert's bounding boxes as SE3 detections with native labels.

    All boxes are written unfiltered (like the legacy per-frame pickles);
    visibility thresholds are applied at read time. Per-box fields that do not
    fit :class:`BoxDetectionAttributes` (brake, radar point count, visible
    pixels, role name, type id) travel in the ``driving_meta`` stream's
    ``box_attributes`` side channel, keyed by track token.
    """

    def __init__(self, expert: "ExpertData") -> None:
        """Initialize recorder and build the box detections metadata.

        Args:
            expert: The expert agent owning the CARLA state to record.
        """
        super().__init__(expert)
        self.box_detections_metadata = BoxDetectionsSE3Metadata(
            box_detection_label_class=CarlaBoxDetectionLabel,
        )

    def record(
        self,
        input_data: dict,
        timestamp: Timestamp,
        ego_state: EgoStateSE3,
    ) -> list[BaseModality]:
        """Convert the expert's bounding boxes into 123D box detections.

        Args:
            input_data: Post-tick sensor data with LEAD's ``bounding_boxes``
                (world-frame 4x4 matrices plus per-box metadata).
            timestamp: Current simulation timestamp.
            ego_state: Ego state of the current tick (unused; boxes are stored
                in the global frame).

        Returns:
            A single-element list with all box detections of this tick (also
            written when no boxes are found).
        """
        id2actor_map = self.expert.id2actor_map
        box_detections: list[BoxDetectionSE3] = []
        for bb in input_data["bounding_boxes"]:
            if bb["class"] == "ego_car":
                continue  # The ego is its own modality stream

            actor = id2actor_map.get(bb["id"])
            if actor is not None:
                velocity_3d = carla_to_123d.get_actor_velocity(actor)
            else:
                velocity_3d = Vector3D(x=0.0, y=0.0, z=0.0)

            box_detections.append(
                BoxDetectionSE3(
                    attributes=BoxDetectionAttributes(
                        label=_CLASS_LABELS[bb["class"]],
                        track_token=str(bb["id"]),
                        num_lidar_points=bb.get("num_points"),
                    ),
                    bounding_box_se3=carla_to_123d.get_bounding_box_se3(bb),
                    velocity_3d=velocity_3d,
                ),
            )

        return [
            BoxDetectionsSE3(
                box_detections=box_detections,
                timestamp=timestamp,
                metadata=self.box_detections_metadata,
            ),
        ]
