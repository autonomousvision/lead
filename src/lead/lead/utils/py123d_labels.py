"""LEAD/CARLA label enums stored by reference in py123d arrow metadata.

Kept dependency-free (no ``carla``, ``lead.config``, ...) so logs can be
read back without the full expert-collection stack installed.
"""

from py123d.datatypes import DefaultBoxDetectionLabel
from py123d.datatypes.detections.box_detection_label import (
    BoxDetectionLabel,
    register_box_detection_label,
)
from py123d.datatypes.sensors.camera_segmentation_label import (
    CameraSegmentationLabel,
    register_camera_segmentation_label,
)


@register_box_detection_label
class CarlaBoxDetectionLabel(BoxDetectionLabel):
    """Native LEAD/CARLA box classes, mirroring the legacy ``bb["class"]`` strings.

    Kept 1:1 with the expert's bounding-box classes so the reader can
    reproduce the legacy CenterNet label mapping exactly.
    """

    CAR = 0
    WALKER = 1
    BICYCLE = 2
    TRAFFIC_LIGHT = 3
    STOP_SIGN = 4
    STATIC = 5
    STATIC_PROP_CAR = 6

    def to_default(self) -> DefaultBoxDetectionLabel:
        """Inherited, see superclass."""
        return {
            CarlaBoxDetectionLabel.CAR: DefaultBoxDetectionLabel.VEHICLE,
            CarlaBoxDetectionLabel.WALKER: DefaultBoxDetectionLabel.PERSON,
            CarlaBoxDetectionLabel.BICYCLE: DefaultBoxDetectionLabel.TWO_WHEELER,
            CarlaBoxDetectionLabel.TRAFFIC_LIGHT: DefaultBoxDetectionLabel.TRAFFIC_LIGHT,
            CarlaBoxDetectionLabel.STOP_SIGN: DefaultBoxDetectionLabel.TRAFFIC_SIGN,
            CarlaBoxDetectionLabel.STATIC: DefaultBoxDetectionLabel.GENERIC_OBJECT,
            CarlaBoxDetectionLabel.STATIC_PROP_CAR: DefaultBoxDetectionLabel.VEHICLE,
        }[self]


@register_camera_segmentation_label
class CarlaCameraSegmentationLabel(CameraSegmentationLabel):
    """Per-pixel semantic classes stored in LEAD 123D logs.

    Mirrors ``constants.TransfuserSemanticSegmentationClass`` — the reduced label
    space produced by ``SEMANTIC_SEGMENTATION_CONVERTER`` at collection time.
    """

    UNLABELED = 0
    VEHICLE = 1
    ROAD = 2
    TRAFFIC_LIGHT = 3
    PEDESTRIAN = 4
    ROAD_LINE = 5
    OBSTACLE = 6
    SPECIAL_VEHICLE = 7
    STOP_SIGN = 8
    BIKER = 9
