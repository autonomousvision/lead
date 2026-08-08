"""Mappings from the recorded CARLA labels to TransFuser's semantic classes."""

import typing

from lead.common.constants import CarlaSemanticSegmentationClass
from lead.config.policy.transfuser.label_classes import PerspectiveSemanticClass

# Mapping from CARLA semantic segmentation classes to TransFuser's, applied at
# load time by the label builders.
SEMANTIC_SEGMENTATION_CONVERTER = {
    CarlaSemanticSegmentationClass.Unlabeled: PerspectiveSemanticClass.UNLABELED,
    CarlaSemanticSegmentationClass.Roads: PerspectiveSemanticClass.ROAD,
    CarlaSemanticSegmentationClass.SideWalks: PerspectiveSemanticClass.UNLABELED,
    CarlaSemanticSegmentationClass.Building: PerspectiveSemanticClass.UNLABELED,
    CarlaSemanticSegmentationClass.Wall: PerspectiveSemanticClass.UNLABELED,
    CarlaSemanticSegmentationClass.Fence: PerspectiveSemanticClass.UNLABELED,
    CarlaSemanticSegmentationClass.Pole: PerspectiveSemanticClass.UNLABELED,
    CarlaSemanticSegmentationClass.TrafficLight: PerspectiveSemanticClass.TRAFFIC_LIGHT,
    CarlaSemanticSegmentationClass.TrafficSign: PerspectiveSemanticClass.UNLABELED,
    CarlaSemanticSegmentationClass.Vegetation: PerspectiveSemanticClass.UNLABELED,
    CarlaSemanticSegmentationClass.Terrain: PerspectiveSemanticClass.UNLABELED,
    CarlaSemanticSegmentationClass.Sky: PerspectiveSemanticClass.UNLABELED,
    CarlaSemanticSegmentationClass.Pedestrian: PerspectiveSemanticClass.PEDESTRIAN,
    CarlaSemanticSegmentationClass.Rider: PerspectiveSemanticClass.BIKER,
    CarlaSemanticSegmentationClass.Car: PerspectiveSemanticClass.VEHICLE,
    CarlaSemanticSegmentationClass.Truck: PerspectiveSemanticClass.VEHICLE,
    CarlaSemanticSegmentationClass.Bus: PerspectiveSemanticClass.VEHICLE,
    CarlaSemanticSegmentationClass.Train: PerspectiveSemanticClass.UNLABELED,
    CarlaSemanticSegmentationClass.Motorcycle: PerspectiveSemanticClass.VEHICLE,
    CarlaSemanticSegmentationClass.Bicycle: PerspectiveSemanticClass.BIKER,
    CarlaSemanticSegmentationClass.Static: PerspectiveSemanticClass.UNLABELED,
    CarlaSemanticSegmentationClass.Dynamic: PerspectiveSemanticClass.UNLABELED,
    CarlaSemanticSegmentationClass.Other: PerspectiveSemanticClass.UNLABELED,
    CarlaSemanticSegmentationClass.Water: PerspectiveSemanticClass.UNLABELED,
    CarlaSemanticSegmentationClass.RoadLine: PerspectiveSemanticClass.ROAD_LINE,
    CarlaSemanticSegmentationClass.Ground: PerspectiveSemanticClass.UNLABELED,
    CarlaSemanticSegmentationClass.Bridge: PerspectiveSemanticClass.UNLABELED,
    CarlaSemanticSegmentationClass.RailTrack: PerspectiveSemanticClass.UNLABELED,
    CarlaSemanticSegmentationClass.GuardRail: PerspectiveSemanticClass.UNLABELED,
}


# Mapping from the class string of a bounding box recorded by the expert to its
# semantic class.
BOX_CLASS_TO_SEMANTIC = {
    "ego_car": PerspectiveSemanticClass.UNLABELED,
    "car": PerspectiveSemanticClass.VEHICLE,
    "walker": PerspectiveSemanticClass.PEDESTRIAN,
    "traffic_light": PerspectiveSemanticClass.TRAFFIC_LIGHT,
    "traffic_light_physical": PerspectiveSemanticClass.TRAFFIC_LIGHT,
    "stop_sign": PerspectiveSemanticClass.UNLABELED,
    "stop_sign_physical": PerspectiveSemanticClass.STOP_SIGN,
    "traffic_sign": PerspectiveSemanticClass.UNLABELED,
    "static_prop_car": PerspectiveSemanticClass.VEHICLE,
}


def semantic_class(box: dict[str, typing.Any]) -> PerspectiveSemanticClass:
    """Return the semantic class of a bounding box.

    Args:
        box: A bounding box dict as produced by the expert.

    Returns:
        The box's semantic class.
    """
    box_class = box["class"]
    if box_class == "static":
        mesh_path = box.get("mesh_path")
        if mesh_path is not None and "Car" in mesh_path:
            return PerspectiveSemanticClass.VEHICLE
        return PerspectiveSemanticClass.OBSTACLE
    return BOX_CLASS_TO_SEMANTIC[box_class]
