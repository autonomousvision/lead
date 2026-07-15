"""Semantic segmentation class enums and the CARLA-to-LEAD mappings."""

import typing
from enum import IntEnum


class CarlaSemanticSegmentationClass(IntEnum):
    """https://carla.readthedocs.io/en/latest/ref_sensors/#semantic-segmentation-camera"""

    Unlabeled = 0
    Roads = 1
    SideWalks = 2
    Building = 3
    Wall = 4
    Fence = 5
    Pole = 6
    TrafficLight = 7
    TrafficSign = 8
    Vegetation = 9
    Terrain = 10
    Sky = 11
    Pedestrian = 12
    Rider = 13
    Car = 14
    Truck = 15
    Bus = 16
    Train = 17
    Motorcycle = 18
    Bicycle = 19
    Static = 20
    Dynamic = 21
    Other = 22
    Water = 23
    RoadLine = 24
    Ground = 25
    Bridge = 26
    RailTrack = 27
    GuardRail = 28


class LeadSemanticSegmentationClass(IntEnum):
    """Semantic segmentation classes used by LEAD.

    ``OBSTACLE``, ``SPECIAL_VEHICLE`` and ``STOP_SIGN`` have no CARLA camera
    class; their pixels are painted at load time from the recorded boxes'
    actor ids via the instance segmentation stream.
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


INSTANCE_ID_MASK = 0xFFFF


# Mapping from CARLA semantic segmentation classes to LEAD semantic segmentation
# classes, applied at load time by model-specific dataloaders.
SEMANTIC_SEGMENTATION_CONVERTER = {
    CarlaSemanticSegmentationClass.Unlabeled: LeadSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.Roads: LeadSemanticSegmentationClass.ROAD,
    CarlaSemanticSegmentationClass.SideWalks: LeadSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.Building: LeadSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.Wall: LeadSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.Fence: LeadSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.Pole: LeadSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.TrafficLight: LeadSemanticSegmentationClass.TRAFFIC_LIGHT,
    CarlaSemanticSegmentationClass.TrafficSign: LeadSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.Vegetation: LeadSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.Terrain: LeadSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.Sky: LeadSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.Pedestrian: LeadSemanticSegmentationClass.PEDESTRIAN,
    CarlaSemanticSegmentationClass.Rider: LeadSemanticSegmentationClass.BIKER,
    CarlaSemanticSegmentationClass.Car: LeadSemanticSegmentationClass.VEHICLE,
    CarlaSemanticSegmentationClass.Truck: LeadSemanticSegmentationClass.VEHICLE,
    CarlaSemanticSegmentationClass.Bus: LeadSemanticSegmentationClass.VEHICLE,
    CarlaSemanticSegmentationClass.Train: LeadSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.Motorcycle: LeadSemanticSegmentationClass.VEHICLE,
    CarlaSemanticSegmentationClass.Bicycle: LeadSemanticSegmentationClass.BIKER,
    CarlaSemanticSegmentationClass.Static: LeadSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.Dynamic: LeadSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.Other: LeadSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.Water: LeadSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.RoadLine: LeadSemanticSegmentationClass.ROAD_LINE,
    CarlaSemanticSegmentationClass.Ground: LeadSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.Bridge: LeadSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.RailTrack: LeadSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.GuardRail: LeadSemanticSegmentationClass.UNLABELED,
}


# Mapping from the class string of a bounding box recorded by the expert to its
# LEAD semantic class.
BOX_CLASS_TO_SEMANTIC = {
    "ego_car": LeadSemanticSegmentationClass.UNLABELED,
    "car": LeadSemanticSegmentationClass.VEHICLE,
    "walker": LeadSemanticSegmentationClass.PEDESTRIAN,
    "traffic_light": LeadSemanticSegmentationClass.TRAFFIC_LIGHT,
    "traffic_light_physical": LeadSemanticSegmentationClass.TRAFFIC_LIGHT,
    "stop_sign": LeadSemanticSegmentationClass.UNLABELED,
    "stop_sign_physical": LeadSemanticSegmentationClass.STOP_SIGN,
    "traffic_sign": LeadSemanticSegmentationClass.UNLABELED,
    "static_prop_car": LeadSemanticSegmentationClass.VEHICLE,
}


def semantic_class(box: dict[str, typing.Any]) -> LeadSemanticSegmentationClass:
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
            return LeadSemanticSegmentationClass.VEHICLE
        return LeadSemanticSegmentationClass.OBSTACLE
    return BOX_CLASS_TO_SEMANTIC[box_class]
