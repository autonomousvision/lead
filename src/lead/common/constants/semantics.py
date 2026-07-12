"""Semantic segmentation class enums and CARLA-to-TransFuser mapping."""

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
    # --- Own classes ---
    ConeAndTrafficWarning = 29
    SpecialVehicles = 30
    StopSign = 31


class TransfuserSemanticSegmentationClass(IntEnum):
    """Semantic segmentation classes used in TransFuser."""

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


# Mapping from CARLA semantic segmentation classes to TransFuser semantic segmentation classes
SEMANTIC_SEGMENTATION_CONVERTER = {
    CarlaSemanticSegmentationClass.Unlabeled: TransfuserSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.Roads: TransfuserSemanticSegmentationClass.ROAD,
    CarlaSemanticSegmentationClass.SideWalks: TransfuserSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.Building: TransfuserSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.Wall: TransfuserSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.Fence: TransfuserSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.Pole: TransfuserSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.TrafficLight: TransfuserSemanticSegmentationClass.TRAFFIC_LIGHT,
    CarlaSemanticSegmentationClass.TrafficSign: TransfuserSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.Vegetation: TransfuserSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.Terrain: TransfuserSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.Sky: TransfuserSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.Pedestrian: TransfuserSemanticSegmentationClass.PEDESTRIAN,
    CarlaSemanticSegmentationClass.Rider: TransfuserSemanticSegmentationClass.BIKER,
    CarlaSemanticSegmentationClass.Car: TransfuserSemanticSegmentationClass.VEHICLE,
    CarlaSemanticSegmentationClass.Truck: TransfuserSemanticSegmentationClass.VEHICLE,
    CarlaSemanticSegmentationClass.Bus: TransfuserSemanticSegmentationClass.VEHICLE,
    CarlaSemanticSegmentationClass.Train: TransfuserSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.Motorcycle: TransfuserSemanticSegmentationClass.VEHICLE,
    CarlaSemanticSegmentationClass.Bicycle: TransfuserSemanticSegmentationClass.BIKER,
    CarlaSemanticSegmentationClass.Static: TransfuserSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.Dynamic: TransfuserSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.Other: TransfuserSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.Water: TransfuserSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.RoadLine: TransfuserSemanticSegmentationClass.ROAD_LINE,
    CarlaSemanticSegmentationClass.Ground: TransfuserSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.Bridge: TransfuserSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.RailTrack: TransfuserSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.GuardRail: TransfuserSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.ConeAndTrafficWarning: TransfuserSemanticSegmentationClass.OBSTACLE,
    CarlaSemanticSegmentationClass.SpecialVehicles: TransfuserSemanticSegmentationClass.SPECIAL_VEHICLE,
    CarlaSemanticSegmentationClass.StopSign: TransfuserSemanticSegmentationClass.STOP_SIGN,
}
