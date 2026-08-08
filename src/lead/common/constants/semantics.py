"""The raw CARLA semantic classes and the encoding of the instance stream."""

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


INSTANCE_ID_MASK = 0xFFFF
