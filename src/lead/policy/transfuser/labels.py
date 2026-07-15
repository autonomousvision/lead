"""Class and index enums for the bounding-box and BEV targets of the TransFuser network."""

from enum import IntEnum


class BoundingBoxIndex(IntEnum):
    """Index to access the bounding box array."""

    X = 0
    Y = 1
    W = 2
    H = 3
    YAW = 4
    VELOCITY = 5
    BRAKE = 6
    CLASS = 7
    SCORE = 8  # Only available for prediction
    NUM_RADAR_POINTS = 8  # Only available for ground truth


class BoundingBoxClass(IntEnum):
    """Bounding box classes.

    Parking vehicles are not a separate class; they are labeled VEHICLE.
    """

    VEHICLE = 0
    WALKER = 1
    TRAFFIC_LIGHT = 2
    STOP_SIGN = 3
    SPECIAL = 4
    OBSTACLE = 5
    BIKER = 6


class BEVSemanticClass(IntEnum):
    """Indices to access the BEV semantic map.

    Parking vehicles are not a separate class; they are labeled VEHICLE.
    """

    UNLABELED = 0
    ROAD = 1
    LANE_MARKERS = 2
    STOP_SIGNS = 3
    VEHICLE = 4
    WALKER = 5
    OBSTACLE = 6
    SPECIAL_VEHICLE = 7
    BIKER = 8
    TRAFFIC_GREEN = 9
    TRAFFIC_RED_NORMAL = 10
    TRAFFIC_RED_NOT_NORMAL = 11


class BEVOccupancyClass(IntEnum):
    """Indices to access the BEV occupancy map.

    Parking vehicles are not a separate class; they are labeled VEHICLE.
    """

    UNLABELED = 0
    VEHICLE = 1
    WALKER = 2
    OBSTACLE = 3
    SPECIAL_VEHICLE = 4
    BIKER = 5
    TRAFFIC_GREEN = 6
    TRAFFIC_RED_NORMAL = 7
    TRAFFIC_RED_NOT_NORMAL = 8
