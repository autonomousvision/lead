"""Bounding box index/class enums used in TransFuser."""

from enum import IntEnum


class TransfuserBoundingBoxIndex(IntEnum):
    """Index to access bounding array of TransFuser."""

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


class TransfuserBoundingBoxClass(IntEnum):
    """Bounding box classes used in TransFuser.

    Parking vehicles are not a separate class; they are labeled VEHICLE.
    """

    VEHICLE = 0
    WALKER = 1
    TRAFFIC_LIGHT = 2
    STOP_SIGN = 3
    SPECIAL = 4
    OBSTACLE = 5
    BIKER = 6
