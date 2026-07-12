"""BEV map class enums produced by TransFuser."""

from enum import IntEnum


class TransfuserBEVSemanticClass(IntEnum):
    """Indices to access BEV semantic map produced by TransFuser.

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


class TransfuserBEVOccupancyClass(IntEnum):
    """Indices to access BEV occupancy map produced by TransFuser.

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
