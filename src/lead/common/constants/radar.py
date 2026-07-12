"""Radar-related index enums."""

from enum import IntEnum


class RadarLabels(IntEnum):
    """Index to access radar label array."""

    X = 0
    Y = 1
    V = 2
    VALID = 3


class RadarDataIndex(IntEnum):
    """Index to access radar data array."""

    X = 0
    Y = 1
    Z = 2
    V = 3
    SENSOR_ID = 4
