"""Dataset identifiers used across data collection, training and evaluation."""

from enum import IntEnum, auto


class TargetDataset(IntEnum):
    """Dataset we target to collect data/train/evaluate on."""

    UNKNOWN = 0
    CARLA_LEADERBOARD2_3CAMERAS = auto()
    CARLA_LEADERBOARD2_6CAMERAS = auto()
    CARLA_LEADERBOARD2_1CAMERA = auto()


class SourceDataset(IntEnum):
    """Dataset intenum a sample originates from."""

    UNKNOWN = 0
    CARLA = auto()


SOURCE_DATASET_NAME_MAP = {
    SourceDataset.CARLA: "carla",
}
