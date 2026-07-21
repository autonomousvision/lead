"""The layout of a LEAD 123D log: modality ids and sensor-id mappings, so
reading a LEAD log needs neither the simulator nor the expert."""

import jaxtyping as jt
import numpy as np
import numpy.typing as npt
from py123d.datatypes import DefaultBoxDetectionLabel, EgoStateSE3Metadata
from py123d.datatypes.detections.box_detection_label import (
    BoxDetectionLabel,
    register_box_detection_label,
)
from py123d.datatypes.sensors.base_camera import CameraID
from py123d.datatypes.sensors.camera_segmentation_label import (
    CameraSegmentationLabel,
    DefaultCameraSegmentationLabel,
    register_camera_segmentation_label,
)
from py123d.datatypes.sensors.radar import RadarID
from py123d.geometry import PoseSE3

# LEAD camera index (1-based; camera ``i`` is ``sensor_rig.cameras[i - 1]``) → 123D camera ID.
# Assigned by mount yaw: index 1 front-left (-57.5°), 2 front (0°), 3 front-right
# (57.5°), 4 rear-right (122.5°), 5 rear (180°), 6 rear-left (-122.5°).
CAMERA_ID_MAPPING: dict[int, CameraID] = {
    1: CameraID.PCAM_L0,
    2: CameraID.PCAM_F0,
    3: CameraID.PCAM_R0,
    4: CameraID.PCAM_R1,
    5: CameraID.PCAM_B0,
    6: CameraID.PCAM_L1,
}

# LEAD radar index (1-based) → 123D radar ID.
RADAR_ID_MAPPING: dict[int, RadarID] = {
    1: RadarID.RADAR_FRONT_LEFT,
    2: RadarID.RADAR_FRONT_RIGHT,
    3: RadarID.RADAR_BACK_RIGHT,
    4: RadarID.RADAR_BACK_LEFT,
}

# Feature key for CARLA's per-point radial (Doppler) velocity in m/s.
RADIAL_VELOCITY_FEATURE = "radial_velocity"

# Split of the nominal sensor rig, and the opt-in split re-recording the
# view-dependent sensors with a perturbated rig.
NORMAL_VIEW_SPLIT = "normal_view"
PERTURBATED_VIEW_SPLIT = "perturbated_view"

# Name of the custom modality stream holding LEAD's per-tick driving state:
# what the expert knew and decided (route, localization, hazards, target speed).
DRIVING_META_MODALITY_ID = "driving_meta"

# Metadata key of the driving-meta stream holding the log's target points, the
# ones the per-tick target point indices index into.
TARGET_POINTS_KEY = "target_points"

# Per-tick driving-meta key holding the box fields the box-detections stream has
# no slot for, keyed by the box's track token.
BOX_ATTRIBUTES_KEY = "box_attributes"

# Per-tick driving-meta key holding the GNSS+IMU-localized ego pose — the noisy
# pose the policy observes — as an SE(3) matrix in the global frame.
LOCALIZED_EGO_STATE_KEY = "localized_ego_state_se3"


def localized_ego_state_se3(
    position: jt.Float[npt.NDArray, " 2"],
    yaw: float,
) -> jt.Float[npt.NDArray, "4 4"]:
    """The localized planar ego pose as an SE(3) matrix in the global frame.

    The localization estimates a planar pose, so the rotation is yaw-only and
    the translation has z = 0.

    Args:
        position: Localized global xy position.
        yaw: Localized yaw angle in radians.

    Returns:
        The 4x4 SE(3) matrix of the pose.
    """
    matrix = np.eye(4)
    cos, sin = np.cos(yaw), np.sin(yaw)
    matrix[:2, :2] = np.array([[cos, -sin], [sin, cos]])
    matrix[:2, 3] = position
    return matrix


def localized_position_yaw(
    matrix: jt.Float[npt.NDArray, "4 4"],
) -> tuple[jt.Float[npt.NDArray, " 2"], float]:
    """Position and yaw of a localized ego-state SE(3) matrix.

    Args:
        matrix: The 4x4 SE(3) matrix of the pose.

    Returns:
        The global xy position and the yaw angle in radians, wrapped to [-pi, pi].
    """
    return matrix[:2, 3], float(np.arctan2(matrix[1, 0], matrix[0, 0]))


# Per-box fields the box-detections stream carries natively (geometry, label,
# identity, velocity, lidar hits); everything else is a box attribute.
NATIVE_BOX_KEYS = frozenset(
    {
        "class",
        "extent",
        "position",
        "yaw",
        "matrix",
        "distance",
        "speed",
        "id",
        "num_points",
    },
)


def ordered_target_points(
    target_points: jt.Float[npt.NDArray, "n 3"],
    target_point_index: int,
) -> tuple[
    jt.Float[npt.NDArray, " 3"],
    jt.Float[npt.NDArray, " 3"],
    jt.Float[npt.NDArray, " 3"],
]:
    """Select the previous, current and next target point.

    At the end of a route the current target point doubles as the next one.

    Args:
        target_points: The route's target points.
        target_point_index: Index of the previous target point.

    Returns:
        The previous, current and next target point.
    """
    last = len(target_points) - 1
    return (
        target_points[min(target_point_index, last)],
        target_points[min(target_point_index + 1, last)],
        target_points[min(target_point_index + 2, last)],
    )


def get_carla_lincoln_mkz_2020_metadata() -> EgoStateSE3Metadata:
    """The ego vehicle every LEAD log is recorded with.

    Returns:
        The geometry of the CARLA Lincoln MKZ 2020, whose frames the logs'
        sensor and ego poses are expressed against.
    """
    # Parameters from the CARLA vehicle model; the rear-axle-to-center transform
    # is derived from them.
    return EgoStateSE3Metadata(
        vehicle_name="carla_lincoln_mkz_2020",
        width=1.83671,
        length=4.89238,
        height=1.49028,
        wheel_base=2.86048,
        center_to_imu_se3=PoseSE3(
            x=1.64855,
            y=0.0,
            z=0.38579,
            qw=1.0,
            qx=0.0,
            qy=0.0,
            qz=0.0,
        ),
        rear_axle_to_imu_se3=PoseSE3.identity(),
    )


@register_box_detection_label
class CarlaBoxDetectionLabel(BoxDetectionLabel):
    """Native LEAD/CARLA box classes, matching the expert's ``bb["class"]`` strings.

    Kept 1:1 so the reader can reproduce the CenterNet label mapping exactly.
    """

    CAR = 0
    WALKER = 1
    BICYCLE = 2
    TRAFFIC_LIGHT = 3
    STOP_SIGN = 4
    STATIC = 5
    STATIC_PROP_CAR = 6
    TRAFFIC_LIGHT_PHYSICAL = 7
    STOP_SIGN_PHYSICAL = 8
    TRAFFIC_SIGN = 9

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
            CarlaBoxDetectionLabel.TRAFFIC_LIGHT_PHYSICAL: DefaultBoxDetectionLabel.TRAFFIC_LIGHT,
            CarlaBoxDetectionLabel.STOP_SIGN_PHYSICAL: DefaultBoxDetectionLabel.TRAFFIC_SIGN,
            CarlaBoxDetectionLabel.TRAFFIC_SIGN: DefaultBoxDetectionLabel.TRAFFIC_SIGN,
        }[self]


@register_camera_segmentation_label
class CarlaCameraSegmentationLabel(CameraSegmentationLabel):
    """Per-pixel semantic classes stored in LEAD 123D logs.

    Matches ``constants.CarlaSemanticSegmentationClass`` — the raw CARLA label
    space of the semantic segmentation camera; reduction to a model's label
    space happens in that model's dataloader.
    """

    UNLABELED = 0
    ROADS = 1
    SIDEWALKS = 2
    BUILDING = 3
    WALL = 4
    FENCE = 5
    POLE = 6
    TRAFFIC_LIGHT = 7
    TRAFFIC_SIGN = 8
    VEGETATION = 9
    TERRAIN = 10
    SKY = 11
    PEDESTRIAN = 12
    RIDER = 13
    CAR = 14
    TRUCK = 15
    BUS = 16
    TRAIN = 17
    MOTORCYCLE = 18
    BICYCLE = 19
    STATIC = 20
    DYNAMIC = 21
    OTHER = 22
    WATER = 23
    ROAD_LINE = 24
    GROUND = 25
    BRIDGE = 26
    RAIL_TRACK = 27
    GUARD_RAIL = 28

    def to_default(self) -> DefaultCameraSegmentationLabel:
        """Inherited, see superclass."""
        return {
            CarlaCameraSegmentationLabel.UNLABELED: DefaultCameraSegmentationLabel.IGNORE,
            CarlaCameraSegmentationLabel.ROADS: DefaultCameraSegmentationLabel.ROAD,
            CarlaCameraSegmentationLabel.SIDEWALKS: DefaultCameraSegmentationLabel.SIDEWALK,
            CarlaCameraSegmentationLabel.BUILDING: DefaultCameraSegmentationLabel.BUILDING,
            CarlaCameraSegmentationLabel.WALL: DefaultCameraSegmentationLabel.BUILDING,
            CarlaCameraSegmentationLabel.FENCE: DefaultCameraSegmentationLabel.BUILDING,
            CarlaCameraSegmentationLabel.POLE: DefaultCameraSegmentationLabel.POLE,
            CarlaCameraSegmentationLabel.TRAFFIC_LIGHT: DefaultCameraSegmentationLabel.TRAFFIC_LIGHT,
            CarlaCameraSegmentationLabel.TRAFFIC_SIGN: DefaultCameraSegmentationLabel.TRAFFIC_SIGN,
            CarlaCameraSegmentationLabel.VEGETATION: DefaultCameraSegmentationLabel.VEGETATION,
            CarlaCameraSegmentationLabel.TERRAIN: DefaultCameraSegmentationLabel.TERRAIN,
            CarlaCameraSegmentationLabel.SKY: DefaultCameraSegmentationLabel.SKY,
            CarlaCameraSegmentationLabel.PEDESTRIAN: DefaultCameraSegmentationLabel.PERSON,
            CarlaCameraSegmentationLabel.RIDER: DefaultCameraSegmentationLabel.RIDER,
            CarlaCameraSegmentationLabel.CAR: DefaultCameraSegmentationLabel.VEHICLE,
            CarlaCameraSegmentationLabel.TRUCK: DefaultCameraSegmentationLabel.VEHICLE,
            CarlaCameraSegmentationLabel.BUS: DefaultCameraSegmentationLabel.VEHICLE,
            CarlaCameraSegmentationLabel.TRAIN: DefaultCameraSegmentationLabel.VEHICLE,
            CarlaCameraSegmentationLabel.MOTORCYCLE: DefaultCameraSegmentationLabel.TWO_WHEELER,
            CarlaCameraSegmentationLabel.BICYCLE: DefaultCameraSegmentationLabel.TWO_WHEELER,
            CarlaCameraSegmentationLabel.STATIC: DefaultCameraSegmentationLabel.OTHER,
            CarlaCameraSegmentationLabel.DYNAMIC: DefaultCameraSegmentationLabel.OTHER,
            CarlaCameraSegmentationLabel.OTHER: DefaultCameraSegmentationLabel.OTHER,
            CarlaCameraSegmentationLabel.WATER: DefaultCameraSegmentationLabel.TERRAIN,
            CarlaCameraSegmentationLabel.ROAD_LINE: DefaultCameraSegmentationLabel.ROAD,
            CarlaCameraSegmentationLabel.GROUND: DefaultCameraSegmentationLabel.TERRAIN,
            CarlaCameraSegmentationLabel.BRIDGE: DefaultCameraSegmentationLabel.BUILDING,
            CarlaCameraSegmentationLabel.RAIL_TRACK: DefaultCameraSegmentationLabel.OTHER,
            CarlaCameraSegmentationLabel.GUARD_RAIL: DefaultCameraSegmentationLabel.BUILDING,
        }[self]
