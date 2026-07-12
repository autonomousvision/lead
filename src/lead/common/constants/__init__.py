"""Project-wide constants, split into categorized submodules.

All names are re-exported here so ``from lead.common import constants`` and
``constants.<NAME>`` keep working.
"""

from lead.common.constants.bev import (
    TransfuserBEVOccupancyClass,
    TransfuserBEVSemanticClass,
)
from lead.common.constants.bounding_boxes import (
    TransfuserBoundingBoxClass,
    TransfuserBoundingBoxIndex,
)
from lead.common.constants.datasets import (
    SOURCE_DATASET_NAME_MAP,
    SourceDataset,
    TargetDataset,
)
from lead.common.constants.meshes import (
    BIKER_MESHES,
    CONSTRUCTION_CONE_BB_SIZE,
    CONSTRUCTION_MESHES,
    EMERGENCY_MESHES,
    LOOKUP_TABLE,
    TRAFFIC_WARNING_BB_SIZE,
)
from lead.common.constants.radar import RadarDataIndex, RadarLabels
from lead.common.constants.scenarios import SCENARIO_TYPES
from lead.common.constants.semantics import (
    SEMANTIC_SEGMENTATION_CONVERTER,
    CarlaSemanticSegmentationClass,
    TransfuserSemanticSegmentationClass,
)
from lead.common.constants.towns import (
    ALL_TOWNS,
    CARLA_MAP_PATHS,
    HIGHWAY_MAX_SPEED_LIMIT,
    OLD_TOWNS,
    SUBURBAN_MAX_SPEED_LIMIT,
    TOWN_NAME_TO_INDEX,
    URBAN_MAX_SPEED_LIMIT,
)
from lead.common.constants.vehicle import CARLA_REAR_AXLE
from lead.common.constants.visualization import (
    CARLA_TRANSFUSER_BEV_SEMANTIC_COLOR_CONVERTER,
    EGO_BB_COLOR,
    GROUND_TRUTH_PAST_WAYPOINT_COLOR,
    GROUNDTRUTH_BB_WP_COLOR,
    GROUNDTRUTH_FUTURE_WAYPOINT_COLOR,
    LIDAR_COLOR,
    PREDICTION_ROUTE_COLOR,
    PREDICTION_ROUTE_RADIUS,
    PREDICTION_WAYPOINT_COLOR,
    PREDICTION_WAYPOINT_RADIUS,
    RADAR_COLOR,
    RADAR_DETECTION_COLOR,
    TP_DEFAULT_COLOR,
    TRANSFUSER_BOUNDING_BOX_COLORS,
    TRANSFUSER_SEMANTIC_COLORS,
    rgb,
)

__all__ = [
    "ALL_TOWNS",
    "BIKER_MESHES",
    "CARLA_MAP_PATHS",
    "CARLA_REAR_AXLE",
    "CARLA_TRANSFUSER_BEV_SEMANTIC_COLOR_CONVERTER",
    "CONSTRUCTION_CONE_BB_SIZE",
    "CONSTRUCTION_MESHES",
    "EGO_BB_COLOR",
    "EMERGENCY_MESHES",
    "GROUNDTRUTH_BB_WP_COLOR",
    "GROUNDTRUTH_FUTURE_WAYPOINT_COLOR",
    "GROUND_TRUTH_PAST_WAYPOINT_COLOR",
    "HIGHWAY_MAX_SPEED_LIMIT",
    "LIDAR_COLOR",
    "LOOKUP_TABLE",
    "OLD_TOWNS",
    "PREDICTION_ROUTE_COLOR",
    "PREDICTION_ROUTE_RADIUS",
    "PREDICTION_WAYPOINT_COLOR",
    "PREDICTION_WAYPOINT_RADIUS",
    "RADAR_COLOR",
    "RADAR_DETECTION_COLOR",
    "SCENARIO_TYPES",
    "SEMANTIC_SEGMENTATION_CONVERTER",
    "SOURCE_DATASET_NAME_MAP",
    "SUBURBAN_MAX_SPEED_LIMIT",
    "TOWN_NAME_TO_INDEX",
    "TP_DEFAULT_COLOR",
    "TRAFFIC_WARNING_BB_SIZE",
    "TRANSFUSER_BOUNDING_BOX_COLORS",
    "TRANSFUSER_SEMANTIC_COLORS",
    "URBAN_MAX_SPEED_LIMIT",
    "CarlaSemanticSegmentationClass",
    "RadarDataIndex",
    "RadarLabels",
    "SourceDataset",
    "TargetDataset",
    "TransfuserBEVOccupancyClass",
    "TransfuserBEVSemanticClass",
    "TransfuserBoundingBoxClass",
    "TransfuserBoundingBoxIndex",
    "TransfuserSemanticSegmentationClass",
    "rgb",
]
