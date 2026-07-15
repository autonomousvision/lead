"""Project-wide constants, split into categorized submodules.

All names are re-exported here so ``from lead.common import constants`` and
``constants.<NAME>`` keep working.
"""

from lead.common.constants.meshes import (
    BIKER_MESHES,
    CONSTRUCTION_CONE_BB_SIZE,
    CONSTRUCTION_MESHES,
    EMERGENCY_MESHES,
    LOOKUP_TABLE,
    NON_SIGN_TRAFFIC_TYPES,
    TRAFFIC_WARNING_BB_SIZE,
)
from lead.common.constants.radar import RadarDataIndex, RadarLabels
from lead.common.constants.scenarios import SCENARIO_TYPES
from lead.common.constants.semantics import (
    BOX_CLASS_TO_SEMANTIC,
    INSTANCE_ID_MASK,
    SEMANTIC_SEGMENTATION_CONVERTER,
    CarlaSemanticSegmentationClass,
    LeadSemanticSegmentationClass,
    semantic_class,
)
from lead.common.constants.towns import (
    ALL_TOWNS,
    CARLA_MAP_PATHS,
    HIGHWAY_MAX_SPEED_LIMIT,
    OLD_TOWNS,
    SUBURBAN_MAX_SPEED_LIMIT,
    URBAN_MAX_SPEED_LIMIT,
)

__all__ = [
    "ALL_TOWNS",
    "BIKER_MESHES",
    "BOX_CLASS_TO_SEMANTIC",
    "CARLA_MAP_PATHS",
    "CONSTRUCTION_CONE_BB_SIZE",
    "CONSTRUCTION_MESHES",
    "EMERGENCY_MESHES",
    "HIGHWAY_MAX_SPEED_LIMIT",
    "INSTANCE_ID_MASK",
    "LOOKUP_TABLE",
    "NON_SIGN_TRAFFIC_TYPES",
    "OLD_TOWNS",
    "SCENARIO_TYPES",
    "SEMANTIC_SEGMENTATION_CONVERTER",
    "SUBURBAN_MAX_SPEED_LIMIT",
    "TRAFFIC_WARNING_BB_SIZE",
    "URBAN_MAX_SPEED_LIMIT",
    "CarlaSemanticSegmentationClass",
    "LeadSemanticSegmentationClass",
    "RadarDataIndex",
    "RadarLabels",
    "semantic_class",
]
