"""Visualization colors for LiDAR, planning and TransFuser class maps."""

from lead.common.constants.bev import TransfuserBEVSemanticClass
from lead.common.constants.bounding_boxes import TransfuserBoundingBoxClass
from lead.common.constants.semantics import TransfuserSemanticSegmentationClass


def rgb(r, g, b):
    """Dummy function to help with visualizing RGB colors by using a VSCode extension."""
    return (r, g, b)


# Other visualization
LIDAR_COLOR = rgb(0, 0, 0)
EGO_BB_COLOR = rgb(151, 15, 48)
TP_DEFAULT_COLOR = rgb(255, 10, 10)
RADAR_COLOR = rgb(24, 237, 3)
RADAR_DETECTION_COLOR = rgb(255, 24, 0)

# Planning visualization
GROUNDTRUTH_BB_WP_COLOR = rgb(15, 60, 255)
GROUNDTRUTH_FUTURE_WAYPOINT_COLOR = rgb(0, 0, 0)
GROUND_TRUTH_PAST_WAYPOINT_COLOR = rgb(0, 0, 0)
PREDICTION_WAYPOINT_COLOR = rgb(255, 0, 0)
PREDICTION_ROUTE_COLOR = rgb(0, 0, 255)
PREDICTION_WAYPOINT_RADIUS = 10
PREDICTION_ROUTE_RADIUS = 6

# TransFuser BEV Semantic class to color
CARLA_TRANSFUSER_BEV_SEMANTIC_COLOR_CONVERTER = {
    TransfuserBEVSemanticClass.UNLABELED: rgb(0, 0, 0),
    TransfuserBEVSemanticClass.ROAD: rgb(250, 250, 250),
    TransfuserBEVSemanticClass.LANE_MARKERS: rgb(255, 255, 0),
    TransfuserBEVSemanticClass.STOP_SIGNS: rgb(160, 160, 0),
    TransfuserBEVSemanticClass.VEHICLE: rgb(15, 60, 255),
    TransfuserBEVSemanticClass.WALKER: rgb(0, 255, 0),
    TransfuserBEVSemanticClass.OBSTACLE: rgb(255, 0, 0),
    TransfuserBEVSemanticClass.SPECIAL_VEHICLE: rgb(255, 0, 255),
    TransfuserBEVSemanticClass.BIKER: rgb(255, 0, 0),
    TransfuserBEVSemanticClass.TRAFFIC_GREEN: rgb(0, 255, 0),
    TransfuserBEVSemanticClass.TRAFFIC_RED_NORMAL: rgb(255, 0, 0),
    TransfuserBEVSemanticClass.TRAFFIC_RED_NOT_NORMAL: rgb(0, 0, 255),
}

# TransFuser++ semantic segmentation colors for CARLA data
TRANSFUSER_SEMANTIC_COLORS = {
    TransfuserSemanticSegmentationClass.UNLABELED: rgb(255, 255, 255),
    TransfuserSemanticSegmentationClass.VEHICLE: rgb(31, 119, 180),
    TransfuserSemanticSegmentationClass.ROAD: rgb(128, 64, 128),
    TransfuserSemanticSegmentationClass.TRAFFIC_LIGHT: rgb(250, 170, 30),
    TransfuserSemanticSegmentationClass.PEDESTRIAN: rgb(0, 255, 60),
    TransfuserSemanticSegmentationClass.ROAD_LINE: rgb(157, 234, 50),
    TransfuserSemanticSegmentationClass.OBSTACLE: rgb(255, 0, 0),
    TransfuserSemanticSegmentationClass.SPECIAL_VEHICLE: rgb(255, 255, 0),
    TransfuserSemanticSegmentationClass.STOP_SIGN: rgb(125, 0, 0),
    TransfuserSemanticSegmentationClass.BIKER: rgb(220, 20, 60),
}

# TransFuser++ bounding box colors
TRANSFUSER_BOUNDING_BOX_COLORS = {
    TransfuserBoundingBoxClass.VEHICLE: rgb(0, 0, 255),
    TransfuserBoundingBoxClass.WALKER: rgb(0, 255, 0),
    TransfuserBoundingBoxClass.TRAFFIC_LIGHT: rgb(255, 0, 0),
    TransfuserBoundingBoxClass.STOP_SIGN: rgb(250, 160, 160),
    TransfuserBoundingBoxClass.SPECIAL: rgb(0, 0, 255),
    TransfuserBoundingBoxClass.OBSTACLE: rgb(0, 255, 13),
    TransfuserBoundingBoxClass.BIKER: rgb(255, 0, 0),
}
