"""Colors and radii used by the TransFuser visualizers."""

from lead.common.constants import LeadSemanticSegmentationClass
from lead.policy.transfuser.labels import BEVSemanticClass, BoundingBoxClass


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
    BEVSemanticClass.UNLABELED: rgb(0, 0, 0),
    BEVSemanticClass.ROAD: rgb(250, 250, 250),
    BEVSemanticClass.LANE_MARKERS: rgb(255, 255, 0),
    BEVSemanticClass.STOP_SIGNS: rgb(160, 160, 0),
    BEVSemanticClass.VEHICLE: rgb(15, 60, 255),
    BEVSemanticClass.WALKER: rgb(0, 255, 0),
    BEVSemanticClass.OBSTACLE: rgb(255, 0, 0),
    BEVSemanticClass.SPECIAL_VEHICLE: rgb(255, 0, 255),
    BEVSemanticClass.BIKER: rgb(255, 0, 0),
    BEVSemanticClass.TRAFFIC_GREEN: rgb(0, 255, 0),
    BEVSemanticClass.TRAFFIC_RED_NORMAL: rgb(255, 0, 0),
    BEVSemanticClass.TRAFFIC_RED_NOT_NORMAL: rgb(0, 0, 255),
}

# TransFuser++ semantic segmentation colors for CARLA data
TRANSFUSER_SEMANTIC_COLORS = {
    LeadSemanticSegmentationClass.UNLABELED: rgb(255, 255, 255),
    LeadSemanticSegmentationClass.VEHICLE: rgb(31, 119, 180),
    LeadSemanticSegmentationClass.ROAD: rgb(128, 64, 128),
    LeadSemanticSegmentationClass.TRAFFIC_LIGHT: rgb(250, 170, 30),
    LeadSemanticSegmentationClass.PEDESTRIAN: rgb(0, 255, 60),
    LeadSemanticSegmentationClass.ROAD_LINE: rgb(157, 234, 50),
    LeadSemanticSegmentationClass.OBSTACLE: rgb(255, 0, 0),
    LeadSemanticSegmentationClass.SPECIAL_VEHICLE: rgb(255, 255, 0),
    LeadSemanticSegmentationClass.STOP_SIGN: rgb(125, 0, 0),
    LeadSemanticSegmentationClass.BIKER: rgb(220, 20, 60),
}

# TransFuser++ bounding box colors
TRANSFUSER_BOUNDING_BOX_COLORS = {
    BoundingBoxClass.VEHICLE: rgb(0, 0, 255),
    BoundingBoxClass.WALKER: rgb(0, 255, 0),
    BoundingBoxClass.TRAFFIC_LIGHT: rgb(255, 0, 0),
    BoundingBoxClass.STOP_SIGN: rgb(250, 160, 160),
    BoundingBoxClass.SPECIAL: rgb(0, 0, 255),
    BoundingBoxClass.OBSTACLE: rgb(0, 255, 13),
    BoundingBoxClass.BIKER: rgb(255, 0, 0),
}
