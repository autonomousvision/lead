"""Bounding-box detection head of the TransFuser model."""

from lead.config.node import ConfigNode
from lead.config.policy.transfuser.label_classes import BoundingBoxClass


class TransfuserBoxesDetectionConfig(ConfigNode):
    """CenterNet bounding-box auxiliary task."""

    # If true use the bounding box auxiliary task.
    detect_boxes: bool = True

    @property
    def predict_box_velocity(self) -> bool:
        """Velocity needs motion cues: predicted only with sweep history beyond the anchor tick."""
        return self._root.policy.transfuser.past_lidar_num_iterations > 0

    # Minimum box center height for detection labels (meters).
    min_box_center_z_meter: float = -4
    # Maximum box center height for detection labels (meters).
    max_box_center_z_meter: float = 4
    # List of static object types to include in bounding box detection.
    detected_static_prop_type_ids: list[str] = [
        "static.prop.constructioncone",
        "static.prop.trafficwarning",
        "static.prop.warningconstruction",
        "static.prop.warningaccident",
    ]
    # Confidence of a bounding box that is needed for the detection to be accepted.
    box_confidence_threshold: float = 0.3
    # Maximum number of bounding boxes our system can detect.
    max_num_boxes: int = 90
    # Number of direction bins for object orientation.
    num_yaw_bins: int = 12
    # Top K center keypoints to consider during detection.
    max_center_net_detections: int = 100
    # Kernel size for CenterNet max pooling operation.
    center_net_max_pooling_kernel_size: int = 3
    # Number of input channels for bounding box detection head.
    box_head_input_channels: int = 64
    # Extra width to add when car doors are open for safety.
    open_door_extra_width_meter: float = 1.2
    # Total number of bounding box classes to detect.
    num_box_classes: int = len(BoundingBoxClass)
