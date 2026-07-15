"""Inference configuration: checkpoint loading and prediction post-processing."""

from lead.config.node import ConfigNode


class InferenceConfig(ConfigNode):
    """Prediction post-processing knobs and which output modality drives each control."""

    # --- Model Settings ---
    # If true lower the target speed using a factor
    lower_target_speed: bool = False
    # Factor to multiply the target speed with when lowering is enabled
    lower_target_speed_factor: float = 0.8
    # Confidence threshold for brake action (full brake applied if confidence exceeds this)
    brake_threshold: float = 0.9
    # If true be strict when load weight
    strict_weight_load: bool = True

    # --- Image Processing ---
    # JPEG quality used in inference (0-100)
    jpeg_quality: int = 90

    # --- Control which output is used for controlling ---
    # Modality used for steering control
    steer_modality: str = "route"
    # Modality used for throttle control
    throttle_modality: str = "target_speed"
    # Modality used for brake control
    brake_modality: str = "target_speed"
