"""Sensor perturbation (augmentation) configuration for data collection."""

from lead.config.node import ConfigNode, overridable_property


class PerturbationConfig(ConfigNode):
    """Camera pose perturbation knobs used during data collection."""

    @overridable_property
    def perturbate_sensors(self) -> bool:
        """If true enable camera perturbation during data collection."""
        return not self._root.expert.data_collection.eval_expert

    # Safety translation perturbation penalty for default scenarios
    default_safety_translation_perturbation_penalty: float = 0.25
    # Safety translation perturbation penalty for urban scenarios with low speed limits
    urban_safety_translation_perturbation_penalty: float = 0.4

    # Minimum value by which the perturbated camera is shifted left and right
    camera_translation_perturbation_min: float = 0.1
    # Maximum value by which the perturbated camera is shifted left and right
    camera_translation_perturbation_max: float = 1.0

    # Minimum value by which the perturbated camera is rotated around the yaw (degrees)
    camera_rotation_perturbation_min: float = 5.0
    # Maximum value by which the perturbated camera is rotated around the yaw (degrees)
    camera_rotation_perturbation_max: float = 12.5
    # Epsilon threshold to ignore rotation augmentation around 0.0 degrees
    camera_rotation_epsilon: float = 0.5
