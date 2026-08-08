"""Ego-status baseline policy configuration."""

from py123d.datatypes.sensors.base_camera import CameraID

from lead.config.node import overridable_property
from lead.config.policy.abstract_policy_config import AbstractPolicyConfig


class EgoStatusConfig(AbstractPolicyConfig):
    """MLP baseline that plans from the target point and ego speed alone."""

    @overridable_property
    def cache_store_dir_name(self) -> str:
        """Directory name of this policy's cache store under the dataset root."""
        return "ego_status_training_cache"

    @property
    def input_cameras(self) -> list[CameraID]:
        """The baseline ingests no cameras."""
        return []

    # The baseline reads no past at all, only the anchor tick and the future.
    past_ego_pose_length_s: float = 0.0
    past_lidar_length_s: float = 0.0
    past_radar_length_s: float = 0.0
    past_rgb_length_s: float = 0.0

    # Width of the MLP hidden layers.
    hidden_dim: int = 256
    # Number of MLP hidden layers.
    num_hidden_layers: int = 2
