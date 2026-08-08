"""TransFuser architecture configuration, combined from the per-section configs."""

from lead.config.node import overridable_property
from lead.config.policy.abstract_policy_config import AbstractPolicyConfig
from lead.config.policy.transfuser.backbone_config import TransfuserBackboneConfig
from lead.config.policy.transfuser.bev_config import TransfuserBevConfig
from lead.config.policy.transfuser.boxes_detection_config import (
    TransfuserBoxesDetectionConfig,
)
from lead.config.policy.transfuser.camera_config import TransfuserCameraConfig
from lead.config.policy.transfuser.lidar_config import TransfuserLidarConfig
from lead.config.policy.transfuser.navigation_config import TransfuserNavigationConfig
from lead.config.policy.transfuser.planning_config import TransfuserPlanningConfig
from lead.config.policy.transfuser.radar_config import TransfuserRadarConfig
from lead.config.policy.transfuser.semantic_config import TransfuserSemanticConfig


class TransfuserConfig(
    TransfuserCameraConfig,
    TransfuserBackboneConfig,
    TransfuserLidarConfig,
    TransfuserRadarConfig,
    TransfuserSemanticConfig,
    TransfuserBevConfig,
    TransfuserBoxesDetectionConfig,
    TransfuserNavigationConfig,
    TransfuserPlanningConfig,
    AbstractPolicyConfig,
):
    """TransFuser architecture, auxiliary heads, and network-input knobs.

    Flat: every per-section attribute is reachable directly on this class.
    """

    @overridable_property
    def cache_store_dir_name(self) -> str:
        """Directory name of this policy's cache store under the dataset root."""
        return "transfuser_training_cache"

    @property
    def future_window_num_iterations(self) -> int:
        """Ticks of future a scene must provide; none unless the planning labels are built."""
        return self.future_ego_pose_num_iterations if self.needs_planning_targets else 0

    @property
    def required_scene_modalities(self) -> list[str]:
        """The spine plus the boxes and cameras the TransFuser sample parts read."""
        return [
            "ego_state_se3",
            "box_detections_se3@initial",
            "custom.driving_meta@initial",
            "camera:all@initial",
        ]

    def per_task_loss_weights(self, _: int) -> dict[str, float]:
        """Computed loss weights for all auxiliary tasks with normalization."""

        weights = {
            "loss_semantic": 1.0,
            "loss_depth": 0.00001,
            "loss_bev_semantic": 1.0,
            "loss_center_net_heatmap": 1.0,
            "loss_center_net_wh": 1.0,
            "loss_center_net_offset": 1.0,
            "loss_center_net_yaw_class": 1.0,
            "loss_center_net_yaw_res": 1.0,
            "loss_center_net_velocity": 1.0,
            "radar_loss": 1.0,
        }

        if self.LTF:
            weights["loss_center_net_velocity"] = 0.0

        if not self.use_semantic:
            weights["loss_semantic"] = 0.0

        if not self.use_depth:
            weights["loss_depth"] = 0.0

        if not self.use_bev_semantic:
            weights["loss_bev_semantic"] = 0.0

        if not self.detect_boxes:
            weights["loss_center_net_heatmap"] = 0.0
            weights["loss_center_net_wh"] = 0.0
            weights["loss_center_net_offset"] = 0.0
            weights["loss_center_net_yaw_class"] = 0.0
            weights["loss_center_net_yaw_res"] = 0.0
            weights["loss_center_net_velocity"] = 0.0

        if not self.predict_box_velocity:
            weights["loss_center_net_velocity"] = 0.0

        if not self.use_radar_detection:
            weights["radar_loss"] = 0.0

        # Planning loss
        weights.update(
            {
                "loss_spatio_temporal_waypoints": 1.0,
                "loss_target_speed": 1.0,
                "loss_spatial_route": 1.0,
            },
        )

        # Disable planning losses during pretraining
        if not self.use_planning_decoder:
            weights["loss_spatio_temporal_waypoints"] = 0.0
            weights["loss_spatial_route"] = 0.0
            weights["loss_target_speed"] = 0.0

        # A planning head that does not run produces no loss, so its weight must
        # leave the normalization sum too.
        if not self.predict_temporal_spatial_waypoints:
            weights["loss_spatio_temporal_waypoints"] = 0.0

        if not self.predict_spatial_path:
            weights["loss_spatial_route"] = 0.0

        if not self.predict_target_speed:
            weights["loss_target_speed"] = 0.0

        return weights
