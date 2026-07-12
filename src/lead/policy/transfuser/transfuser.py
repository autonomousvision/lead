from __future__ import annotations

import typing
from dataclasses import dataclass

import jaxtyping as jt
import torch
from torch.utils.data import Dataset

from lead.common.constants import SourceDataset
from lead.config import LeadConfig
from lead.policy.abstract_policy import AbstractPolicy
from lead.policy.transfuser.decoder.bev_decoder import BEVDecoder
from lead.policy.transfuser.decoder.center_net_decoder import (
    CenterNetBoundingBoxPrediction,
    CenterNetDecoder,
)
from lead.policy.transfuser.decoder.perspective_decoder import PerspectiveDecoder
from lead.policy.transfuser.decoder.planning_decoder import PlanningDecoder
from lead.policy.transfuser.decoder.radar_detector import RadarDetector
from lead.policy.transfuser.encoder.transfuser_backbone import TransfuserBackbone
from lead.policy.transfuser.utils import transfuser_utils as fn

if typing.TYPE_CHECKING:
    from lead.policy.transfuser.visualization.feature_map_visualizer import (
        FeatureMapVisualizer,
    )
    from lead.policy.transfuser.visualization.ground_truth_visualizer import (
        GroundTruthVisualizer,
    )
    from lead.policy.transfuser.visualization.prediction_visualizer import (
        PredictionVisualizer,
    )


class Transfuser(AbstractPolicy):
    """TransFuser policy: image + LiDAR fusion backbone with task-specific decoders."""

    def __init__(
        self,
        device: torch.device,
        lead_config: LeadConfig,
    ) -> None:
        super().__init__(device, lead_config)
        self.config = lead_config.agent.transfuser
        self.log = {}

        self.backbone = TransfuserBackbone(self.device, lead_config)

        if self.config.use_semantic:
            self.semantic_decoder = PerspectiveDecoder(
                lead_config=lead_config,
                in_channels=self.backbone.num_image_features,
                out_channels=self.config.num_semantic_classes,
                perspective_upsample_factor=self.backbone.perspective_upsample_factor,
                modality="semantic",
                device=self.device,
                source_data=SourceDataset.CARLA,
            )

        if self.config.use_depth:
            self.depth_decoder = PerspectiveDecoder(
                lead_config=lead_config,
                in_channels=self.backbone.num_image_features,
                out_channels=1,
                perspective_upsample_factor=self.backbone.perspective_upsample_factor,
                modality="depth",
                device=self.device,
                source_data=SourceDataset.CARLA,
            )

        if self.config.use_bev_semantic:
            self.bev_semantic_decoder = BEVDecoder(
                lead_config,
                self.config.num_bev_semantic_classes,
                self.device,
                source_data=SourceDataset.CARLA,
            )

        if self.config.detect_boxes:
            self.center_net_decoder = CenterNetDecoder(
                self.config.num_bb_classes,
                lead_config,
                self.device,
                source_data=SourceDataset.CARLA,
            )

        if self.config.use_radar_detection:
            self.radar_detector = RadarDetector(
                bev_input_dim=self.backbone.num_lidar_features,
                lead_config=lead_config,
                device=self.device,
            )

        if self.config.use_planning_decoder:
            self.planning_decoder = PlanningDecoder(
                input_bev_channels=self.backbone.num_lidar_features,
                lead_config=lead_config,
                device=self.device,
            ).to(self.device)

    def forward(self, data: dict[str, typing.Any]) -> Prediction:
        self.log = {}
        pred_route = pred_future_waypoints = pred_target_speed_distribution = (
            pred_target_speed_scalar
        ) = None
        pred_semantic = pred_depth = pred_bounding_box = pred_bev_semantic = None

        # Backbone
        bev_features, image_features = self.backbone(data)

        # Radar detection
        radar_features = radar_predictions = None
        if self.config.use_radar_detection:
            radar_features, radar_predictions = self.radar_detector(bev_features, data)

        # Planning heads
        if self.config.use_planning_decoder:
            planner_radar_features = radar_features
            planner_radar_predictions = radar_predictions
            if not self.config.use_radar_detection:
                planner_radar_features = planner_radar_predictions = None
            (
                pred_route,
                pred_future_waypoints,
                pred_target_speed_distribution,
                pred_target_speed_scalar,
            ) = self.planning_decoder(
                bev_features,
                planner_radar_features,
                planner_radar_predictions,
                data,
                log=self.log,
            )

        # Semantic segmentation forward pass
        if self.config.use_semantic:
            pred_semantic = self.semantic_decoder(data, image_features, self.log)

        # Depth estimation forward pass
        if self.config.use_depth:
            pred_depth = self.depth_decoder(data, image_features, self.log)

        # Bounding box detection forward pass
        bev_feature_grid = self.backbone.top_down(bev_features)
        if self.config.detect_boxes:
            pred_bounding_box = self.center_net_decoder(
                data,
                bev_feature_grid,
                self.log,
            )

        # BEV semantic segmentation forward pass
        if self.config.use_bev_semantic:
            pred_bev_semantic = self.bev_semantic_decoder(
                bev_feature_grid,
                self.log,
            )

        # Collect predictions
        return Prediction(
            # Planning prediction
            pred_future_waypoints=pred_future_waypoints,
            pred_target_speed_distribution=pred_target_speed_distribution,
            pred_target_speed_scalar=pred_target_speed_scalar,
            pred_route=pred_route,
            # CARLA perception prediction
            pred_semantic=pred_semantic,
            pred_depth=pred_depth,
            pred_bounding_box=pred_bounding_box,
            pred_bev_semantic=pred_bev_semantic,
            pred_radar_features=radar_features,
            pred_radar_predictions=radar_predictions,
        )

    def compute_loss(
        self,
        predictions: Prediction,
        data: dict[str, typing.Any],
    ) -> tuple[dict[str, torch.Tensor], dict[str, typing.Any]]:
        loss = {}
        # Semantic segmentation loss
        if self.config.use_semantic:
            self.semantic_decoder.compute_loss(
                predictions.pred_semantic,
                data,
                loss,
                log=self.log,
            )

        # Depth estimation loss
        if self.config.use_depth:
            self.depth_decoder.compute_loss(
                predictions.pred_depth,
                data,
                loss,
                log=self.log,
            )

        # BEV semantic segmentation loss
        if self.config.use_bev_semantic:
            self.bev_semantic_decoder.compute_loss(
                predictions.pred_bev_semantic,
                data,
                loss,
                log=self.log,
            )

        # Bounding box detection loss
        if self.config.detect_boxes:
            self.center_net_decoder.compute_loss(
                data=data,
                bounding_box_features=predictions.pred_bounding_box,
                losses=loss,
                log=self.log,
            )

        # Radar detection loss
        if self.config.use_radar_detection:
            self.radar_detector.compute_loss(
                pred=predictions.pred_radar_predictions,
                data=data,
                loss=loss,
                log=self.log,
            )

        # Planning loss
        if self.config.use_planning_decoder:
            self.planning_decoder.compute_loss(
                data=data,
                predictions=predictions,
                loss=loss,
                log=self.log,
            )

        return loss, self.log

    def build_dataset(self) -> Dataset:
        # Imported here so evaluation-time policy imports skip the data pipeline.
        from lead.policy.transfuser.dataloader.dataset import TransfuserDataset

        return TransfuserDataset(lead_config=self.lead_config)

    def detailed_loss_weights(self, epoch: int) -> dict[str, float]:
        return self.config.detailed_loss_weights(epoch)

    def visualize_prediction(
        self,
        data: dict[str, typing.Any],
        prediction: Prediction,
    ) -> PredictionVisualizer:
        """Build the visualizer of the model predictions for one sample.

        Args:
            data: Dictionary containing batched input data tensors.
            prediction: Model outputs to visualize.

        Returns:
            The prediction visualizer.
        """
        # Imported here: the visualization package imports the evaluation
        # ensemble, which imports this module.
        from lead.policy.transfuser.visualization.prediction_visualizer import (
            PredictionVisualizer,
        )

        return PredictionVisualizer(
            lead_config=self.lead_config,
            data=data,
            prediction=prediction,
        )

    def visualize_ground_truth(
        self,
        data: dict[str, typing.Any],
    ) -> GroundTruthVisualizer:
        """Build the visualizer of the ground-truth labels for one sample.

        Args:
            data: Dictionary containing batched input data tensors.

        Returns:
            The ground-truth visualizer.
        """
        # Imported here: the visualization package imports the evaluation
        # ensemble, which imports this module.
        from lead.policy.transfuser.visualization.ground_truth_visualizer import (
            GroundTruthVisualizer,
        )

        return GroundTruthVisualizer(lead_config=self.lead_config, data=data)

    def visualize_features(
        self,
        data: dict[str, typing.Any],
        prediction: Prediction,
    ) -> FeatureMapVisualizer:
        """Build the visualizer of the label and prediction feature maps for one sample.

        Args:
            data: Dictionary containing batched input data tensors.
            prediction: Model outputs to visualize.

        Returns:
            The feature-map visualizer.
        """
        # Imported here: the visualization package imports the evaluation
        # ensemble, which imports this module.
        from lead.policy.transfuser.visualization.feature_map_visualizer import (
            FeatureMapVisualizer,
        )

        return FeatureMapVisualizer(
            lead_config=self.lead_config,
            data=data,
            prediction=prediction,
        )

    def prepare_for_training(self) -> None:
        """Patch norm layers to run in fp32 and optionally freeze the backbone."""
        fn.patch_norm_fp32(self)
        self.backbone.requires_grad_(not self.config.freeze_backbone)


@dataclass
class Prediction:
    """Raw output predictions from the model."""

    # Planning prediction
    pred_future_waypoints: jt.Float[torch.Tensor, "bs n_waypoints 2"] | None
    pred_target_speed_distribution: (
        jt.Float[torch.Tensor, "bs num_speed_classes"] | None
    )
    pred_target_speed_scalar: jt.Float[torch.Tensor, " bs"] | None
    pred_route: jt.Float[torch.Tensor, "bs n_checkpoints 2"] | None

    # CARLA perception prediction
    pred_semantic: (
        jt.Float[torch.Tensor, "bs num_semantic_classes img_height img_width"] | None
    )
    pred_bev_semantic: (
        jt.Float[torch.Tensor, "bs num_bev_classes bev_height bev_width"] | None
    )
    pred_depth: jt.Float[torch.Tensor, "bs img_height img_width"] | None
    pred_bounding_box: CenterNetBoundingBoxPrediction | None
    pred_radar_features: jt.Float[torch.Tensor, "B Q C"] | None
    pred_radar_predictions: jt.Float[torch.Tensor, "B Q 4"] | None
