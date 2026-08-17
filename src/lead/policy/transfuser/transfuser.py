from __future__ import annotations

import typing
from collections.abc import Mapping
from dataclasses import dataclass, field

import jaxtyping as jt
import numpy as np
import torch
from py123d.datatypes.sensors.base_camera import CameraID

from lead.api.abstract_dataset import SizedDataset
from lead.api.abstract_policy import (
    AbstractPolicy,
    AuxiliaryLog,
    TaskLosses,
)
from lead.config import LeadConfig
from lead.config.policy.transfuser.transfuser_config import TransfuserConfig
from lead.log_reader import SceneData
from lead.log_reader.scene_loader import SceneLoader
from lead.policy.transfuser.dataloader.sample import TransfuserForwardBatch
from lead.policy.transfuser.network.bev_decoder import BEVDecoder
from lead.policy.transfuser.network.center_net_decoder import (
    CenterNetBoundingBoxPrediction,
    CenterNetDecoder,
)
from lead.policy.transfuser.network.perspective_decoder import PerspectiveDecoder
from lead.policy.transfuser.network.planning_decoder import PlanningDecoder
from lead.policy.transfuser.network.radar_detector import RadarDetector
from lead.policy.transfuser.network.transfuser_backbone import TransfuserBackbone
from lead.policy.transfuser.utils import precision
from lead.policy.transfuser.utils.gpu_augmentation import augment_rgb_batch

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


class Transfuser(AbstractPolicy[TransfuserForwardBatch, "Prediction"]):
    """TransFuser policy: image + LiDAR fusion backbone with task-specific decoders."""

    def __init__(self, lead_config: LeadConfig) -> None:
        super().__init__(lead_config)
        self.config = lead_config.policy.transfuser

        self.backbone = TransfuserBackbone(lead_config)

        if self.config.use_semantic:
            self.semantic_decoder = PerspectiveDecoder(
                lead_config=lead_config,
                in_channels=self.backbone.num_image_features,
                out_channels=self.config.num_semantic_classes,
                perspective_upsample_factor=self.backbone.perspective_upsample_factor,
                modality="semantic",
            )

        if self.config.use_depth:
            self.depth_decoder = PerspectiveDecoder(
                lead_config=lead_config,
                in_channels=self.backbone.num_image_features,
                out_channels=1,
                perspective_upsample_factor=self.backbone.perspective_upsample_factor,
                modality="depth",
            )

        if self.config.use_bev_semantic:
            self.bev_semantic_decoder = BEVDecoder(
                lead_config,
                self.config.num_bev_semantic_classes,
            )

        if self.config.detect_boxes:
            self.center_net_decoder = CenterNetDecoder(
                self.config.num_box_classes,
                lead_config,
            )

        if self.config.use_radar_detection:
            self.radar_detector = RadarDetector(
                bev_input_dim=self.backbone.num_lidar_features,
                lead_config=lead_config,
            )

        if self.config.use_planning_decoder:
            self.planning_decoder = PlanningDecoder(
                input_bev_channels=self.backbone.num_lidar_features,
                lead_config=lead_config,
            )

    def augment_batch(self, batch: TransfuserForwardBatch) -> TransfuserForwardBatch:
        """Inherited, see superclass."""
        data_config = self.lead_config.training.data
        if not self.training or not data_config.use_color_augmentation:
            return batch
        if "rgb" in batch:
            batch["rgb"] = augment_rgb_batch(
                batch["rgb"],
                data_config.color_augmentation_probability,
            )
        return batch

    def forward(self, batch: TransfuserForwardBatch) -> Prediction:
        auxiliary_log: AuxiliaryLog = {}
        pred_route = pred_future_waypoints = pred_target_speed_distribution = (
            pred_target_speed_scalar
        ) = None
        pred_semantic = pred_depth = pred_bounding_box = pred_bev_semantic = None

        # Backbone
        bev_features, image_features = self.backbone(batch)

        # Radar detection
        radar_features = radar_predictions = None
        if self.config.use_radar_detection:
            radar_features, radar_predictions = self.radar_detector(bev_features, batch)

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
                batch,
                log=auxiliary_log,
            )

        # Semantic segmentation forward pass
        if self.config.use_semantic:
            pred_semantic = self.semantic_decoder(
                batch,
                image_features,
                auxiliary_log,
            )

        # Depth estimation forward pass
        if self.config.use_depth:
            pred_depth = self.depth_decoder(batch, image_features, auxiliary_log)

        # Bounding box detection and BEV semantic segmentation forward pass,
        # the two heads reading the top-down BEV feature grid.
        if self.backbone.builds_bev_feature_grid:
            bev_feature_grid = self.backbone.top_down(bev_features)

            if self.config.detect_boxes:
                pred_bounding_box = self.center_net_decoder(
                    batch,
                    bev_feature_grid,
                    auxiliary_log,
                )

            if self.config.use_bev_semantic:
                pred_bev_semantic = self.bev_semantic_decoder(
                    bev_feature_grid,
                    auxiliary_log,
                )

        # Collect predictions
        return Prediction(
            # Planning prediction
            future_waypoints=pred_future_waypoints,
            target_speed_distribution=pred_target_speed_distribution,
            target_speed_scalar=pred_target_speed_scalar,
            route=pred_route,
            # CARLA perception prediction
            semantic=pred_semantic,
            depth=pred_depth,
            bounding_box=pred_bounding_box,
            bev_semantic=pred_bev_semantic,
            radar_features=radar_features,
            radar_predictions=radar_predictions,
            auxiliary_log=auxiliary_log,
        )

    def compute_loss(
        self,
        predictions: Prediction,
        batch: TransfuserForwardBatch,
    ) -> tuple[TaskLosses, AuxiliaryLog]:
        loss: TaskLosses = {}
        auxiliary_log = predictions.auxiliary_log
        # Semantic segmentation loss
        if self.config.use_semantic:
            assert predictions.semantic is not None
            self.semantic_decoder.compute_loss(
                predictions.semantic,
                batch,
                loss,
                log=auxiliary_log,
            )

        # Depth estimation loss
        if self.config.use_depth:
            assert predictions.depth is not None
            self.depth_decoder.compute_loss(
                predictions.depth,
                batch,
                loss,
                log=auxiliary_log,
            )

        # BEV semantic segmentation loss
        if self.config.use_bev_semantic:
            assert predictions.bev_semantic is not None
            self.bev_semantic_decoder.compute_loss(
                predictions.bev_semantic,
                batch,
                loss,
                log=auxiliary_log,
            )

        # Bounding box detection loss
        if self.config.detect_boxes:
            assert predictions.bounding_box is not None
            self.center_net_decoder.compute_loss(
                data=batch,
                bounding_box_features=predictions.bounding_box,
                losses=loss,
                log=auxiliary_log,
            )

        # Radar detection loss
        if self.config.use_radar_detection:
            assert predictions.radar_predictions is not None
            self.radar_detector.compute_loss(
                pred=predictions.radar_predictions,
                data=batch,
                loss=loss,
                log=auxiliary_log,
            )

        # Planning loss
        if self.config.use_planning_decoder:
            self.planning_decoder.compute_loss(
                data=batch,
                predictions=predictions,
                loss=loss,
                log=auxiliary_log,
            )

        return loss, auxiliary_log

    def build_features(self, scene_data: SceneData) -> Mapping[str, typing.Any]:
        from lead.policy.transfuser.dataloader.features import build_features

        return build_features(scene_data, self.lead_config)

    def features_to_batch(
        self,
        features: Mapping[str, typing.Any],
        device: torch.device,
    ) -> TransfuserForwardBatch:
        # The model consumes float tensors with a leading batch dimension; the
        # town stays a list of strings, as the training collate leaves it.
        batch: TransfuserForwardBatch = {
            "town": [features["town"]],
        }
        for key in (
            "rgb",
            "rasterized_lidar",
            "radar",
            "previous_target_point",
            "target_point",
            "next_target_point",
            "speed",
        ):
            if key not in features:
                continue
            batch[key] = torch.as_tensor(
                np.asarray(features[key]),
                dtype=torch.float32,
                device=device,
            )[None]
        return batch

    def get_policy_config(self) -> TransfuserConfig:
        """Inherited, see superclass."""
        return self.config

    def build_scene_loader(self) -> SceneLoader:
        """The loader over this policy's training scenes, reading what it ingests.

        Returns:
            The scene loader handed to this policy's dataset.
        """
        data = self.lead_config.training.data
        return SceneLoader(
            data.py123d_data_root,
            self.build_scene_filter(),
            perturbation_probability=data.sensor_perturbation_probability,
            target_point_pop_distance=self.config.target_point_pop_distance,
            past_tick_ages=self.config.past_lidar_tick_ages,
            tick_duration_us=round(1e6 / self.lead_config.expert.simulation.carla_fps),
            camera_ids=[CameraID(camera) for camera in self.config.input_cameras],
        )

    def build_dataset(self) -> SizedDataset:
        """Build the TransFuser dataset; the cache store is opened per config."""
        from lead.policy.transfuser.dataloader.dataset import TransfuserDataset

        return TransfuserDataset(
            lead_config=self.lead_config,
            scene_loader=self.build_scene_loader(),
        )

    def per_task_loss_weights(self, epoch: int) -> dict[str, float]:
        return self.config.per_task_loss_weights(epoch)

    def visualize_prediction(
        self,
        data: TransfuserForwardBatch,
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
        # policy runner, which imports this module.
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
        data: TransfuserForwardBatch,
    ) -> GroundTruthVisualizer:
        """Build the visualizer of the ground-truth labels for one sample.

        Args:
            data: Dictionary containing batched input data tensors.

        Returns:
            The ground-truth visualizer.
        """
        # Imported here: the visualization package imports the evaluation
        # policy runner, which imports this module.
        from lead.policy.transfuser.visualization.ground_truth_visualizer import (
            GroundTruthVisualizer,
        )

        return GroundTruthVisualizer(lead_config=self.lead_config, data=data)

    def visualize_features(
        self,
        data: TransfuserForwardBatch,
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
        # policy runner, which imports this module.
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
        if self.config.norm_layers_in_fp32:
            precision.patch_norm_fp32(self)
        self.backbone.requires_grad_(not self.config.freeze_backbone)


@dataclass
class Prediction:
    """Raw output predictions from the model."""

    # Planning prediction
    future_waypoints: jt.Float[torch.Tensor, "bs n_waypoints 2"] | None
    target_speed_distribution: jt.Float[torch.Tensor, "bs num_speed_classes"] | None
    target_speed_scalar: jt.Float[torch.Tensor, " bs"] | None
    route: jt.Float[torch.Tensor, "bs n_checkpoints 2"] | None

    # CARLA perception prediction
    semantic: (
        jt.Float[torch.Tensor, "bs num_semantic_classes img_height img_width"] | None
    )
    bev_semantic: (
        jt.Float[torch.Tensor, "bs num_bev_classes bev_height bev_width"] | None
    )
    depth: jt.Float[torch.Tensor, "bs img_height img_width"] | None
    bounding_box: CenterNetBoundingBoxPrediction | None
    radar_features: jt.Float[torch.Tensor, "B Q C"] | None
    radar_predictions: jt.Float[torch.Tensor, "B Q 4"] | None

    # Auxiliary values the heads recorded during the forward pass; compute_loss
    # appends to it and hands it to the trainer for scalar logging.
    auxiliary_log: AuxiliaryLog = field(default_factory=dict)


@dataclass
class AgentPrediction:
    """A prediction extended with the controls the driving agent tracked from it."""

    prediction: Prediction

    # Target speed after the test-time brake threshold and optional lowering.
    target_speed: torch.Tensor | None
    steer: float
    throttle: float
    brake: float
    # Per-modality controls: None when the model did not predict that modality.
    waypoints_steer: float | None
    waypoints_throttle: float | None
    waypoints_brake: float | None
    route_steer: float | None
    target_speed_throttle: float | None
    target_speed_brake: float | None
