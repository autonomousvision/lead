from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import jaxtyping as jt
import numpy as np
import torch
import torch.nn.functional as F

from lead.common.constants import TransfuserBoundingBoxIndex
from lead.config import LeadConfig
from lead.evaluation.inference import nms
from lead.policy.abstract_policy import AbstractPolicy, build_policy
from lead.policy.transfuser.dataloader import dataset_utils as carla_dataset_utils
from lead.policy.transfuser.decoder.center_net_decoder import PredictedBoundingBox
from lead.policy.transfuser.decoder.planning_decoder import decode_two_hot
from lead.policy.transfuser.transfuser import Prediction

np.set_printoptions(suppress=True)

LOG = logging.getLogger(__name__)


class Ensemble:
    def __init__(
        self,
        lead_config: LeadConfig,
        model_path: str,
        device: torch.device,
        prefix: str = "model",
    ) -> None:
        """
        Ensemble constructor: loads all matching checkpoints from ``model_path``.

        Args:
            lead_config: The config tree the models were trained with.
            model_path: Path to the trained model weights.
            device: Device to run inference on.
            prefix: Prefix of the model weights files to load.
        """
        self.lead_config = lead_config
        self.device = device

        # Loading models
        self.nets: list[AbstractPolicy] = []
        for file in sorted(os.listdir(model_path)):
            if file.startswith(prefix) and file.endswith(".pth"):
                LOG.info(f"Loading model weight from {os.path.join(model_path, file)}")
                net = build_policy(self.device, lead_config)
                if self.lead_config.training.optimization.sync_batchnorm:
                    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
                state_dict = torch.load(
                    os.path.join(model_path, file),
                    map_location=self.device,
                    weights_only=True,
                )
                net.load_state_dict(
                    state_dict,
                    strict=lead_config.evaluation.inference.strict_weight_load,
                )
                net.cuda(device=self.device).eval()
                self.nets.append(net)
        self.step = 4  # Constant so produced images start with 5, not really important

    def aggregate_planning_decoder(
        self,
        predictions: list[Prediction],
    ) -> tuple[
        jt.Float[torch.Tensor, "1 num_waypoints 2"] | None,
        jt.Float[torch.Tensor, "1 num_checkpoints 2"] | None,
        jt.Float[torch.Tensor, " 1 1"] | None,
        jt.Float[torch.Tensor, "1 num_speed_classes"] | None,
    ]:
        """Ensemble the outputs of the planning decoder from multiple models.

        Args:
            predictions: List of dictionaries containing the predictions of each model
        Returns:
            pred_routes: The aggregated route.
            pred_future_waypoints: The aggregated future waypoints.
            pred_target_speed_scalar: The aggregated target speed.
            pred_target_speed_distribution: The aggregated target speed distribution.
        """
        pred_routes = pred_future_waypoints = pred_target_speed_scalar = (
            pred_target_speed_distribution
        ) = None

        if self.lead_config.agent.transfuser.use_planning_decoder:
            if self.lead_config.agent.transfuser.predict_target_speed:
                pred_target_speed_logits = torch.stack(
                    [pred.pred_target_speed_distribution[0] for pred in predictions],
                ).mean(dim=0, keepdim=True)  # Average target speed logits.

                pred_target_speed_distribution = F.softmax(
                    pred_target_speed_logits,
                    dim=-1,
                )  # softmax probabilities.
                pred_target_speed_scalar = decode_two_hot(
                    pred_target_speed_distribution,
                    self.lead_config.agent.transfuser.target_speed_classes,
                    self.device,
                ).reshape(1, 1)  # Decode to scalar.
                if (
                    pred_target_speed_distribution[0, 0]
                    > self.lead_config.evaluation.inference.brake_threshold
                ):  # Brake if we are confident enough.
                    pred_target_speed_scalar = torch.Tensor([0.0]).reshape(1, -1)
                if (
                    self.lead_config.evaluation.inference.lower_target_speed
                ):  # Optionally lower the target speed.
                    pred_target_speed_scalar *= (
                        self.lead_config.evaluation.inference.lower_target_speed_factor
                    )

            if self.lead_config.agent.transfuser.predict_temporal_spatial_waypoints:
                pred_future_waypoints = torch.stack(
                    [pred.pred_future_waypoints[0] for pred in predictions],
                ).mean(dim=0, keepdim=True)  # Average waypoints.

            if self.lead_config.agent.transfuser.predict_spatial_path:
                pred_routes = torch.stack(
                    [pred.pred_route[0] for pred in predictions],
                ).mean(dim=0, keepdim=True)  # Average route.

        return (
            pred_routes,
            pred_future_waypoints,
            pred_target_speed_scalar,
            pred_target_speed_distribution,
        )

    def aggregate_bounding_boxes(
        self,
        predictions: list[Prediction],
    ) -> tuple[list[PredictedBoundingBox], list[PredictedBoundingBox]]:
        """
        Args:
            predictions: List of dictionaries containing the predictions of each model
        Returns:
            List of aggregated bounding boxes in vehicle system.
            List of aggregated bounding boxes in image system.
        """
        pred_bounding_boxes_vehicle_system, pred_bounding_boxes_image_system = [], []
        if self.lead_config.agent.transfuser.detect_boxes:
            for prediction in predictions:
                pred_bb = prediction.pred_bounding_box.pred_bounding_box_vehicle_system.squeeze().reshape(
                    -1,
                    9,
                )
                if len(pred_bb) > 0:
                    pred_bounding_boxes_vehicle_system.append(pred_bb)

        if len(pred_bounding_boxes_vehicle_system) > 0:
            pred_bounding_boxes_vehicle_system = nms.non_maximum_suppression(
                pred_bounding_boxes_vehicle_system,
                float(self.lead_config.evaluation.inference.iou_threshold_nms),
            )

            pred_bounding_boxes_image_system = (
                carla_dataset_utils.bb_vehicle_to_image_system(
                    pred_bounding_boxes_vehicle_system,
                    self.lead_config.expert.data_collection.pixels_per_meter,
                    self.lead_config.expert.data_collection.min_x_meter,
                    self.lead_config.expert.data_collection.min_y_meter,
                )
            )

            pred_bounding_boxes_vehicle_system = [
                PredictedBoundingBox(
                    x=float(bb[TransfuserBoundingBoxIndex.X]),
                    y=float(bb[TransfuserBoundingBoxIndex.Y]),
                    w=float(bb[TransfuserBoundingBoxIndex.W]),
                    h=float(bb[TransfuserBoundingBoxIndex.H]),
                    yaw=float(bb[TransfuserBoundingBoxIndex.YAW]),
                    velocity=float(bb[TransfuserBoundingBoxIndex.VELOCITY]),
                    brake=float(bb[TransfuserBoundingBoxIndex.BRAKE]),
                    clazz=int(bb[TransfuserBoundingBoxIndex.CLASS]),
                    score=float(bb[TransfuserBoundingBoxIndex.SCORE]),
                )
                for bb in pred_bounding_boxes_vehicle_system
            ]

            pred_bounding_boxes_image_system = [
                PredictedBoundingBox(
                    x=float(bb[TransfuserBoundingBoxIndex.X]),
                    y=float(bb[TransfuserBoundingBoxIndex.Y]),
                    w=float(bb[TransfuserBoundingBoxIndex.W]),
                    h=float(bb[TransfuserBoundingBoxIndex.H]),
                    yaw=float(bb[TransfuserBoundingBoxIndex.YAW]),
                    velocity=float(bb[TransfuserBoundingBoxIndex.VELOCITY]),
                    brake=float(bb[TransfuserBoundingBoxIndex.BRAKE]),
                    clazz=int(bb[TransfuserBoundingBoxIndex.CLASS]),
                    score=float(bb[TransfuserBoundingBoxIndex.SCORE]),
                )
                for bb in pred_bounding_boxes_image_system
            ]

        return pred_bounding_boxes_vehicle_system, pred_bounding_boxes_image_system

    def aggregate_bev_semantic(
        self,
        predictions: list[Prediction],
    ) -> jt.Float[torch.Tensor, "B num_classes bev_height bev_width"] | None:
        """
        Args:
            predictions: List of dictionaries containing the predictions of each model
        Returns:
            pred_bev_semantic: Tensor containing the aggregated BEV semantic map
        """
        if self.lead_config.agent.transfuser.use_bev_semantic:
            pred_bev_semantic = []
            for prediction in predictions:
                pred_bev_semantic.append(prediction.pred_bev_semantic)
            stacked = torch.stack(
                pred_bev_semantic,
                dim=0,
            )  # (num_models, num_batches, num_classes, H, W)
            ch0 = (
                stacked[:, :, 0].min(dim=0).values.unsqueeze(1)
            )  # (num_batches, 1, H, W)
            others = (
                stacked[:, :, 1:].max(dim=0).values
            )  # (num_batches, num_classes-1, H, W)
            return torch.cat([ch0, others], dim=1)  # (num_batches, num_classes, H, W)
        return None

    def aggregate_depth(
        self,
        predictions: list[Prediction],
    ) -> jt.Float[torch.Tensor, "B img_height img_width"] | None:
        """
        Args:
            predictions: List of dictionaries containing the predictions of each model
        Returns:
            pred_depth: Tensor containing the aggregated depth map
        """
        if self.lead_config.agent.transfuser.use_depth:
            pred_depth = []
            for prediction in predictions:
                pred_depth.append(prediction.pred_depth)
            stacked = torch.stack(pred_depth, dim=0)  # (num_models, num_batches, H, W)
            return stacked.mean(dim=0)  # (num_batches, H, W)
        return None

    def aggregate_semantic_segmentation(
        self,
        predictions: list[Prediction],
    ) -> jt.Float[torch.Tensor, "B num_classes img_height img_width"] | None:
        """
        Args:
            predictions: List of dictionaries containing the predictions of each model
        Returns:
            pred_semantic: Tensor containing the aggregated semantic segmentation map
        """
        if self.lead_config.agent.transfuser.use_semantic:
            pred_semantic = []
            for prediction in predictions:
                pred_semantic.append(prediction.pred_semantic)
            stacked = torch.stack(
                pred_semantic,
                dim=0,
            )  # (num_models, num_batches, num_classes, H, W)
            ch0 = (
                stacked[:, :, 0].min(dim=0).values.unsqueeze(1)
            )  # (num_batches, 1, H, W)
            others = (
                stacked[:, :, 1:].max(dim=0).values
            )  # (num_batches, num_classes-1, H, W)
            return torch.cat([ch0, others], dim=1)  # (num_batches, num_classes, H, W)
        return None

    def aggregate(self, _, predictions: list[Prediction]) -> EnsemblePrediction:
        """
        Args:
            predictions: List of dictionaries containing the predictions of each model
        Returns:
            EnsemblePrediction object containing the aggregated predictions
        """
        # Bounding boxes
        pred_bounding_boxes_vehicle_system, pred_bounding_boxes_image_system = (
            self.aggregate_bounding_boxes(predictions)
        )

        # BEV semantic map
        pred_bev_semantic = self.aggregate_bev_semantic(predictions)

        # Semantic segmentation
        pred_semantic = self.aggregate_semantic_segmentation(predictions)

        # Depth
        pred_depth = self.aggregate_depth(predictions)

        # Planning
        (
            pred_route,
            pred_future_waypoints,
            pred_target_speed_scalar,
            pred_target_speed_distribution,
        ) = self.aggregate_planning_decoder(predictions)

        return EnsemblePrediction(
            pred_future_waypoints=pred_future_waypoints,
            pred_target_speed_scalar=pred_target_speed_scalar,
            pred_target_speed_distribution=pred_target_speed_distribution,
            pred_route=pred_route,
            pred_semantic=pred_semantic,
            pred_depth=pred_depth,
            pred_bev_semantic=pred_bev_semantic,
            pred_bounding_box_vehicle_system=pred_bounding_boxes_vehicle_system,
            pred_bounding_box_image_system=pred_bounding_boxes_image_system,
            pred_radar_predictions=None,
        )

    @torch.inference_mode()
    def forward(self, data: dict[str, torch.Tensor]) -> EnsemblePrediction:
        """Run inference on the ensemble of models.
        Args:
            data: Dictionary containing the input data for the model

        Returns:
            EnsemblePrediction object containing the aggregated predictions
        """
        self.step += 1
        with torch.amp.autocast(
            device_type="cuda",
            dtype=self.lead_config.training.optimization.torch_float_type,
            enabled=self.lead_config.training.optimization.use_mixed_precision_training,
        ):
            self.predictions: list[Prediction] = [net(data) for net in self.nets]
        return self.aggregate(data, self.predictions)

    def __getitem__(self, index):
        return self.nets[index]


@dataclass
class EnsemblePrediction:
    """Aggregated raw predictions of the model ensemble."""

    pred_future_waypoints: jt.Float[torch.Tensor, "bs n_waypoints 2"] | None
    pred_target_speed_scalar: jt.Float[torch.Tensor, "bs 1"] | None
    pred_target_speed_distribution: (
        jt.Float[torch.Tensor, "bs num_speed_classes"] | None
    )
    pred_route: jt.Float[torch.Tensor, "bs n_checkpoints 2"] | None
    pred_semantic: (
        jt.Float[torch.Tensor, "bs num_sem_classes img_height img_width"] | None
    )
    pred_depth: jt.Float[torch.Tensor, "bs img_height img_width"] | None
    pred_bev_semantic: (
        jt.Float[torch.Tensor, "bs num_bev_classes bev_height bev_width"] | None
    )
    pred_bounding_box_vehicle_system: list[PredictedBoundingBox] | None
    pred_bounding_box_image_system: list[PredictedBoundingBox] | None
    pred_radar_predictions: None


@dataclass
class AgentPrediction(EnsemblePrediction):
    """Ensemble prediction extended with the tracked and selected vehicle controls."""

    steer: float
    throttle: float
    brake: float
    waypoints_steer: float
    waypoints_throttle: float
    waypoints_brake: float
    route_steer: float
    target_speed_throttle: float
    target_speed_brake: float
