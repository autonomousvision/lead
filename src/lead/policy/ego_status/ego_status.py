"""Ego-status baseline policy: an MLP from navigation state to waypoints."""

import typing
from collections.abc import Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional

from lead.api.abstract_dataset import SizedDataset
from lead.api.abstract_policy import (
    AbstractPolicy,
    AuxiliaryLog,
    TaskLosses,
)
from lead.config import LeadConfig
from lead.config.policy.ego_status.ego_status_config import EgoStatusConfig
from lead.log_reader import SceneData
from lead.log_reader.carla_decoding import carla_forward_speed
from lead.log_reader.scene_loader import SceneLoader
from lead.policy.ego_status.dataloader.sample import (
    EgoStatusForwardBatch,
    EgoStatusPrediction,
)


class EgoStatus(AbstractPolicy[EgoStatusForwardBatch, EgoStatusPrediction]):
    """MLP that plans waypoints from the target point and ego speed."""

    def __init__(self, lead_config: LeadConfig) -> None:
        """Build the MLP heads.

        Args:
            lead_config: Root config tree.
        """
        super().__init__(lead_config)
        config = lead_config.policy.ego_status
        self.config = config

        layers: list[nn.Module] = []
        input_dim = 3  # target point (x, y) + speed
        for _ in range(config.num_hidden_layers):
            layers += [nn.Linear(input_dim, config.hidden_dim), nn.ReLU()]
            input_dim = config.hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.waypoints_head = nn.Linear(
            config.hidden_dim,
            config.num_ego_pose_prediction * 2,
        )

    def forward(self, batch: EgoStatusForwardBatch) -> EgoStatusPrediction:
        """Predict the plan for one batch.

        Args:
            batch: The batch of model inputs and labels.

        Returns:
            The predicted waypoints.
        """
        features = torch.cat(
            [
                batch["target_point"].float(),
                batch["speed"].float().reshape(-1, 1),
            ],
            dim=1,
        )
        hidden = self.backbone(features)
        return {
            "waypoints": self.waypoints_head(hidden).reshape(hidden.shape[0], -1, 2),
        }

    def compute_loss(
        self,
        predictions: EgoStatusPrediction,
        batch: EgoStatusForwardBatch,
    ) -> tuple[TaskLosses, AuxiliaryLog]:
        """L1 waypoint loss against the logged expert future.

        Args:
            predictions: Model predictions for the batch.
            batch: The batch of model inputs and labels.

        Returns:
            The per-task losses and an empty log dict.
        """
        losses = {
            "loss_waypoints": functional.l1_loss(
                predictions["waypoints"],
                batch["future_waypoints"].float(),
            ),
        }
        return losses, {}

    def build_features(self, scene_data: SceneData) -> Mapping[str, typing.Any]:
        """The navigation inputs of one scene.

        Args:
            scene_data: One tick of raw driving data in the ego view frame.

        Returns:
            The model inputs of one unbatched sample.
        """
        assert scene_data.target_point is not None
        assert scene_data.ego_state is not None
        return {
            "target_point": scene_data.target_point.astype(np.float32),
            "speed": carla_forward_speed(scene_data.ego_state),
        }

    def features_to_batch(
        self,
        features: Mapping[str, typing.Any],
        device: torch.device,
    ) -> EgoStatusForwardBatch:
        """Turn one sample's features into a batch of one on the given device.

        Args:
            features: Model inputs of one sample.
            device: Device the policy runs on.

        Returns:
            The batched model inputs.
        """
        batch: EgoStatusForwardBatch = {}
        for key in ("target_point", "speed"):
            if key in features:
                batch[key] = torch.as_tensor(  # type: ignore[literal-required]
                    np.asarray(features[key]),
                    dtype=torch.float32,
                    device=device,
                )[None]
        return batch

    def get_policy_config(self) -> EgoStatusConfig:
        """Inherited, see superclass."""
        return self.config

    def build_scene_loader(self) -> SceneLoader:
        """The loader over this policy's training scenes, reading what it ingests.

        Returns:
            The scene loader handed to this policy's dataset.
        """
        return SceneLoader(
            self.lead_config.training.data.py123d_data_root,
            self.build_scene_filter(),
            # The baseline reads no sensors, so which rig it draws is moot; the
            # nominal and shifted views share the ego pose and the driving meta.
            perturbation_probability=0.5,
        )

    def build_dataset(self) -> SizedDataset:
        """Build the ego-status training dataset."""
        from lead.policy.ego_status.dataloader.dataset import EgoStatusDataset

        return EgoStatusDataset(
            lead_config=self.lead_config,
            scene_loader=self.build_scene_loader(),
        )

    def per_task_loss_weights(self, epoch: int) -> dict[str, float]:
        """The waypoint loss weight.

        Args:
            epoch: The current training epoch.

        Returns:
            The weight of every per-task loss.
        """
        del epoch
        return {"loss_waypoints": 1.0}
