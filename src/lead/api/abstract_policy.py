"""Abstract interface of a learned driving policy, swappable via ``policy.target``."""

import abc
import importlib
import typing
from collections.abc import Mapping
from typing import Generic, TypeVar

import torch
from py123d.api.scene.scene_filter import SceneFilter
from py123d.datatypes.sensors.base_camera import CameraID
from torch import nn

from lead.api.abstract_dataset import SizedDataset
from lead.api.scene_data import SceneData
from lead.config import LeadConfig
from lead.config.policy.abstract_policy_config import AbstractPolicyConfig

# The per-task losses of one batch, keyed exactly as
# :meth:`AbstractPolicy.per_task_loss_weights` keys its weights.
TaskLosses = dict[str, torch.Tensor]
# Auxiliary scalars the trainer logs alongside the losses; a tensor value is
# reduced to its mean before logging.
AuxiliaryLog = dict[str, float | torch.Tensor]

# The batch a policy consumes and the prediction it returns. A policy names
# both when it subclasses, e.g. ``AbstractPolicy[TransfuserForwardBatch, Prediction]``.
BatchT = TypeVar("BatchT", bound=Mapping[str, typing.Any])
PredictionT = TypeVar("PredictionT")


class AbstractPolicy(nn.Module, abc.ABC, Generic[BatchT, PredictionT]):
    """Interface a learned driving policy implements for training and evaluation.

    Implementations are selected by the ``module:Class`` path in
    ``lead_config.policy.target`` and built via :func:`build_policy`.
    """

    @abc.abstractmethod
    def __init__(self, lead_config: LeadConfig) -> None:
        super().__init__()
        self.lead_config = lead_config

    @abc.abstractmethod
    def get_policy_config(self) -> AbstractPolicyConfig:
        """This policy's config section of the tree, ``policy.<name>``.

        Returns:
            The section holding this policy's knobs.
        """
        raise NotImplementedError

    @property
    def input_cameras(self) -> list[CameraID]:
        """The cameras this policy ingests, in left-to-right stitch order.

        Returns:
            The camera ids, empty for a policy that reads no cameras.
        """
        return self.get_policy_config().input_cameras

    @abc.abstractmethod
    def forward(self, batch: BatchT) -> PredictionT:
        """Compute predictions for one batch of model inputs."""
        raise NotImplementedError

    def augment_batch(self, batch: BatchT) -> BatchT:
        """Augment one collated training batch in place on its device.

        Called by the trainer before the forward pass and outside any compiled
        graph; inference never calls it.

        Args:
            batch: The collated batch of model inputs and labels.

        Returns:
            The batch with the policy's augmentations applied; no-op by default.
        """
        return batch

    @abc.abstractmethod
    def compute_loss(
        self,
        predictions: PredictionT,
        batch: BatchT,
    ) -> tuple[TaskLosses, AuxiliaryLog]:
        """Compute training losses for predictions against the batch labels.

        Args:
            predictions: Model predictions for the batch.
            batch: The batch of model inputs and labels.

        Returns:
            The per-task losses and a log dict of auxiliary values.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def build_features(self, scene_data: SceneData) -> Mapping[str, typing.Any]:
        """Turn one scene of raw driving data into this policy's model inputs.

        The one featurization path of the policy: the training dataset calls it
        on scenes read from the logs, the driving agent on scene data assembled
        from the simulator. Labels are not built here — the training dataset
        builds them separately from the privileged fields.

        Args:
            scene_data: One tick of raw driving data in the ego view frame.

        Returns:
            The model inputs of one unbatched sample.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def features_to_batch(
        self,
        features: Mapping[str, typing.Any],
        device: torch.device,
    ) -> BatchT:
        """Turn one sample's features into a batch of one on the given device.

        Training batches sample features with the dataloader's collate; driving
        runs a single scene at a time and calls this to produce the same batched
        layout :meth:`forward` consumes.

        Args:
            features: Model inputs of one sample, as built by :meth:`build_features`.
            device: Device the policy runs on.

        Returns:
            The batched model inputs.
        """
        raise NotImplementedError

    def build_scene_filter(self) -> SceneFilter:
        """The scenes this policy trains on: the dataset-wide selection plus its margins.

        Assembled once here from ``training.data`` and the config contract; a
        policy shapes it through :meth:`get_policy_config`, not by overriding.

        Returns:
            The scene filter handed to this policy's scene loader.
        """
        data = self.lead_config.training.data
        policy_config = self.get_policy_config()
        return SceneFilter(
            split_names=[data.py123d_split],
            log_names=list(data.py123d_log_names) or None,
            log_locations=list(data.towns) or None,
            max_num_scenes=data.max_num_scenes or None,
            num_chunks=data.num_chunks if data.num_chunks > 1 else None,
            chunk_idx=data.chunk_index if data.num_chunks > 1 else None,
            shuffle=data.shuffle_scenes,
            # Both margins must be set: py123d enumerates nothing when a
            # history is asked for against an unset future.
            history_num_iterations=policy_config.required_history_num_iterations,
            future_num_iterations=policy_config.future_window_num_iterations,
            required_scene_modalities=policy_config.required_scene_modalities,
        )

    @abc.abstractmethod
    def build_dataset(self) -> SizedDataset:
        """Build the training dataset producing this policy's model inputs.

        The policy owns the whole read side: which scenes to enumerate (its
        ``SceneFilter``) and how to read each one (its ``SceneLoader``), both
        built here and handed to the dataset. The dataset only turns what
        comes back into tensors.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def per_task_loss_weights(self, epoch: int) -> dict[str, float]:
        """Unnormalized loss weights for one epoch, keyed like the losses of :meth:`compute_loss`.

        Args:
            epoch: The current training epoch.

        Returns:
            The weight of every per-task loss.
        """
        raise NotImplementedError

    def visualize_prediction(
        self,
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> typing.Any:
        """Build a visualizer rendering the policy's predictions for one batch.

        The exact signature is policy-specific; the returned visualizer renders
        an image via its ``visualize()`` method, leaving any saving or logging
        to the caller. Optional: the trainer skips visualization when unimplemented.
        """
        raise NotImplementedError

    def visualize_ground_truth(
        self,
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> typing.Any:
        """Build a visualizer rendering the ground-truth labels of one batch.

        The exact signature is policy-specific; the returned visualizer renders
        an image via its ``visualize()`` method, leaving any saving or logging
        to the caller. Optional: the trainer skips visualization when unimplemented.
        """
        raise NotImplementedError

    def visualize_features(
        self,
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> typing.Any:
        """Build a visualizer rendering the policy's feature maps for one batch.

        The exact signature is policy-specific; the returned visualizer renders
        an image via its ``visualize()`` method, leaving any saving or logging
        to the caller. Optional: the trainer skips visualization when unimplemented.
        """
        raise NotImplementedError

    def prepare_for_training(self) -> None:
        """Apply policy-specific model preparation before training; no-op by default."""


def build_policy(
    lead_config: LeadConfig,
) -> "AbstractPolicy[typing.Any, typing.Any]":
    """Instantiate the policy named by ``lead_config.policy.target``.

    The policy is built wherever the caller is; whoever runs it places it, so a
    Lightning trainer can own placement and an evaluation run can call ``to()``.

    Args:
        lead_config: The config tree.

    Returns:
        The constructed policy.
    """
    module_name, _, class_name = lead_config.policy.target.partition(":")
    policy_class = getattr(importlib.import_module(module_name), class_name)
    if not issubclass(policy_class, AbstractPolicy):
        raise TypeError(
            f"policy.target '{lead_config.policy.target}' is not an AbstractPolicy subclass.",
        )
    return policy_class(lead_config)
