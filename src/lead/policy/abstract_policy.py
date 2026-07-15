"""Abstract interface of a learned driving policy, swappable via ``policy.target``."""

import abc
import importlib
import typing
from collections.abc import Sized

import torch
from torch import nn
from torch.utils.data import Dataset

from lead.config import LeadConfig
from lead.dataloader import Frame


class SizedDataset(Dataset, Sized, abc.ABC):
    """A map-style :class:`Dataset` reporting its length, as training requires."""


class AbstractPolicy(nn.Module, abc.ABC):
    """Interface a learned driving policy implements for training and evaluation.

    Implementations are selected by the ``module:Class`` path in
    ``lead_config.policy.target`` and built via :func:`build_policy`.
    """

    @abc.abstractmethod
    def __init__(self, device: torch.device, lead_config: LeadConfig) -> None:
        super().__init__()
        self.device = device
        self.lead_config = lead_config

    @abc.abstractmethod
    def forward(self, data: dict[str, typing.Any]) -> typing.Any:
        """Compute predictions for one batch of model inputs."""

    @abc.abstractmethod
    def compute_loss(
        self,
        predictions: typing.Any,
        data: dict[str, typing.Any],
    ) -> tuple[dict[str, torch.Tensor], dict[str, typing.Any]]:
        """Compute training losses for predictions against the batch labels.

        Args:
            predictions: Model predictions for the batch.
            data: The batch of model inputs and labels.

        Returns:
            The per-task losses and a log dict of auxiliary values.
        """

    @abc.abstractmethod
    def build_features(self, frame: Frame) -> dict[str, typing.Any]:
        """Turn one frame of raw driving data into this policy's model inputs.

        The one featurization path of the policy: the training dataset calls it
        on frames read from the logs, the driving agent on frames assembled from
        the simulator, and privileged fields (labels, futures, the map) are read
        only when :attr:`~lead.dataloader.frame.Frame.is_privileged` holds.

        Args:
            frame: One tick of raw driving data in the ego view frame.

        Returns:
            The model inputs of one unbatched sample, plus the training labels
            when the frame is privileged.
        """

    @abc.abstractmethod
    def batch_features(
        self,
        features: dict[str, typing.Any],
        device: torch.device,
    ) -> dict[str, typing.Any]:
        """Turn one sample's features into a batch of one on the given device.

        Training batches sample features with the dataloader's collate; driving
        runs a single frame at a time and calls this to produce the same batched
        layout :meth:`forward` consumes.

        Args:
            features: Model inputs of one sample, as built by :meth:`build_features`.
            device: Device the policy runs on.

        Returns:
            The batched model inputs.
        """

    @abc.abstractmethod
    def build_dataset(self) -> SizedDataset:
        """Build the training dataset producing this policy's model inputs."""

    @abc.abstractmethod
    def detailed_loss_weights(self, epoch: int) -> dict[str, float]:
        """Unnormalized loss weights for one epoch, keyed like the losses of :meth:`compute_loss`.

        Args:
            epoch: The current training epoch.

        Returns:
            The weight of every per-task loss.
        """

    @abc.abstractmethod
    def visualize_prediction(
        self,
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> typing.Any:
        """Build a visualizer rendering the policy's predictions for one batch.

        The exact signature is policy-specific; the returned visualizer renders
        an image via its ``visualize()`` method, leaving any saving or logging
        to the caller.
        """

    @abc.abstractmethod
    def visualize_ground_truth(
        self,
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> typing.Any:
        """Build a visualizer rendering the ground-truth labels of one batch.

        The exact signature is policy-specific; the returned visualizer renders
        an image via its ``visualize()`` method, leaving any saving or logging
        to the caller.
        """

    @abc.abstractmethod
    def visualize_features(
        self,
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> typing.Any:
        """Build a visualizer rendering the policy's feature maps for one batch.

        The exact signature is policy-specific; the returned visualizer renders
        an image via its ``visualize()`` method, leaving any saving or logging
        to the caller.
        """

    def prepare_for_training(self) -> None:
        """Apply policy-specific model preparation before training; no-op by default."""


def build_policy(device: torch.device, lead_config: LeadConfig) -> AbstractPolicy:
    """Instantiate the policy named by ``lead_config.policy.target``.

    Args:
        device: Device the policy runs on.
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
    return policy_class(device, lead_config)
