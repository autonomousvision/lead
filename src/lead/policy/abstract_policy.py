"""Abstract interface of a learned driving policy, swappable via ``agent.target``."""

import abc
import importlib
import typing

import torch
from torch import nn
from torch.utils.data import Dataset

from lead.config import LeadConfig


class AbstractPolicy(nn.Module, abc.ABC):
    """Interface a learned driving policy implements for training and evaluation.

    A policy is the trainable model together with its training contracts: the
    dataset producing its model inputs (:meth:`build_dataset`), the forward
    pass (:meth:`forward`), the per-task losses and their weights
    (:meth:`compute_loss`, :meth:`detailed_loss_weights`) and visualizers of
    its predictions, labels and feature maps (:meth:`visualize_prediction`,
    :meth:`visualize_ground_truth`, :meth:`visualize_features`).

    Implementations are selected by the ``module:Class`` path in
    ``lead_config.agent.target`` (usually set by an agent config profile) and
    constructed via :func:`build_policy`, so swapping the policy for training
    and evaluation only requires pointing at a different yaml. For driving in
    CARLA, a policy is wrapped by an
    :class:`~lead.evaluation.abstract_driving_agent.AbstractDrivingAgent`.
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
    def build_dataset(self) -> Dataset:
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

        The exact signature is policy-specific. The returned visualizer
        renders an image via its ``visualize()`` method; callers own any
        saving or logging.
        """

    @abc.abstractmethod
    def visualize_ground_truth(
        self,
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> typing.Any:
        """Build a visualizer rendering the ground-truth labels of one batch.

        The exact signature is policy-specific. The returned visualizer
        renders an image via its ``visualize()`` method; callers own any
        saving or logging.
        """

    @abc.abstractmethod
    def visualize_features(
        self,
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> typing.Any:
        """Build a visualizer rendering the policy's feature maps for one batch.

        The exact signature is policy-specific. The returned visualizer
        renders an image via its ``visualize()`` method; callers own any
        saving or logging.
        """

    def prepare_for_training(self) -> None:
        """Apply policy-specific model preparation before training; no-op by default."""


def build_policy(device: torch.device, lead_config: LeadConfig) -> AbstractPolicy:
    """Instantiate the policy named by ``lead_config.agent.target``.

    Args:
        device: Device the policy runs on.
        lead_config: The config tree.

    Returns:
        The constructed policy.
    """
    module_name, _, class_name = lead_config.agent.target.partition(":")
    policy_class = getattr(importlib.import_module(module_name), class_name)
    if not issubclass(policy_class, AbstractPolicy):
        raise TypeError(
            f"agent.target '{lead_config.agent.target}' is not an AbstractPolicy subclass.",
        )
    return policy_class(device, lead_config)
