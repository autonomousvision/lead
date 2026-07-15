"""Loads the trained policy and runs it at test time."""

from __future__ import annotations

import logging
import os
import typing

import torch
from torch.amp.autocast_mode import autocast

from lead.config import LeadConfig
from lead.policy.abstract_policy import AbstractPolicy, build_policy

LOG = logging.getLogger(__name__)


class PolicyRunner:
    """Loads a single trained policy and runs it in closed loop."""

    def __init__(
        self,
        lead_config: LeadConfig,
        model_path: str,
        device: torch.device,
        prefix: str = "model",
    ) -> None:
        """Load the model checkpoint found in ``model_path``.

        Args:
            lead_config: The config tree the model was trained with.
            model_path: Directory holding the trained model weights.
            device: Device to run inference on.
            prefix: Prefix of the model weights file to load.

        Raises:
            ValueError: If ``model_path`` does not hold exactly one weights file.
        """
        self.lead_config = lead_config
        self.device = device

        weight_files = sorted(
            file
            for file in os.listdir(model_path)
            if file.startswith(prefix) and file.endswith(".pth")
        )
        if len(weight_files) != 1:
            raise ValueError(
                f"Expected exactly one '{prefix}*.pth' weight file in {model_path}, "
                f"found {len(weight_files)}: {weight_files}",
            )
        weight_path = os.path.join(model_path, weight_files[0])
        LOG.info(f"Loading model weight from {weight_path}")

        self.policy: AbstractPolicy = build_policy(self.device, lead_config)
        if self.lead_config.training.optimization.sync_batchnorm:
            # convert_sync_batchnorm's stub widens the return type to nn.Module;
            # it always preserves the input module's actual class at runtime.
            self.policy = typing.cast(
                AbstractPolicy,
                torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.policy),
            )
        self.policy.load_state_dict(
            torch.load(weight_path, map_location=self.device, weights_only=True),
            strict=lead_config.evaluation.inference.strict_weight_load,
        )
        self.policy.cuda(device=self.device).eval()

    @torch.inference_mode()
    def forward(self, data: dict[str, typing.Any]) -> typing.Any:
        """Run the policy on one batch of model inputs.

        Args:
            data: The batched model inputs, as built by the policy's
                ``batch_features``.

        Returns:
            The raw prediction of the policy.
        """
        with autocast(
            device_type="cuda",
            dtype=self.lead_config.training.optimization.torch_float_type,
            enabled=self.lead_config.training.optimization.use_mixed_precision_training,
        ):
            return self.policy(data)
