"""Lightning module wrapping the configured policy for training."""

import logging
import os
import typing

import lightning.pytorch as pl
import torch
import wandb
from lightning.pytorch.utilities import rank_zero_only
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from lead.config import LeadConfig
from lead.policy.abstract_policy import build_policy

LOG = logging.getLogger(__name__)


class LeadLightningModule(pl.LightningModule):
    """Training wrapper around the configured :class:`~lead.policy.abstract_policy.AbstractPolicy`.

    Owns loss weighting, optimizer/scheduler construction, scalar and image
    logging, and the raw ``model_{epoch:04d}.pth`` checkpoints consumed by the
    evaluation :class:`lead.evaluation.inference.ensemble.Ensemble`. Everything
    policy-specific — dataset, losses, visualizers — is reached through the
    policy's contracts.
    """

    def __init__(self, config: LeadConfig) -> None:
        super().__init__()
        self.config = config

        model = build_policy(config.training.device, config)
        model.prepare_for_training()

        if config.training.experiment.load_file is not None:
            LOG.info(
                f"Loading model weights from {config.training.experiment.load_file}",
            )
            model.load_state_dict(
                torch.load(
                    config.training.experiment.load_file,
                    map_location="cpu",
                    weights_only=True,
                ),
                strict=config.training.experiment.continue_failed_training,
            )

        LOG.info(
            f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters",
        )
        if config.training.optimization.channel_last:
            model = model.to(memory_format=torch.channels_last)
            LOG.info("Using channel last memory format")

        # Uncompiled reference kept for clean state-dict keys in checkpoints.
        self._raw_model = model
        if config.training.optimization.compile:
            model = torch.compile(
                model,
                fullgraph=True,  # require entire model to be compiled, fail if not
                dynamic=False,  # aggressively specialize to current input shapes
                backend="inductor",
                mode="max-autotune",  # highest autotune + CUDA graph
            )
        self.model = model
        self.detailed_loss_weights: dict[str, float] = {}

    def train_dataloader(self) -> DataLoader:
        """Build the training DataLoader over the 123D CARLA scenes.

        Lightning wraps the loader in a ``DistributedSampler`` (preserving
        shuffle and drop_last) when training on multiple GPUs.

        Returns:
            The training DataLoader.
        """
        dataset = self._raw_model.build_dataset()
        LOG.info(f"Dataset size: {len(dataset)} samples")

        num_gpus = max(1, torch.cuda.device_count())
        num_workers = (
            int(self.config.training.assigned_cpu_cores / num_gpus)
            * self.config.training.data.workers_per_cpu_cores
        )
        return DataLoader(
            dataset,
            batch_size=self.config.training.optimization.batch_size // num_gpus,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=self.config.training.data.prefetch_factor
            if num_workers > 0
            else None,
            persistent_workers=num_workers > 0,
        )

    def configure_optimizers(self) -> dict:
        """Build the AdamW optimizer and per-step cosine annealing scheduler.

        Returns:
            Lightning optimizer configuration with a step-interval scheduler.
        """
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.optimization.lr,
            amsgrad=True,
            weight_decay=self.config.training.optimization.weight_decay,
            fused=True,
        )
        total_steps = int(self.trainer.estimated_stepping_batches)
        steps_per_epoch = max(1, total_steps // max(1, self.trainer.max_epochs))
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=steps_per_epoch,
            T_mult=2,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def on_train_epoch_start(self) -> None:
        """Recompute the normalized loss weights for the current epoch."""
        weights = self._raw_model.detailed_loss_weights(self.current_epoch)
        total_loss_weight = sum(weights.values())
        self.detailed_loss_weights = {
            key: value / total_loss_weight for key, value in weights.items()
        }

    def training_step(self, data: dict, batch_idx: int) -> torch.Tensor:
        """Run forward pass, weight the losses, and log scalars and images.

        Args:
            data: Collated training batch from the policy's dataset.
            batch_idx: Index of the batch within the epoch.

        Returns:
            The weighted total loss.
        """
        data["iteration"] = batch_idx
        data["training_step"] = self.global_step

        predictions = self.model(data=data)
        losses, log = self.model.compute_loss(predictions=predictions, data=data)

        loss = torch.zeros(1, device=self.device)
        for key, value in losses.items():
            # Reshape as sanity check if the loss is a scalar
            loss = loss + self.detailed_loss_weights[key] * value.float().reshape(1)

        self._log_scalars(losses, log, data)
        self._visualize(predictions, data, batch_idx)
        return loss

    def _log_scalars(self, losses: dict, log: dict, data: dict) -> None:
        """Log loss and debug scalars through the Lightning loggers.

        Args:
            losses: Unscaled per-task losses.
            log: Extra metrics returned by ``compute_loss``.
            data: The training batch (for loading-time diagnostics).
        """
        message: dict[str, float | torch.Tensor] = {}
        for name, value in losses.items():
            message[f"unscaled_loss/{name}"] = value.float()
            message[f"scaled_loss/{name}"] = (
                self.detailed_loss_weights[name] * value.float()
            )
        for name, value in log.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().float().mean()
            message[name] = value

        message["debug/epoch"] = float(self.current_epoch)
        message["debug/batch_size_per_gpu"] = float(data["source_dataset"].shape[0])
        message["debug/max_gpu_mem"] = torch.cuda.max_memory_allocated(
            self.device,
        ) / (1024**3)  # Convert to GB
        message["debug/average_loading_time"] = data["loading_time"].float().mean()
        message["debug/average_loading_meta_time"] = (
            data["loading_meta_time"].float().mean()
        )
        message["debug/average_loading_sensor_time"] = (
            data["loading_sensor_time"].float().mean()
        )
        self.log_dict(message, on_step=True, on_epoch=False, rank_zero_only=True)

    def _visualize(
        self,
        predictions: typing.Any,
        data: dict[str, typing.Any],
        batch_idx: int,
    ) -> None:
        """Render and log training visualizations at the configured frequency.

        Args:
            predictions: Model predictions for the current batch.
            data: The training batch.
            batch_idx: Index of the batch within the epoch.
        """
        should_viz = (
            self.trainer.is_global_zero
            and self.config.training.experiment.visualize_training
            and (
                (batch_idx + 1) % self.config.training.experiment.log_images_frequency
                == 0
                or batch_idx <= 1
            )
        )
        if not should_viz:
            return

        policy = self._raw_model
        with torch.no_grad():
            images = {
                "pred": policy.visualize_prediction(
                    data=data,
                    prediction=predictions,
                ).visualize(),
                "gt": policy.visualize_ground_truth(data=data).visualize(),
                "feature_maps": policy.visualize_features(
                    data=data,
                    prediction=predictions,
                ).visualize(),
            }

        if self.config.debug_mode:
            LOG.info(f"Visualizing training sample at step {self.global_step}.")
            save_path = os.path.join("outputs", "training_viz")
            os.makedirs(save_path, exist_ok=True)
            for name, image in images.items():
                Image.fromarray(image).save(
                    os.path.join(
                        save_path,
                        f"{name}_{str(self.global_step).zfill(5)}.png",
                    ),
                )
        if self.config.training.experiment.log_wandb:
            LOG.info(f"Logging training sample to WandB at step {self.global_step}.")
            wandb.log(
                {
                    f"viz/train_{name}": wandb.Image(image)
                    for name, image in images.items()
                },
                commit=False,
            )

    def on_train_epoch_end(self) -> None:
        """Save the raw model checkpoint for the finished epoch."""
        self._save_model_checkpoint()

    @rank_zero_only
    def _save_model_checkpoint(self) -> None:
        """Write ``model_{epoch:04d}.pth`` and prune the previous epoch's file.

        The raw state dict (without Lightning prefixes) is what the evaluation
        ensemble loads; optimizer state for failure resume lives in Lightning's
        ``last.ckpt`` instead.
        """
        if (
            not self.config.training.optimization.save_model_checkpoint
            or self.config.training.experiment.logdir is None
        ):
            return

        epoch = self.current_epoch
        torch.save(
            self._raw_model.state_dict(),
            os.path.join(
                self.config.training.experiment.logdir,
                f"model_{epoch:04d}.pth",
            ),
        )
        LOG.info(f"Saved model checkpoint for epoch {epoch}.")

        last_model_file = os.path.join(
            self.config.training.experiment.logdir,
            f"model_{epoch - 1:04d}.pth",
        )
        if (
            epoch > 0
            and os.path.isfile(last_model_file)
            and epoch not in self.config.training.optimization.epoch_checkpoints_keep
        ):
            os.remove(last_model_file)
            LOG.info(f"Removed model checkpoint for epoch {epoch - 1}.")
