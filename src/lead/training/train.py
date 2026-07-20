import logging
import os
import sys
import traceback
import typing
import warnings

import lightning.pytorch as pl
import matplotlib
import torch
import wandb
from lightning.pytorch import Callback
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import Logger, TensorBoardLogger, WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.utilities import rank_zero_only
from PIL import Image
from torch.distributed.elastic.multiprocessing.errors import record
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from lead.common import runtime
from lead.common.logging import setup_logging
from lead.config import LeadConfig
from lead.policy.abstract_policy import build_policy
from lead.training import config_io

matplotlib.use("Agg")  # non-GUI backend for headless servers

setup_logging()
LOG = logging.getLogger(__name__)

warnings.filterwarnings(
    "ignore",
    message="Grad strides do not match bucket view strides",
)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r".*argument 'device' of Tensor\.pin_memory\(\) is deprecated.*",
)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r".*argument 'device' of Tensor\.is_pinned\(\) is deprecated.*",
)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r".*Caught DeprecationWarning in pin memory thread.*",
)
warnings.filterwarnings("ignore", category=UserWarning, message=".*epoch parameter.*")
warnings.filterwarnings(
    "ignore",
    category=ResourceWarning,
    message="Implicitly cleaning up.*TemporaryDirectory.*",
)

# Trade float32 matmul precision for tensor-core speed (TF32)
torch.set_float32_matmul_precision("high")


class LeadLightningModule(pl.LightningModule):
    """Training wrapper around the configured :class:`~lead.policy.abstract_policy.AbstractPolicy`.

    Owns loss weighting, optimizer/scheduler construction, scalar and image
    logging, and the raw ``model_{epoch:04d}.pth`` checkpoints consumed by the
    evaluation :class:`lead.evaluation.inference.policy_runner.PolicyRunner`;
    everything policy-specific — dataset, losses, visualizers — is reached
    through the policy's contracts.
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
            # nn.Module.to()'s stub overloads don't cover a memory_format-only
            # call, even though torch documents and supports it at runtime.
            model = model.to(  # pyright: ignore[reportCallIssue]
                memory_format=torch.channels_last,
            )
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
        assert self.trainer.max_epochs is not None, (
            "Trainer must be built with max_epochs set"
        )
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
        message["debug/batch_size_per_gpu"] = float(data["loading_time"].shape[0])
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
            save_path = str(runtime.output_dir_root() / "training_viz")
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
        policy runner loads; optimizer state for failure resume lives in Lightning's
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


def build_loggers(config: LeadConfig) -> list[Logger]:
    """Build the Lightning loggers (TensorBoard always, WandB when enabled).

    Args:
        config: Training configuration.

    Returns:
        The loggers, empty when no logdir is configured.
    """
    if config.training.experiment.logdir is None:
        return []
    loggers: list[Logger] = [
        TensorBoardLogger(
            save_dir=config.training.experiment.logdir,
            name="",
            version="",
        ),
    ]
    if config.training.experiment.log_wandb:
        loggers.append(
            WandbLogger(
                project=config.training.experiment.wandb_project_name,
                name=config.training.experiment.description,
                id=config.training.experiment.wandb_id,
                resume=config.training.experiment.wandb_resume,
                save_dir=config.training.experiment.logdir,
                config=config_io.serializable_config(config),
            ),
        )
    else:
        LOG.info("WandB is disabled. Only TensorBoard logging is enabled.")
    return loggers


def build_trainer(config: LeadConfig) -> pl.Trainer:
    """Build the Lightning trainer from the training configuration.

    Args:
        config: Training configuration.

    Returns:
        The configured trainer.
    """
    callbacks: list[Callback] = [LearningRateMonitor(logging_interval="step")]
    if (
        config.training.optimization.save_model_checkpoint
        and config.training.experiment.logdir is not None
    ):
        # Full trainer state (optimizer, scheduler, loop) for failure resume;
        # raw model_{epoch:04d}.pth files are saved by LeadLightningModule.
        callbacks.append(
            ModelCheckpoint(
                dirpath=config.training.experiment.logdir,
                save_last=True,
                save_top_k=0,
            ),
        )
    return pl.Trainer(
        accelerator="gpu",
        devices="auto",
        strategy=DDPStrategy(broadcast_buffers=False)
        if torch.cuda.device_count() > 1
        else "auto",
        sync_batchnorm=config.training.optimization.sync_batchnorm,
        precision="bf16-mixed"
        if config.training.optimization.use_mixed_precision_training
        else "32-true",
        max_epochs=config.training.optimization.epochs,
        logger=build_loggers(config) or False,
        callbacks=callbacks,
        log_every_n_steps=config.training.experiment.log_scalars_frequency,
        num_sanity_val_steps=0,
        enable_checkpointing=config.training.optimization.save_model_checkpoint,
        benchmark=True,
        default_root_dir=config.training.experiment.logdir,
    )


def _fast_fail_excepthook(exc_type, exc_value, exc_tb) -> None:
    # Print the traceback, then hard-exit to skip slow DataLoader worker
    # teardown (persistent_workers + pin_memory thread can hang for minutes).
    traceback.print_exception(exc_type, exc_value, exc_tb)
    os._exit(1)


@record  # Records error and tracebacks in case of failure
def main() -> None:
    sys.excepthook = _fast_fail_excepthook

    config = config_io.initialize_config()
    pl.seed_everything(config.training.experiment.seed, workers=True)
    config_io.save_config(config)

    module = LeadLightningModule(config)
    trainer = build_trainer(config)

    ckpt_path = None
    if (
        config.training.experiment.continue_failed_training
        and config.training.experiment.logdir is not None
    ):
        last_ckpt = os.path.join(config.training.experiment.logdir, "last.ckpt")
        if os.path.isfile(last_ckpt):
            ckpt_path = last_ckpt
            LOG.info(f"Resuming training from {last_ckpt}")

    trainer.fit(module, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
