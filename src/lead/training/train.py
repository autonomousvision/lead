import datetime
import faulthandler
import glob
import logging
import os
import sys
import time
import traceback
import typing
import warnings

import lightning.pytorch as pl
import matplotlib
import torch
import torch._inductor.config
from lightning.pytorch import Callback
from lightning.pytorch.callbacks import ThroughputMonitor, TQDMProgressBar
from lightning.pytorch.callbacks.progress.tqdm_progress import Tqdm
from lightning.pytorch.loggers import Logger, WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from torch.distributed.elastic.multiprocessing.errors import record
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from lead.api.abstract_policy import AbstractPolicy, build_policy
from lead.common.logging_setup import setup_logging
from lead.config import LeadConfig
from lead.training import run_config

matplotlib.use("Agg")  # non-GUI backend for headless servers

setup_logging()
LOG = logging.getLogger(__name__)

# A rank killed by SIGSEGV/SIGBUS/SIGABRT dies before Python can report it; this
# prints the C-level stack instead of exiting in silence.
faulthandler.enable()

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
    """Training wrapper around the configured :class:`~lead.api.abstract_policy.AbstractPolicy`.

    Owns loss weighting, optimizer/scheduler construction, scalar and image
    logging, and the raw ``model_{epoch:04d}.pth`` checkpoints consumed by the
    evaluation :class:`lead.evaluation.inference.policy_runner.PolicyRunner`;
    everything policy-specific — dataset, losses, visualizers — is reached
    through the policy's contracts.
    """

    def __init__(self, config: LeadConfig) -> None:
        super().__init__()
        self.config = config

        model = build_policy(config)
        model.prepare_for_training()

        if config.training.experiment.initial_weights_file is not None:
            LOG.info(
                f"Loading model weights from {config.training.experiment.initial_weights_file}",
            )
            model.load_state_dict(
                torch.load(
                    config.training.experiment.initial_weights_file,
                    map_location="cpu",
                    weights_only=True,
                ),
                strict=config.training.experiment.resume_from_last_checkpoint,
            )

        LOG.info(
            f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters",
        )
        if config.training.optimization.use_channels_last_memory_format:
            # nn.Module.to()'s stub overloads don't cover a memory_format-only
            # call, even though torch documents and supports it at runtime.
            model = model.to(  # pyright: ignore[reportCallIssue]
                memory_format=torch.channels_last,
            )
            LOG.info("Using channel last memory format")

        # Dynamo traces beartype's own error reporting and dies inside it, so the
        # two cannot be on together; the type checks are the more useful of the
        # pair on a run that asked for them.
        type_checking = (
            os.environ.get("LEAD_RUNTIME_TYPE_CHECKING", "true").lower() == "true"
        )
        if config.training.optimization.use_torch_compile and type_checking:
            LOG.info("Runtime type checking is on, so the model is not compiled")
        if config.training.optimization.use_torch_compile and not type_checking:
            # Cudagraph trees pool the memory backing graph outputs, so a tensor
            # still referenced after the next step's replay reads overwritten
            # memory and raises. Logging and visualization both outlive the step
            # they came from, so the pooling is off and each graph keeps its own
            # output buffers.
            torch._inductor.config.triton.cudagraph_trees = False  # noqa: SLF001
            # torch.compile wraps the module in an OptimizedModule that keeps
            # its interface; the stubs erase it to a bare callable.
            model = typing.cast(
                "AbstractPolicy[typing.Any, typing.Any]",
                torch.compile(
                    model,
                    dynamic=False,  # aggressively specialize to current input shapes
                    backend="inductor",
                    mode=config.training.optimization.torch_compile_mode,
                    fullgraph=config.training.optimization.torch_compile_fullgraph,
                ),
            )
        self.model = model
        self.normalized_loss_weights: dict[str, float] = {}
        self._num_workers = 0
        self._step_timer = 0.0
        self._last_logged_step = 0
        self._h2d_stream: torch.cuda.Stream | None = None
        self._h2d_event: torch.cuda.Event | None = None

    @property
    def _raw_model(self) -> "AbstractPolicy[typing.Any, typing.Any]":
        """The policy beneath any torch.compile wrapper.

        A property rather than an attribute: assigning the module would
        register it as a second submodule and duplicate every state-dict key.
        """
        return getattr(self.model, "_orig_mod", self.model)

    def train_dataloader(self) -> DataLoader:
        """Build the training DataLoader over the 123D CARLA scenes.

        Lightning wraps the loader in a ``DistributedSampler`` (preserving
        shuffle and drop_last) when training on multiple GPUs.

        Returns:
            The training DataLoader.
        """
        dataset = self._raw_model.build_dataset()
        LOG.info(f"Dataset size: {len(dataset)} samples")

        # The cores are this rank's own, so they are not divided by the rank count;
        # batch_size is the global batch, so each rank loads its share of it.
        num_workers = (
            self.config.training.num_assigned_cpu_cores
            * self.config.training.data.workers_per_cpu_core
        )
        self._num_workers = num_workers
        return DataLoader(
            dataset,
            batch_size=self.config.training.optimization.batch_size
            // self.trainer.world_size,
            shuffle=True,
            drop_last=True,
            collate_fn=getattr(dataset, "collate_fn", None),
            num_workers=num_workers,
            pin_memory=self.config.training.data.pin_memory,
            prefetch_factor=self.config.training.data.prefetch_batches_per_worker
            if num_workers > 0
            else None,
            persistent_workers=num_workers > 0,
            in_order=self.config.training.data.loader_in_order,
        )

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        """Build the AdamW optimizer and per-step cosine annealing scheduler.

        Returns:
            Lightning optimizer configuration with a step-interval scheduler.
        """
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.optimization.learning_rate,
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

    def on_train_start(self) -> None:
        """Log what decides this run's throughput, once, before the first step."""
        optimization = self.config.training.optimization
        data = self.config.training.data
        LOG.info(
            f"Devices: {self.trainer.world_size} x {self.trainer.strategy.__class__.__name__} | "
            f"precision {self.trainer.precision} | "
            f"compile {optimization.use_torch_compile}",
        )
        LOG.info(
            f"Batch: {optimization.batch_size} global, "
            f"{optimization.batch_size // self.trainer.world_size} per device | "
            f"{self.trainer.estimated_stepping_batches} steps over "
            f"{self.trainer.max_epochs} epochs",
        )
        LOG.info(
            f"Loader: {self._num_workers} workers per rank | "
            f"pin_memory {data.pin_memory} | "
            f"prefetch {data.prefetch_batches_per_worker} batches per worker | "
            f"cache store {data.read_from_cache_store}",
        )
        self._step_timer = time.perf_counter()
        self._last_logged_step = 0

    def on_train_batch_end(
        self,
        outputs: torch.Tensor | typing.Mapping[str, typing.Any] | None,
        batch: dict,
        batch_idx: int,
    ) -> None:
        """Log the first step's cost and the rate since the previous log.

        Args:
            outputs: What ``training_step`` returned.
            batch: The batch just trained on.
            batch_idx: Index of the batch within the epoch.
        """
        del outputs, batch
        now = time.perf_counter()
        if batch_idx == 0:
            LOG.info(
                f"First step took {now - self._step_timer:.1f} s "
                f"(worker startup and any compilation included)",
            )
            self._step_timer = now
            return
        steps = self.global_step - self._last_logged_step
        if steps < self.config.training.experiment.log_scalars_every_n_steps:
            return
        elapsed = now - self._step_timer
        batch_size = self.config.training.optimization.batch_size
        LOG.info(
            f"step {self.global_step} | {steps / elapsed:5.2f} it/s | "
            f"{steps * batch_size / elapsed:6.0f} samples/s | "
            f"{torch.cuda.max_memory_allocated() / 1e9:4.1f} GB peak GPU memory",
        )
        self._step_timer = now
        self._last_logged_step = self.global_step

    def on_train_epoch_start(self) -> None:
        """Recompute the normalized loss weights for the current epoch."""
        unnormalized_loss_weights = self._raw_model.per_task_loss_weights(
            self.current_epoch,
        )
        total_loss_weight = sum(unnormalized_loss_weights.values())
        if total_loss_weight == 0:
            raise ValueError(
                "Every task loss weight is zero, so there is nothing to train. "
                "Enable at least one head or the planning decoder.",
            )
        self.normalized_loss_weights = {
            key: value / total_loss_weight
            for key, value in unnormalized_loss_weights.items()
        }

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """Fail on the first step if any parameter takes no gradient from the loss.

        A module that runs but whose loss is switched off trains on nothing.
        DDP's reducer already refuses that; this makes a single device agree
        instead of training the dead head silently.

        Args:
            optimizer: The optimizer about to step.

        Raises:
            ValueError: If a trainable parameter received no gradient.
        """
        del optimizer
        if self.global_step > 0:
            return
        starved = [
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad and parameter.grad is None
        ]
        if starved:
            raise ValueError(
                f"{len(starved)} trainable parameters received no gradient on the "
                f"first step, starting with {starved[:5]}. A head is running "
                f"without contributing to the loss.",
            )

    def transfer_batch_to_device(
        self,
        batch: dict,
        device: torch.device,
        dataloader_idx: int,
    ) -> dict:
        """Upload the batch on a dedicated copy stream.

        The default upload runs on the compute stream, so it queues behind the
        previous step's kernels; a side stream lets the copy engine run the
        transfer while that step is still executing. ``on_train_batch_start``
        fences the compute stream on the recorded event before any kernel
        touches the batch.

        Args:
            batch: The collated batch, on pinned host memory.
            device: The target device of this rank.
            dataloader_idx: Index of the dataloader the batch came from.

        Returns:
            The batch with every tensor on the target device.
        """
        if (
            device.type != "cuda"
            or not self.config.training.data.copy_batch_on_side_stream
        ):
            return super().transfer_batch_to_device(batch, device, dataloader_idx)
        if self._h2d_stream is None:
            self._h2d_stream = torch.cuda.Stream(device=device)
            self._h2d_event = torch.cuda.Event()
        with torch.cuda.stream(self._h2d_stream):
            moved = super().transfer_batch_to_device(batch, device, dataloader_idx)
        assert self._h2d_event is not None
        self._h2d_event.record(self._h2d_stream)
        return moved

    def on_train_batch_start(self, batch: dict, batch_idx: int) -> None:
        """Make the compute stream wait for the batch upload to finish.

        Also marks every batch tensor as used by the compute stream, so the
        caching allocator does not hand their side-stream memory to the next
        upload while this step still reads it.

        Args:
            batch: The batch ``transfer_batch_to_device`` returned.
            batch_idx: Index of the batch within the epoch.
        """
        if self._h2d_event is None:
            return
        compute_stream = torch.cuda.current_stream()
        self._h2d_event.wait(compute_stream)
        for value in batch.values():
            if isinstance(value, torch.Tensor):
                value.record_stream(compute_stream)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Run forward pass, weight the losses, and log scalars and images.

        Args:
            batch: Collated training batch from the policy's dataset.
            batch_idx: Index of the batch within the epoch.

        Returns:
            The weighted total loss.
        """
        # Cumulative optimizer steps, so a decoder can gate its own debug
        # logging on the same cadence the trainer logs scalars at.
        batch["current_gradient_step"] = self.global_step

        # Through the uncompiled module: augmentation is RNG-driven and must
        # stay outside the compiled graph.
        batch = self._raw_model.augment_batch(batch)
        predictions = self.model(batch)
        losses, extra_metrics = self.model.compute_loss(predictions, batch)

        total_loss = torch.zeros(1, device=self.device)
        for key, value in losses.items():
            # Reshape as sanity check if the loss is a scalar
            total_loss = total_loss + self.normalized_loss_weights[
                key
            ] * value.float().reshape(1)

        self._log_scalars(losses, extra_metrics, batch)
        self._visualize(predictions, batch, batch_idx)
        return total_loss

    def _log_scalars(
        self,
        losses: dict,
        extra_metrics: dict,
        batch: dict,
    ) -> None:
        """Log loss and training scalars through the Lightning loggers."""
        scalars_to_log: dict[str, float | torch.Tensor] = {}
        for name, value in losses.items():
            task = name.removeprefix("loss_")
            # ``.float()`` on an fp32 loss returns the tensor itself, which would
            # hand the loggers a view of the model's own output to hold past the
            # end of the step.
            scalars_to_log[f"losses/unscaled_{task}"] = value.detach().float().clone()
            scalars_to_log[f"losses/scaled_{task}"] = (
                self.normalized_loss_weights[name] * value.detach().float()
            )
        for name, value in extra_metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().float().mean()
            scalars_to_log[name] = value

        scalars_to_log["trainer/lr"] = self.trainer.optimizers[0].param_groups[0]["lr"]
        scalars_to_log["trainer/max_gpu_mem"] = torch.cuda.max_memory_allocated(
            self.device,
        ) / (1024**3)  # Convert to GB
        scalars_to_log["trainer/batch_size_per_gpu"] = float(
            batch["loading_time"].shape[0],
        )
        scalars_to_log["trainer/average_loading_time"] = (
            batch["loading_time"].float().mean()
        )
        self.log_dict(scalars_to_log, on_step=True, on_epoch=False, rank_zero_only=True)

    def _visualize(
        self,
        predictions: typing.Any,
        batch: dict[str, typing.Any],
        batch_idx: int,
    ) -> None:
        """Render and log training visualizations at the configured frequency."""
        should_visualize = (
            self.trainer.is_global_zero
            and self.config.training.experiment.visualize_training
            and (
                (batch_idx + 1)
                % self.config.training.experiment.log_images_every_n_steps
                == 0
                or batch_idx <= 1
            )
        )
        if not should_visualize:
            return

        policy = self._raw_model
        try:
            with torch.no_grad():
                images = {
                    "pred": policy.visualize_prediction(
                        data=batch,
                        prediction=predictions,
                    ).visualize(),
                    "gt": policy.visualize_ground_truth(data=batch).visualize(),
                    "feature_maps": policy.visualize_features(
                        data=batch,
                        prediction=predictions,
                    ).visualize(),
                }
        except NotImplementedError:
            return

        self._log_images(images)

    def _log_images(self, images: dict[str, typing.Any]) -> None:
        """Send rendered visualizations to every logger that takes images.

        Scalars reach the loggers through ``log_dict``, which has no image
        equivalent, so each logger's own image call is made here.
        """
        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                for name, image in images.items():
                    logger.log_image(
                        key=f"visualization/train_{name}",
                        images=[image],
                        step=self.global_step,
                    )

    def on_train_epoch_end(self) -> None:
        """Save this epoch's checkpoint files, then prune the previous epoch's.

        One Lightning checkpoint is dumped and split into three files:
        ``model_{epoch:04d}.pth`` (the raw state dict the evaluation policy
        runner loads), ``optimizer_{epoch:04d}.pth`` and
        ``trainer_state_{epoch:04d}.pth`` (scheduler, loops and counters).
        Pruning runs only after the new trio is on disk: the previous model
        file survives when its epoch is listed in
        ``epochs_to_keep_checkpoints_for``; the previous optimizer and trainer
        state are always removed.
        """
        if not _is_checkpointing_enabled(self.config):
            return

        # _is_checkpointing_enabled already rejects a missing output dir.
        output_dir = self.config.training.experiment.output_dir
        assert output_dir is not None

        # Every rank enters the dump (Lightning coordinates the rank-0 write);
        # splitting and pruning are plain file work for rank 0 alone.
        monolith_path = os.path.join(output_dir, ".checkpoint_dump.pth")
        self.trainer.save_checkpoint(monolith_path)
        if not self.trainer.is_global_zero:
            return
        epoch = self.current_epoch
        self._split_checkpoint(monolith_path, output_dir, epoch)
        LOG.info(f"Saved checkpoint files for epoch {epoch}.")
        self._prune_checkpoint(output_dir, epoch - 1)

    def _split_checkpoint(
        self,
        monolith_path: str,
        output_dir: str,
        epoch: int,
    ) -> None:
        """Split a dumped Lightning checkpoint into the three epoch files.

        Args:
            monolith_path: The dumped Lightning checkpoint; removed when split.
            output_dir: Directory the epoch files are written to.
            epoch: The epoch the files belong to.
        """
        checkpoint = torch.load(monolith_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.pop("state_dict")

        # The monolith prefixes every weight key with the module path down to
        # the policy (e.g. ``model.`` or ``model._orig_mod.`` when compiled);
        # stripped for the model file, kept for the resume merge.
        first_raw_key = next(iter(self._raw_model.state_dict()))
        first_full_key = next(iter(state_dict))
        assert first_full_key.endswith(first_raw_key)
        prefix = first_full_key[: len(first_full_key) - len(first_raw_key)]
        assert all(key.startswith(prefix) for key in state_dict)

        model_path, optimizer_path, state_path = _checkpoint_paths(output_dir, epoch)
        torch.save(
            {key.removeprefix(prefix): value for key, value in state_dict.items()},
            model_path,
        )
        torch.save(checkpoint.pop("optimizer_states"), optimizer_path)
        checkpoint["state_dict_prefix"] = prefix
        torch.save(checkpoint, state_path)
        os.remove(monolith_path)

    def _prune_checkpoint(self, output_dir: str, epoch: int) -> None:
        """Remove a previous epoch's checkpoint files.

        Also drops any merged resume checkpoint, which has served its purpose
        once a newer epoch is on disk.

        Args:
            output_dir: Directory holding the checkpoint files.
            epoch: The epoch to prune; negative is a no-op.
        """
        removable = glob.glob(os.path.join(output_dir, ".resume_merged_*.pth"))
        if epoch >= 0:
            model_path, optimizer_path, state_path = _checkpoint_paths(
                output_dir,
                epoch,
            )
            removable += [optimizer_path, state_path]
            keep = self.config.training.optimization.epochs_to_keep_checkpoints_for
            if epoch not in keep:
                removable.append(model_path)
        for path in removable:
            if os.path.isfile(path):
                os.remove(path)


# How long a non-zero rank waits for rank 0's merged resume checkpoint; covers
# reading and rewriting the full trainer state on a slow shared filesystem.
_RESUME_MERGE_TIMEOUT_SECONDS = 1800


def _is_checkpointing_enabled(config: LeadConfig) -> bool:
    """Whether this run writes checkpoints (needs a save flag and an output dir)."""
    return (
        config.training.optimization.save_model_checkpoint
        and config.training.experiment.output_dir is not None
    )


def _checkpoint_paths(output_dir: str, epoch: int) -> tuple[str, str, str]:
    """The model, optimizer and trainer-state file paths of one epoch.

    Args:
        output_dir: Directory holding the checkpoint files.
        epoch: The epoch the files belong to.

    Returns:
        The three file paths.
    """
    return (
        os.path.join(output_dir, f"model_{epoch:04d}.pth"),
        os.path.join(output_dir, f"optimizer_{epoch:04d}.pth"),
        os.path.join(output_dir, f"trainer_state_{epoch:04d}.pth"),
    )


def _merged_resume_checkpoint(output_dir: str) -> str | None:
    """Merge the newest checkpoint trio back into Lightning's monolith format.

    Rank zero writes the merged file (atomically, via rename) into the shared
    output directory; every other rank waits for it to appear, since all ranks
    load the resume checkpoint themselves. Runs before the process group
    exists, so the coordination is filesystem-based.

    Args:
        output_dir: The run's output directory holding the checkpoint files.

    Returns:
        Path of the merged checkpoint, or None when there is nothing to resume.
    """
    state_files = sorted(glob.glob(os.path.join(output_dir, "trainer_state_*.pth")))
    if not state_files:
        return None
    epoch = int(
        os.path.basename(state_files[-1]).removesuffix(".pth").rsplit("_", 1)[-1],
    )
    model_path, optimizer_path, state_path = _checkpoint_paths(output_dir, epoch)
    if not (os.path.isfile(model_path) and os.path.isfile(optimizer_path)):
        LOG.warning(f"Incomplete checkpoint trio for epoch {epoch}, starting fresh.")
        return None

    merged_path = os.path.join(output_dir, f".resume_merged_{epoch:04d}.pth")
    rank = int(os.environ.get("SLURM_PROCID", os.environ.get("LOCAL_RANK", "0")))
    if rank != 0:
        deadline = time.monotonic() + _RESUME_MERGE_TIMEOUT_SECONDS
        while not os.path.isfile(merged_path):
            if time.monotonic() > deadline:
                raise TimeoutError(f"Rank 0 never produced {merged_path}.")
            time.sleep(1.0)
        return merged_path
    if os.path.isfile(merged_path):
        # A completed merge of the same epoch from an earlier attempt; the
        # atomic rename below guarantees it is never half-written.
        return merged_path

    checkpoint = torch.load(state_path, map_location="cpu", weights_only=False)
    prefix = checkpoint.pop("state_dict_prefix")
    weights = torch.load(model_path, map_location="cpu", weights_only=True)
    checkpoint["state_dict"] = {prefix + key: value for key, value in weights.items()}
    checkpoint["optimizer_states"] = torch.load(
        optimizer_path,
        map_location="cpu",
        weights_only=False,
    )
    partial_path = f"{merged_path}.tmp"
    torch.save(checkpoint, partial_path)
    os.replace(partial_path, merged_path)
    return merged_path


def _build_loggers(config: LeadConfig) -> list[Logger]:
    """Build the Lightning loggers (WandB when enabled)."""
    if config.training.experiment.output_dir is None:
        return []
    if not config.training.experiment.log_wandb:
        LOG.info("WandB is disabled; metrics go to the run log only.")
        return []
    return [
        WandbLogger(
            project=config.training.experiment.wandb_project_name,
            name=config.training.experiment.wandb_run_name,
            id=config.training.experiment.wandb_id,
            resume=config.training.experiment.wandb_resume_mode,
            save_dir=config.training.experiment.output_dir,
            config=run_config.to_serializable_dict(config),
        ),
    ]


class _BatchesPerSecondPostfix:
    """Live per-iteration rate readout for a bar's postfix.

    An object rather than a string because tqdm prefixes ``", "`` to any
    string postfix; a non-string object is rendered verbatim.
    """

    def __init__(self, bar: Tqdm) -> None:
        self._bar = bar

    def __str__(self) -> str:
        # tqdm keeps its smoothed rate in raw per-iteration units; only the
        # rendered main rate is scaled by unit_scale. The rate is None between
        # smoothing updates, where tqdm itself falls back to the overall
        # average.
        bar_state = self._bar.format_dict
        rate = bar_state["rate"]
        if rate is None and bar_state["elapsed"]:
            rate = bar_state["n"] / bar_state["elapsed"]
        return f" {rate:.2f} batches/s" if rate else ""


class _SamplesProgressBar(TQDMProgressBar):
    """Progress bar counting samples instead of iterations.

    The global batch size scales the main rate, so runs with different batch
    sizes report directly comparable samples/s; the per-iteration rate follows
    as a ``batches/s`` readout.
    """

    def __init__(self, samples_per_iteration: int) -> None:
        super().__init__()
        self._samples_per_iteration = samples_per_iteration
        self._rate_postfix: _BatchesPerSecondPostfix | None = None

    def init_train_tqdm(self) -> Tqdm:
        """Inherited, see superclass."""
        bar = super().init_train_tqdm()
        # The leading space separates the rate number from its unit.
        bar.unit = " samples"
        bar.unit_scale = self._samples_per_iteration
        self._rate_postfix = _BatchesPerSecondPostfix(bar)
        bar.postfix = self._rate_postfix
        return bar

    def get_metrics(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> dict[str, int | str | float | dict[str, float]]:
        """Inherited, see superclass; drops the logger-version entry.

        Args:
            trainer: The running trainer.
            pl_module: The trained module.

        Returns:
            The postfix metrics of the bar.
        """
        metrics = super().get_metrics(trainer, pl_module)
        # The truncated logger version; the output dir and wandb page identify
        # the run already.
        metrics.pop("v_num", None)
        return metrics

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: typing.Any,
        batch: typing.Any,
        batch_idx: int,
    ) -> None:
        """Inherited, see superclass; keeps the rate postfix in place.

        The superclass pushes ``get_metrics`` through ``set_postfix`` here,
        which would replace the rate object with a metrics string.

        Args:
            trainer: The running trainer.
            pl_module: The trained module.
            outputs: The training step outputs.
            batch: The finished batch.
            batch_idx: Index of the finished batch.
        """
        n = batch_idx + 1
        if self._should_update(n, self.train_progress_bar.total):
            self.train_progress_bar.n = n
            self.train_progress_bar.refresh()

    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Inherited, see superclass; restores the rate postfix it replaces.

        Args:
            trainer: The running trainer.
            pl_module: The trained module.
        """
        super().on_train_epoch_end(trainer, pl_module)
        if self._train_progress_bar is not None:
            self.train_progress_bar.postfix = self._rate_postfix

    def on_validation_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Inherited, see superclass; restores the rate postfix it replaces.

        Args:
            trainer: The running trainer.
            pl_module: The trained module.
        """
        super().on_validation_end(trainer, pl_module)
        if self._train_progress_bar is not None:
            self.train_progress_bar.postfix = self._rate_postfix


def _build_trainer(config: LeadConfig) -> pl.Trainer:
    """Build the Lightning trainer from the training configuration."""
    # Steps and samples per second, so throughput is a logged metric rather than
    # something to read off a progress bar.
    callbacks: list[Callback] = [
        ThroughputMonitor(batch_size_fn=lambda batch: len(batch["loading_time"])),
    ]
    if config.training.lightning.enable_progress_bar:
        callbacks.append(
            _SamplesProgressBar(config.training.optimization.batch_size),
        )
    lightning_config = config.training.lightning
    return pl.Trainer(
        accelerator=lightning_config.accelerator,
        devices=lightning_config.devices,
        num_nodes=lightning_config.num_nodes,
        # "auto" builds DDP from the ddp_* knobs; a named strategy is passed
        # through as the user spelled it.
        strategy=DDPStrategy(
            broadcast_buffers=lightning_config.ddp_broadcast_buffers,
            static_graph=lightning_config.ddp_static_graph,
            gradient_as_bucket_view=lightning_config.ddp_gradient_as_bucket_view,
            timeout=datetime.timedelta(seconds=lightning_config.ddp_timeout_seconds),
        )
        if lightning_config.strategy == "auto" and torch.cuda.device_count() > 1
        else lightning_config.strategy,
        sync_batchnorm=config.training.optimization.sync_batchnorm,
        precision="bf16-mixed"
        if config.training.optimization.use_mixed_precision_training
        else "32-true",
        max_epochs=config.training.optimization.num_epochs,
        max_steps=lightning_config.max_steps,
        limit_train_batches=lightning_config.limit_train_batches,
        logger=_build_loggers(config) or False,
        callbacks=callbacks,
        log_every_n_steps=config.training.experiment.log_scalars_every_n_steps,
        num_sanity_val_steps=lightning_config.num_sanity_val_steps,
        accumulate_grad_batches=lightning_config.accumulate_grad_batches,
        gradient_clip_val=lightning_config.gradient_clip_val or None,
        gradient_clip_algorithm=lightning_config.gradient_clip_algorithm,
        deterministic=lightning_config.deterministic,
        enable_progress_bar=lightning_config.enable_progress_bar,
        enable_model_summary=lightning_config.enable_model_summary,
        profiler=lightning_config.profiler or None,
        fast_dev_run=lightning_config.fast_dev_run or False,
        overfit_batches=lightning_config.overfit_batches,
        detect_anomaly=lightning_config.detect_anomaly,
        # Checkpointing is owned by LeadLightningModule.on_train_epoch_end;
        # True would make Lightning add its own default ModelCheckpoint.
        enable_checkpointing=False,
        benchmark=lightning_config.benchmark,
        default_root_dir=config.training.experiment.output_dir,
    )


def _fast_fail_excepthook(exc_type, exc_value, exc_tb) -> None:
    # Print the traceback, then hard-exit to skip slow DataLoader worker
    # teardown (persistent_workers + pin_memory thread can hang for minutes).
    # os._exit skips the interpreter's flush, so the traceback is pushed out
    # first; without this a failing rank dies without saying why.
    traceback.print_exception(exc_type, exc_value, exc_tb)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)


@record  # Records error and tracebacks in case of failure
def main() -> None:
    sys.excepthook = _fast_fail_excepthook

    config = run_config.initialize_config()
    pl.seed_everything(config.training.experiment.seed, workers=True)
    run_config.save_config(config)

    module = LeadLightningModule(config)
    trainer = _build_trainer(config)

    resume_checkpoint_path = None
    if (
        config.training.experiment.resume_from_last_checkpoint
        and config.training.experiment.output_dir is not None
    ):
        resume_checkpoint_path = _merged_resume_checkpoint(
            config.training.experiment.output_dir,
        )
        if resume_checkpoint_path is not None:
            LOG.info(f"Resuming training from {resume_checkpoint_path}")

    # SystemExit bypasses sys.excepthook and a fatal signal bypasses the interpreter,
    # so a rank that dies either way says nothing about why; catch both here.
    try:
        trainer.fit(module, ckpt_path=resume_checkpoint_path)
    except BaseException:
        rank = os.environ.get("SLURM_PROCID", "?")
        sys.stderr.write(f"=== rank {rank} exiting ===\n")
        traceback.print_exc()
        sys.stderr.flush()
        raise


if __name__ == "__main__":
    main()
