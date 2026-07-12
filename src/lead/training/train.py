import logging
import os
import sys
import traceback
import warnings

import lightning.pytorch as pl
import matplotlib
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from torch.distributed.elastic.multiprocessing.errors import record

from lead.common.logging_config import setup_logging
from lead.config import LeadConfig
from lead.training import config_utils
from lead.training.lightning.lightning_module import LeadLightningModule

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


def build_loggers(config: LeadConfig) -> list:
    """Build the Lightning loggers (TensorBoard always, WandB when enabled).

    Args:
        config: Training configuration.

    Returns:
        The loggers, empty when no logdir is configured.
    """
    if config.training.experiment.logdir is None:
        return []
    loggers = [
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
                config=config_utils.serializable_config(config),
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
    callbacks = [LearningRateMonitor(logging_interval="step")]
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

    config = config_utils.initialize_config()
    pl.seed_everything(config.training.experiment.seed, workers=True)
    config_utils.save_config(config, config.training.rank)

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
