"""Experiment identity, logging, and visualization configuration."""

from lead.common.env import read_dotenv_int
from lead.config.node import ConfigNode, overridable_property


class ExperimentConfig(ConfigNode):
    """Experiment identity, WandB logging, and visualization knobs."""

    # Training Seed
    seed: int = 0
    # Description of the experiment.
    wandb_run_name: str = "An example experiment description."
    # File to continue training from
    initial_weights_file: str | None = None
    # If true continue the training from a failed training checkpoint.
    resume_from_last_checkpoint: bool = False

    # The one directory a run writes to: checkpoints, configs, logs, visualizations.
    output_dir: str | None = None

    # WandB ID for the experiment. If None, it will be generated automatically.
    wandb_id: str | None = None
    # must, allow, never
    wandb_resume_mode: str = "never"
    # Name of the WandB project to log to.
    wandb_project_name: str = "lead"

    @overridable_property
    def log_wandb(self) -> bool:
        """If true log metrics to Weights & Biases."""
        if self._root.debug_mode:
            return False
        return True

    # If true produce images during training for visualization.
    visualize_training: bool = True
    # Flag to visualize the dataset and deactivate randomization and augmentation.
    visualize_dataset: bool = False

    @property
    def log_scalars_every_n_steps(self) -> int:
        """How often to log scalar values during training.

        Live-adjustable: re-read from ``.env`` on each access so the frequency
        can be changed mid-run.
        """
        if self._root.debug_mode:
            return 1
        return read_dotenv_int("WANDB_LOG_FREQUENCY_TRAINING_SCALAR", default=50)

    @property
    def log_images_every_n_steps(self) -> int:
        """How often to log images during training.

        Live-adjustable like ``log_scalars_every_n_steps``; falls back to 100.
        """
        if self._root.debug_mode:
            return 100
        return read_dotenv_int("WANDB_LOG_FREQUENCY_TRAINING_IMAGES", default=100)
