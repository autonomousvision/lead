"""Experiment identity, logging, and visualization configuration."""

from lead.common.dotenv import read_dotenv_int
from lead.config.node import ConfigNode, overridable_property


class ExperimentConfig(ConfigNode):
    """Experiment identity, WandB logging, and visualization knobs."""

    # Training Seed
    seed: int = 0
    # Unique experiment identifier.
    id: str = "Experiment 1"
    # Description of the experiment.
    description: str = "An example experiment description."
    # Directory to log data to.
    logdir: str | None = None
    # File to continue training from
    load_file: str | None = None
    # If true continue the training from a failed training checkpoint.
    continue_failed_training: bool = False

    # WandB ID for the experiment. If None, it will be generated automatically.
    wandb_id: str | None = None
    # must, allow, never
    wandb_resume: str = "never"

    @property
    def wandb_project_name(self) -> str:
        """Name of the WandB project based on the training phase."""
        if self._root.training.is_pretraining:
            return "lead_pretrain"
        return "lead_posttrain"

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
    # Flag to visualize the failed scenarios and deactivate randomization and augmentation.
    visualize_failed_scenarios: bool = False

    @property
    def log_scalars_frequency(self) -> int:
        """How often to log scalar values during training.

        Live-adjustable: re-read from ``.env`` on each access so the frequency
        can be changed mid-run.
        """
        if self._root.debug_mode:
            return 1
        return read_dotenv_int("WANDB_LOG_FREQUENCY_TRAINING_SCALAR", default=1)

    @property
    def log_images_frequency(self) -> int:
        """How often to log images during training.

        Live-adjustable like ``log_scalars_frequency``; falls back to 100.
        """
        if self._root.debug_mode:
            return 100
        return read_dotenv_int("WANDB_LOG_FREQUENCY_TRAINING_IMAGES", default=100)
