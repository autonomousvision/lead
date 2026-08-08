"""Config resolution and serialization for training runs."""

from __future__ import annotations

import logging
import os

import yaml
from lightning.pytorch.utilities import rank_zero_only

from lead.config import (
    PROCESS_RUNTIME_CONFIG_KEYS,
    LeadConfig,
    load_lead_config,
    yaml_filtered,
)

LOG = logging.getLogger(__name__)


def initialize_config() -> LeadConfig:
    """Build the training config tree.

    The expert section comes from the dataset's stored expert config (falling
    back to the expert class defaults); when resuming, the checkpoint's
    ``config.yaml`` is merged on top; env/CLI overrides win over everything.

    Returns:
        The resolved config tree.
    """
    # Bootstrap pass: env/CLI may relocate the dataset or set initial_weights_file.
    bootstrap_config = load_lead_config(use_cli=True)

    checkpoint_config = None
    initial_weights_file = bootstrap_config.training.experiment.initial_weights_file
    if initial_weights_file is not None:
        with open(
            os.path.join(os.path.dirname(initial_weights_file), "config.yaml"),
            encoding="utf-8",
        ) as f:
            checkpoint_config = yaml.safe_load(f)

    return load_lead_config(
        loaded_config=checkpoint_config,
        dataset_expert_config=_read_dataset_expert_config(bootstrap_config),
        use_cli=True,
        # Tolerate checkpoint configs whose keys don't match the current schema.
        raise_on_unknown_key=checkpoint_config is None,
    )


def to_serializable_dict(config: LeadConfig) -> dict:
    """Resolve the config tree into a yaml-serializable dict.

    Process-runtime values (rank, device, ...) and machine-local dataset roots
    are excluded so serialized configs stay reproducible across machines.

    Args:
        config: The config tree.

    Returns:
        Nested dict of all sections, non-serializable leaves removed.
    """
    config_dict = yaml_filtered(config.to_dict())
    for key in PROCESS_RUNTIME_CONFIG_KEYS:
        config_dict["training"].pop(key, None)
    config_dict["training"]["data"].pop("py123d_data_root", None)
    return config_dict


def _read_dataset_expert_config(config: LeadConfig) -> dict | None:
    """Read the expert config stored with the dataset being trained on."""
    config_path = os.path.join(
        config.training.data.py123d_data_root,
        "config.yaml",
    )
    if not os.path.isfile(config_path):
        LOG.info(
            "No expert config stored at %s; using the expert class defaults.",
            config_path,
        )
        return None
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@rank_zero_only
def save_config(config: LeadConfig) -> None:
    """Serialize the config tree to ``output_dir/config.yaml`` on rank 0.

    Args:
        config: The config tree.
    """
    output_dir = config.training.experiment.output_dir
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "config.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(to_serializable_dict(config), f, sort_keys=False)
