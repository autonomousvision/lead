"""Config resolution and serialization for training runs."""

from __future__ import annotations

import logging
import os

import yaml

from lead.config import RUNTIME_KEYS, LeadConfig, load_lead_config, yaml_filtered

LOG = logging.getLogger(__name__)


def _read_dataset_expert_config(config: LeadConfig) -> dict | None:
    """Read the expert config stored with the dataset being trained on.

    Args:
        config: Bootstrap config used to resolve the dataset location.

    Returns:
        The stored expert config dict, or None when the dataset has none (the
        default expert config profile applies then).
    """
    config_path = os.path.join(
        os.path.dirname(config.training.data.py123d_logs_root),
        "config.yaml",
    )
    if not os.path.isfile(config_path):
        LOG.info(
            "No expert config stored at %s; using the '%s' expert config profile.",
            config_path,
            config.expert.config_profile,
        )
        return None
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def initialize_config() -> LeadConfig:
    """Build the training config tree.

    The expert section comes from the dataset's stored expert config (falling
    back to the default expert config profile); when resuming, the checkpoint's
    ``config.yaml`` is merged on top; env/CLI overrides win over everything.

    Returns:
        The resolved config tree.
    """
    # Bootstrap pass: env/CLI may relocate the dataset or set load_file.
    bootstrap = load_lead_config(use_cli=True)

    loaded_config = None
    load_file = bootstrap.training.experiment.load_file
    if load_file is not None:
        with open(
            os.path.join(os.path.dirname(load_file), "config.yaml"),
            encoding="utf-8",
        ) as f:
            loaded_config = yaml.safe_load(f)

    return load_lead_config(
        loaded_config=loaded_config,
        dataset_expert_config=_read_dataset_expert_config(bootstrap),
        use_cli=True,
        # The checkpoint config may carry keys renamed since it was written.
        raise_error_on_missing_key=loaded_config is None,
    )


def serializable_config(config: LeadConfig) -> dict:
    """Resolve the config tree into a yaml-serializable dict.

    Process-runtime values (rank, device, ...) and machine-local dataset roots
    are excluded so serialized configs stay reproducible across machines.

    Args:
        config: The config tree.

    Returns:
        Nested dict of all sections, non-serializable leaves removed.
    """
    tree = yaml_filtered(config.to_dict())
    for key in RUNTIME_KEYS:
        tree["training"].pop(key, None)
    for key in ("py123d_logs_root", "py123d_maps_root"):
        tree["training"]["data"].pop(key, None)
    return tree


def save_config(config: LeadConfig, rank: int) -> None:
    """Serialize the config tree to ``logdir/config.yaml`` on rank 0.

    Args:
        config: The config tree.
        rank: Process rank; only rank 0 writes.
    """
    logdir = config.training.experiment.logdir
    if rank == 0 and logdir is not None:
        os.makedirs(logdir, exist_ok=True)
        with open(os.path.join(logdir, "config.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(serializable_config(config), f, sort_keys=False)
