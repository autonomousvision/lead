"""The global config tree (``lead_config``), yaml profile loading and serialization."""

import enum
import logging
import os
import sys
from pathlib import Path
from typing import Any, cast

import yaml
from omegaconf import OmegaConf

from lead.config.evaluation.evaluation_config import EvaluationConfig
from lead.config.expert.expert_config import ExpertConfig
from lead.config.node import ConfigNode, child_node
from lead.config.policy.policy_config import PolicyConfig
from lead.config.training.training_config import TrainingConfig

LOG = logging.getLogger(__name__)

# Environment variable holding a space-separated dotlist of overrides with
# fully qualified paths, e.g.
# ``LEAD_CONFIG="debug_mode=true expert.config_profile=default"``.
ENV_KEY = "LEAD_CONFIG"

# Config sections selectable via yaml profiles in ``src/lead/config/profiles/<section>/``.
PROFILE_SECTIONS = ("expert", "policy")

CONFIG_PROFILES_ROOT = Path(__file__).resolve().parent / "profiles"


class LeadConfig(ConfigNode):
    """Root of the config tree, the single config object passed around.

    Override sources (highest priority first): CLI dotlist, ``LEAD_CONFIG``
    environment dotlist, loaded config file, yaml config profile, class
    default.
    """

    # If true, run code in debug mode with settings that allow for
    # faster iteration and easier debugging (e.g., lower resolution, fewer data points saved, etc.)
    debug_mode: bool = False

    expert = child_node(ExpertConfig)
    policy = child_node(PolicyConfig)
    training = child_node(TrainingConfig)
    evaluation = child_node(EvaluationConfig)


# --- Config profiles ---
def available_config_profiles(section: str) -> list[str]:
    """Names of the yaml profiles available for a config section.

    Args:
        section: Profile directory name, e.g. ``"expert"`` or ``"policy"``.

    Returns:
        Sorted profile names (yaml file stems).
    """
    section_dir = CONFIG_PROFILES_ROOT / section
    return sorted(path.stem for path in section_dir.glob("*.yaml"))


def load_config_profile(section: str, name: str) -> dict[str, Any]:
    """Load a yaml config profile as a nested override dict.

    Args:
        section: Profile directory name, e.g. ``"expert"`` or ``"policy"``.
        name: Profile name (yaml file stem).

    Returns:
        The profile's overrides; an empty dict for an empty yaml file.
    """
    path = CONFIG_PROFILES_ROOT / section / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(
            f"Unknown {section} config profile '{name}'. "
            f"Available profiles: {available_config_profiles(section)}",
        )
    loaded = cast(
        "dict[str, Any] | None",
        OmegaConf.to_container(OmegaConf.load(path), resolve=True),
    )
    return loaded or {}


# --- Override sources ---
def _parse_dotlist(dotlist: list[str]) -> dict[str, Any]:
    """Parse fully qualified ``section.path.key=value`` tokens into a nested dict."""
    if not dotlist:
        return {}
    parsed = OmegaConf.create(OmegaConf.from_dotlist(dotlist))
    return cast("dict[str, Any]", OmegaConf.to_container(parsed, resolve=True))


def _cli_overrides() -> dict[str, Any]:
    """Overrides from ``sys.argv``.

    Only dotlist-shaped arguments are overrides; other arguments (paths,
    flags of the host program such as pytest) are not for us.
    """
    return _parse_dotlist(
        [arg for arg in sys.argv[1:] if "=" in arg and not arg.startswith("-")],
    )


def _env_overrides() -> dict[str, Any]:
    """Overrides from the ``LEAD_CONFIG`` environment dotlist."""
    return _parse_dotlist(os.getenv(ENV_KEY, "").split())


# --- Loading ---
def apply_stored_expert_config(
    config: LeadConfig,
    stored_config: dict[str, Any],
) -> None:
    """Apply the expert config stored with a dataset onto ``config.expert``.

    Unknown keys are ignored so datasets stay loadable after knob renames.

    Args:
        config: The config tree to update.
        stored_config: The dataset's stored expert config dict.
    """
    config.expert.apply_overrides(
        stored_config,
        user_override=False,
        raise_error_on_missing_key=False,
    )


def load_lead_config(
    loaded_config: dict[str, Any] | None = None,
    dataset_expert_config: dict[str, Any] | None = None,
    use_cli: bool = False,
    raise_error_on_missing_key: bool = True,
) -> LeadConfig:
    """Build the config tree from defaults, profiles and override sources.

    Application order (later wins): class defaults, yaml config profiles,
    ``dataset_expert_config``, ``loaded_config``, ``LEAD_CONFIG`` environment
    dotlist, CLI dotlist.

    Args:
        loaded_config: Nested config dict from a stored file, e.g. a
            checkpoint's ``config.yaml``.
        dataset_expert_config: Expert config stored with the dataset being
            loaded; applied onto the ``expert`` section only.
        use_cli: Whether to read overrides from ``sys.argv``.
        raise_error_on_missing_key: Whether unknown keys in ``loaded_config``
            raise.

    Returns:
        The resolved config tree.
    """
    env_overrides = _env_overrides()
    cli_overrides = _cli_overrides() if use_cli else {}
    config = LeadConfig()

    # Profiles first: their deltas are the base other sources override.
    for section_name in PROFILE_SECTIONS:
        profile_name = None
        for source in (cli_overrides, env_overrides, loaded_config or {}):
            section_overrides = source.get(section_name)
            if isinstance(section_overrides, dict):
                profile_name = profile_name or section_overrides.get("config_profile")
        section = getattr(config, section_name)
        profile_name = profile_name or section.config_profile
        section.apply_overrides(
            load_config_profile(section_name, profile_name),
            user_override=False,
        )
        section.config_profile = profile_name

    if dataset_expert_config:
        apply_stored_expert_config(config, dataset_expert_config)

    if loaded_config:
        config.apply_overrides(
            loaded_config,
            user_override=False,
            raise_error_on_missing_key=raise_error_on_missing_key,
        )

    config.apply_overrides(env_overrides, user_override=True)
    config.apply_overrides(cli_overrides, user_override=True)
    return config


# --- Serialization ---
def _yaml_serializable(value: Any) -> bool:
    """Check whether a value can be serialized to yaml.

    Args:
        value: Value to check.

    Returns:
        True if ``yaml.safe_dump`` accepts the value.
    """
    try:
        yaml.safe_dump(value)
        return True
    except yaml.YAMLError:
        return False


def yaml_filtered(tree: dict[str, Any]) -> dict[str, Any]:
    """Recursively drop values that cannot be serialized to yaml.

    Enums are stored as their values and tuples as lists; loading coerces them
    back to the knob's declared type.

    Args:
        tree: Nested config dict, e.g. from :meth:`ConfigNode.to_dict`.

    Returns:
        The tree with non-serializable leaves removed.
    """
    out: dict[str, Any] = {}
    for key, value in tree.items():
        if isinstance(value, dict):
            out[key] = yaml_filtered(value)
            continue
        if isinstance(value, enum.Enum):
            value = value.value
        elif isinstance(value, tuple):
            value = list(value)
        if _yaml_serializable(value):
            out[key] = value
    return out
