# Configuration System

Many configuration values of our system are interdependent and need to be computed dynamically based on other settings (e.g., number of cameras depends on the target dataset, BEV grid dimensions depend on planning area boundaries). While frameworks like Hydra offer powerful composition capabilities, our use case benefits more from the simplicity and directness of Python classes with computed properties.

This page explains how our configuration system works, including the override hierarchy, config inheritance structure, and the dynamic property system.

> **Note**: This documentation is primarily relevant if you want to train models. For evaluation only, the default configurations are typically sufficient.

## Override Default Configurations

Configurations can be loaded from multiple sources with a clear priority hierarchy from highest to lowest. The higher priority source always wins.

1. **CLI Arguments** (via `load_from_args`)
2. **Environment Variables** (via `load_from_environment`)
3. **Config Files** (via `load_from_file`)
4. **Class Defaults** (defined in the class)

For example, consider the `batch_size` configuration:
- If the class default is `64`
- And you load from a config file that specifies `"batch_size": 32`, then `batch_size` becomes `32`
- But if you also set `LEAD_TRAINING_CONFIG="batch_size=48"`, then `batch_size` becomes `48`
- However, if you pass `batch_size=128` via CLI, then the final `batch_size` is `128` (CLI has highest priority)

Different config classes use different loading combinations based on their use case:

| Config Class | Loading Methods | Priority Order | Reason |
|-------------|----------------|----------------|--------|
| `TrainingConfig` | `load_from_environment` + `load_from_args` | Args > Environment | Training scripts support both CLI args (highest) and environment variables |
| `ClosedLoopConfig` | `load_from_environment` only | Environment only | CARLA evaluation runs disallow direct CLI access |
| `OpenLoopConfig` | `load_from_environment` only | Environment only | CARLA evaluation runs disallow direct CLI access |
| `ExpertConfig` | `load_from_environment` only | Environment only | CARLA evaluation runs disallow direct CLI access |

**Note:** `TrainingConfig` is unique in that it supports **both** environment variables and CLI arguments, with CLI arguments taking precedence. This allows flexible configuration during training while still supporting environment-based configuration when needed.

The option `load_from_file` is primarily used for continue failed training or fine-tuning,
 where you want to load a complete configuration from a previous experiment.

## Config Hierarchy

Different use cases only require a subset of every possible configuration. Especially helpful here is the use-case where
we are training model on one sensor calibration and collecting data on parallel with another sensor calibration.

```
BaseConfig (common/config_base.py)
├── Camera and sensor configuration
├── Loading mechanisms (file, environment, CLI)

TrainingConfig (training/config_training.py)
├── Inherits from BaseConfig
├── Dataset-specific properties
├── Model architecture settings
├── Training hyperparameters

ExpertConfig (expert/config_expert.py)
├── Inherits from BaseConfig
├── Dataset-specific properties
├── Expert logic settings

OpenLoopConfig (inference/config_open_loop.py)
├── Inherits from BaseConfig
└── Offline evaluation settings

ClosedLoopConfig (inference/config_closed_loop.py)
├── Inherits from OpenLoopConfig
└── Online CARLA simulation settings
```

## Dynamic Property System

We use two types of dynamic properties:

**1. Standard `@property`**

Used for computed values that should never be overridden:

```python
@property
def num_used_cameras(self):
    """Computed from the used_cameras list."""
    return sum(int(use) for use in self.used_cameras)
```

**2. Custom `@overridable_property`**

Those properties can be overridden by users. For example, number of epochs depends on each dataset we are using:

```python
@overridable_property
def epochs(self):
    """Default epochs by dataset, but can be overridden."""
    if self.carla_leaderboard_mode:
        return 31
    if self.target_dataset == TargetDataset.NAVSIM_4CAMERAS:
        return 61
    return 20
```

This can be overridden by setting `epochs=61` in CLI.
