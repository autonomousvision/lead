"""Training section of the config tree."""

from typing import TYPE_CHECKING

from lead.common import runtime_variables
from lead.config.node import ConfigNode, config_child_node, overridable_property
from lead.config.training.data_config import TrainingDataConfig
from lead.config.training.experiment_config import ExperimentConfig
from lead.config.training.lightning_config import LightningConfig
from lead.config.training.optimization_config import OptimizationConfig

if TYPE_CHECKING:
    pass

# Process-specific values excluded from serialized configs: they describe the
# machine/job, not the experiment (see src/lead/common/runtime_variables.py).
PROCESS_RUNTIME_CONFIG_KEYS: frozenset[str] = frozenset({"num_assigned_cpu_cores"})


class TrainingConfig(ConfigNode):
    """Configuration of a training run (experiment identity, optimization, data)."""

    experiment = config_child_node(ExperimentConfig)
    optimization = config_child_node(OptimizationConfig)
    lightning = config_child_node(LightningConfig)
    data = config_child_node(TrainingDataConfig)

    # --- Process runtime (delegated, excluded from serialization) ---
    @overridable_property
    def num_assigned_cpu_cores(self) -> int:
        """Number of CPU cores assigned to this job."""
        return runtime_variables.num_assigned_cpu_cores()
