"""Training section of the config tree."""

from typing import TYPE_CHECKING

from lead.common import runtime
from lead.config.node import ConfigNode, child_node, overridable_property
from lead.config.training.data_config import TrainingDataConfig
from lead.config.training.experiment_config import ExperimentConfig
from lead.config.training.optimization_config import OptimizationConfig

if TYPE_CHECKING:
    import torch

# Process-specific values excluded from serialized configs: they describe the
# machine/job, not the experiment (see src/lead/common/runtime.py).
RUNTIME_KEYS: frozenset[str] = frozenset({"device", "assigned_cpu_cores"})


class TrainingConfig(ConfigNode):
    """Configuration of a training run (experiment identity, optimization, data)."""

    experiment = child_node(ExperimentConfig)
    optimization = child_node(OptimizationConfig)
    data = child_node(TrainingDataConfig)

    @property
    def is_pretraining(self) -> bool:
        """If true indicates pretraining phase."""
        return not self._root.policy.transfuser.use_planning_decoder

    # --- Process runtime (delegated, excluded from serialization) ---
    @property
    def device(self) -> "torch.device":
        """PyTorch device to use for training."""
        return runtime.device()

    @overridable_property
    def assigned_cpu_cores(self) -> int:
        """Number of CPU cores assigned to this job."""
        return runtime.assigned_cpu_cores()
