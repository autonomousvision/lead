"""Properties of the running process (not the experiment).

Distributed rank, device, CPU assignment, and scratch locations describe the
machine and job the code happens to run on. They live here — not in the config
— so experiment configs stay reproducible across machines. ``TrainingConfig``
exposes thin delegating properties for call-site convenience, but excludes
these values from serialization.
"""

import os
from pathlib import Path

import torch


def rank() -> int:
    """Current process rank in distributed training."""
    return int(os.environ.get("RANK", "0"))


def world_size() -> int:
    """Total number of processes in distributed training."""
    return int(os.environ.get("WORLD_SIZE", "1"))


def local_rank() -> int:
    """Local rank of current process on the node."""
    return int(os.environ.get("LOCAL_RANK", "0"))


def device() -> torch.device:
    """PyTorch device of this process."""
    return torch.device(
        f"cuda:{local_rank()}" if torch.cuda.is_available() else "cpu",
    )


def is_on_tcml() -> bool:
    """Check if running on Training Center for Machine Learning of Tübingen."""
    return os.environ.get("TCML") is not None


def assigned_cpu_cores(default: int = 8) -> int:
    """Number of CPU cores assigned to this job.

    Args:
        default: Core count assumed outside of SLURM.

    Returns:
        The SLURM-assigned core count, or the default.
    """
    cpus_per_task = os.environ.get("SLURM_CPUS_PER_TASK")
    if "SLURM_JOB_ID" in os.environ and cpus_per_task:
        return int(cpus_per_task)
    return default


def project_root() -> Path:
    """Repository root of the running checkout, derived from the package location."""
    return Path(__file__).resolve().parents[3]
