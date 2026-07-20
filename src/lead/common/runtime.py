"""Properties of the running process (not the experiment): device, CPU assignment, scratch locations.

Kept out of the config so experiment configs stay reproducible across machines.
"""

from __future__ import annotations

import os
import typing
from pathlib import Path

if typing.TYPE_CHECKING:
    import torch


def device() -> torch.device:
    """PyTorch device of this process."""
    import torch

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return torch.device(
        f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu",
    )


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


def output_dir_root() -> Path:
    """Root directory for generated outputs, read hot from ``LEAD_OUTPUT_DIR_ROOT`` in ``.env``."""
    from lead.common.dotenv import read_dotenv

    return Path(read_dotenv("LEAD_OUTPUT_DIR_ROOT"))
