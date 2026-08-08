"""Properties of the running process (not the experiment): CPU assignment, scratch locations.

Kept out of the config so experiment configs stay reproducible across machines.
"""

from __future__ import annotations

import multiprocessing
import os
from pathlib import Path


def num_assigned_cpu_cores(default: int = 8) -> int:
    """Number of CPU cores this process may use.

    A worker process of a pool shares the job's cores with its siblings, so it
    reports one: a pool it sized from the job's count would multiply the node's
    threads by the number of workers. Ranks launched as subprocesses, as
    Lightning's DDP launches them, are not worker processes and get the job's
    full count.

    Args:
        default: Core count assumed outside of SLURM.

    Returns:
        One inside a worker process, else the SLURM-assigned core count, else
        the default.
    """
    if multiprocessing.parent_process() is not None:
        return 1
    cpus_per_task = os.environ.get("SLURM_CPUS_PER_TASK")
    if "SLURM_JOB_ID" in os.environ and cpus_per_task:
        return int(cpus_per_task)
    return default


def project_root() -> Path:
    """Repository root of the running checkout, derived from the package location."""
    return Path(__file__).resolve().parents[3]


def output_dir_root() -> Path:
    """Root directory for generated outputs, read hot from ``LEAD_OUTPUT_DIR_ROOT`` in ``.env``."""
    from lead.common.env import read_dotenv

    return Path(read_dotenv("LEAD_OUTPUT_DIR_ROOT"))
