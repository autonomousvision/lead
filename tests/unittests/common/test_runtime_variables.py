"""Tests of the process's reported CPU assignment."""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import pytest

from lead.common import runtime_variables


def _reported_cores(_index: int) -> int:
    return runtime_variables.num_assigned_cpu_cores()


def test_reports_the_slurm_assignment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "1")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "144")

    assert runtime_variables.num_assigned_cpu_cores() == 144


def test_reports_the_default_outside_slurm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)

    assert runtime_variables.num_assigned_cpu_cores(default=7) == 7


def test_a_worker_process_reports_one_core(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sizing a pool per worker from the job's cores oversubscribes the node.

    A cache build runs one worker per core and each enumerates scenes, so the
    job's count here would square the node's thread count.
    """
    monkeypatch.setenv("SLURM_JOB_ID", "1")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "144")

    with ProcessPoolExecutor(
        max_workers=2,
        mp_context=multiprocessing.get_context("fork"),
    ) as executor:
        reported = list(executor.map(_reported_cores, range(2)))

    assert reported == [1, 1]
