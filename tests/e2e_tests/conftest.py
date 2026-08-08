"""Shared fixtures for the end-to-end tests, which need a collected dataset.

The dataset root is ``PY123D_DATA_ROOT``; importing ``lead`` bootstraps it from
``.env`` when the environment does not already set it. Without it every test
here skips, so the suite stays runnable on a machine with no data.
"""

import os
import pathlib

import numpy as np
import pytest

import lead  # noqa: F401 — imported for its PY123D_DATA_ROOT bootstrap

# Samples inspected per test. Constant, so runtime does not grow with the
# dataset; the suite runs in seconds, so this is deliberately generous.
MAX_SAMPLES = 64


@pytest.fixture(scope="session")
def data_root() -> str:
    """The collected dataset root, or skip when none is configured.

    Returns:
        The dataset root path.
    """
    root = os.environ.get("PY123D_DATA_ROOT")
    if not root:
        pytest.skip("PY123D_DATA_ROOT not set")
    if not (pathlib.Path(root) / "logs").is_dir():
        pytest.skip(f"No logs/ under PY123D_DATA_ROOT={root}")
    return root


@pytest.fixture(scope="session")
def lead_config(data_root: str):
    """The config tree resolved against the configured dataset.

    Args:
        data_root: The dataset root, so the skip applies here too.

    Returns:
        The config tree.
    """
    from lead.config import load_lead_config

    return load_lead_config()


def sample_indices(count: int, limit: int = MAX_SAMPLES) -> list[int]:
    """Up to ``limit`` indices spread evenly over a sequence.

    Args:
        count: Length of the sequence.
        limit: Maximum number of indices to return.

    Returns:
        The sorted, unique indices.
    """
    if count <= 0:
        return []
    return sorted(set(np.linspace(0, count - 1, limit, dtype=int).tolist()))
