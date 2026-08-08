"""The pipeline on a dataset that ships the normal view alone.

The perturbated split is an optional download, so every stage must fall back to
the nominal rig when ``logs/perturbated_view`` does not exist, even with
perturbation enabled. The root here mirrors the collected logs with symlinks,
which costs no storage.
"""

import pathlib

import pytest
from py123d.api.scene.scene_filter import SceneFilter

from lead.api.py123d_log_api import NORMAL_SENSOR_VIEW, PERTURBATED_SENSOR_VIEW
from lead.config import LeadConfig, load_lead_config
from lead.log_reader import SceneLoader
from tests.e2e_tests.conftest import sample_indices

# Logs mirrored into the root. Enough for the pairing to have something to find,
# and a bound so the fixture stays cheap on a full dataset.
MAX_LOGS = 2
# Samples inspected; every one of them decodes a full camera rig.
MAX_SAMPLES = 8


def _mirror_files(source: pathlib.Path, destination: pathlib.Path) -> None:
    """Recreate a directory tree with its files symlinked, not copied.

    Args:
        source: Directory to mirror.
        destination: Directory to create the mirror in.
    """
    destination.mkdir(parents=True, exist_ok=True)
    for entry in source.iterdir():
        if entry.is_dir():
            _mirror_files(entry, destination / entry.name)
        else:
            (destination / entry.name).symlink_to(entry)


@pytest.fixture(scope="module")
def normal_view_only_root(
    data_root: str,
    tmp_path_factory: pytest.TempPathFactory,
) -> str:
    """A dataset root holding the normal-view logs and the maps, nothing else.

    Args:
        data_root: The collected dataset root to mirror from.
        tmp_path_factory: Pytest's temporary directory factory.

    Returns:
        The mirrored root.
    """
    source = pathlib.Path(data_root)
    logs = source / "logs" / NORMAL_SENSOR_VIEW
    scenario_dirs = sorted(path for path in logs.iterdir() if path.is_dir())
    log_dirs = [
        log
        for scenario in scenario_dirs
        for log in sorted(path for path in scenario.iterdir() if path.is_dir())
    ][:MAX_LOGS]
    if not log_dirs:
        pytest.skip(f"No logs under {logs}")

    root = tmp_path_factory.mktemp(NORMAL_SENSOR_VIEW)
    for log in log_dirs:
        _mirror_files(
            log,
            root / "logs" / NORMAL_SENSOR_VIEW / log.parent.name / log.name,
        )
    _mirror_files(source / "maps", root / "maps")
    assert not (root / "logs" / PERTURBATED_SENSOR_VIEW).exists()
    return str(root)


@pytest.fixture(scope="module")
def normal_view_only_config(normal_view_only_root: str) -> LeadConfig:
    """The config tree pointed at the normal-view-only root.

    Args:
        normal_view_only_root: The mirrored root.

    Returns:
        The config tree.
    """
    return load_lead_config(
        {"training": {"data": {"py123d_data_root": normal_view_only_root}}},
    )


def test_loader_reads_normal_views_when_perturbation_is_requested(
    normal_view_only_root: str,
) -> None:
    """A loader asked for the perturbated rig falls back to the nominal one."""
    loader = SceneLoader(
        normal_view_only_root,
        SceneFilter(future_num_iterations=40),
        perturbation_probability=1.0,
    )
    assert len(loader) > 0, f"No scenes under {normal_view_only_root}"
    assert not loader.any_perturbated_sensor_views

    for index in sample_indices(len(loader), MAX_SAMPLES):
        assert not loader.has_perturbated_sensor_view(index)
        assert loader.draw_sensor_view(index) == NORMAL_SENSOR_VIEW
        scene_data = loader[index]
        assert scene_data.cameras
        assert scene_data.rig_perturbation is None


def test_dataset_builds_samples_and_caches_the_normal_view_only(
    normal_view_only_config: LeadConfig,
) -> None:
    """The policy dataset builds its samples, and caches one view per sample."""
    from lead.policy.transfuser.transfuser import Transfuser

    dataset = Transfuser(normal_view_only_config).build_dataset()
    assert len(dataset) > 0

    for index in sample_indices(len(dataset), MAX_SAMPLES):
        assert dataset.sample_sensor_views(index) == [NORMAL_SENSOR_VIEW]
        sample = dataset[index]
        assert sample.rgb is not None
        assert sample.rasterized_lidar is not None
