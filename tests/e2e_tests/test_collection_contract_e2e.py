"""The collected dataset matches what data collection was asked to produce.

Collection can fail per-route without failing the collection process, so these
tests pin the artifacts instead: every requested route must have left a
readable log, and both sensor views must exist.
"""

import os

import pytest
from py123d.api.scene.scene_filter import SceneFilter

from lead.log_reader import SceneLoader


def _loader(data_root: str, perturbation_probability: float) -> SceneLoader:
    """A minimal loader over one sensor view.

    Args:
        data_root: The dataset root.
        perturbation_probability: 0.0 for the normal view, 1.0 for the
            perturbated one.

    Returns:
        The loader.
    """
    return SceneLoader(
        data_root,
        SceneFilter(shuffle=False, history_num_iterations=0, future_num_iterations=0),
        perturbation_probability=perturbation_probability,
        read_lidar_sweeps=False,
        read_radar_sweeps=False,
        read_semantic_cameras=False,
        read_depth_cameras=False,
        read_map_api=False,
    )


@pytest.mark.e2e
def test_the_perturbated_split_exists(data_root) -> None:
    """Collection writes both views, so a missing perturbated split is a failure."""
    loader = _loader(data_root, 1.0)
    assert loader.any_perturbated_sensor_views, (
        f"no perturbated sensor views under {data_root}"
    )


@pytest.mark.e2e
def test_every_requested_route_left_a_readable_log(data_root) -> None:
    """The readable log count reaches the route count the collection ran."""
    minimum = os.environ.get("LEAD_E2E_MIN_LOGS")
    if minimum is None:
        pytest.skip("LEAD_E2E_MIN_LOGS not set: not running against a fresh collection")
    log_names = {scene.log_name for scene in _loader(data_root, 0.0).scenes}
    assert len(log_names) >= int(minimum), (
        f"collection was asked for {minimum} routes but only "
        f"{len(log_names)} readable logs exist: {sorted(log_names)}"
    )
