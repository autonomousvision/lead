"""The camera history the log reader serves, checked against real logs.

The camera streams are anchor-locked to the save ticks, so a history request is
only ever served at the ages the dataset actually stored a frame at.
"""

import pytest
from py123d.api.scene.scene_filter import SceneFilter

from lead.api.scene_loading_spec import SceneLoadingSpec
from lead.log_reader import SceneLoader
from tests.e2e_tests.conftest import sample_indices

TICK_US = 50_000
# Period of the save ticks, where the camera streams exist.
SAVE_PERIOD = 5
# Samples inspected; every one decodes a full camera rig per requested age.
MAX_SAMPLES = 4


@pytest.fixture(scope="module")
def loader(data_root: str) -> SceneLoader:
    """A loader with enough history for two save ticks of camera past.

    Args:
        data_root: The dataset root.

    Returns:
        The loader.
    """
    return SceneLoader(
        data_root,
        # Both margins are set: py123d enumerates no scene when a history is
        # asked for against an unset future.
        SceneFilter(
            history_num_iterations=2 * SAVE_PERIOD,
            future_num_iterations=0,
        ),
        read_radar_sweeps=False,
        read_semantic_cameras=False,
        read_depth_cameras=False,
        read_map_api=False,
    )


@pytest.mark.e2e
def test_camera_history_is_read_at_the_requested_ages(loader) -> None:
    """A request for the save-tick ages comes back with a full rig at each."""
    ages = (0, SAVE_PERIOD, 2 * SAVE_PERIOD)
    spec = SceneLoadingSpec(rgb_tick_ages=ages)
    for index in sample_indices(len(loader), MAX_SAMPLES):
        scene_data = loader.read(index, False, spec)
        assert scene_data.past_cameras is not None
        # Ages near a log's start have no frame; the rest must all be served.
        assert set(scene_data.past_cameras) <= set(ages)
        for age, rig in scene_data.past_cameras.items():
            assert len(rig) == len(scene_data.cameras), (
                f"age {age} returned a partial rig"
            )
            for camera in rig:
                assert camera.image is not None


@pytest.mark.e2e
def test_camera_history_serves_every_stored_age(loader) -> None:
    """Every requested past save tick the log stores actually comes back."""
    ages = (0, SAVE_PERIOD, 2 * SAVE_PERIOD)
    spec = SceneLoadingSpec(rgb_tick_ages=ages)
    fully_served = 0
    for index in sample_indices(len(loader), MAX_SAMPLES):
        scene_data = loader.read(index, False, spec)
        assert scene_data.past_cameras is not None
        # Age 0 is served by ``cameras``; a past age is stored whenever the log
        # reaches that far back from the anchor.
        stored = {
            age for age in ages if 0 < age <= scene_data.scene_metadata.initial_idx
        }
        assert set(scene_data.past_cameras) == stored, (
            f"sample {index}: served {sorted(scene_data.past_cameras)}, "
            f"stored history is {sorted(stored)}"
        )
        fully_served += stored == set(ages) - {0}
    assert fully_served, "no sample had the full camera history to serve"


@pytest.mark.e2e
def test_the_anchor_rig_is_decoded_once(loader) -> None:
    """Age 0 is served by ``cameras``; the history never repeats that decode."""
    spec = SceneLoadingSpec(rgb_tick_ages=(0, SAVE_PERIOD))
    for index in sample_indices(len(loader), MAX_SAMPLES):
        scene_data = loader.read(index, False, spec)
        assert scene_data.cameras, "the anchor rig must come back in cameras"
        assert scene_data.past_cameras is not None
        assert 0 not in scene_data.past_cameras
        anchor = loader.scenes[index].get_timestamp_at_iteration(0).time_us
        for camera in scene_data.cameras:
            assert camera.timestamp.time_us == anchor


@pytest.mark.e2e
def test_camera_history_ages_are_distinct_frames(loader) -> None:
    """An older age must be an older frame, or the read is not moving in time."""
    spec = SceneLoadingSpec(rgb_tick_ages=(0, SAVE_PERIOD))
    anchor = SAVE_PERIOD
    compared = 0
    for index in sample_indices(len(loader), MAX_SAMPLES):
        scene_data = loader.read(index, False, spec)
        assert scene_data.past_cameras is not None
        if anchor not in scene_data.past_cameras:
            continue
        past = scene_data.past_cameras[anchor][0]
        assert past.timestamp.time_us == (
            scene_data.cameras[0].timestamp.time_us - anchor * TICK_US
        )
        compared += 1
    assert compared, "no sample carried a past camera frame to compare"


@pytest.mark.e2e
def test_no_camera_history_is_read_when_none_is_asked_for(loader) -> None:
    """The history costs a decode per age, so it stays off unless requested."""
    scene_data = loader.read(0, False, SceneLoadingSpec(rgb_tick_ages=(0,)))
    assert scene_data.past_cameras is None
    assert scene_data.cameras


@pytest.mark.e2e
def test_no_cameras_are_read_when_the_ages_are_empty(loader) -> None:
    """An empty tuple is the whole statement: this read touches no camera."""
    scene_data = loader.read(0, False, SceneLoadingSpec())
    assert scene_data.cameras == []
    assert scene_data.past_cameras is None


@pytest.mark.e2e
def test_pose_history_covers_every_requested_sweep_age(loader) -> None:
    """Sweep alignment indexes the pose history by tick age, so it must reach."""
    ages = (0, 1, 2, 3, 4)
    spec = SceneLoadingSpec(lidar_tick_ages=ages)
    for index in sample_indices(len(loader), MAX_SAMPLES):
        scene_data = loader.read(index, False, spec)
        assert scene_data.lidar_sweeps is not None
        assert scene_data.past_ego_positions is not None
        # A sweep whose age has no pose would silently drop during alignment.
        assert len(scene_data.past_ego_positions) > max(
            scene_data.lidar_sweeps,
            default=0,
        )


@pytest.mark.e2e
def test_sweeps_are_read_only_at_the_requested_ages(loader) -> None:
    """A thinned sweep window serves exactly the stored requested ages."""
    ages = (0, 2, 4)
    spec = SceneLoadingSpec(lidar_tick_ages=ages)
    fully_served = 0
    for index in sample_indices(len(loader), MAX_SAMPLES):
        scene_data = loader.read(index, False, spec)
        assert scene_data.lidar_sweeps is not None
        # Lidar is stored every tick, so a requested age is served whenever the
        # log reaches that far back from the anchor — and nothing between them.
        stored = {age for age in ages if age <= scene_data.scene_metadata.initial_idx}
        assert set(scene_data.lidar_sweeps) == stored
        fully_served += stored == set(ages)
    assert fully_served, "no sample had the full sweep history to serve"
