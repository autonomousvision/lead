"""Invariants every log of the dataset under ``LEAD_TEST_DATA_ROOT`` must satisfy:
storage frequencies, save-tick anchoring, age-0 alignment, and consistency of the
derived quantities."""

import os

import numpy as np
import pytest
from py123d.api.scene.scene_filter import SceneFilter
from py123d.datatypes import CustomModality, LidarID, ModalityType

from lead.api.py123d_log_api import (
    BOX_ATTRIBUTES_KEY,
    CAMERA_ID_MAPPING,
    DRIVING_META_MODALITY_ID,
    LOCALIZED_EGO_STATE_KEY,
    localized_position_yaw,
)
from lead.dataloader import Py123DDataLoader, scene_index
from lead.dataloader.box_decoding import carla_ego_pose

TICK_US = 50_000
SAVE_PERIOD = 5
SWEEP_WINDOW = 10
# Bound on |ground truth - GNSS+IMU estimate|; generous for filter transients.
MAX_LOCALIZATION_ERROR_M = 20.0
MAX_SAMPLES = 12


@pytest.fixture(scope="module")
def data_root():
    """The collected dataset root, or skip when none is configured.

    Returns:
        The dataset root path.
    """
    root = os.environ.get("LEAD_TEST_DATA_ROOT")
    if not root:
        pytest.skip("LEAD_TEST_DATA_ROOT not set")
    return root


@pytest.fixture(scope="module")
def loader(data_root: str) -> Py123DDataLoader:
    """The generic loader over the dataset, sensor extras disabled for speed.

    Args:
        data_root: The dataset root.

    Returns:
        The loader.
    """
    loader = Py123DDataLoader(
        data_root,
        radar_sweeps=False,
        semantics=False,
        depths=False,
        map=False,
    )
    assert len(loader) > 0, f"No scenes under {data_root}"
    return loader


@pytest.fixture(scope="module")
def windowed_loader(data_root: str) -> Py123DDataLoader:
    """A loader whose scenes carry two save periods of future.

    Args:
        data_root: The dataset root.

    Returns:
        The loader.
    """
    return Py123DDataLoader(
        data_root,
        SceneFilter(future_num_iterations=2 * SAVE_PERIOD),
        radar_sweeps=False,
        semantics=False,
        depths=False,
        map=False,
    )


def sample_indices(count: int) -> list[int]:
    """Up to ``MAX_SAMPLES`` indices spread evenly over a sequence.

    Args:
        count: Length of the sequence.

    Returns:
        The sorted, unique indices.
    """
    return sorted(set(np.linspace(0, count - 1, MAX_SAMPLES, dtype=int).tolist()))


def anchor_us(loader: Py123DDataLoader, index: int) -> int:
    """The anchor timestamp of a sample.

    Args:
        loader: The loader owning the scene.
        index: Sample index.

    Returns:
        The anchor timestamp in microseconds.
    """
    return loader.scenes[index].get_timestamp_at_iteration(0).time_us


@pytest.mark.e2e
def test_scene_counts_match_the_stored_ticks(data_root, loader, windowed_loader):
    """Sample counts follow exactly from the ticks each log stores."""
    all_scenes = scene_index.get_scenes(
        data_root,
        SceneFilter(history_num_iterations=0, future_num_iterations=0),
    )
    ticks_by_log: dict[str, set[int]] = {}
    for scene in all_scenes:
        ticks_by_log.setdefault(scene.log_name, set()).add(
            scene.get_scene_metadata().initial_idx,
        )
    anchors_by_log: dict[str, list[int]] = {}
    for scene in loader.scenes:
        anchors_by_log.setdefault(scene.log_name, []).append(
            scene.get_scene_metadata().initial_idx,
        )
    windowed_by_log: dict[str, list[int]] = {}
    for scene in windowed_loader.scenes:
        windowed_by_log.setdefault(scene.log_name, []).append(
            scene.get_scene_metadata().initial_idx,
        )

    assert set(anchors_by_log) == set(ticks_by_log)
    for log_name, ticks in ticks_by_log.items():
        num_ticks = len(ticks)
        assert ticks == set(range(num_ticks)), f"{log_name}: sync rows have holes"

        anchors = sorted(anchors_by_log[log_name])
        expected_anchors = list(range(anchors[0], num_ticks, SAVE_PERIOD))
        assert anchors == expected_anchors, f"{log_name}: missing or extra anchors"

        expected_windowed = [
            anchor
            for anchor in expected_anchors
            if anchor + 2 * SAVE_PERIOD <= num_ticks - 1
        ]
        assert sorted(windowed_by_log.get(log_name, [])) == expected_windowed, (
            f"{log_name}: future-windowed scene count is off"
        )


@pytest.mark.e2e
def test_anchors_form_the_save_tick_grid(loader):
    """Within a log, anchors are unique and spaced by whole save periods."""
    by_log: dict[str, list[int]] = {}
    for index, scene in enumerate(loader.scenes):
        by_log.setdefault(scene.log_name, []).append(anchor_us(loader, index))

    for log_name, timestamps in by_log.items():
        deltas = np.diff(np.array(sorted(timestamps)))
        assert (deltas > 0).all(), f"{log_name}: duplicated scene anchors"
        assert (deltas % (SAVE_PERIOD * TICK_US) == 0).all(), (
            f"{log_name}: anchors off the save-tick grid"
        )


@pytest.mark.e2e
def test_every_tick_is_stored_between_save_ticks(windowed_loader):
    """State streams exist on every tick; cameras exactly on the save ticks."""
    scenes = windowed_loader.scenes
    assert scenes, "No scene has two save periods of future"
    camera_id = next(
        camera_id
        for camera_id in CAMERA_ID_MAPPING.values()
        if camera_id in scenes[0].get_camera_metadatas()
    )
    for index in sample_indices(len(scenes)):
        scene = scenes[index]
        anchor = scene.get_timestamp_at_iteration(0).time_us
        for iteration in range(2 * SAVE_PERIOD + 1):
            assert scene.get_timestamp_at_iteration(iteration).time_us == (
                anchor + iteration * TICK_US
            ), "scene iterations are not simulator ticks"
            assert scene.get_ego_state_se3_at_iteration(iteration) is not None
            assert (
                scene.get_lidar_at_iteration(iteration, LidarID.LIDAR_TOP) is not None
            )
            assert (
                scene.get_traffic_light_detections_at_iteration(iteration) is not None
            )
            assert (
                scene.get_custom_modality_at_iteration(
                    iteration,
                    DRIVING_META_MODALITY_ID,
                )
                is not None
            )
            camera = scene.get_camera_at_iteration(iteration, camera_id)
            if iteration % SAVE_PERIOD == 0:
                assert camera is not None, f"camera missing on save tick {iteration}"
            else:
                assert camera is None, f"camera present off the save tick {iteration}"


@pytest.mark.e2e
def test_frames_are_aligned_at_age_zero(loader):
    """Every sensor of a frame carries the anchor tick's timestamp."""
    for index in sample_indices(len(loader)):
        anchor = anchor_us(loader, index)
        frame = loader[index]
        assert frame.cameras, "frame without cameras"
        for camera in frame.cameras:
            assert camera.timestamp.time_us == anchor
        assert frame.lidar_sweeps is not None and 0 in frame.lidar_sweeps
        for age, sweep in frame.lidar_sweeps.items():
            assert sweep.timestamp.time_us == anchor - age * TICK_US
            assert len(sweep.point_cloud_3d) > 0
        expected_ages = min(SWEEP_WINDOW, frame.scene_metadata.initial_idx + 1)
        assert set(frame.lidar_sweeps) == set(range(expected_ages))


@pytest.mark.e2e
def test_past_motion_matches_the_stored_localization(loader):
    """Past positions/yaws re-derive from the raw driving-meta stream."""
    for index in sample_indices(len(loader)):
        frame = loader[index]
        positions, yaws = frame.past_positions, frame.past_yaws
        assert positions is not None and yaws is not None
        expected_len = min(SWEEP_WINDOW, frame.scene_metadata.initial_idx + 1)
        assert positions.shape == (expected_len, 2)
        assert yaws.shape == (expected_len,)
        np.testing.assert_allclose(positions[0], [0.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(yaws[0], 0.0, atol=1e-9)

        scene = loader.scenes[index]
        anchor = scene.get_timestamp_at_iteration(0).time_us
        poses = {}
        for modality in scene.get_modality_between_timestamps(
            anchor - (SWEEP_WINDOW - 1) * TICK_US,
            anchor,
            ModalityType.CUSTOM,
            DRIVING_META_MODALITY_ID,
            inclusive="both",
        ):
            assert isinstance(modality, CustomModality)
            age = round((anchor - modality.timestamp.time_us) / TICK_US)
            poses[age] = localized_position_yaw(
                np.asarray(modality.data[LOCALIZED_EGO_STATE_KEY], dtype=np.float64),
            )
        anchor_position, anchor_yaw = poses[0]
        cos, sin = np.cos(anchor_yaw), np.sin(anchor_yaw)
        to_anchor = np.array([[cos, sin], [-sin, cos]])
        for age in range(expected_len):
            position, yaw = poses[age]
            np.testing.assert_allclose(
                positions[age],
                to_anchor @ (position - anchor_position),
                atol=1e-6,
            )
            np.testing.assert_allclose(
                np.mod(yaws[age] - (yaw - anchor_yaw) + np.pi, 2 * np.pi) - np.pi,
                0.0,
                atol=1e-6,
            )


@pytest.mark.e2e
def test_localized_pose_is_a_planar_se3_near_the_ground_truth(loader):
    """The stored localization is a valid planar SE(3) close to the true pose."""
    for index in sample_indices(len(loader)):
        frame = loader[index]
        assert frame.meta is not None
        matrix = np.asarray(frame.meta[LOCALIZED_EGO_STATE_KEY], dtype=np.float64)
        assert matrix.shape == (4, 4)
        np.testing.assert_allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-12)
        rotation = matrix[:3, :3]
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-9)
        np.testing.assert_allclose(rotation[2], [0.0, 0.0, 1.0], atol=1e-12)
        assert matrix[2, 3] == 0.0

        assert frame.ego_state is not None
        gt_position, _ = carla_ego_pose(frame.ego_state)
        localized_position, _ = localized_position_yaw(matrix)
        assert np.linalg.norm(localized_position - gt_position) < (
            MAX_LOCALIZATION_ERROR_M
        ), "localized pose implausibly far from the ground truth"


@pytest.mark.e2e
def test_driving_meta_carries_the_core_contract(loader):
    """The meta fields the loaders depend on exist with their documented shapes."""
    for index in sample_indices(len(loader)):
        frame = loader[index]
        meta = frame.meta
        assert meta is not None
        assert str(loader.tp_pop_distance) in meta["target_point_indices"]
        for attribute in ("target_point_previous", "target_point", "target_point_next"):
            point = getattr(frame, attribute)
            assert point is not None and np.isfinite(point).all()
        route = np.asarray(meta["route"])
        assert route.ndim == 2 and route.shape[1] == 2 and np.isfinite(route).all()
        assert isinstance(meta[BOX_ATTRIBUTES_KEY], dict)
        for token in meta[BOX_ATTRIBUTES_KEY]:
            assert token.lstrip("-").isdigit(), "box key is not a stringified actor id"
        if frame.box_detections is not None:
            for detection in frame.box_detections.box_detections:
                assert detection.attributes.track_token.lstrip("-").isdigit()


@pytest.mark.e2e
def test_future_accessors_walk_the_tick_grid(windowed_loader):
    """Futures at iteration ``i`` are the states exactly ``i`` ticks ahead."""
    scenes = windowed_loader.scenes
    assert scenes, "No scene has two save periods of future"
    iterations = [1, SAVE_PERIOD, 2 * SAVE_PERIOD]
    for index in sample_indices(len(scenes)):
        anchor = scenes[index].get_timestamp_at_iteration(0).time_us
        states = windowed_loader.future_ego_states(index, iterations)
        metas = windowed_loader.future_metas(index, iterations)
        previous_position, _ = carla_ego_pose(states[1])
        for iteration in iterations:
            state = states[iteration]
            assert state is not None
            assert state.timestamp.time_us == anchor + iteration * TICK_US
            position, _ = carla_ego_pose(state)
            # CARLA speeds stay far below 40 m/s; positions cannot jump.
            assert np.linalg.norm(position - previous_position) < 40.0 * 0.5
            previous_position = position
            meta = metas[iteration]
            assert meta is not None and LOCALIZED_EGO_STATE_KEY in meta


@pytest.mark.e2e
def test_both_views_load_and_pair_by_timestamp(data_root):
    """Both views load for the same ticks; the sensors differ, the tick does not."""
    normal = Py123DDataLoader(
        data_root,
        perturbation_prob=0.0,
        lidar_sweeps=False,
        radar_sweeps=False,
        map=False,
    )
    perturbated = Py123DDataLoader(
        data_root,
        perturbation_prob=1.0,
        lidar_sweeps=False,
        radar_sweeps=False,
        map=False,
    )
    if not perturbated.perturbated_scenes:
        pytest.skip("Dataset has no perturbated split")
    assert len(normal) == len(perturbated)

    paired = 0
    # Both views decode full images per sample; keep the pass small.
    for index in sample_indices(len(perturbated))[:3]:
        anchor = anchor_us(perturbated, index)
        perturbated_frame = perturbated[index]
        if perturbated_frame.perturbation is None:
            continue
        paired += 1
        normal_frame = normal[index]
        assert normal_frame.perturbation is None

        assert 0.0 < abs(perturbated_frame.perturbation.translation) <= 2.0
        assert abs(perturbated_frame.perturbation.rotation) <= 20.0
        for camera in (*perturbated_frame.cameras, *normal_frame.cameras):
            assert camera.timestamp.time_us == anchor
        assert not np.array_equal(
            perturbated_frame.cameras[0].image,
            normal_frame.cameras[0].image,
        )
        assert perturbated_frame.semantics is not None
        assert perturbated_frame.depths is not None
        # Ego-frame outputs are re-projected into the perturbated rig's frame.
        assert not np.allclose(
            perturbated_frame.target_point,
            normal_frame.target_point,
        )
    assert paired > 0, "perturbated split present but no scene paired"
