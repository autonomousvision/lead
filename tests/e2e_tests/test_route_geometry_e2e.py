"""The navigation geometry of a log agrees with itself.

The target points come from the GNSS global plan the ego navigates by; the
dense route comes from the privileged planner's map waypoints. They are two
independent descriptions of the same drive, so a target point the ego actually
walked past must lie on the route the ego actually drove.
"""

import numpy as np
import pytest
from py123d.api.scene.scene_filter import SceneFilter
from py123d.datatypes import CustomModality

from lead.api.py123d_log_api import (
    DRIVING_META_MODALITY_ID,
    LOCALIZED_EGO_STATE_KEY,
    TARGET_POINTS_METADATA_KEY,
    ordered_target_points,
    se3_matrix_to_localized_pose,
)
from lead.log_reader import SceneLoader
from tests.e2e_tests.conftest import sample_indices

# The pop distance whose target-point indices the loader defaults to.
POP_DISTANCE = 3.25
# The bounds below guard against gross errors — a wrong frame, a wrong town, a
# swapped index — not against the metre-level disagreement two independent
# descriptions of the same drive legitimately have: the simulated GNSS carries
# one to two metres of noise and the route is sampled a metre apart.
MAX_OFF_ROUTE_M = 10.0
# A target point is popped once the ego is within the pop distance of it, but
# only every fifth tick is stored, so the nearest stored pose is that much
# further out again.
MAX_DISTANCE_TO_DRIVEN_PATH_M = 10.0
# The route is sampled one point per metre along the lane.
MAX_ROUTE_STEP_M = 5.0
# How far the route may sit from the original while still counting as "not
# rerouted". The two are sampled from different arrays and stored as float32,
# so they agree to well under a lane width rather than exactly.
MAX_UNCHANGED_ROUTE_OFFSET_M = 2.5
# Logs inspected. These assertions are per-log and need every tick of a log, so
# the bound is on logs rather than on ticks.
MAX_LOGS = 3


@pytest.fixture(scope="module")
def loader(data_root: str) -> SceneLoader:
    """A loader over every stored tick, sensors off.

    Only the driving-meta stream and the log metadata are read here, so every
    sensor modality stays disabled and all ticks are cheap to walk.

    Args:
        data_root: The dataset root.

    Returns:
        The loader.
    """
    loader = SceneLoader(
        data_root,
        SceneFilter(shuffle=False, history_num_iterations=0, future_num_iterations=0),
        target_point_pop_distance=POP_DISTANCE,
        read_lidar_sweeps=False,
        read_radar_sweeps=False,
        read_semantic_cameras=False,
        read_depth_cameras=False,
        read_map_api=False,
    )
    assert len(loader) > 0, f"No scenes under {data_root}"
    return loader


@pytest.fixture(scope="module")
def inspected_scenes(loader: SceneLoader) -> list:
    """The scenes of up to ``MAX_LOGS`` logs, every tick of each.

    Args:
        loader: The loader owning the logs.

    Returns:
        The scenes, in the loader's order.
    """
    names: list[str] = []
    for scene in loader.scenes:
        if scene.log_name not in names:
            names.append(scene.log_name)
        if len(names) > MAX_LOGS:
            break
    selected = set(names[:MAX_LOGS])
    return [scene for scene in loader.scenes if scene.log_name in selected]


@pytest.fixture(scope="module")
def navigation(inspected_scenes: list) -> dict[str, dict]:
    """Every inspected log's target points, dense route and driven path.

    The route is truncated to a horizon ahead of the ego, so the union over all
    ticks of a log is what the ego actually drove along.

    Args:
        inspected_scenes: The scenes to walk.

    Returns:
        Per log name: ``target_points`` (n, 3), ``route`` (m, 2),
        ``route_original`` (m, 2), ``driven`` (t, 2) and the set of visited
        target-point indices.
    """
    logs: dict[str, dict] = {}
    for scene in inspected_scenes:
        log = logs.setdefault(
            scene.log_name,
            {
                "target_points": np.asarray(
                    scene.get_all_custom_modality_metadatas()[
                        DRIVING_META_MODALITY_ID
                    ].metadata[TARGET_POINTS_METADATA_KEY],
                    dtype=np.float64,
                ),
                "route": [],
                "route_original": [],
                "driven": [],
                "visited": set(),
            },
        )
        modality = scene.get_custom_modality_at_iteration(0, DRIVING_META_MODALITY_ID)
        assert isinstance(modality, CustomModality)
        meta = modality.data

        route = np.asarray(meta["route"], dtype=np.float64)
        log["route"].append(route)
        log["route_original"].append(
            np.asarray(meta.get("route_original", route), dtype=np.float64),
        )
        # The pose the ego navigated by, which is what popped the target points.
        position, _ = se3_matrix_to_localized_pose(
            np.asarray(meta[LOCALIZED_EGO_STATE_KEY], dtype=np.float64),
        )
        log["driven"].append(position)
        log["visited"].add(int(meta["target_point_indices"][str(POP_DISTANCE)]))

    for log in logs.values():
        log["route"] = np.concatenate(log["route"])
        log["route_original"] = np.concatenate(log["route_original"])
        log["driven"] = np.asarray(log["driven"])
    return logs


def _min_distances(
    points: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """Distance from every point to the nearest reference point.

    Args:
        points: Query points, shape (k, 2).
        reference: Reference points, shape (m, 2).

    Returns:
        The per-point minimum distances, shape (k,).
    """
    return np.linalg.norm(points[:, None, :] - reference[None, :, :], axis=2).min(
        axis=1,
    )


@pytest.mark.e2e
def test_visited_target_points_lie_on_the_dense_route(navigation) -> None:
    """A target point the ego popped is a point of the route it drove."""
    checked = 0
    for log_name, log in navigation.items():
        target_points = log["target_points"]
        # The last visited index is the last point the ego came within the pop
        # distance of; everything up to it is behind the ego and covered.
        popped = sorted(index for index in log["visited"] if index > 0)
        if not popped:
            continue
        indices = np.arange(popped[-1] + 1)
        distances = _min_distances(
            target_points[indices, :2],
            log["route_original"],
        )
        worst = int(np.argmax(distances))
        assert distances[worst] < MAX_OFF_ROUTE_M, (
            f"{log_name}: target point {indices[worst]} is "
            f"{distances[worst]:.2f} m off the dense route"
        )
        checked += len(indices)
    assert checked, "no log popped a target point; nothing was verified"


@pytest.mark.e2e
def test_popped_target_points_sit_on_the_path_the_ego_drove(navigation) -> None:
    """A target point is popped by driving up to it, so the ego was there."""
    for log_name, log in navigation.items():
        driven = log["driven"]
        # Index 0 is the route's start, which the ego never has to reach.
        popped = [index for index in sorted(log["visited"]) if index > 0]
        for index in popped:
            previous, _, _ = ordered_target_points(log["target_points"], index)
            distance = float(_min_distances(previous[None, :2], driven)[0])
            assert distance < MAX_DISTANCE_TO_DRIVEN_PATH_M, (
                f"{log_name}: target point {index} was popped, but the ego "
                f"never came closer than {distance:.2f} m to it"
            )


@pytest.mark.e2e
def test_target_point_indices_advance_monotonically(inspected_scenes) -> None:
    """The planner only ever pops forward, and never past the route's end."""
    by_log: dict[str, list[tuple[int, int]]] = {}
    num_points_by_log: dict[str, int] = {}
    for scene in inspected_scenes:
        modality = scene.get_custom_modality_at_iteration(0, DRIVING_META_MODALITY_ID)
        assert isinstance(modality, CustomModality)
        by_log.setdefault(scene.log_name, []).append(
            (
                scene.get_timestamp_at_iteration(0).time_us,
                int(modality.data["target_point_indices"][str(POP_DISTANCE)]),
            ),
        )
        num_points_by_log[scene.log_name] = len(
            scene.get_all_custom_modality_metadatas()[
                DRIVING_META_MODALITY_ID
            ].metadata[TARGET_POINTS_METADATA_KEY],
        )

    for log_name, entries in by_log.items():
        indices = [index for _, index in sorted(entries)]
        assert indices == sorted(indices), (
            f"{log_name}: the target-point index moved backwards"
        )
        # The planner keeps the last target point ahead as the destination.
        assert indices[-1] <= num_points_by_log[log_name] - 2, (
            f"{log_name}: the last target point was popped, leaving no "
            f"destination ahead"
        )


@pytest.mark.e2e
def test_the_reroute_stays_close_to_the_original_route(loader) -> None:
    """Evasive reroutes shift the route within a lane, not onto another road."""
    for index in sample_indices(len(loader.scenes)):
        scene = loader.scenes[index]
        modality = scene.get_custom_modality_at_iteration(0, DRIVING_META_MODALITY_ID)
        assert isinstance(modality, CustomModality)
        meta = modality.data
        if "route_original" not in meta:
            pytest.skip("logs predate the route_original key")
        route = np.asarray(meta["route"], dtype=np.float64)
        original = np.asarray(meta["route_original"], dtype=np.float64)
        assert route.shape == original.shape
        # Both are sampled from the same index along the plan, so compare
        # pointwise rather than to the nearest neighbour.
        offsets = np.linalg.norm(route - original, axis=1)
        assert (offsets < MAX_OFF_ROUTE_M).all(), (
            f"scene {index}: the route deviates up to {offsets.max():.2f} m "
            f"from the original"
        )
        # The flag is read at the ego's own route index, so it constrains the
        # first saved point only; the horizon ahead may already be shifted.
        if not meta.get("changed_route", False):
            assert offsets[0] < MAX_UNCHANGED_ROUTE_OFFSET_M, (
                f"scene {index}: the route starts {offsets[0]:.2f} m off the "
                f"original though nothing was rerouted"
            )


@pytest.mark.e2e
def test_the_stored_route_is_a_gapless_plan_ahead_of_the_ego(loader) -> None:
    """The route is the remaining plan, sampled continuously along the lane."""
    for index in sample_indices(len(loader)):
        scene = loader.scenes[index]
        modality = scene.get_custom_modality_at_iteration(0, DRIVING_META_MODALITY_ID)
        assert isinstance(modality, CustomModality)
        route = np.asarray(modality.data["route"], dtype=np.float64)
        assert route.ndim == 2 and route.shape[1] == 2
        assert len(route) > 1, "the stored route holds no direction"
        assert np.isfinite(route).all()
        # Consecutive plan points are map samples a fixed step apart.
        steps = np.linalg.norm(np.diff(route, axis=0), axis=1)
        assert (steps > 0.0).all(), "the route repeats a point"
        assert steps.max() < MAX_ROUTE_STEP_M, "the route has a gap"
