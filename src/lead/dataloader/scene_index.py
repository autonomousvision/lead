"""Scene enumeration over 123D logs: one scene anchored at every save tick, with history/future margins."""

import logging
from dataclasses import replace
from pathlib import Path

from py123d.api.scene.arrow.arrow_scene_builder import ArrowSceneBuilder
from py123d.api.scene.scene_api import SceneAPI
from py123d.api.scene.scene_filter import SceneFilter
from py123d.common.execution.thread_pool_executor import ThreadPoolExecutor
from py123d.common.runtime import get_dataset_paths

from lead.api.py123d_log_api import NORMAL_VIEW_SPLIT, PERTURBATED_VIEW_SPLIT
from lead.common import runtime_variables

LOG = logging.getLogger(__name__)


def resolve_roots(data_root: str | Path | None) -> tuple[Path, Path]:
    """Resolve the logs and maps roots of a dataset root.

    Args:
        data_root: Dataset root holding ``logs/`` and ``maps/``; None uses
            py123d's ``PY123D_DATA_ROOT``.

    Returns:
        The logs root and the maps root.
    """
    if data_root is None:
        paths = get_dataset_paths()
        assert (
            paths.py123d_logs_root is not None and paths.py123d_maps_root is not None
        ), "Provide data_root or set PY123D_DATA_ROOT."
        return Path(paths.py123d_logs_root), Path(paths.py123d_maps_root)
    return Path(data_root) / "logs", Path(data_root) / "maps"


def get_scenes(
    data_root: str | Path | None,
    scene_filter: SceneFilter,
) -> list[SceneAPI]:
    """Enumerate the scenes a filter selects from a dataset root.

    py123d discovers logs at any depth beneath a split directory, so a split
    name covers its scenario-type subdirectories.

    Args:
        data_root: Dataset root holding ``logs/`` and ``maps/``.
        scene_filter: The scenes to read. When it names no splits, the
            normal-view split is used.

    Returns:
        All scenes matching the filter, ordered by log and iteration.
    """
    logs_root, maps_root = resolve_roots(data_root)
    scene_filter = replace(
        scene_filter,
        split_names=scene_filter.split_names or [NORMAL_VIEW_SPLIT],
    )

    scene_builder = ArrowSceneBuilder(logs_root=logs_root, maps_root=maps_root)
    scenes = scene_builder.get_scenes(
        scene_filter,
        ThreadPoolExecutor(max_workers=runtime_variables.assigned_cpu_cores()),
    )
    LOG.info(f"Scene index: {len(scenes)} scenes in {scene_filter.split_names}")
    return scenes


def get_perturbated_scene_lookup(
    data_root: str | Path | None,
) -> dict[tuple[str, int], SceneAPI]:
    """Enumerate the perturbated-view scenes, keyed for pairing.

    The perturbated split holds only the view-dependent sensor streams
    (cameras, semantic, depth, radar) written tick-synchronously with the
    normal split, so a normal-view scene is paired with the perturbated scene
    anchored at the same simulation timestamp.

    Args:
        data_root: Dataset root holding ``logs/`` and ``maps/``.

    Returns:
        Mapping from (log name, anchor timestamp in µs) to the perturbated
        scene; empty when the perturbated split does not exist.
    """
    logs_root, maps_root = resolve_roots(data_root)
    if not (logs_root / PERTURBATED_VIEW_SPLIT).is_dir():
        LOG.info(
            f"No perturbated split at {logs_root / PERTURBATED_VIEW_SPLIT}, "
            "using normal views only",
        )
        return {}
    scene_builder = ArrowSceneBuilder(logs_root=logs_root, maps_root=maps_root)
    scene_filter = SceneFilter(
        split_names=[PERTURBATED_VIEW_SPLIT],
        history_num_iterations=0,
        future_num_iterations=0,
        required_scene_modalities=["camera:all@initial"],
    )
    scenes = scene_builder.get_scenes(
        scene_filter,
        ThreadPoolExecutor(max_workers=runtime_variables.assigned_cpu_cores()),
    )
    LOG.info(
        f"Scene index: {len(scenes)} perturbated scenes in {PERTURBATED_VIEW_SPLIT}",
    )
    return {
        (scene.log_name, scene.get_timestamp_at_iteration(0).time_us): scene
        for scene in scenes
    }
