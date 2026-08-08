"""Scene enumeration over 123D logs: one scene anchored at every save tick, with history/future margins."""

import logging
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
import numpy.typing as npt
from py123d.api.scene.arrow.arrow_scene_api import ArrowSceneAPI
from py123d.api.scene.arrow.arrow_scene_builder import (
    LazyArrowSceneBuilder,
    _parse_valid_log_dirs,
)
from py123d.api.scene.arrow.lazy_scene_sequence import (
    LazySceneSequence,
    LogSceneIndex,
    build_log_scene_index,
)
from py123d.api.scene.scene_api import SceneAPI
from py123d.api.scene.scene_filter import SceneFilter
from py123d.common.execution.thread_pool_executor import ThreadPoolExecutor
from py123d.common.runtime import get_dataset_paths

from lead.api.py123d_log_api import NORMAL_SENSOR_VIEW, PERTURBATED_SENSOR_VIEW
from lead.common import runtime_variables

LOG = logging.getLogger(__name__)

# Indexing a log is mostly opening its sync table, so a few threads saturate the
# reads while more of them only contend.
_INDEX_THREADS = 8


def _resolve_roots(data_root: str | Path | None) -> tuple[Path, Path]:
    """Resolve the logs and maps roots of a dataset root."""
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
) -> Sequence[SceneAPI]:
    """Enumerate the scenes a filter selects from a dataset root.

    py123d discovers logs at any depth beneath a split directory, so a split
    name covers its scenario-type subdirectories.

    Args:
        data_root: Dataset root holding ``logs/`` and ``maps/``.
        scene_filter: The scenes to read. When it names no splits, the
            normal-view split is used.

    Returns:
        All scenes matching the filter, ordered by log and iteration.

    Raises:
        FileNotFoundError: If the logs root does not exist.
        RuntimeError: If no scene matches, which means a wrong data root or split.
    """
    logs_root, maps_root = _resolve_roots(data_root)
    if not logs_root.is_dir():
        raise FileNotFoundError(
            f"logs root {logs_root} does not exist; check PY123D_DATA_ROOT",
        )
    scene_filter = replace(
        scene_filter,
        split_names=scene_filter.split_names or [NORMAL_SENSOR_VIEW],
    )

    # Lazy: a scene is built when it is indexed, not one object per scene here.
    scenes = LazyArrowSceneBuilder(
        logs_root=logs_root,
        maps_root=maps_root,
    ).get_scenes(
        scene_filter,
        ThreadPoolExecutor(
            max_workers=min(_INDEX_THREADS, runtime_variables.num_assigned_cpu_cores()),
        ),
    )
    if not scenes:
        raise RuntimeError(
            f"no scenes under {logs_root} match splits {scene_filter.split_names}; "
            f"check PY123D_DATA_ROOT and the split",
        )
    LOG.info(f"Scene index: {len(scenes)} scenes in {scene_filter.split_names}")
    return scenes


def scene_log_is_covered(
    scenes: Sequence[SceneAPI],
    log_names: set[str],
) -> npt.NDArray[np.bool_]:
    """Flag the scenes whose log is one of the given ones.

    A lazily enumerated sequence names its logs as columns, so no scene has to
    be built to answer this.

    Args:
        scenes: The scenes to flag.
        log_names: The logs that count as covered.

    Returns:
        One boolean per scene, in input order.
    """
    if isinstance(scenes, LazySceneSequence):
        scene_log_names, name_indices, _ = scenes.anchor_columns()
        covered = np.array([name in log_names for name in scene_log_names], dtype=bool)
        return covered[name_indices] if len(name_indices) else covered[:0]
    return np.array([scene.log_name in log_names for scene in scenes], dtype=bool)


class PerturbatedViewPairing:
    """The perturbated-view counterpart of a normal-view scene.

    The two splits are written tick-synchronously, so a counterpart is found by
    log and anchor timestamp. Only the perturbated logs' directories are listed
    up front; a log's ticks are indexed the first time one of them is read.
    """

    def __init__(self, logs_root: Path, log_names: list[str] | None) -> None:
        """List the logs the perturbated split provides.

        Args:
            logs_root: Root holding one directory per split.
            log_names: Restrict to these logs; None covers every log.
        """
        self._log_dir_by_name = {
            log_dir.name: log_dir
            for log_dir in _parse_valid_log_dirs(
                logs_root,
                SceneFilter(
                    split_names=[PERTURBATED_SENSOR_VIEW],
                    log_names=log_names,
                ),
            )
        }
        self._scene_filter = SceneFilter(
            split_names=[PERTURBATED_SENSOR_VIEW],
            history_num_iterations=0,
            future_num_iterations=0,
            required_scene_modalities=["camera:all@initial"],
        )
        self._index_by_log: dict[str, LogSceneIndex | None] = {}

    def __len__(self) -> int:
        return len(self._log_dir_by_name)

    def covered_logs(self) -> set[str]:
        """Logs the perturbated split provides.

        Returns:
            The log names, by directory name.
        """
        return set(self._log_dir_by_name)

    def scene_at(self, log_name: str, timestamp_us: int) -> SceneAPI:
        """Read the perturbated scene of a log's tick.

        Args:
            log_name: Log the normal-view scene comes from.
            timestamp_us: Anchor timestamp of the normal-view scene.

        Returns:
            The perturbated scene anchored at the same tick.

        Raises:
            KeyError: If the log or the tick has no perturbated counterpart.
        """
        if log_name not in self._index_by_log:
            log_dir = self._log_dir_by_name.get(log_name)
            self._index_by_log[log_name] = (
                None
                if log_dir is None
                else build_log_scene_index(log_dir, self._scene_filter)
            )
        log_index = self._index_by_log[log_name]
        if log_index is None:
            raise KeyError(f"no perturbated view for log {log_name}")

        timestamps = log_index.anchor_timestamps_us
        position = int(np.searchsorted(timestamps, timestamp_us))
        if position >= len(timestamps) or timestamps[position] != timestamp_us:
            raise KeyError(f"no perturbated view for {log_name} at tick {timestamp_us}")
        anchor_index = int(log_index.anchor_indices[position])
        return ArrowSceneAPI(
            log_dir=log_index.log_dir,
            scene_metadata=log_index.scene_metadata_at(anchor_index),
        )


def get_perturbated_view_pairing(
    data_root: str | Path | None,
    log_names: list[str] | None = None,
) -> PerturbatedViewPairing | None:
    """Find the perturbated-view logs a normal-view loader can pair with.

    The perturbated split holds only the view-dependent sensor streams
    (cameras, semantic, depth, radar) written tick-synchronously with the
    normal split, so a normal-view scene is paired with the perturbated scene
    anchored at the same simulation timestamp.

    Args:
        data_root: Dataset root holding ``logs/`` and ``maps/``.
        log_names: Restrict the pairing to these logs; None pairs every log.
            A loader over some logs pairs only those, never the whole split.

    Returns:
        The pairing, or None when the perturbated split does not exist.
    """
    logs_root, _ = _resolve_roots(data_root)
    if not (logs_root / PERTURBATED_SENSOR_VIEW).is_dir():
        LOG.info(
            f"No perturbated split at {logs_root / PERTURBATED_SENSOR_VIEW}, "
            "using normal views only",
        )
        return None
    pairing = PerturbatedViewPairing(logs_root, log_names)
    LOG.info(f"Scene index: {len(pairing)} logs in {PERTURBATED_SENSOR_VIEW}")
    return pairing
