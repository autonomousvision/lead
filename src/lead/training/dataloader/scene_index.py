"""Scene enumeration over 123D logs.

Training samples are 123D scenes anchored at every save tick, with
history/future margins replacing the legacy frame skipping.
"""

import logging
import os

from py123d.api.scene.arrow.arrow_scene_builder import ArrowSceneBuilder
from py123d.api.scene.scene_api import SceneAPI
from py123d.api.scene.scene_filter import SceneFilter
from py123d.common.execution.thread_pool_executor import ThreadPoolExecutor

from lead.common import runtime
from lead.config import LeadConfig

LOG = logging.getLogger(__name__)


def build_scene_filter(
    config: LeadConfig,
    history_num_iterations: int,
    future_num_iterations: int,
    log_names: list[str] | None = None,
) -> SceneFilter:
    """Build the scene filter for LEAD training scenes.

    Args:
        config: Training configuration with the 123D split selection.
        history_num_iterations: Save ticks of history each scene must provide.
        future_num_iterations: Save ticks of future each scene must provide.
        log_names: Optional explicit log subset (e.g. for held-out splits).

    Returns:
        SceneFilter selecting one scene per save tick with full margins.
    """
    return SceneFilter(
        split_names=[config.training.data.py123d_split_name],
        log_names=log_names,
        history_num_iterations=history_num_iterations,
        future_num_iterations=future_num_iterations,
        required_scene_modalities=[
            "ego_state_se3",
            "box_detections_se3@initial",
            "custom.driving_meta@initial",
            "camera:all@initial",
        ],
    )


def get_scenes(
    config: LeadConfig,
    history_num_iterations: int,
    future_num_iterations: int,
    log_names: list[str] | None = None,
) -> list[SceneAPI]:
    """Enumerate training scenes from the 123D logs root.

    Args:
        config: Training configuration with data roots.
        history_num_iterations: Save ticks of history each scene must provide.
        future_num_iterations: Save ticks of future each scene must provide.
        log_names: Optional explicit log subset (e.g. for held-out splits).

    Returns:
        All scenes matching the filter, ordered by log and iteration.
    """
    scene_builder = ArrowSceneBuilder(
        logs_root=config.training.data.py123d_logs_root,
        maps_root=config.training.data.py123d_maps_root,
    )
    scene_filter = build_scene_filter(
        config,
        history_num_iterations=history_num_iterations,
        future_num_iterations=future_num_iterations,
        log_names=log_names,
    )
    scenes = scene_builder.get_scenes(
        scene_filter,
        ThreadPoolExecutor(max_workers=runtime.assigned_cpu_cores()),
    )
    if config.training.data.hold_out_town13_routes:
        scenes = [
            scene
            for scene in scenes
            if not scene.get_log_metadata().log_name.startswith("Town13")
        ]
    LOG.info(
        f"Scene index: {len(scenes)} scenes in {config.training.data.py123d_split_name}",
    )
    return scenes


def get_perturbated_scene_lookup(
    config: LeadConfig,
) -> dict[tuple[str, int], SceneAPI]:
    """Enumerate the perturbated-view scenes, keyed for pairing.

    The perturbated split holds only the view-dependent sensor streams
    (cameras, semantic, depth, radar) written tick-synchronously with the
    normal split, so a normal-view scene is paired with the perturbated scene
    anchored at the same log frame.

    Args:
        config: Training configuration with data roots and split names.

    Returns:
        Mapping from (log name, anchor frame index) to the perturbated scene;
        empty when the perturbated split does not exist.
    """
    split_dir = os.path.join(
        config.training.data.py123d_logs_root,
        config.training.data.py123d_perturbated_split,
    )
    if not os.path.isdir(split_dir):
        LOG.info(f"No perturbated split at {split_dir}, using normal views only")
        return {}
    scene_builder = ArrowSceneBuilder(
        logs_root=config.training.data.py123d_logs_root,
        maps_root=config.training.data.py123d_maps_root,
    )
    scene_filter = SceneFilter(
        split_names=[config.training.data.py123d_perturbated_split],
        history_num_iterations=0,
        future_num_iterations=0,
        required_scene_modalities=["camera:all@initial"],
    )
    scenes = scene_builder.get_scenes(
        scene_filter,
        ThreadPoolExecutor(max_workers=runtime.assigned_cpu_cores()),
    )
    LOG.info(
        f"Scene index: {len(scenes)} perturbated scenes in "
        f"{config.training.data.py123d_perturbated_split}",
    )
    return {
        (scene.log_name, scene.get_scene_metadata().initial_idx): scene
        for scene in scenes
    }
