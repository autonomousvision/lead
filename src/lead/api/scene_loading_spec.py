"""The scene loading spec: which modalities one read of the generic loader populates."""

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class SceneLoadingSpec:
    """Read switches of one :class:`~lead.api.scene_data.SceneData` request.

    The ego state, boxes, driving meta, target points and past motion are part
    of every read; the spec gates the expensive modality streams only.

    A ``*_tick_ages`` tuple says *when* to read a stream, in ticks before the
    anchor, and whether to read it at all: empty reads nothing, ``(0,)`` is the
    anchor alone, ``(0, 5)`` the anchor and five ticks back. They come from the
    policy's ``past_<modality>_length_s`` and ``past_<modality>_frequency`` (see
    :class:`~lead.config.policy.abstract_policy_config.AbstractPolicyConfig`).
    """

    # Tick ages of the RGB camera rigs to read. Age 0 lands in
    # ``SceneData.cameras``, the older ones in ``SceneData.past_cameras``.
    rgb_tick_ages: tuple[int, ...] = ()
    # Tick ages of the lidar sweeps to read.
    lidar_tick_ages: tuple[int, ...] = ()
    # Tick ages of the radar sweeps to read.
    radar_tick_ages: tuple[int, ...] = ()
    # Tick ages the ego-pose history must reach beyond the sweeps' own. The
    # sweep alignment indexes that history by tick age, so a read spans the
    # sweep ages whether or not this asks for them.
    ego_pose_tick_ages: tuple[int, ...] = ()
    # Depth cameras at the anchor tick.
    read_depth_cameras: bool = False
    # Semantic and instance cameras at the anchor tick.
    read_semantic_cameras: bool = False
    # Map API of the town.
    read_map_api: bool = False
    # Future scene iterations whose driving metas and ego states the scene data
    # carries (empty = none).
    future_iterations: tuple[int, ...] = ()

    @staticmethod
    def union(specs: Iterable["SceneLoadingSpec"]) -> "SceneLoadingSpec":
        """Merge specs into the one read that satisfies all of them.

        Args:
            specs: The specs to merge.

        Returns:
            The spec requesting every modality any input spec requests.
        """
        specs = list(specs)
        return SceneLoadingSpec(
            rgb_tick_ages=_merged(spec.rgb_tick_ages for spec in specs),
            lidar_tick_ages=_merged(spec.lidar_tick_ages for spec in specs),
            radar_tick_ages=_merged(spec.radar_tick_ages for spec in specs),
            ego_pose_tick_ages=_merged(spec.ego_pose_tick_ages for spec in specs),
            read_depth_cameras=any(spec.read_depth_cameras for spec in specs),
            read_semantic_cameras=any(spec.read_semantic_cameras for spec in specs),
            read_map_api=any(spec.read_map_api for spec in specs),
            future_iterations=_merged(spec.future_iterations for spec in specs),
        )


def _merged(iterations: Iterable[tuple[int, ...]]) -> tuple[int, ...]:
    """The ascending union of the iterations every spec asked for.

    Args:
        iterations: One tuple of tick ages or scene iterations per spec.

    Returns:
        The ascending union, empty when no spec asked for any.
    """
    return tuple(sorted({iteration for group in iterations for iteration in group}))
