"""Tests of the frame spec union."""

from lead.log_reader import SceneLoadingSpec


def test_union_of_nothing_requests_nothing() -> None:
    spec = SceneLoadingSpec.union([])
    assert spec == SceneLoadingSpec()


def test_union_merges_switches_and_future_iterations() -> None:
    spec = SceneLoadingSpec.union(
        [
            SceneLoadingSpec(rgb_tick_ages=(0,), future_iterations=(4, 8)),
            SceneLoadingSpec(lidar_tick_ages=(0,), radar_tick_ages=(0,)),
            SceneLoadingSpec(read_semantic_cameras=True, future_iterations=(8, 12)),
        ],
    )
    assert spec == SceneLoadingSpec(
        rgb_tick_ages=(0,),
        lidar_tick_ages=(0,),
        radar_tick_ages=(0,),
        read_semantic_cameras=True,
        future_iterations=(4, 8, 12),
    )


def test_union_is_idempotent() -> None:
    spec = SceneLoadingSpec(
        read_depth_cameras=True,
        read_map_api=True,
        future_iterations=(2,),
    )
    assert SceneLoadingSpec.union([spec, spec]) == spec


def test_union_merges_the_past_tick_ages_of_every_part() -> None:
    spec = SceneLoadingSpec.union(
        [
            SceneLoadingSpec(lidar_tick_ages=(0, 2, 4)),
            SceneLoadingSpec(lidar_tick_ages=(0, 1)),
            SceneLoadingSpec(rgb_tick_ages=(0, 5)),
            SceneLoadingSpec(ego_pose_tick_ages=(0, 1, 2)),
        ],
    )
    # One read has to satisfy every part, so the ages are the union.
    assert spec.lidar_tick_ages == (0, 1, 2, 4)
    assert spec.rgb_tick_ages == (0, 5)
    assert spec.ego_pose_tick_ages == (0, 1, 2)


def test_a_stream_no_part_asks_for_is_not_read() -> None:
    # The tick ages are the whole statement: empty means "read nothing".
    spec = SceneLoadingSpec.union(
        [SceneLoadingSpec(lidar_tick_ages=(0, 1)), SceneLoadingSpec()],
    )
    assert spec.lidar_tick_ages == (0, 1)
    assert spec.radar_tick_ages == ()
    assert spec.rgb_tick_ages == ()
