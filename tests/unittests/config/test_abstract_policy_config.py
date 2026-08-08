"""Tests of the policy-config contract and its temporal window: seconds and Hz in, tick ages out."""

import pytest

from lead.config import load_lead_config

# The dataset the defaults describe: 20 Hz ticks, cameras only on save ticks.
CARLA_FPS = 20
CAMERA_STORE_FREQ = 5


def transfuser(overrides: dict | None = None, carla_fps: int = CARLA_FPS):
    """The TransFuser config section at a given tick rate.

    Args:
        overrides: ``policy.transfuser`` overrides to apply.
        carla_fps: Simulator tick rate the dataset was logged at.

    Returns:
        The config section.
    """
    return load_lead_config(
        {
            "expert": {"simulation": {"carla_fps": carla_fps}},
            "policy": {"transfuser": dict(overrides or {})},
        },
    ).policy.transfuser


class TestDerivation:
    """Tick ages and counts follow from the seconds and the rate."""

    def test_past_span_includes_the_anchor(self):
        # 0.2 s at 20 Hz is 4 ticks back, plus the anchor itself.
        assert transfuser().past_lidar_tick_ages == (0, 1, 2, 3, 4)

    def test_a_zero_past_span_is_the_anchor_alone(self):
        assert transfuser({"past_lidar_length_s": 0.0}).past_lidar_tick_ages == (0,)

    def test_future_span_excludes_the_anchor(self):
        # The anchor is the present, not a prediction.
        assert transfuser().future_ego_pose_iterations == (
            5,
            10,
            15,
            20,
            25,
            30,
            35,
            40,
        )

    def test_prediction_count_derives_from_the_span(self):
        config = transfuser()
        assert config.num_ego_pose_prediction == len(config.future_ego_pose_iterations)
        assert config.num_ego_pose_prediction == 8

    def test_a_lower_frequency_thins_the_span_it_does_not_shorten_it(self):
        config = transfuser({"past_lidar_length_s": 0.4, "past_lidar_frequency": 5})
        # Still 0.4 s of history, sampled every 4th tick rather than every tick.
        assert config.past_lidar_tick_ages == (0, 4, 8)

    def test_past_window_is_the_widest_modality(self):
        config = transfuser({"past_lidar_length_s": 0.2, "past_rgb_length_s": 0.5})
        assert config.past_window_num_iterations == 10  # 0.5 s at 20 Hz

    def test_required_history_is_the_camera_history_alone(self):
        # Sweeps and poses degrade at a route's start; only cameras are exact.
        assert transfuser().required_history_num_iterations == 0
        assert (
            transfuser({"past_rgb_length_s": 0.5}).required_history_num_iterations == 10
        )
        assert (
            transfuser({"past_lidar_length_s": 1.0}).required_history_num_iterations
            == 0
        )

    def test_frequencies_default_to_the_fastest_the_dataset_stores(self):
        config = transfuser()
        # Lidar is stored every tick, the cameras only on save ticks.
        assert config.past_lidar_frequency == CARLA_FPS
        assert config.past_rgb_frequency == CARLA_FPS // CAMERA_STORE_FREQ


class TestTickRateInvariance:
    """The same seconds mean the same thing whatever rate the log was written at."""

    @pytest.mark.parametrize("carla_fps", [10, 20])
    def test_same_seconds_give_the_same_plan(self, carla_fps):
        # 5 Hz divides both rates, so the plan is identical either way.
        config = transfuser({"future_ego_pose_frequency": 5}, carla_fps=carla_fps)
        assert config.future_ego_pose_length_s == 2.0
        assert config.num_ego_pose_prediction == 10

    def test_only_the_tick_counts_follow_the_rate(self):
        fast = transfuser({"future_ego_pose_frequency": 5}, carla_fps=20)
        slow = transfuser({"future_ego_pose_frequency": 5}, carla_fps=10)
        assert fast.future_ego_pose_iterations == tuple(range(4, 41, 4))
        assert slow.future_ego_pose_iterations == tuple(range(2, 21, 2))
        assert fast.past_lidar_tick_ages == (0, 1, 2, 3, 4)
        assert slow.past_lidar_tick_ages == (0, 1, 2)

    def test_default_frequencies_hold_at_another_rate(self):
        # Defaults derive from the rate, so they cannot go stale against it.
        config = transfuser(carla_fps=10)
        assert config.past_lidar_frequency == 10
        assert config.past_rgb_frequency == 10 // CAMERA_STORE_FREQ


class TestValidation:
    """A window the dataset cannot serve raises rather than rounding silently."""

    def test_frequency_must_divide_the_tick_rate(self):
        config = transfuser({"future_ego_pose_frequency": 3})
        with pytest.raises(ValueError, match="divide the simulator"):
            _ = config.future_ego_pose_iterations

    def test_frequency_cannot_beat_the_storage_period(self):
        # The cameras only exist on save ticks, however fast the simulator ran.
        config = transfuser({"past_rgb_frequency": CARLA_FPS})
        with pytest.raises(ValueError, match="fastest available rate is 4 Hz"):
            _ = config.past_rgb_tick_ages

    def test_a_span_must_be_a_whole_number_of_ticks(self):
        config = transfuser({"past_lidar_length_s": 0.13})
        with pytest.raises(ValueError, match="whole number of ticks"):
            _ = config.past_lidar_tick_ages

    def test_a_negative_span_raises(self):
        config = transfuser({"past_lidar_length_s": -0.2})
        with pytest.raises(ValueError, match="whole number of ticks"):
            _ = config.past_lidar_tick_ages

    def test_lidar_may_run_at_the_full_tick_rate(self):
        # Unlike the cameras, lidar is stored every tick.
        assert transfuser({"past_lidar_frequency": CARLA_FPS}).past_lidar_tick_ages


class TestDerivedValuesAreNotConfigurable:
    """The counts fall out of the spans, so they cannot drift from them."""

    def test_prediction_count_rejects_a_user_override(self):
        config = load_lead_config()
        with pytest.raises(AttributeError, match="derived"):
            config.apply_overrides(
                {"policy": {"ego_status": {"num_ego_pose_prediction": 12}}},
                is_user_override=True,
            )

    def test_a_stored_config_reload_keeps_the_derived_value(self):
        # Configs written before a rename must still load; the stored value loses.
        config = load_lead_config()
        config.apply_overrides(
            {"policy": {"ego_status": {"num_ego_pose_prediction": 12}}},
            is_user_override=False,
        )
        assert config.policy.ego_status.num_ego_pose_prediction == 8

    def test_the_span_drives_the_count(self):
        assert (
            transfuser({"future_ego_pose_length_s": 1.0}).num_ego_pose_prediction == 4
        )


class TestPolicyContract:
    """What each policy publishes to its scene filter."""

    def test_transfuser_requires_no_history_despite_its_sweep_window(self):
        from lead.policy.transfuser.transfuser import Transfuser

        lead_config = load_lead_config(
            {"policy": {"transfuser": {"use_planning_decoder": True}}},
        )
        scene_filter = Transfuser(lead_config).build_scene_filter()
        # The sweep window is a preference, not a requirement: a route's first
        # frames train on the shorter history their log offers.
        assert lead_config.policy.transfuser.past_window_num_iterations == 4
        assert scene_filter.history_num_iterations == 0
        assert scene_filter.future_num_iterations == 40

    def test_ego_status_reads_no_past(self):
        from lead.policy.ego_status.ego_status import EgoStatus

        lead_config = load_lead_config()
        assert lead_config.policy.ego_status.past_window_num_iterations == 0
        scene_filter = EgoStatus(lead_config).build_scene_filter()
        assert scene_filter.history_num_iterations == 0
        assert scene_filter.future_num_iterations == 40

    def test_pretraining_needs_no_future(self):
        from lead.policy.transfuser.transfuser import Transfuser

        lead_config = load_lead_config(
            {"policy": {"transfuser": {"use_planning_decoder": False}}},
        )
        assert Transfuser(lead_config).build_scene_filter().future_num_iterations == 0

    def test_box_velocity_needs_sweep_history(self):
        assert transfuser().predict_box_velocity is True
        assert transfuser({"past_lidar_length_s": 0.0}).predict_box_velocity is False


# ``build_scene_filter`` lives once on AbstractPolicy, assembled from the
# ``training.data`` selection and the config contract; the tests below pin
# that mapping and the contract itself. Add a policy here when you add one.
POLICY_CLASSES = [
    "lead.policy.transfuser.transfuser:Transfuser",
    "lead.policy.ego_status.ego_status:EgoStatus",
]


def scene_filter_of(target: str, data_overrides: dict | None = None):
    """The scene filter one policy builds under the given data selection.

    Args:
        target: ``module:Class`` path of the policy.
        data_overrides: ``training.data`` overrides to apply.

    Returns:
        The policy's scene filter.
    """
    import importlib

    module_name, _, class_name = target.partition(":")
    policy_class = getattr(importlib.import_module(module_name), class_name)
    lead_config = load_lead_config(
        {
            "policy": {"target": target},
            "training": {"data": dict(data_overrides or {})},
        },
    )
    return policy_class(lead_config).build_scene_filter(), lead_config


DATA_SELECTION = {
    "towns": ["Town03", "Town05"],
    "py123d_log_names": ["log_a", "log_b"],
    "max_num_scenes": 500,
    "num_chunks": 4,
    "chunk_index": 2,
    "shuffle_scenes": True,
}


@pytest.mark.parametrize("target", POLICY_CLASSES)
def test_every_policy_honors_the_dataset_wide_selection(target):
    """A policy's filter must carry every ``training.data`` selection key."""
    scene_filter, lead_config = scene_filter_of(target, DATA_SELECTION)

    assert scene_filter.log_locations == DATA_SELECTION["towns"]
    assert scene_filter.log_names == DATA_SELECTION["py123d_log_names"]
    assert scene_filter.max_num_scenes == DATA_SELECTION["max_num_scenes"]
    assert scene_filter.num_chunks == DATA_SELECTION["num_chunks"]
    assert scene_filter.chunk_idx == DATA_SELECTION["chunk_index"]
    assert scene_filter.shuffle is True
    assert scene_filter.split_names == [lead_config.training.data.py123d_split]


@pytest.mark.parametrize("target", POLICY_CLASSES)
def test_every_policy_leaves_an_unset_selection_unfiltered(target):
    """The defaults must not narrow the scene list; None means "no filter"."""
    scene_filter, _ = scene_filter_of(target)

    assert scene_filter.log_locations is None
    assert scene_filter.log_names is None
    assert scene_filter.max_num_scenes is None
    # Chunking is off entirely rather than "chunk 0 of 1".
    assert scene_filter.num_chunks is None
    assert scene_filter.chunk_idx is None


@pytest.mark.parametrize("target", POLICY_CLASSES)
def test_every_policy_sets_both_temporal_margins(target):
    """py123d enumerates nothing when a history meets an unset future."""
    scene_filter, _ = scene_filter_of(target)

    assert scene_filter.history_num_iterations is not None
    assert scene_filter.future_num_iterations is not None


@pytest.mark.parametrize("target", POLICY_CLASSES)
def test_every_policy_publishes_its_config_section(target):
    """``get_policy_config`` is the typed section policy-agnostic code reads."""
    import importlib

    from lead.config.policy.abstract_policy_config import AbstractPolicyConfig

    module_name, _, class_name = target.partition(":")
    policy_class = getattr(importlib.import_module(module_name), class_name)
    lead_config = load_lead_config({"policy": {"target": target}})
    policy_config = policy_class(lead_config).get_policy_config()

    assert isinstance(policy_config, AbstractPolicyConfig)


@pytest.mark.parametrize("target", POLICY_CLASSES)
def test_every_policy_requires_the_modalities_its_readers_index(target):
    """Every policy reads the ego pose and the driving meta of the anchor tick."""
    required = scene_filter_of(target)[0].required_scene_modalities

    assert required is not None
    assert "ego_state_se3" in required
    assert "custom.driving_meta@initial" in required
