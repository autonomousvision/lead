"""Tests for the LeadConfig tree: hierarchy, profiles, overrides and serialization."""

import pytest
import yaml

from lead.config import (
    apply_stored_expert_config,
    load_lead_config,
    yaml_filtered,
)


@pytest.fixture
def config():
    """Default config tree without any overrides."""
    return load_lead_config()


class TestHierarchy:
    """Structure and derived values of the tree."""

    def test_sections_exist(self, config):
        assert config.expert.sensor_rig is not None
        assert config.policy.transfuser is not None
        assert config.training.experiment is not None
        assert config.evaluation.controller is not None

    def test_cross_section_derived_values(self, config):
        sensor_rig = config.expert.sensor_rig
        assert sensor_rig.num_cameras == len(sensor_rig.cameras)
        assert config.policy.transfuser.final_image_width == sensor_rig.image_width
        data_collection = config.expert.data_collection
        assert config.policy.transfuser.lidar_horz_anchors == (
            data_collection.lidar_width_pixel // 32
        )

    def test_points_per_meter_propagates(self, config):
        config.expert.simulation.points_per_meter = 20
        assert config.expert.driving.transition_smoothness_distance == 8 * 20
        assert config.expert.pid.lateral_pid_minimum_lookahead_distance == 2.4 * 20

    def test_unknown_attribute_set_raises(self, config):
        with pytest.raises(AttributeError, match="renamed"):
            config.expert.pid.lateral_pid_kpp = 1.0


class TestProfiles:
    """Yaml config profile selection and application."""

    def test_default_profile_is_the_six_camera_rig(self, config):
        assert config.expert.config_profile == "default"
        assert config.expert.sensor_rig.num_cameras == 6
        assert config.policy.transfuser.final_image_width == 6 * 384

    def test_expert_profile_changes_camera_rig(self, monkeypatch):
        monkeypatch.setenv(
            "LEAD_CONFIG",
            "expert.config_profile=leaderboard2_3cameras",
        )
        config = load_lead_config()
        assert config.expert.config_profile == "leaderboard2_3cameras"
        assert config.expert.sensor_rig.num_cameras == 3
        assert config.policy.transfuser.final_image_width == 3 * 384

    def test_unknown_profile_raises(self, monkeypatch):
        monkeypatch.setenv("LEAD_CONFIG", "expert.config_profile=nope")
        with pytest.raises(FileNotFoundError, match="Available profiles"):
            load_lead_config()

    def test_policy_profile_overrides_architecture(self, monkeypatch):
        monkeypatch.setenv(
            "LEAD_CONFIG",
            "policy.config_profile=transfuser_regnety032",
        )
        config = load_lead_config()
        assert config.policy.transfuser.image_architecture == "regnety_032"
        assert config.policy.target == "lead.policy.transfuser.transfuser:Transfuser"


class TestOverrides:
    """Env/CLI dotlist override resolution."""

    def test_fully_qualified_keys(self, monkeypatch):
        monkeypatch.setenv(
            "LEAD_CONFIG",
            "training.experiment.logdir=/tmp/x "
            "expert.pid.lateral_pid_kp=1.5 "
            "expert.sensor_rig.use_radars=false",
        )
        config = load_lead_config()
        assert config.training.experiment.logdir == "/tmp/x"
        assert config.expert.pid.lateral_pid_kp == 1.5
        assert config.expert.sensor_rig.use_radars is False

    def test_abbreviated_key_raises(self, monkeypatch):
        monkeypatch.setenv("LEAD_CONFIG", "logdir=/tmp/x")
        with pytest.raises(AttributeError, match="Unknown"):
            load_lead_config()

    def test_unknown_key_raises(self, monkeypatch):
        monkeypatch.setenv("LEAD_CONFIG", "expert.pid.not_a_real_knob=1")
        with pytest.raises(AttributeError, match="Unknown"):
            load_lead_config()

    def test_derived_override_raises(self, monkeypatch):
        monkeypatch.setenv("LEAD_CONFIG", "expert.sensor_rig.num_cameras=3")
        with pytest.raises(AttributeError, match="derived"):
            load_lead_config()

    def test_overridable_property_override(self, monkeypatch):
        monkeypatch.setenv("LEAD_CONFIG", "training.optimization.batch_size=8")
        config = load_lead_config()
        assert config.training.optimization.batch_size == 8

    def test_lossy_int_override_raises(self, monkeypatch):
        monkeypatch.setenv("LEAD_CONFIG", "training.optimization.epochs=0.5")
        config = load_lead_config()
        with pytest.raises(TypeError, match="integer"):
            _ = config.training.optimization.epochs

    def test_per_index_list_override_raises(self, monkeypatch):
        monkeypatch.setenv(
            "LEAD_CONFIG",
            "expert.sensor_rig.cameras.0.width=512",
        )
        with pytest.raises(TypeError, match="expects a list"):
            load_lead_config()

    def test_mutating_one_config_does_not_leak(self, config):
        config.expert.sensor_rig.cameras[0]["width"] = 9999
        assert load_lead_config().expert.sensor_rig.cameras[0]["width"] == 384


class TestSerialization:
    """Round trips through nested yaml dumps."""

    def test_nested_round_trip(self, config):
        dump = yaml.safe_load(yaml.safe_dump(yaml_filtered(config.to_dict())))
        reloaded = load_lead_config(loaded_config=dump)
        assert reloaded.expert.pid.lateral_pid_kp == config.expert.pid.lateral_pid_kp
        assert reloaded.expert.sensor_rig.num_cameras == 6

    def test_profile_survives_round_trip(self, monkeypatch):
        monkeypatch.setenv(
            "LEAD_CONFIG",
            "expert.config_profile=leaderboard2_3cameras",
        )
        dump = yaml_filtered(load_lead_config().to_dict())
        monkeypatch.delenv("LEAD_CONFIG")
        reloaded = load_lead_config(loaded_config=yaml.safe_load(yaml.safe_dump(dump)))
        assert reloaded.expert.config_profile == "leaderboard2_3cameras"
        assert reloaded.expert.sensor_rig.num_cameras == 3


class TestDatasetExpertConfig:
    """Dataset-stored expert config loading."""

    def test_nested_dataset_store(self, config):
        stored = yaml_filtered(config.expert.to_dict())
        stored["data_collection"]["pixels_per_meter"] = 8.0
        loaded = load_lead_config(dataset_expert_config=stored)
        assert loaded.expert.data_collection.pixels_per_meter == 8.0
        assert loaded.expert.data_collection.lidar_width_pixel == int(96 * 8.0)

    def test_env_override_beats_dataset_store(self, monkeypatch):
        monkeypatch.setenv(
            "LEAD_CONFIG",
            "expert.data_collection.pixels_per_meter=2.0",
        )
        config = load_lead_config(
            dataset_expert_config={"data_collection": {"pixels_per_meter": 8.0}},
        )
        assert config.expert.data_collection.pixels_per_meter == 2.0

    def test_stored_config_tolerates_renamed_knobs(self, config):
        apply_stored_expert_config(
            config,
            {"data_collection": {"pixels_per_meter": 8.0, "old_gone_key": True}},
        )
        assert config.expert.data_collection.pixels_per_meter == 8.0
