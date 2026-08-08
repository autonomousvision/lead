"""Tests of the ego-status baseline policy."""

import numpy as np
import pytest
import torch

from lead.api.abstract_policy import build_policy
from lead.config.lead_config import load_lead_config
from lead.policy.ego_status.dataloader.sample import EgoStatusTrainingSample

_TARGET = "lead.policy.ego_status.ego_status:EgoStatus"


@pytest.fixture
def config():
    config = load_lead_config()
    config.apply_overrides({"policy": {"target": _TARGET}})
    return config


@pytest.fixture
def policy(config):
    return build_policy(config)


def _batch(config, batch_size: int = 4) -> dict[str, torch.Tensor]:
    ego_status = config.policy.ego_status
    return {
        "target_point": torch.randn(batch_size, 2),
        "speed": torch.rand(batch_size),
        "future_waypoints": torch.randn(
            batch_size,
            ego_status.num_ego_pose_prediction,
            2,
        ),
    }


def test_policy_publishes_its_config_section(config, policy) -> None:
    assert policy.get_policy_config() is config.policy.ego_status
    assert (
        policy.get_policy_config().cache_store_dir_name == "ego_status_training_cache"
    )


def test_forward_shapes(config, policy) -> None:
    ego_status = config.policy.ego_status
    predictions = policy(_batch(config))
    assert predictions["waypoints"].shape == (
        4,
        ego_status.num_ego_pose_prediction,
        2,
    )


def test_loss_keys_match_weights_and_are_finite(config, policy) -> None:
    batch = _batch(config)
    losses, log = policy.compute_loss(policy(batch), batch)
    assert set(losses) == set(policy.per_task_loss_weights(epoch=0))
    assert log == {}
    for loss in losses.values():
        assert torch.isfinite(loss)


def test_batch_features_shapes(policy) -> None:
    features = {"target_point": np.array([1.0, 2.0]), "speed": 3.0}
    batched = policy.features_to_batch(features, torch.device("cpu"))
    assert batched["target_point"].shape == (1, 2)
    assert batched["speed"].shape == (1,)


def test_sample_round_trip_and_collate(config) -> None:
    ego_status = config.policy.ego_status
    outputs = {
        "target_point": np.array([1.0, 2.0], dtype=np.float32),
        "speed": 3.0,
        "future_waypoints": np.zeros(
            (ego_status.num_ego_pose_prediction, 2),
            dtype=np.float32,
        ),
        "log_name": "Town01_log",
    }
    sample = EgoStatusTrainingSample.from_outputs(outputs)
    assert sample.speed == 3.0
    assert set(sample.extras) == {"log_name"}
    batch = EgoStatusTrainingSample.collate([sample, sample])
    assert batch["future_waypoints"].shape == (
        2,
        ego_status.num_ego_pose_prediction,
        2,
    )
    assert batch["log_name"] == ["Town01_log", "Town01_log"]
