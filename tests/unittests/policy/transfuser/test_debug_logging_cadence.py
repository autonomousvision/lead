"""The decoders' debug logging fires on the trainer's gradient-step cadence."""

import pytest
import torch

from lead.config import LeadConfig, load_lead_config
from lead.policy.transfuser.decoder.bev_decoder import BEVDecoder


@pytest.fixture
def config() -> LeadConfig:
    """Fixture providing the config tree."""
    return load_lead_config()


@pytest.fixture
def decoder(config: LeadConfig) -> BEVDecoder:
    """A BEV decoder, built wherever the test runs."""
    return BEVDecoder(
        config,
        config.policy.transfuser.num_bev_semantic_classes,
    )


def _compute_loss(
    decoder: BEVDecoder,
    config: LeadConfig,
    gradient_step: int | None,
) -> dict:
    """Run one loss computation and return the log dict it filled.

    Args:
        decoder: The decoder under test.
        config: The config tree.
        gradient_step: Cumulative optimizer step, or None to mimic inference.

    Returns:
        The auxiliary log dict.
    """
    num_classes = config.policy.transfuser.num_bev_semantic_classes
    data = {"bev_semantic": torch.zeros(2, 16, 16, dtype=torch.long)}
    if gradient_step is not None:
        data["current_gradient_step"] = gradient_step
    log: dict = {}
    decoder.compute_loss(torch.randn(2, num_classes, 16, 16), data, {}, log)
    return log


def test_logs_on_the_configured_cadence(decoder, config) -> None:
    every = config.training.experiment.log_scalars_every_n_steps
    assert _compute_loss(decoder, config, every - 1)
    assert _compute_loss(decoder, config, 2 * every - 1)


def test_stays_silent_between_cadence_steps(decoder, config) -> None:
    every = config.training.experiment.log_scalars_every_n_steps
    assert _compute_loss(decoder, config, 0) == {}
    assert _compute_loss(decoder, config, every) == {}


def test_stays_silent_without_a_gradient_step(decoder, config) -> None:
    """Driving has no trainer, so the key is absent and nothing is logged."""
    assert _compute_loss(decoder, config, None) == {}
