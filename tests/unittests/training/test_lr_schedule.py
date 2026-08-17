"""Tests of the warmup cosine schedule: cycle geometry and the no-warmup case."""

from collections.abc import Callable

import pytest
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LRScheduler

from lead.training.train import CosineAnnealingWarmRestartsWithWarmup

_BASE_LR = 3.674e-4
_T_0 = 100
_T_MULT = 2
# T_0 * (1 + 2 + 4 + 8 + 16): five whole cycles, the shape a 31-epoch run has.
_TOTAL_STEPS = _T_0 * 31
_CYCLE_STARTS = [0, _T_0, _T_0 * 3, _T_0 * 7, _T_0 * 15]


def _trace(
    scheduler_factory: Callable[[torch.optim.Optimizer], LRScheduler],
    num_steps: int = _TOTAL_STEPS,
) -> list[float]:
    """Record the learning rate a schedule sets at each step.

    Args:
        scheduler_factory: Builds the schedule from an optimizer.
        num_steps: Steps to run.

    Returns:
        The learning rate in force at every step.
    """
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([parameter], lr=_BASE_LR)
    scheduler = scheduler_factory(optimizer)
    rates = []
    for _ in range(num_steps):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    return rates


def _with_warmup(warmup_fraction: float, eta_min: float = 0.0):
    """Build the factory for a warmup schedule with the given fraction."""
    return lambda optimizer: CosineAnnealingWarmRestartsWithWarmup(
        optimizer,
        T_0=_T_0,
        T_mult=_T_MULT,
        warmup_fraction=warmup_fraction,
        eta_min=eta_min,
    )


def test_zero_warmup_matches_cosine_annealing_warm_restarts() -> None:
    """A zero fraction reproduces torch's schedule step for step."""
    reference = _trace(
        lambda optimizer: CosineAnnealingWarmRestarts(
            optimizer,
            T_0=_T_0,
            T_mult=_T_MULT,
        ),
    )
    assert _trace(_with_warmup(0.0)) == pytest.approx(reference, rel=1e-12, abs=1e-18)


def test_restarts_reach_the_base_rate_without_warmup() -> None:
    """Every cycle of the unwarmed schedule opens at the full rate."""
    rates = _trace(_with_warmup(0.0))
    for start in _CYCLE_STARTS:
        assert rates[start] == pytest.approx(_BASE_LR)


def test_warmup_ramps_each_cycle_from_near_zero_to_the_base_rate() -> None:
    """Each cycle opens far below the base rate and peaks at the ramp's end."""
    warmup_fraction = 0.1
    rates = _trace(_with_warmup(warmup_fraction))
    lengths = [_T_0 * 2**index for index in range(len(_CYCLE_STARTS))]
    for start, length in zip(_CYCLE_STARTS, lengths, strict=True):
        warmup_steps = int(length * warmup_fraction)
        assert rates[start] < _BASE_LR / 10
        assert rates[start + warmup_steps] == pytest.approx(_BASE_LR)
        # Strictly rising across the ramp, so no cycle re-enters at full rate.
        ramp = rates[start : start + warmup_steps + 1]
        assert all(a < b for a, b in zip(ramp, ramp[1:], strict=False))


def test_warmup_leaves_the_mean_rate_unchanged() -> None:
    """A linear ramp and the cosine it displaces average the same, so decay holds."""
    unwarmed = _trace(_with_warmup(0.0))
    warmed = _trace(_with_warmup(0.1))
    assert sum(warmed) == pytest.approx(sum(unwarmed), rel=1e-3)


def test_peak_never_exceeds_the_base_rate() -> None:
    """Warmup only ever holds the rate below its peak."""
    assert max(_trace(_with_warmup(0.1))) == pytest.approx(_BASE_LR)


def test_eta_min_is_the_floor_every_cycle_anneals_to() -> None:
    """A non-zero floor ends each cycle at that rate rather than at zero."""
    eta_min = _BASE_LR / 10
    rates = _trace(_with_warmup(0.1, eta_min=eta_min))
    # A cycle's last step still sits one step short of its minimum, so the floor
    # is approached from above rather than landed on.
    assert min(rates) >= eta_min
    assert min(rates) == pytest.approx(eta_min, rel=1e-2)
    for start in _CYCLE_STARTS[1:]:
        assert rates[start - 1] == pytest.approx(eta_min, rel=1e-2)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"T_0": 0}, "at least 1"),
        ({"T_mult": 0}, "at least 1"),
        ({"warmup_fraction": 1.0}, r"\[0, 1\)"),
        ({"warmup_fraction": -0.1}, r"\[0, 1\)"),
    ],
)
def test_invalid_arguments_are_rejected(kwargs: dict, message: str) -> None:
    """Out-of-range cycle lengths and fractions raise rather than train oddly."""
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([parameter], lr=_BASE_LR)
    arguments = {"T_0": _T_0, "T_mult": _T_MULT, "warmup_fraction": 0.1, **kwargs}
    with pytest.raises(ValueError, match=message):
        CosineAnnealingWarmRestartsWithWarmup(optimizer, **arguments)
