"""Tests of the typed TransFuser training sample and its collation."""

import numpy as np
import pytest
import torch

from lead.policy.transfuser.dataloader.sample import TransfuserTrainingSample


def _sample(seed: int) -> TransfuserTrainingSample:
    rng = np.random.default_rng(seed)
    return TransfuserTrainingSample.from_outputs(
        {
            "rgb": rng.integers(0, 255, size=(3, 4, 6), dtype=np.uint8),
            "rasterized_lidar": rng.random((1, 8, 8)).astype(np.float32),
            "speed": 3.5,
            "town": "Town01",
            "steer": 0.1,
            "loading_time": 0.0,
        },
    )


def _runtime_checked() -> bool:
    """Whether the jaxtyping import hook instrumented the sample dataclass."""
    return hasattr(TransfuserTrainingSample.__init__, "__wrapped__")


def test_from_outputs_splits_principal_fields_and_extras() -> None:
    sample = _sample(seed=0)
    assert sample.rgb is not None
    assert sample.rasterized_lidar is not None
    assert sample.speed == 3.5
    assert sample.semantic is None
    assert set(sample.extras) == {"steer", "loading_time"}


def test_to_batch_dict_drops_unset_fields_and_keeps_extras() -> None:
    sample = _sample(seed=0)
    flat = sample.to_batch_dict()
    assert set(flat) == {
        "rgb",
        "rasterized_lidar",
        "speed",
        "town",
        "steer",
        "loading_time",
    }


def test_collate_stacks_into_the_batch_layout() -> None:
    batch = TransfuserTrainingSample.collate([_sample(seed=0), _sample(seed=1)])
    assert isinstance(batch, dict)
    assert batch["rgb"].shape == (2, 3, 4, 6)
    assert batch["rasterized_lidar"].shape == (2, 1, 8, 8)
    assert torch.is_tensor(batch["speed"])
    assert batch["speed"].shape == (2,)
    assert batch["town"] == ["Town01", "Town01"]


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("rasterized_lidar", np.zeros((1, 8, 8), dtype=np.float64)),
        ("rasterized_lidar", np.zeros((8, 8), dtype=np.float32)),
        ("rgb", np.zeros((3, 4, 6), dtype=np.int32)),
    ],
    ids=["wrong dtype", "missing channel axis", "wrong image dtype"],
)
def test_wrong_dtype_or_shape_is_rejected(field_name: str, value: np.ndarray) -> None:
    """The import hook instruments dataclasses, so a bad array fails at construction."""
    if not _runtime_checked():
        pytest.skip("LEAD_RUNTIME_TYPE_CHECKING is off; fields are documentation only")
    with pytest.raises(Exception, match="[Tt]ype"):
        TransfuserTrainingSample(**{field_name: value})


def test_avg_factor_is_zero_dimensional_so_it_collates_to_one_per_sample() -> None:
    """A 0-d array keeps the batch axis flat; a (1,) array would add a second one."""
    batch = TransfuserTrainingSample.collate(
        [
            TransfuserTrainingSample(
                center_net_avg_factor=np.asarray(3, dtype=np.int64),
            ),
            TransfuserTrainingSample(
                center_net_avg_factor=np.asarray(5, dtype=np.int64),
            ),
        ],
    )
    assert batch["center_net_avg_factor"].shape == (2,)
