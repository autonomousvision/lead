"""The typed ego-status sample and the batch it collates into."""

import typing
from dataclasses import dataclass, field
from typing import TypedDict

import numpy as np
import torch
from jaxtyping import Float, Float32

from lead.api.training_sample import PolicyTrainingSample


@dataclass
class EgoStatusTrainingSample(PolicyTrainingSample):
    """One unbatched ego-status sample: numpy arrays, no batch axis."""

    target_point: Float32[np.ndarray, " 2"] | None = None
    speed: float | None = None
    future_waypoints: Float32[np.ndarray, "n_waypoints 2"] | None = None
    # Everything the builders emit beyond the typed fields.
    extras: dict[str, typing.Any] = field(default_factory=dict)


class EgoStatusForwardBatch(TypedDict, total=False):
    """What :meth:`EgoStatusTrainingSample.collate` produces and the model indexes."""

    target_point: Float32[torch.Tensor, "b 2"]
    speed: Float[torch.Tensor, " b"]
    future_waypoints: Float32[torch.Tensor, "b n_waypoints 2"]


class EgoStatusPrediction(TypedDict):
    """The MLP's plan for one batch."""

    waypoints: Float[torch.Tensor, "b n_waypoints 2"]
