"""The typed TransFuser sample and the batch it collates into.

Fields the config gates off stay None; the meta lifts ride in ``extras``.
"""

import typing
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TypedDict

import numpy as np
import torch
from jaxtyping import Float32, Float64, Int32, Int64, UInt, UInt8

from lead.api.array_types import (
    BevSemanticGrid,
    BoxRowsPixelFrame,
    Point2D,
    StitchedRgb,
)
from lead.api.training_sample import PolicyTrainingSample


@dataclass
class TransfuserTrainingSample(PolicyTrainingSample):
    """One unbatched TransFuser sample: numpy arrays, no batch axis."""

    # --- Sensor inputs ---
    rgb: StitchedRgb | None = None
    rasterized_lidar: Float32[np.ndarray, "1 bev_h bev_w"] | None = None
    # Per-sensor returns concatenated: [x, y, z, radial_velocity, sensor_id].
    radar: Float32[np.ndarray, "n_radar_points 5"] | None = None

    # --- Perspective targets, at the stitched camera resolution ---
    semantic: UInt8[np.ndarray, "sem_h sem_w"] | None = None
    # Quantized depth codes as stored; ops.dequantize_depth decodes on device.
    depth: UInt[np.ndarray, "sem_h sem_w"] | None = None

    # --- BEV target ---
    bev_semantic: BevSemanticGrid | None = None

    # --- CenterNet targets, on the BEV grid downsampled by ``bev_downsample_factor`` ---
    center_net_heatmap: Float32[np.ndarray, "n_box_classes cell_h cell_w"] | None = None
    center_net_wh: Float32[np.ndarray, "2 cell_h cell_w"] | None = None
    center_net_offset: Float32[np.ndarray, "2 cell_h cell_w"] | None = None
    center_net_yaw_class: Int32[np.ndarray, "cell_h cell_w"] | None = None
    center_net_yaw_res: Float32[np.ndarray, "1 cell_h cell_w"] | None = None
    center_net_velocity: Float32[np.ndarray, "1 cell_h cell_w"] | None = None
    center_net_brake: Int32[np.ndarray, "cell_h cell_w"] | None = None
    center_net_pixel_weight: Float32[np.ndarray, "2 cell_h cell_w"] | None = None
    center_net_bounding_boxes: BoxRowsPixelFrame | None = None
    # Positive-pixel count of the sample; 0-d, so it collates to (batch,).
    center_net_avg_factor: Int64[np.ndarray, ""] | None = None
    # Radar detection labels: [x, y, velocity, valid] per query.
    radar_detections: Float32[np.ndarray, "n_queries 4"] | None = None

    # --- Planning targets, in the sample's view frame ---
    future_waypoints: Float64[np.ndarray, "n_waypoints 2"] | None = None
    future_yaws: Float64[np.ndarray, " n_waypoints"] | None = None
    route: Float64[np.ndarray, "n_checkpoints 2"] | None = None

    # --- Navigation and ego state ---
    previous_target_point: Point2D | None = None
    target_point: Point2D | None = None
    next_target_point: Point2D | None = None
    speed: float | None = None

    # --- Identity ---
    town: str | None = None
    scenario_type: str | None = None

    extras: dict[str, typing.Any] = field(default_factory=dict)

    @classmethod
    def collate(
        cls,
        samples: Sequence[PolicyTrainingSample],
    ) -> dict[str, typing.Any]:
        """Collate into float32, the precision the model runs its inputs at.

        Autocast leaves float64 alone by design, so a float64 input would reach
        a reduced-precision layer unconverted; a ``float`` field collates to
        float64 unless narrowed here.

        Args:
            samples: The samples of one batch.

        Returns:
            The batch, with every float64 tensor narrowed to float32.
        """
        batch = super().collate(samples)
        return {
            key: value.to(torch.float32)
            if isinstance(value, torch.Tensor) and value.dtype == torch.float64
            else value
            for key, value in batch.items()
        }


class TransfuserForwardBatch(TypedDict, total=False):
    """What :meth:`TransfuserTrainingSample.collate` produces and the model indexes.

    Collation gives every array a leading batch axis, turns a ``str`` field into
    a ``list[str]`` and every float field into a float32 tensor. Runtime shape
    checking does not reach a ``TypedDict``; these annotations serve the reader
    and the static checker.
    """

    rgb: UInt8[torch.Tensor, "b 3 img_h img_w"]
    rasterized_lidar: Float32[torch.Tensor, "b 1 bev_h bev_w"]
    radar: Float32[torch.Tensor, "b n_radar_points 5"]

    semantic: UInt8[torch.Tensor, "b sem_h sem_w"]
    depth: UInt[torch.Tensor, "b sem_h sem_w"]
    bev_semantic: UInt8[torch.Tensor, "b bev_h bev_w"]

    center_net_heatmap: Float32[torch.Tensor, "b n_box_classes cell_h cell_w"]
    center_net_wh: Float32[torch.Tensor, "b 2 cell_h cell_w"]
    center_net_offset: Float32[torch.Tensor, "b 2 cell_h cell_w"]
    center_net_yaw_class: Int32[torch.Tensor, "b cell_h cell_w"]
    center_net_yaw_res: Float32[torch.Tensor, "b 1 cell_h cell_w"]
    center_net_velocity: Float32[torch.Tensor, "b 1 cell_h cell_w"]
    center_net_brake: Int32[torch.Tensor, "b cell_h cell_w"]
    center_net_pixel_weight: Float32[torch.Tensor, "b 2 cell_h cell_w"]
    center_net_bounding_boxes: Float32[torch.Tensor, "b n_boxes 9"]
    center_net_avg_factor: Int64[torch.Tensor, " b"]
    radar_detections: Float32[torch.Tensor, "b n_queries 4"]

    future_waypoints: Float32[torch.Tensor, "b n_waypoints 2"]
    future_yaws: Float32[torch.Tensor, " b n_waypoints"]
    route: Float32[torch.Tensor, "b n_checkpoints 2"]

    previous_target_point: Float32[torch.Tensor, "b 2"]
    target_point: Float32[torch.Tensor, "b 2"]
    next_target_point: Float32[torch.Tensor, "b 2"]
    speed: Float32[torch.Tensor, " b"]

    town: list[str]
    scenario_type: list[str]

    # Expert driving-meta lifted into the batch as planning supervision.
    brake: Float32[torch.Tensor, " b"]
    target_speed: Float32[torch.Tensor, " b"]

    # Diagnostics the trainer injects, not the sample builders.
    # Cumulative optimizer steps; absent outside training.
    current_gradient_step: int
    loading_time: Float32[torch.Tensor, " b"]


class TransfuserOutputs(TypedDict, total=False):
    """What a sample builder emits: sample fields as numpy, before collation.

    The same keys as :class:`TransfuserTrainingSample`, so a builder emitting a name the
    sample has no field for is a static error rather than a silent ``extras``
    entry. The dtypes are checked at runtime when the sample is built.
    """

    rgb: StitchedRgb
    rasterized_lidar: Float32[np.ndarray, "1 bev_h bev_w"]
    radar: Float32[np.ndarray, "n_radar_points 5"]

    semantic: UInt8[np.ndarray, "sem_h sem_w"]
    depth: UInt[np.ndarray, "sem_h sem_w"]
    bev_semantic: BevSemanticGrid

    center_net_heatmap: Float32[np.ndarray, "n_box_classes cell_h cell_w"]
    center_net_wh: Float32[np.ndarray, "2 cell_h cell_w"]
    center_net_offset: Float32[np.ndarray, "2 cell_h cell_w"]
    center_net_yaw_class: Int32[np.ndarray, "cell_h cell_w"]
    center_net_yaw_res: Float32[np.ndarray, "1 cell_h cell_w"]
    center_net_velocity: Float32[np.ndarray, "1 cell_h cell_w"]
    center_net_brake: Int32[np.ndarray, "cell_h cell_w"]
    center_net_pixel_weight: Float32[np.ndarray, "2 cell_h cell_w"]
    center_net_bounding_boxes: BoxRowsPixelFrame
    center_net_avg_factor: Int64[np.ndarray, ""]
    radar_detections: Float32[np.ndarray, "n_queries 4"]

    future_waypoints: Float64[np.ndarray, "n_waypoints 2"]
    future_yaws: Float64[np.ndarray, " n_waypoints"]
    route: Float64[np.ndarray, "n_checkpoints 2"]

    previous_target_point: Point2D
    target_point: Point2D
    next_target_point: Point2D
    speed: float

    town: str
    scenario_type: str
