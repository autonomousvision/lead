"""Tensor ops shared by the TransFuser encoder and decoders."""

import math
import typing

import jaxtyping as jt
import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F

from lead.config import LeadConfig, TransfuserConfig

# Constrained so the normalization body type-checks once per array flavour,
# instead of against an unnarrowable union.
_BevArrayT = typing.TypeVar("_BevArrayT", np.ndarray, torch.Tensor)

# Constant frequency ladder of gen_sineembed_for_position, built once per
# (dimension, device) instead of on every forward.
_SINEEMBED_DIM_T: dict[tuple[int, torch.device], torch.Tensor] = {}


def _sineembed_dim_t(half_hidden_dim: int, device: torch.device) -> torch.Tensor:
    key = (half_hidden_dim, device)
    dim_t = _SINEEMBED_DIM_T.get(key)
    if dim_t is None:
        dim_t = torch.arange(half_hidden_dim, dtype=torch.float32, device=device)
        dim_t = 10000 ** (2 * (dim_t // 2) / half_hidden_dim)
        _SINEEMBED_DIM_T[key] = dim_t
    return dim_t


def dequantize_depth(
    raw: jt.UInt[torch.Tensor, "*shape"],
    max_depth: float,
) -> jt.Float32[torch.Tensor, "*shape"]:
    """Decode a linearly quantized, sentinel-free depth raster to metres.

    The inverse of the dataset's depth encoding; the batch ships the stored
    integer codes and this runs on the device.

    Args:
        raw: The integer depth codes, any unsigned dtype.
        max_depth: The far plane of the quantization, in metres.

    Returns:
        Metric depth in metres.
    """
    return raw.to(torch.float32) * (max_depth / torch.iinfo(raw.dtype).max)


def normalize_imagenet(
    x: jt.Float[torch.Tensor, "B 3 H W"],
) -> jt.Float[torch.Tensor, "B 3 H W"]:
    """Normalize input images according to ImageNet standards.
    Args:
        x: Input images batch.

    Returns:
        Normalized images batch.
    """
    x = x.clone()
    x[:, 0] = ((x[:, 0] / 255.0) - 0.485) / 0.229
    x[:, 1] = ((x[:, 1] / 255.0) - 0.456) / 0.224
    x[:, 2] = ((x[:, 2] / 255.0) - 0.406) / 0.225
    return x


def gen_sineembed_for_position(
    pos_tensor: jt.Float[torch.Tensor, "B 2"],
    hidden_dim: int = 64,
):
    """Mostly copy-paste from https://github.com/IDEA-opensource/DAB-DETR
    Args:
        pos_tensor: Last dimension is (x, y). Values are expected to be in range [0, 1].
        hidden_dim: Dimension of the output positional embedding. Must be even.
    Returns:
        Positional embedding with shape (B, hidden_dim)
    """
    if not torch.compiler.is_compiling():
        assert 0 <= pos_tensor.min() and pos_tensor.max() <= 1, (
            "pos_tensor values should be in range [0, 1]"
        )
    half_hidden_dim = hidden_dim // 2
    scale = 2 * math.pi
    dim_t = _sineembed_dim_t(half_hidden_dim, pos_tensor.device)
    x_embed = pos_tensor[..., 0] * scale
    y_embed = pos_tensor[..., 1] * scale
    pos_x = x_embed[..., None] / dim_t
    pos_y = y_embed[..., None] / dim_t
    pos_x = torch.stack(
        (pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()),
        dim=-1,
    ).flatten(-2)
    pos_y = torch.stack(
        (pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()),
        dim=-1,
    ).flatten(-2)
    return torch.cat((pos_y, pos_x), dim=-1)


def _unit_normalize_in_place(
    points: _BevArrayT,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
) -> _BevArrayT:
    """Scale the x and y columns of BEV points into [0, 1], in place."""
    # ndarray and Tensor share this indexing API but no common static type.
    columns: typing.Any = points
    columns[..., 0] = (columns[..., 0] - min_x) / (max_x - min_x)
    columns[..., 1] = (columns[..., 1] - min_y) / (max_y - min_y)
    return points


@typing.overload
def unit_normalize_bev_points(
    points: jt.Float[torch.Tensor, "... 2"],
    lead_config: LeadConfig,
) -> jt.Float[torch.Tensor, "... 2"]: ...
@typing.overload
def unit_normalize_bev_points(
    points: jt.Float[npt.NDArray, "... 2"],
    lead_config: LeadConfig,
) -> jt.Float[npt.NDArray, "... 2"]: ...
def unit_normalize_bev_points(
    points: jt.Float[npt.NDArray | torch.Tensor, "... 2"],
    lead_config: LeadConfig,
) -> jt.Float[npt.NDArray | torch.Tensor, "... 2"]:
    """Unit normalize BEV points to range [0, 1].

    Args:
        points: BEV points in meters.
        lead_config: Root config tree with the BEV area geometry.
    Returns:
        Normalized BEV points of shape in range [0, 1].
    """
    config = lead_config.policy.transfuser
    bounds = (
        float(config.bev_min_x_meter),
        float(config.bev_max_x_meter),
        float(config.bev_min_y_meter),
        float(config.bev_max_y_meter),
    )
    if isinstance(points, torch.Tensor):
        return _unit_normalize_in_place(points.clone(), *bounds)
    return _unit_normalize_in_place(points.copy(), *bounds)


def bev_grid_sample(
    bev: jt.Float[torch.Tensor, "B D H W"],
    ref_points: jt.Float[torch.Tensor, "B N 2"],  # absolute coords (x, y)
    lead_config: LeadConfig,
) -> jt.Float[torch.Tensor, "B N D"]:
    """
    Deterministic bilinear sampling of BEV features at given reference points.

    Args:
        bev: BEV feature map in ego space.
        ref_points: Absolute coordinates in ego space.
        lead_config: Root config tree with the BEV area geometry.

    Returns:
        sampled: interpolated BEV features at given points (B, N, D)
    """
    config = lead_config.policy.transfuser
    B, D, H, W = bev.shape
    N = ref_points.shape[1]

    x = ref_points[..., 0]
    y = ref_points[..., 1]

    # Normalize to [-1, 1]
    u = (
        2
        * (y - config.bev_min_y_meter)
        / (config.bev_max_y_meter - config.bev_min_y_meter)
        - 1
    )
    v = (
        2
        * (x - config.bev_min_x_meter)
        / (config.bev_max_x_meter - config.bev_min_x_meter)
        - 1
    )

    grid = torch.stack([u, v], dim=-1)  # (B, N, 2)
    grid = grid.view(B, N, 1, 2)  # (B, N, 1, 2)

    sampled = F.grid_sample(
        bev,
        grid,
        mode="bilinear",
        align_corners=True,
    )  # (B, D, N, 1)

    return sampled.squeeze(-1).permute(0, 2, 1)  # (B, N, D)


def class2angle(
    angle_cls: torch.Tensor,
    angle_res: torch.Tensor,
    config: TransfuserConfig,
    limit_period: bool = True,
) -> torch.Tensor:
    """Convert discrete angle class and residual back to continuous angle.

    Inverse function to angle_to_yaw_class for decoding predicted angle values.

    Args:
        angle_cls: Discrete angle class tensor to decode.
        angle_res: Angle residual tensor to decode.
        config: Transfuser configuration containing num_yaw_bins.
        limit_period: Whether to limit angle to [-π, π] range.

    Returns:
        Decoded continuous angle tensor.
    """
    angle_per_class = 2 * np.pi / float(config.num_yaw_bins)
    angle_center = angle_cls.float() * angle_per_class
    angle = angle_center + angle_res
    if limit_period:
        angle[angle > np.pi] -= 2 * np.pi
    return angle
