"""Tensor ops shared by the TransFuser encoder and decoders."""

import math
import typing

import jaxtyping as jt
import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F

from lead.config import LeadConfig, TransfuserConfig


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
    assert 0 <= pos_tensor.min() and pos_tensor.max() <= 1, (
        "pos_tensor values should be in range [0, 1]"
    )
    half_hidden_dim = hidden_dim // 2
    scale = 2 * math.pi
    dim_t = torch.arange(half_hidden_dim, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / half_hidden_dim)
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
    pos = torch.cat((pos_y, pos_x), dim=-1)
    return pos


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
    data_config = lead_config.expert.data_collection
    min_x, max_x, min_y, max_y = (
        data_config.min_x_meter,
        data_config.max_x_meter,
        data_config.min_y_meter,
        data_config.max_y_meter,
    )
    if isinstance(points, torch.Tensor):
        points = points.clone()
    else:
        points = points.copy()
    points[..., 0] = (points[..., 0] - min_x) / (max_x - min_x)
    points[..., 1] = (points[..., 1] - min_y) / (max_y - min_y)
    return points


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
    data_config = lead_config.expert.data_collection
    B, D, H, W = bev.shape
    N = ref_points.shape[1]

    x = ref_points[..., 0]
    y = ref_points[..., 1]

    # Normalize to [-1, 1]
    u = (
        2
        * (y - data_config.min_y_meter)
        / (data_config.max_y_meter - data_config.min_y_meter)
        - 1
    )
    v = (
        2
        * (x - data_config.min_x_meter)
        / (data_config.max_x_meter - data_config.min_x_meter)
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

    Inverse function to angle2class for decoding predicted angle values.

    Args:
        angle_cls: Discrete angle class tensor to decode.
        angle_res: Angle residual tensor to decode.
        config: Transfuser configuration containing num_dir_bins.
        limit_period: Whether to limit angle to [-π, π] range.

    Returns:
        Decoded continuous angle tensor.
    """
    angle_per_class = 2 * np.pi / float(config.num_dir_bins)
    angle_center = angle_cls.float() * angle_per_class
    angle = angle_center + angle_res
    if limit_period:
        angle[angle > np.pi] -= 2 * np.pi
    return angle
