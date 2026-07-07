"""Differentiable collision cost for the reactive control head (P2.2).

Penalizes predicted waypoints that pass through occupied BEV cells (vehicles,
pedestrians, obstacles, ...). The occupancy is blurred into a soft "danger
field" so ``grid_sample`` yields a smooth gradient that pushes waypoints away
from obstacles (a binary map would only have gradient exactly at the edge).

This is an OPEN-LOOP soft prior (obstacles come from the recorded BEV semantic),
NOT true reactive safety -- see P2 doc §5. Self-contained; run
``python -m lead.tfv6.collision_cost`` for a gradient smoke test (no data).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from beartype import beartype

from lead.common.constants import TransfuserBEVSemanticClass as C
from lead.training.config_training import TrainingConfig

# BEV-semantic class ids that count as physical obstacles for path collision.
OBSTACLE_CLASSES: tuple[int, ...] = (
    int(C.VEHICLE),
    int(C.WALKER),
    int(C.OBSTACLE),
    int(C.PARKING_VEHICLE),
    int(C.SPECIAL_VEHICLE),
    int(C.BIKER),
)


@beartype
def occupancy_from_bev_semantic(bev_semantic: torch.Tensor) -> torch.Tensor:
    """``(B, H, W)`` class ids -> ``(B, 1, H, W)`` binary obstacle occupancy."""
    if bev_semantic.dim() == 4:  # (B,1,H,W) -> (B,H,W)
        bev_semantic = bev_semantic[:, 0]
    occ = torch.zeros_like(bev_semantic, dtype=torch.float32)
    for cls in OBSTACLE_CLASSES:
        occ = occ + (bev_semantic == cls).float()
    return occ.clamp(0.0, 1.0).unsqueeze(1)  # (B, 1, H, W)


def _gaussian_kernel(sigma_px: float, device, dtype) -> torch.Tensor:
    radius = max(int(3 * sigma_px), 1)
    xs = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    k1 = torch.exp(-(xs**2) / (2 * sigma_px**2))
    k1 = k1 / k1.sum()
    k2 = torch.outer(k1, k1)  # (K, K)
    return k2.to(dtype).view(1, 1, *k2.shape)


@beartype
def soft_danger_field(
    occ: torch.Tensor,
    config: TrainingConfig,
    sigma_m: float = 2.0,
) -> torch.Tensor:
    """Blur binary occupancy into a smooth danger field in ``[0, 1]``.

    High near obstacles, decaying outward -> gives waypoints a gradient ramp.
    """
    sigma_px = max(sigma_m * config.pixels_per_meter, 1.0)
    kernel = _gaussian_kernel(sigma_px, occ.device, occ.dtype)
    pad = kernel.shape[-1] // 2
    danger = F.conv2d(occ, kernel, padding=pad)
    # normalize so a lone obstacle cell peaks near 1
    peak = danger.amax(dim=(2, 3), keepdim=True).clamp(min=1e-6)
    return (danger / peak).clamp(0.0, 1.0)


@beartype
def waypoints_to_grid(waypoints: torch.Tensor, config: TrainingConfig) -> torch.Tensor:
    """``(B, N, 2)`` ego metres (x=long, y=lat) -> ``(B, N, 1, 2)`` grid_sample coords in [-1,1].

    Uses the same BEV convention as the intent rasterizer: x->width, y->height.
    """
    ppm = config.pixels_per_meter
    w = config.lidar_width_pixel
    h = config.lidar_height_pixel
    col = (waypoints[..., 0] - config.min_x_meter) * ppm  # [0, W] along width
    row = (waypoints[..., 1] - config.min_y_meter) * ppm  # [0, H] along height
    gx = col / (w - 1) * 2.0 - 1.0
    gy = row / (h - 1) * 2.0 - 1.0
    grid = torch.stack([gx, gy], dim=-1)  # (B, N, 2)
    return grid.unsqueeze(2)  # (B, N, 1, 2)


@beartype
def differentiable_collision(
    waypoints: torch.Tensor,  # (B, N, 2) ego metres, requires_grad for training
    bev_semantic: torch.Tensor,  # (B, H, W) or (B,1,H,W) class ids
    config: TrainingConfig,
    sigma_m: float = 2.0,
) -> torch.Tensor:
    """Mean danger-field value sampled at the predicted waypoints (scalar loss).

    Differentiable w.r.t. ``waypoints`` -> gradient pushes the path off obstacles.
    """
    occ = occupancy_from_bev_semantic(bev_semantic)
    danger = soft_danger_field(occ, config, sigma_m=sigma_m).to(waypoints.dtype)
    grid = waypoints_to_grid(waypoints, config).to(waypoints.dtype)
    sampled = F.grid_sample(
        danger, grid, mode="bilinear", padding_mode="zeros", align_corners=True,
    )  # (B, 1, N, 1)
    return sampled.mean()


def _smoke_test() -> None:
    torch.manual_seed(0)
    config = TrainingConfig()
    h, w = config.lidar_height_pixel, config.lidar_width_pixel
    b = 2
    # fake BEV semantic with a VEHICLE block ahead of ego
    sem = torch.zeros(b, h, w, dtype=torch.long)
    sem[:, 150:180, 150:190] = int(C.VEHICLE)
    # waypoints that walk forward through the block (ego metres)
    wp = torch.zeros(b, 8, 2)
    wp[:, :, 0] = torch.linspace(2.0, 12.0, 8)  # x forward
    wp.requires_grad_(True)

    loss = differentiable_collision(wp, sem, config)
    loss.backward()
    print("collision loss:", float(loss))
    print("waypoints grad norm:", float(wp.grad.norm()))
    print("occ obstacle cells:", int(occupancy_from_bev_semantic(sem).sum()))
    assert wp.grad is not None and wp.grad.norm() > 0, "no gradient to waypoints!"
    print("COLLISION COST OK (gradient flows to waypoints)")


if __name__ == "__main__":
    _smoke_test()
