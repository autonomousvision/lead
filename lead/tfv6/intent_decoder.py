"""Visual-intent decoder (scaffold).

Route-1 of the "implicit visual intent + reactive control" plan
(see ``方案_visual_intent_reactive_control.md``).

The intent head mirrors :class:`lead.tfv6.bev_decoder.BEVDecoder` but emits a
single-channel *soft* BEV heatmap that represents where the ego *intends* to go.
It is supervised by rasterizing the expert's future path onto the BEV grid as a
Gaussian corridor, so the representation stays distributional (fuzzy) instead of
collapsing to a single committed trajectory.

This module is self-contained and does not touch the existing model wiring.
Run ``python -m lead.tfv6.intent_decoder`` for a shape smoke test (no data needed).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from beartype import beartype
from torch.nn import functional as F

from lead.common.constants import ChaffeurNetBEVSemanticClass
from lead.training.config_training import TrainingConfig


@beartype
def rasterize_reachable_support(
    hdmap: npt.NDArray,  # raw ChaffeurNet hdmap raster (H0, W0), ego-centred, ROAD==1
    config: TrainingConfig,
    horizon_m: float = 30.0,
    sigma_px: float = 2.0,
) -> npt.NDArray:
    """Rasterize the forward-reachable drivable region into a soft BEV support field.

    P5 multimodal intent target: instead of the single expert route, light up every
    drivable arm the ego can reach going forward within ``horizon_m`` (at a junction
    this fans into all feasible turns). Derived purely from the hdmap ROAD class, so
    it needs no simulator; it does not distinguish lane direction (oncoming lanes are
    included -- a known v0 limitation to be refined with a lane graph).

    Args:
        hdmap: Raw ego-centred hdmap raster (single channel), ROAD encoded as
            ``ChaffeurNetBEVSemanticClass.ROAD``.
        config: Training config (BEV grid geometry).
        horizon_m: Forward horizon for the reachable flood fill, in metres.
        sigma_px: Gaussian smoothing to soften the support into a corridor.

    Returns:
        A ``(1, H, W)`` float32 field in ``[0, 1]`` aligned with the intent grid.
    """
    from scipy import ndimage

    if hdmap.ndim == 3:
        hdmap = hdmap[..., 0]
    h_px = config.lidar_height_pixel
    w_px = config.lidar_width_pixel
    ppm = config.pixels_per_meter
    ppm_c = config.pixels_per_meter_collection
    row_ego = int((0 - config.min_y_meter) * ppm)
    col_ego = int((0 - config.min_x_meter) * ppm)

    # Resample the raw hdmap ROAD mask into the ego BEV intent grid (metric-aligned).
    rows, cols = np.meshgrid(np.arange(h_px), np.arange(w_px), indexing="ij")
    x_m = cols / ppm + config.min_x_meter  # longitudinal (forward)
    y_m = rows / ppm + config.min_y_meter  # lateral
    hx = np.round(hdmap.shape[1] / 2 + x_m * ppm_c).astype(np.int64)
    hy = np.round(hdmap.shape[0] / 2 + y_m * ppm_c).astype(np.int64)
    inside = (hx >= 0) & (hx < hdmap.shape[1]) & (hy >= 0) & (hy < hdmap.shape[0])
    road = np.zeros((h_px, w_px), dtype=bool)
    road[inside] = hdmap[hy[inside], hx[inside]] == ChaffeurNetBEVSemanticClass.ROAD

    # Forward-reachable drivable region = the road component containing the ego,
    # intersected with the forward horizon window.
    labels, _ = ndimage.label(road, structure=np.ones((3, 3)))
    ego_label = labels[row_ego, col_ego]
    if ego_label == 0:  # ego pixel just off the rasterized road; snap to nearest label
        window = labels[row_ego - 8 : row_ego + 9, col_ego - 8 : col_ego + 9]
        nonzero = window[window > 0]
        ego_label = int(nonzero[0]) if nonzero.size else 0
    reachable = (labels == ego_label) & (ego_label > 0)
    forward = np.zeros_like(reachable)
    forward[:, col_ego : int(col_ego + horizon_m * ppm)] = True

    support = (reachable & forward).astype(np.float32)
    support = ndimage.gaussian_filter(support, sigma_px)
    peak = support.max()
    if peak > 0:
        support /= peak
    return support[None]  # (1, H, W)


@beartype
def rasterize_waypoints_to_bev(
    waypoints: torch.Tensor,  # (B, N, 2) ego-frame metres: [:,:,0]=x (long), [:,:,1]=y (lat)
    config: TrainingConfig,
    sigma_m: float = 1.5,
) -> torch.Tensor:
    """Rasterize an ego-frame path into a soft BEV intent heatmap.

    Returns a ``(B, 1, H, W)`` tensor in ``[0, 1]`` (Gaussian corridor along the
    path, aggregated by max over path points).

    NOTE: BEV orientation is derived from config (x->width, y->height). The exact
    axis flip should be confirmed with the P0 visualization; flip ``row``/``col``
    here if the overlay looks mirrored.
    """
    device = waypoints.device
    b, n, _ = waypoints.shape
    h = config.lidar_height_pixel
    w = config.lidar_width_pixel
    ppm = config.pixels_per_meter
    sigma_px = max(sigma_m * ppm, 1.0)

    # ego metres -> BEV pixel: x (longitudinal) spans width, y (lateral) spans height
    col = (waypoints[..., 0] - config.min_x_meter) * ppm  # (B, N)  along W
    row = (waypoints[..., 1] - config.min_y_meter) * ppm  # (B, N)  along H

    ys = torch.arange(h, device=device, dtype=torch.float32)
    xs = torch.arange(w, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # (H, W)
    grid_y = grid_y.view(1, 1, h, w)
    grid_x = grid_x.view(1, 1, h, w)

    r = row.view(b, n, 1, 1)
    c = col.view(b, n, 1, 1)
    d2 = (grid_y - r) ** 2 + (grid_x - c) ** 2  # (B, N, H, W)
    heat = torch.exp(-d2 / (2.0 * sigma_px * sigma_px))  # (B, N, H, W)
    heat = heat.amax(dim=1, keepdim=True)  # (B, 1, H, W) corridor = max over points
    return heat.clamp(0.0, 1.0)


class IntentDecoder(nn.Module):
    """Single-channel soft BEV intent head (clone of BEVDecoder, 1-ch output)."""

    @beartype
    def __init__(self, config: TrainingConfig, device: torch.device):
        super().__init__()
        self.config = config
        self.device = device
        self.net = nn.Sequential(
            nn.Conv2d(
                config.bev_features_chanels,
                config.bev_features_chanels,
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1),
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                config.bev_features_chanels,
                1,  # single-channel intent occupancy
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.Upsample(
                size=(config.lidar_height_pixel, config.lidar_width_pixel),
                mode="bilinear",
                align_corners=False,
            ),
        )

    @beartype
    def forward(self, bev_feature_grid: torch.Tensor) -> torch.Tensor:
        """``(B, D, h, w)`` BEV feature grid -> ``(B, 1, H, W)`` intent logits."""
        return self.net(bev_feature_grid)

    @beartype
    def _intent_label(self, data: dict) -> torch.Tensor:
        """Build the soft intent target from the expert path in ``data``.

        Prefers the label rasterized on the dataloader (``visual_intent_label``);
        falls back to rasterizing the route/waypoints here if it is absent.
        """
        if data.get("visual_intent_label") is not None:
            return data["visual_intent_label"].to(
                self.device,
                dtype=torch.float32,
                non_blocking=True,
            )  # (B, 1, H, W)
        path = data["route"] if data.get("route") is not None else data["future_waypoints"]
        path = path.to(self.device, dtype=torch.float32, non_blocking=True)
        with torch.no_grad():
            return rasterize_waypoints_to_bev(path, self.config)

    @beartype
    def compute_loss(
        self,
        pred: torch.Tensor,  # (B, 1, H, W) logits
        data: dict,
        loss: dict,
        log: dict,
    ) -> None:
        target = self._intent_label(data)  # (B, 1, H, W) in [0, 1]
        with torch.amp.autocast(device_type="cuda", enabled=False):
            # pos_weight to counter sparsity of the intent corridor
            pos = target.gt(0.1).float().mean().clamp(min=1e-4)
            pos_weight = ((1.0 - pos) / pos).clamp(max=50.0)
            loss["loss_visual_intent"] = F.binary_cross_entropy_with_logits(
                pred.float(),
                target.float(),
                pos_weight=pos_weight,
            )
        if (
            "iteration" in data
            and ((data["iteration"] + 1) % self.config.log_scalars_frequency) == 0
        ):
            prob = torch.sigmoid(pred.float())
            log["visual_intent/pred_max"] = prob.max().item()
            log["visual_intent/pred_mean"] = prob.mean().item()
            log["visual_intent/target_coverage"] = pos.item()


def _smoke_test() -> None:
    """Random-tensor shape check; no dataset required."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = TrainingConfig()
    b = 2
    grid = torch.randn(
        b,
        config.bev_features_chanels,
        config.lidar_height_pixel // config.bev_down_sample_factor,
        config.lidar_width_pixel // config.bev_down_sample_factor,
        device=device,
    )
    dec = IntentDecoder(config, device).to(device)
    pred = dec(grid)
    print("intent logits:", tuple(pred.shape))  # expect (B, 1, H, W)

    data = {
        "route": torch.randn(b, config.num_route_points_prediction, 2, device=device),
        "iteration": 0,
    }
    loss, log = {}, {}
    dec.compute_loss(pred, data, loss, log)
    print("loss_visual_intent:", float(loss["loss_visual_intent"]))
    print("log:", {k: round(v, 4) for k, v in log.items()})
    label = dec._intent_label(data)
    print("intent label:", tuple(label.shape), "range", (float(label.min()), float(label.max())))


if __name__ == "__main__":
    _smoke_test()
