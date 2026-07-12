"""P4a: lightweight decoder from cached Qwen-VL hidden states to BEV visual intent.

Architecture:
    Qwen hidden (12,12,2560) → Conv upsample → image-space intent H_img
                              → differentiable depth projection → BEV intent (1,320,384)

Training: distill the same expert route Gaussian heatmap (visual_intent_label) P1 used.
Qwen is frozen & pre-cached; only this decoder is trained (fast, H20 single-GPU ok).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VLMIntentDecoder(nn.Module):
    """Decode Qwen-VL image-token hidden states into BEV visual intent via learned upsampling + projection.

    Input:  (B, h', w', D_vlm) cached Qwen hidden states, typically (B, 12, 12, 2560)
    Output: (B, 1, H_bev, W_bev) BEV intent heatmap logits, e.g. (B, 1, 320, 384)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        D_vlm = 2560  # Qwen3-VL-4B hidden dim

        # Upsample to image resolution (384x384 front camera)
        # 12x12 → 24x24 → 48x48 → 96x96 → 192x192 → 384x384  (5 steps, stride 2 each)
        self.upsample = nn.Sequential(
            nn.Conv2d(D_vlm, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # → 24

            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # → 48

            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # → 96

            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # → 192

            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # → 384

            nn.Conv2d(32, 1, 1),  # final 1x1 conv to 1-channel intent logits
        )

    def forward(self, vlm_hidden: torch.Tensor, depth: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            vlm_hidden: (B, h', w', D_vlm) cached Qwen hidden states
            depth: (B, 1, H_img, W_img) GT depth for projection, optional for now

        Returns:
            (B, 1, H_bev, W_bev) BEV intent logits
        """
        B, h, w, D = vlm_hidden.shape
        x = vlm_hidden.permute(0, 3, 1, 2)  # → (B, D, h, w)

        # Upsample to image-space intent
        H_img = self.upsample(x)  # (B, 1, 384, 384)

        # TODO P4a.1: add differentiable depth projection H_img → H_bev
        # For now: bilinear resize to BEV grid as placeholder (P1 did direct BEV decode, we match that first)
        H_bev = F.interpolate(
            H_img,
            size=(self.config.lidar_height_pixel, self.config.lidar_width_pixel),
            mode="bilinear",
            align_corners=False,
        )
        return H_bev

    def compute_loss(
        self,
        pred_intent: torch.Tensor,
        data: dict,
        loss: dict,
        log: dict,
    ) -> None:
        """Distill expert route Gaussian heatmap (visual_intent_label).

        Reuses IntentDecoder.compute_loss logic: BCE + pos_weight.
        """
        from lead.tfv6.intent_decoder import rasterize_waypoints_to_bev

        # Get or build visual_intent_label (P1 precomputed it in dataloader if available)
        if "visual_intent_label" in data:
            target = data["visual_intent_label"]  # (B, 1, H, W)
        else:
            # Fallback: rasterize expert route on-the-fly
            target = rasterize_waypoints_to_bev(
                data["route"],  # (B, num_route_pts, 3)
                self.config,
                sigma_m=1.5,
            )
        target = target.to(pred_intent.device, pred_intent.dtype)

        # BCE with pos_weight (positive pixels are rare)
        pos_weight = torch.tensor([10.0], device=pred_intent.device, dtype=pred_intent.dtype)
        intent_loss = F.binary_cross_entropy_with_logits(
            pred_intent, target, pos_weight=pos_weight, reduction="mean"
        )

        loss["loss_visual_intent"] = intent_loss * self.config.visual_intent_loss_weight
        log["unscaled_loss/loss_visual_intent"] = intent_loss.item()

        # P5: recall-weighted Tversky term. BCE is per-pixel and barely penalises a
        # whole dropped arm (its pixels average out); Tversky is a region overlap
        # ratio, so a missed arm (false negatives, weighted by beta) hurts a lot ->
        # forces the head to cover every feasible arm instead of collapsing to one.
        if self.config.intent_tversky_weight > 0:
            p = torch.sigmoid(pred_intent.float())
            g = target.float()
            dims = (1, 2, 3)  # per-sample sums over (C, H, W)
            tp = (p * g).sum(dims)
            fp = (p * (1.0 - g)).sum(dims)
            fn = ((1.0 - p) * g).sum(dims)
            smooth = 1.0
            tversky = (tp + smooth) / (
                tp
                + self.config.intent_tversky_alpha * fp
                + self.config.intent_tversky_beta * fn
                + smooth
            )
            tversky_loss = (1.0 - tversky).mean()
            loss["loss_visual_intent"] = (
                loss["loss_visual_intent"]
                + self.config.intent_tversky_weight * tversky_loss
            )
            log["unscaled_loss/loss_intent_tversky"] = tversky_loss.item()


def main():
    """Smoke test: random tensors through forward + loss."""
    import sys
    sys.path.insert(0, "/mmu_mllm_hdd_3/liuzihan08/vla/lead")
    from lead.training.config_training import TrainingConfig

    cfg = TrainingConfig()
    cfg.use_intent_decoder = True

    model = VLMIntentDecoder(cfg).cuda()
    B = 2
    vlm_hidden = torch.randn(B, 12, 12, 2560, device="cuda")

    pred = model(vlm_hidden)
    print(f"forward ok: {vlm_hidden.shape} → {pred.shape}")

    # Mock data for loss
    data = {
        "route": torch.randn(B, 10, 3, device="cuda"),  # mock expert route
    }
    loss, log = {}, {}
    model.compute_loss(pred, data, loss, log)
    print(f"loss ok: {loss}, {log}")
    print("✓ VLMIntentDecoder smoke test passed")


if __name__ == "__main__":
    main()
