"""P1 acceptance viz: predicted visual-intent vs ground-truth, overlaid on BEV.

Loads a trained intent checkpoint, runs the model on a few samples, and saves
3-panel images: [BEV + GT intent] | [BEV + pred intent] | [GT(green) vs pred(red)].

Runs on CPU (slow but fine for a handful of samples) or GPU.
Example:
    CUDA_VISIBLE_DEVICES="" python scripts/viz_intent_pred.py \
        --ckpt outputs/local_training/intent_p1/model_0000.pth --n 6
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from lead.data_loader.carla_dataset import CARLAData
from lead.tfv6.tfv6 import TFv6
from lead.training.config_training import TrainingConfig
from lead.training.mixed_training_utils import mixed_data_collate_fn

_BEV_COLORS = np.array(
    [
        [0, 0, 0], [70, 70, 70], [128, 64, 128], [244, 35, 232], [107, 142, 35],
        [70, 130, 180], [220, 20, 60], [0, 0, 142], [0, 0, 70], [0, 60, 100],
        [0, 80, 100], [119, 11, 32], [250, 170, 30], [190, 153, 153], [220, 220, 0],
    ],
    dtype=np.uint8,
)


def colorize_bev(sem: np.ndarray) -> np.ndarray:
    sem = sem.astype(np.int64)
    pal = _BEV_COLORS
    if sem.max() >= len(pal):
        pal = np.concatenate([pal, np.full((sem.max() - len(pal) + 1, 3), 128, np.uint8)])
    return pal[sem]


def overlay(bev_rgb: np.ndarray, heat: np.ndarray, color) -> np.ndarray:
    a = np.clip(heat, 0, 1)[..., None]
    out = bev_rgb.astype(np.float32) * (1 - a) + np.array(color, np.float32) * a
    return out.clip(0, 255).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/local_training/intent_p1/model_0000.pth")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", default="outputs/viz_intent_pred")
    ap.add_argument("--stride", type=int, default=61237)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    try:
        import cv2  # noqa: PLC0415
        write = lambda p, im: cv2.imwrite(p, im[..., ::-1])
    except Exception:
        from PIL import Image  # noqa: PLC0415
        write = lambda p, im: Image.fromarray(im).save(p)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = TrainingConfig()
    config.use_persistent_cache = True
    config.use_training_session_cache = False
    config.use_intent_decoder = True
    config.use_planning_decoder = False
    config.use_mixed_precision_training = False  # fp32 for CPU-safe inference

    model = TFv6(device, config).to(device)
    sd = torch.load(args.ckpt, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"loaded {args.ckpt} | missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()

    ds = CARLAData(root=config.carla_data, config=config)
    idxs = [(k * args.stride) % len(ds) for k in range(args.n)]
    loader = DataLoader(
        Subset(ds, idxs), batch_size=args.n, shuffle=False,
        num_workers=0, collate_fn=mixed_data_collate_fn,
    )
    batch = next(iter(loader))

    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
        pred = model(data=batch)
    prob = torch.sigmoid(pred.pred_visual_intent.float())[:, 0].cpu().numpy()  # (B,H,W)
    gt = np.asarray(batch["visual_intent_label"])[:, 0]                        # (B,H,W)
    sem = np.asarray(batch["bev_semantic"])                                    # (B,H,W)

    er = int((0.0 - config.min_y_meter) * config.pixels_per_meter)
    ec = int((0.0 - config.min_x_meter) * config.pixels_per_meter)
    sep = np.full((sem.shape[1], 3, 3), 255, np.uint8)
    for i, idx in enumerate(idxs):
        bev = colorize_bev(sem[i])
        p_gt = overlay(bev, gt[i], [40, 255, 40])            # GT green
        p_pd = overlay(bev, prob[i], [255, 40, 40])          # pred red
        p_bo = overlay(overlay(bev, gt[i], [40, 255, 40]), prob[i] * 0.8, [255, 40, 40])
        for pan in (p_gt, p_pd, p_bo):
            pan[max(er - 2, 0):er + 3, max(ec - 2, 0):ec + 3] = [255, 255, 0]
        vis = np.concatenate([p_gt, sep, p_pd, sep, p_bo], axis=1)
        out = os.path.join(args.out, f"pred_{idx:07d}.png")
        write(out, vis)
        print(f"[{idx}] pred_max={prob[i].max():.2f} gt_cov={ (gt[i]>0.1).mean():.4f} -> {out}")

    print(f"panels: [BEV+GT(green)] | [BEV+pred(red)] | [both]  ego=yellow, saved to {args.out}")


if __name__ == "__main__":
    main()
