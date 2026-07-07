"""P0 visualization: expert path -> soft BEV intent corridor, overlaid on bev_semantic.

Pure data + rasterization + drawing. No GPU, no model, no training.
Confirms that the intent label / BEV grid orientation is correct before P1.

Run (CPU is enough):
    CUDA_VISIBLE_DEVICES="" python scripts/viz_visual_intent.py --n 8 --out outputs/viz_visual_intent
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from lead.data_loader.carla_dataset import CARLAData
from lead.tfv6.intent_decoder import rasterize_waypoints_to_bev
from lead.training.config_training import TrainingConfig


# a small distinct color per BEV-semantic class id (fallback grey if out of range)
_BEV_COLORS = np.array(
    [
        [0, 0, 0], [70, 70, 70], [128, 64, 128], [244, 35, 232], [107, 142, 35],
        [70, 130, 180], [220, 20, 60], [0, 0, 142], [0, 0, 70], [0, 60, 100],
        [0, 80, 100], [119, 11, 32], [250, 170, 30], [190, 153, 153], [220, 220, 0],
    ],
    dtype=np.uint8,
)


def colorize_bev_semantic(sem: np.ndarray) -> np.ndarray:
    sem = sem.astype(np.int64)
    palette = _BEV_COLORS
    if sem.max() >= len(palette):
        palette = np.concatenate(
            [palette, np.full((sem.max() - len(palette) + 1, 3), 128, np.uint8)]
        )
    return palette[sem]  # (H, W, 3)


def overlay_intent(bev_rgb: np.ndarray, intent: np.ndarray) -> np.ndarray:
    """Red overlay of the intent heatmap (intent in [0,1], HxW)."""
    out = bev_rgb.astype(np.float32)
    a = np.clip(intent, 0.0, 1.0)[..., None]  # (H, W, 1)
    red = np.array([255, 40, 40], np.float32)
    out = out * (1 - a) + red * a
    return out.clip(0, 255).astype(np.uint8)


def get_path(data: dict) -> np.ndarray | None:
    for key in ("route", "future_waypoints"):
        v = data.get(key)
        if v is not None:
            return np.asarray(v, dtype=np.float32).reshape(-1, 2)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, help="number of samples to render")
    ap.add_argument("--out", type=str, default="outputs/viz_visual_intent")
    ap.add_argument("--stride", type=int, default=9973, help="index stride for variety")
    ap.add_argument("--sigma_m", type=float, default=1.5)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    try:
        import cv2  # noqa: PLC0415
        writer = lambda p, im: cv2.imwrite(p, im[..., ::-1])  # RGB->BGR
    except Exception:
        from PIL import Image  # noqa: PLC0415
        writer = lambda p, im: Image.fromarray(im).save(p)

    config = TrainingConfig()
    config.use_persistent_cache = True
    config.use_training_session_cache = False
    config.use_planning_decoder = True  # posttrain mode exposes route/future_waypoints labels

    ds = CARLAData(root=config.carla_data, config=config)
    n_total = len(ds)
    print(f"dataset size: {n_total}")

    ego_col = int((0.0 - config.min_x_meter) * config.pixels_per_meter)  # x=0 -> width
    ego_row = int((0.0 - config.min_y_meter) * config.pixels_per_meter)  # y=0 -> height

    saved = 0
    for k in range(args.n):
        idx = (k * args.stride) % n_total
        data = ds[idx]
        path = get_path(data)
        sem = data.get("bev_semantic")
        if path is None or sem is None:
            print(f"[{idx}] missing route/bev_semantic (keys: {sorted(data.keys())[:12]}...)")
            continue
        sem = np.asarray(sem)
        if sem.ndim == 3:
            sem = sem[0]

        intent = rasterize_waypoints_to_bev(
            torch.tensor(path)[None], config, sigma_m=args.sigma_m
        )[0, 0].numpy()  # (H, W)

        bev_rgb = colorize_bev_semantic(sem)
        vis = overlay_intent(bev_rgb, intent)
        # mark ego + raw path pixels for cross-check
        vis[max(ego_row - 2, 0):ego_row + 3, max(ego_col - 2, 0):ego_col + 3] = [0, 255, 0]

        out_path = os.path.join(args.out, f"intent_{idx:07d}.png")
        writer(out_path, vis)
        saved += 1
        print(
            f"[{idx}] path pts={len(path)} "
            f"x[{path[:,0].min():.1f},{path[:,0].max():.1f}] "
            f"y[{path[:,1].min():.1f},{path[:,1].max():.1f}] "
            f"bev{sem.shape} intent_max={intent.max():.2f} -> {out_path}"
        )

    print(f"saved {saved}/{args.n} to {args.out}")
    print(f"ego marker (green) at row={ego_row}, col={ego_col}; forward=+x spans width")


if __name__ == "__main__":
    main()
