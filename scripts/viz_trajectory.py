"""P2 acceptance viz: predicted trajectory + intent field + obstacles on BEV.

Overlays, per sample, on the BEV semantic map:
  - intent field (red heatmap, sigmoid of pred_visual_intent)
  - predicted waypoints (yellow polyline+dots)
  - expert GT waypoints (green polyline+dots)
  - ego (cyan)
Obstacles are already colored in the BEV semantic backdrop.

Example:
    CUDA_VISIBLE_DEVICES=0 python scripts/viz_trajectory.py \
        --ckpt outputs/local_training/control_p2_full/model_0030.pth --intent --cond --n 6
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
        [0, 0, 0], [70, 70, 70], [128, 64, 128], [244, 35, 232], [220, 20, 60],
        [255, 140, 0], [255, 255, 0], [200, 0, 0], [150, 0, 0], [255, 0, 255],
        [0, 200, 0], [200, 0, 0], [120, 0, 0],
    ], dtype=np.uint8,
)


def colorize(sem):
    sem = sem.astype(np.int64)
    pal = _BEV_COLORS
    if sem.max() >= len(pal):
        pal = np.concatenate([pal, np.full((sem.max() - len(pal) + 1, 3), 128, np.uint8)])
    return pal[sem]


def to_px(wp, config):
    col = (wp[:, 0] - config.min_x_meter) * config.pixels_per_meter
    row = (wp[:, 1] - config.min_y_meter) * config.pixels_per_meter
    return np.stack([col, row], axis=1).astype(np.int32)  # (N,2) as (x=col,y=row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--intent", action="store_true")
    ap.add_argument("--cond", action="store_true")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", default="outputs/viz_trajectory")
    ap.add_argument("--stride", type=int, default=54121)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    import cv2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = TrainingConfig()
    config.use_persistent_cache = True
    config.use_training_session_cache = False
    config.use_planning_decoder = True
    config.use_intent_decoder = args.intent
    config.use_control_conditioning = args.cond
    config.use_mixed_precision_training = False

    model = TFv6(device, config).to(device)
    sd = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(sd, strict=False)
    model.eval()

    ds = CARLAData(root=config.carla_data, config=config)
    idxs = [(k * args.stride) % len(ds) for k in range(args.n)]
    loader = DataLoader(Subset(ds, idxs), batch_size=args.n, shuffle=False,
                        num_workers=0, collate_fn=mixed_data_collate_fn)
    batch = next(iter(loader))
    with torch.no_grad():
        pred = model(data=batch)
    wp = pred.pred_future_waypoints.float().cpu().numpy()          # (B,n,2)
    gt = np.asarray(batch["future_waypoints"])[:, :wp.shape[1]]    # (B,n,2)
    sem = np.asarray(batch["bev_semantic"])                        # (B,H,W)
    intent = (
        torch.sigmoid(pred.pred_visual_intent.float())[:, 0].cpu().numpy()
        if pred.pred_visual_intent is not None else None
    )
    er = int((0 - config.min_y_meter) * config.pixels_per_meter)
    ec = int((0 - config.min_x_meter) * config.pixels_per_meter)

    for i, idx in enumerate(idxs):
        img = colorize(sem[i]).astype(np.float32)
        if intent is not None:  # red intent field
            a = np.clip(intent[i], 0, 1)[..., None]
            img = img * (1 - 0.6 * a) + np.array([255, 40, 40], np.float32) * (0.6 * a)
        img = img.clip(0, 255).astype(np.uint8)
        img = np.ascontiguousarray(img)
        gpx, ppx = to_px(gt[i], config), to_px(wp[i], config)
        cv2.polylines(img, [gpx], False, (40, 255, 40), 1)   # GT green
        cv2.polylines(img, [ppx], False, (255, 255, 0), 1)   # pred yellow
        for p in gpx:
            cv2.circle(img, tuple(p), 2, (40, 255, 40), -1)
        for p in ppx:
            cv2.circle(img, tuple(p), 2, (255, 255, 0), -1)
        cv2.circle(img, (ec, er), 3, (0, 255, 255), -1)      # ego cyan
        cv2.imwrite(os.path.join(args.out, f"traj_{idx:07d}.png"), img[..., ::-1])
        print(f"[{idx}] saved")
    print(f"legend: intent=red, GT=green, pred=yellow, ego=cyan -> {args.out}")


if __name__ == "__main__":
    main()
