"""Open-loop collision-rate + ADE eval for a trained model.

Runs the model on N samples, projects predicted waypoints onto the GT BEV
occupancy, and reports:
  - waypoint collision rate: fraction of predicted waypoints landing in an
    obstacle cell
  - trajectory collision rate: fraction of trajectories with >=1 such waypoint
  - ADE / FDE vs expert future waypoints
Run 3x for LEAD / P2.1 / P2.2 and compare. GPU recommended.

Examples:
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_collision_rate.py \
        --ckpt outputs/local_training/posttrain/model_0030.pth --n 400          # LEAD
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_collision_rate.py \
        --ckpt outputs/local_training/control_p2_full/model_0030.pth \
        --intent --cond --n 400                                                 # P2.2
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from lead.data_loader.carla_dataset import CARLAData
from lead.tfv6.collision_cost import occupancy_from_bev_semantic, waypoints_to_grid
from lead.tfv6.tfv6 import TFv6
from lead.training.config_training import TrainingConfig
from lead.training.mixed_training_utils import mixed_data_collate_fn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--intent", action="store_true", help="model has intent decoder")
    ap.add_argument("--cond", action="store_true", help="control conditioned on intent")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--bs", type=int, default=8)
    args = ap.parse_args()

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
    miss, unexp = model.load_state_dict(sd, strict=False)
    model.eval()
    print(f"loaded {args.ckpt} | intent={args.intent} cond={args.cond} | missing={len(miss)} unexpected={len(unexp)}")

    ds = CARLAData(root=config.carla_data, config=config)
    idxs = [(k * 2411) % len(ds) for k in range(args.n)]
    loader = DataLoader(
        Subset(ds, idxs), batch_size=args.bs, shuffle=False,
        num_workers=8, collate_fn=mixed_data_collate_fn,
    )

    wp_hits = wp_total = traj_hits = traj_total = 0
    ade_sum = fde_sum = 0.0
    nwp = config.num_way_points_prediction
    with torch.no_grad():
        for batch in loader:
            pred = model(data=batch)
            wp = pred.pred_future_waypoints.float()  # (B, n, 2) ego metres
            occ = occupancy_from_bev_semantic(
                batch["bev_semantic"].to(device),
            ).float()  # (B,1,H,W)
            grid = waypoints_to_grid(wp, config).to(occ.dtype)  # (B,n,1,2)
            hit = torch.nn.functional.grid_sample(
                occ, grid, mode="nearest", padding_mode="zeros", align_corners=True,
            )[:, 0, :, 0]  # (B, n) in {0,1}
            wp_hits += int((hit > 0.5).sum())
            wp_total += hit.numel()
            traj_hit = (hit > 0.5).any(dim=1)
            traj_hits += int(traj_hit.sum())
            traj_total += traj_hit.numel()

            gt = batch["future_waypoints"].to(device).float()[:, :nwp]
            d = (wp - gt).norm(dim=-1)  # (B, n)
            ade_sum += float(d.mean(dim=1).sum())
            fde_sum += float(d[:, -1].sum())

    print(f"samples={traj_total}")
    print(f"waypoint  collision rate: {wp_hits/max(wp_total,1)*100:.2f}%  ({wp_hits}/{wp_total})")
    print(f"trajectory collision rate: {traj_hits/max(traj_total,1)*100:.2f}%  ({traj_hits}/{traj_total})")
    print(f"ADE={ade_sum/max(traj_total,1):.4f}  FDE={fde_sum/max(traj_total,1):.4f}")


if __name__ == "__main__":
    main()
