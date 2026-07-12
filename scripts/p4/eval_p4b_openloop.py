"""P4b open-loop ablation: swap the planner's intent source, everything else frozen.

Loads the P2 backbone + planner (frozen) and the P4a VLMIntentDecoder (frozen),
then evaluates open-loop ADE/FDE twice on the SAME frames:
    - baseline: intent from the TFv6 IntentDecoder (P2's own head)
    - P4b:      intent from the Qwen-VL VLMIntentDecoder (P4a head)

Only the intent field fed to the planner differs -> a clean paired ablation that
isolates "where the intent comes from". Frames are the VLM-cache subset (stride-10),
since P4b can only run where cached Qwen features exist.

Usage (multi-GPU via torchrun; single-GPU also works):
    CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/p4/eval_p4b_openloop.sh
    CUDA_VISIBLE_DEVICES=0 bash scripts/p4/eval_p4b_openloop.sh --limit 64  # smoke
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm


def _errors(pred: torch.Tensor, label: torch.Tensor, common_utils) -> tuple[float, float]:
    """Return (ADE, FDE) for a predicted vs. ground-truth trajectory batch."""
    return (
        common_utils.average_displacement_error(pred, label),
        common_utils.final_displacement_error(pred, label),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p2-ckpt", default="outputs/local_training/control_p2_full/model_0030.pth")
    ap.add_argument("--p4a-ckpt", default="outputs/local_training/vlm_intent_p4a/model_0014.pth")
    ap.add_argument("--vlm-cache", default="data/p4/vlm_cache")
    ap.add_argument("--manifest", default="data/p4/manifest.jsonl",
                    help="extraction manifest; cache membership is tested against it "
                         "in memory (avoids a per-frame stat storm on shared storage)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None, help="cap #frames for smoke test")
    ap.add_argument("--out", default="outputs/local_evaluation/p4b_openloop/metrics.json")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from lead.common import common_utils
    from lead.data_loader.carla_dataset import CARLAData
    from lead.data_loader.vlm_intent_dataset import VLMIntentDataset
    from lead.tfv6.tfv6 import TFv6
    from lead.tfv6.vlm_intent_decoder import VLMIntentDecoder
    from lead.training.config_training import TrainingConfig
    from lead.training.mixed_training_utils import mixed_data_collate_fn

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # --- DDP setup (torchrun sets RANK/WORLD_SIZE/LOCAL_RANK) ---
    # gloo backend: the only collective is a 15-scalar all_reduce at the end, which
    # NCCL routes through /dev/shm -- unavailable/too-small in this container. gloo
    # uses TCP on CPU tensors and sidesteps the shared-memory segment entirely.
    is_ddp = "RANK" in os.environ
    if is_ddp:
        dist.init_process_group(backend="gloo")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank, world_size, local_rank = 0, 1, 0
    is_main = rank == 0

    # --- Config: reconstruct the exact P2 config from the checkpoint's config.json ---
    with open(os.path.join(os.path.dirname(args.p2_ckpt), "config.json")) as f:
        loaded_config = json.load(f)
    config = TrainingConfig(loaded_config, raise_error_on_missing_key=False)

    # The ResNet image backbone is built with pretrained=True, which timm fetches
    # from the HF hub -- impossible on this offline node (and node-local caches don't
    # follow us across machines). It's pointless here anyway: the full P2 checkpoint
    # is loaded over the whole model right after. Force pretrained=False so
    # construction never touches the network; the checkpoint supplies the real weights.
    import timm
    _timm_create_model = timm.create_model
    timm.create_model = lambda *a, **k: _timm_create_model(*a, **{**k, "pretrained": False})

    # --- Models (both frozen) ---
    model = TFv6(device, config).to(device)
    model.load_state_dict(
        torch.load(args.p2_ckpt, map_location=device, weights_only=True),
        strict=True,
    )
    model.eval().requires_grad_(False)

    vlm_decoder = VLMIntentDecoder(config).to(device)
    vlm_ckpt = torch.load(args.p4a_ckpt, map_location=device, weights_only=True)
    vlm_decoder.load_state_dict(vlm_ckpt["model"])
    vlm_decoder.eval().requires_grad_(False)
    if is_main:
        print(f"Loaded P2 model from {args.p2_ckpt}")
        print(f"Loaded P4a VLMIntentDecoder from {args.p4a_ckpt}")

    # --- Data: full-sensor CARLAData + per-frame VLM cache (paired) ---
    carla_ds = CARLAData(root=config.carla_data, config=config)
    vlm_ds = VLMIntentDataset(carla_ds, vlm_cache_dir=args.vlm_cache, manifest_path=args.manifest)
    sampler = (
        DistributedSampler(vlm_ds, num_replicas=world_size, rank=rank, shuffle=False)
        if is_ddp
        else None
    )
    loader = DataLoader(
        vlm_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=mixed_data_collate_fn,
        pin_memory=True,
    )
    # --limit is a GLOBAL frame cap; split it across ranks for smoke tests.
    local_limit = None if args.limit is None else math.ceil(args.limit / world_size)

    n_wp = config.num_way_points_prediction

    # Sum-of-(metric * batch) accumulators, normalized by frame count at the end.
    keys = ["route_ade", "route_fde", "wp_ade", "wp_fde"]
    base = {k: 0.0 for k in keys}
    p4b = {k: 0.0 for k in keys}
    zero = {k: 0.0 for k in keys}  # control: planner fed an all-zero intent
    intent_l1 = 0.0  # mean |sigmoid(TFv6) - sigmoid(VLM)| over the BEV field
    intent_corr = 0.0  # Pearson corr between the two sigmoid fields
    n_seen = 0

    for data in tqdm(loader, desc="P4b open-loop eval", disable=not is_main):
        if local_limit is not None and n_seen >= local_limit:
            break
        bs = data["vlm_hidden"].shape[0]
        route_label = data["route"].float()
        wp_label = data["future_waypoints"].float()[:, :n_wp]

        with torch.no_grad():
            # VLMIntentDecoder was trained in fp32; the planner casts the intent to
            # its own dtype internally. The P2 model runs under the same bf16 autocast
            # it was trained with (else fp32 weights clash with bf16 sensor inputs).
            vlm_intent = vlm_decoder(data["vlm_hidden"].to(device).float())
            with torch.amp.autocast(
                device_type="cuda",
                dtype=config.torch_float_type,
                enabled=config.use_mixed_precision_training,
            ):
                pred_base = model(data)
                pred_p4b = model(data, external_intent=vlm_intent)
                pred_zero = model(data, external_intent=torch.zeros_like(vlm_intent))

        for acc, pred in [(base, pred_base), (p4b, pred_p4b), (zero, pred_zero)]:
            r_ade, r_fde = _errors(pred.pred_route, route_label, common_utils)
            w_ade, w_fde = _errors(pred.pred_future_waypoints, wp_label, common_utils)
            acc["route_ade"] += r_ade * bs
            acc["route_fde"] += r_fde * bs
            acc["wp_ade"] += w_ade * bs
            acc["wp_fde"] += w_fde * bs

        # Intent-field agreement: how close is the VLM intent to the TFv6 intent the
        # planner was trained on? (pred_base.pred_visual_intent is the TFv6 field.)
        s_tfv6 = torch.sigmoid(pred_base.pred_visual_intent.float()).flatten(1)
        s_vlm = torch.sigmoid(vlm_intent.float()).flatten(1)
        intent_l1 += (s_tfv6 - s_vlm).abs().mean().item() * bs
        intent_corr += torch.corrcoef(
            torch.stack([s_tfv6.flatten(), s_vlm.flatten()])
        )[0, 1].item() * bs
        n_seen += bs

    # --- Aggregate sums across ranks BEFORE normalizing (gloo -> CPU tensor) ---
    if is_ddp:
        packed = torch.tensor(
            [base[k] for k in keys]
            + [p4b[k] for k in keys]
            + [zero[k] for k in keys]
            + [intent_l1, intent_corr, float(n_seen)],
            device="cpu",
            dtype=torch.float64,
        )
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        vals = packed.tolist()
        base = {k: vals[i] for i, k in enumerate(keys)}
        p4b = {k: vals[4 + i] for i, k in enumerate(keys)}
        zero = {k: vals[8 + i] for i, k in enumerate(keys)}
        intent_l1, intent_corr, n_seen = vals[12], vals[13], int(round(vals[14]))

    for acc in (base, p4b, zero):
        for k in keys:
            acc[k] /= n_seen
    intent_l1 /= n_seen
    intent_corr /= n_seen

    if not is_main:
        dist.destroy_process_group()
        return

    # --- Report (rank 0 only) ---
    print(f"\n=== P4b open-loop ablation ({n_seen} frames) ===")
    print(f"{'metric':<12}{'baseline(TFv6)':>16}{'P4b(VLM)':>12}{'zero-intent':>13}{'Δ(P4b-base)':>14}")
    for k in keys:
        d = p4b[k] - base[k]
        print(f"{k:<12}{base[k]:>16.4f}{p4b[k]:>12.4f}{zero[k]:>13.4f}{d:>+14.4f}")
    print(f"\nintent-field agreement (TFv6 vs VLM): "
          f"sigmoid L1 = {intent_l1:.4f}, Pearson corr = {intent_corr:.4f}")
    print("Read: if zero-intent ≈ baseline, the planner barely uses intent (swap is "
          "uninformative); if zero-intent is worse while P4b ≈ baseline, intent "
          "matters and the VLM field reproduces the TFv6 one.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(
            {
                "n_frames": n_seen,
                "p2_ckpt": args.p2_ckpt,
                "p4a_ckpt": args.p4a_ckpt,
                "baseline_tfv6_intent": base,
                "p4b_vlm_intent": p4b,
                "zero_intent_control": zero,
                "delta_p4b_minus_base": {k: p4b[k] - base[k] for k in keys},
                "delta_zero_minus_base": {k: zero[k] - base[k] for k in keys},
                "intent_field_sigmoid_l1": intent_l1,
                "intent_field_pearson_corr": intent_corr,
            },
            f,
            indent=2,
        )
    print(f"\n✓ Wrote {args.out}")

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
