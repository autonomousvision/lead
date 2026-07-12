"""P4a training script: train VLMIntentDecoder to distill expert route intent from Qwen-VL features.

Multi-GPU via torchrun DDP (only trains lightweight decoder, no backbone forward -> fast).

Usage (8 GPUs):
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/p4/train_vlm_intent_ddp.sh
Or directly:
    torchrun --standalone --nproc_per_node=8 scripts/p4/train_vlm_intent.py \
        --vlm-cache data/p4/vlm_cache --epochs 15 --batch-size 128 --lr 3e-4 \
        --logdir outputs/local_training/vlm_intent_p4a
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm-cache", default="data/p4/vlm_cache")
    ap.add_argument("--manifest", default=None,
                    help="extraction manifest; if given, cache membership is tested in "
                         "memory instead of one os.path.exists per frame (avoids a stat storm)")
    ap.add_argument("--logdir", default="outputs/local_training/vlm_intent_p4a")
    ap.add_argument("--batch-size", type=int, default=128, help="GLOBAL batch size (split across GPUs)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--resume", type=str, default=None, help="checkpoint to resume from")
    ap.add_argument("--multimodal-intent", action="store_true",
                    help="P5a: distill the multimodal drivable-support field (all "
                         "reachable arms) instead of the single expert route")
    ap.add_argument("--tversky-weight", type=float, default=0.0,
                    help="P5: weight of the recall-weighted Tversky term (0 = BCE only)")
    args = ap.parse_args()

    # --- DDP setup ---
    is_ddp = "RANK" in os.environ
    if is_ddp:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank, world_size, local_rank = 0, 1, 0
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    is_main = rank == 0

    if is_main:
        os.makedirs(args.logdir, exist_ok=True)

    # Config + dataset
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from lead.data_loader.carla_dataset import CARLAData
    from lead.data_loader.vlm_intent_dataset import VLMIntentDataset, vlm_intent_collate_fn
    from lead.training.config_training import TrainingConfig
    from lead.tfv6.vlm_intent_decoder import VLMIntentDecoder

    config = TrainingConfig()
    config.use_planning_decoder = True
    config.use_intent_decoder = True
    # P5a: distill the multimodal drivable-support field instead of the expert route.
    config.use_multimodal_intent = args.multimodal_intent
    config.intent_tversky_weight = args.tversky_weight
    # "plant" model_type makes CARLAData.__getitem__ return right after building the
    # visual_intent_label, skipping the heavy sensor path (lidar .laz). P4a only needs
    # the label, so this makes the dataloader fast.
    config.model_type = "plant"

    carla_ds = CARLAData(root=config.carla_data, config=config)
    vlm_ds = VLMIntentDataset(carla_ds, vlm_cache_dir=args.vlm_cache, manifest_path=args.manifest)

    per_gpu_bs = max(1, args.batch_size // world_size)
    sampler = DistributedSampler(vlm_ds, shuffle=True) if is_ddp else None
    loader = DataLoader(
        vlm_ds,
        batch_size=per_gpu_bs,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=vlm_intent_collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    if is_main:
        print(f"world_size={world_size}, per_gpu_bs={per_gpu_bs}, "
              f"{len(vlm_ds)} samples, {len(loader)} batches/epoch/gpu")

    # Model
    model = VLMIntentDecoder(config).to(device)
    if is_main:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"VLMIntentDecoder: {n_params/1e6:.2f}M trainable params")
    if is_ddp:
        model = DDP(model, device_ids=[local_rank])
    core = model.module if is_ddp else model

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Resume
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        core.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        if is_main:
            print(f"Resumed from {args.resume}, starting epoch {start_epoch}")

    # Training loop
    model.train()
    for epoch in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", disable=not is_main)
        epoch_loss = 0.0
        for batch in pbar:
            vlm_hidden = batch["vlm_hidden"].to(device)
            for k in ["visual_intent_label", "route"]:
                if k in batch:
                    batch[k] = batch[k].to(device)

            pred_intent = model(vlm_hidden)

            loss_dict, log_dict = {}, {}
            core.compute_loss(pred_intent, batch, loss_dict, log_dict)
            loss = loss_dict["loss_visual_intent"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if is_main:
                pbar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(loader)
        if is_main:
            print(f"Epoch {epoch}: avg loss = {avg_loss:.4f}")
            if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
                ckpt_path = os.path.join(args.logdir, f"model_{epoch:04d}.pth")
                torch.save(
                    {"epoch": epoch, "model": core.state_dict(),
                     "optimizer": optimizer.state_dict(), "loss": avg_loss},
                    ckpt_path,
                )
                print(f"Saved {ckpt_path}")

    if is_ddp:
        dist.destroy_process_group()
    if is_main:
        print(f"✓ P4a training done. Checkpoints in {args.logdir}")


if __name__ == "__main__":
    main()
