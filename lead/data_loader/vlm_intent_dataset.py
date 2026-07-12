"""P4a dataset: cached Qwen-VL hidden states + visual_intent_label for distillation."""

from __future__ import annotations

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class VLMIntentDataset(Dataset):
    """Read cached Qwen-VL hidden states + expert intent label for P4a training.

    Each sample:
        - vlm_hidden: (h', w', D_vlm) from data/p4/vlm_cache/<scenario>/<route>/<frame>.npy
        - visual_intent_label: (1, H_bev, W_bev) expert route Gaussian heatmap
        - route: (num_pts, 3) expert waypoints for fallback label generation
    """

    def __init__(
        self,
        carla_dataset,
        vlm_cache_dir: str = "data/p4/vlm_cache",
        manifest_path: str | None = None,
    ):
        """
        Args:
            carla_dataset: CARLAData instance (already built, gives us frame list + route/label loader)
            vlm_cache_dir: root dir of cached .npy files
            manifest_path: optional path to the extraction manifest (.jsonl with
                scenario/route/frame per cached frame). When given, cache membership
                is tested against the manifest in memory instead of one
                ``os.path.exists`` per frame -- this avoids a metadata storm when the
                CARLA index is the full (non-strided) dataset on shared storage.
        """
        self.carla_ds = carla_dataset
        self.vlm_cache_dir = vlm_cache_dir
        self.config = carla_dataset.config

        cached_keys = None
        if manifest_path is not None:
            with open(manifest_path) as f:
                cached_keys = {
                    (e["scenario"], e["route"], e["frame"])
                    for e in (json.loads(line) for line in f)
                }

        # Filter to frames that have cached VLM features
        self.valid_indices = []
        for i in range(len(carla_dataset.images)):
            p = str(carla_dataset.images[i], encoding="utf-8")
            parts = p.split("/")
            scenario, route, frame = parts[-4], parts[-3], parts[-1].split(".")[0]
            if cached_keys is not None:
                has_cache = (scenario, route, frame) in cached_keys
            else:
                npy_path = os.path.join(vlm_cache_dir, scenario, route, frame + ".npy")
                has_cache = os.path.exists(npy_path)
            if has_cache:
                self.valid_indices.append(i)

        print(f"VLMIntentDataset: {len(self.valid_indices)} / {len(carla_dataset.images)} frames have VLM cache")


    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> dict:
        carla_idx = self.valid_indices[idx]

        # Load full CARLA sample (has route, visual_intent_label if precomputed, etc.)
        data = self.carla_ds[carla_idx]

        # Load cached VLM hidden states
        p = str(self.carla_ds.images[carla_idx], encoding="utf-8")
        parts = p.split("/")
        scenario, route, frame = parts[-4], parts[-3], parts[-1].split(".")[0]
        npy_path = os.path.join(self.vlm_cache_dir, scenario, route, frame + ".npy")
        vlm_hidden = np.load(npy_path)  # (h', w', D_vlm), fp16
        data["vlm_hidden"] = torch.from_numpy(vlm_hidden).to(torch.float32)

        return data


def vlm_intent_collate_fn(batch: list[dict]) -> dict:
    """Collate for VLMIntentDataset: keep only intent-relevant fields (vlm_hidden + label)."""
    data = {}

    def _to_tensor(x):
        return x if torch.is_tensor(x) else torch.from_numpy(np.asarray(x))

    # vlm_hidden is already a float tensor from __getitem__
    data["vlm_hidden"] = torch.stack([b["vlm_hidden"] for b in batch], dim=0)  # (B, h', w', D)

    # visual_intent_label: numpy (1,H,W) from CARLAData rasterization -> tensor
    if batch[0].get("visual_intent_label") is not None:
        data["visual_intent_label"] = torch.stack(
            [_to_tensor(b["visual_intent_label"]).float() for b in batch], dim=0
        )

    # route: fallback for label generation
    if batch[0].get("route") is not None:
        data["route"] = torch.stack([_to_tensor(b["route"]).float() for b in batch], dim=0)

    return data


def main():
    """Smoke: load a few samples."""
    import sys
    sys.path.insert(0, "/mmu_mllm_hdd_3/liuzihan08/vla/lead")
    from lead.data_loader.carla_dataset import CARLAData
    from lead.training.config_training import TrainingConfig

    cfg = TrainingConfig()
    cfg.use_planning_decoder = True
    cfg.use_intent_decoder = True
    cfg.model_type = "plant"  # skip heavy sensor load, only need visual_intent_label

    carla_ds = CARLAData(root=cfg.carla_data, config=cfg)
    vlm_ds = VLMIntentDataset(carla_ds, vlm_cache_dir="data/p4/vlm_cache")
    print(f"vlm_ds len: {len(vlm_ds)}")

    sample = vlm_ds[0]
    print(f"sample keys: {sample.keys()}")
    print(f"vlm_hidden shape: {sample['vlm_hidden'].shape}")
    if "visual_intent_label" in sample:
        print(f"visual_intent_label shape: {sample['visual_intent_label'].shape}")
    print("✓ VLMIntentDataset smoke test passed")


if __name__ == "__main__":
    main()
