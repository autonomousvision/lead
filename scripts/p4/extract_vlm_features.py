"""P4 step 2 (qwenvl env): extract command-conditioned VLM image-token hidden states.

For each manifest frame: read the raw strip image, crop the FRONT camera, run a single
prefill forward through a FROZEN Qwen3-VL with the nav command placed BEFORE the image
(causal LM -> image tokens can only attend to preceding tokens, so the command must come
first to condition them), grab the last-layer hidden states at the image-token positions,
reshape to the (h', w', D) patch grid, and cache as fp16 .npy at
    <out>/<scenario>/<route>/<frame>.npy

Resumable (skips existing) and shardable across GPUs.

Run (qwenvl env). Example: 4 GPUs, 4 shards:
    for i in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=$i python scripts/p4/extract_vlm_features.py \
        --manifest data/p4/manifest.jsonl --out data/p4/vlm_cache \
        --model /mmu_mllm_hdd_3/liuzihan08/vla/models/Qwen3-VL-4B-Instruct \
        --num-shards 4 --shard $i &
    done; wait

Smoke (a few frames, prints shapes):
    CUDA_VISIBLE_DEVICES=0 python scripts/p4/extract_vlm_features.py \
        --manifest data/p4/manifest_smoke.jsonl --out data/p4/vlm_cache_smoke --limit 4
"""

from __future__ import annotations

import argparse
import json
import os


PROMPT_TEMPLATE = (
    "You are the motion planner of a car. Navigation command: {cmd}. "
    "Look at the front camera image and determine where the ego vehicle should drive "
    "to follow this command."
)

# P5: command-agnostic prompt. No navigation command -> the image-token hidden states
# encode ALL drivable directions (the feasible set), leaving the command to be resolved
# later at the planner. This matches the multimodal drivable-support distillation target.
PROMPT_DRIVABLE = (
    "You are the motion planner of a car. Look at the front camera image and identify "
    "every direction the car could drive from here -- all lanes and turns that are "
    "drivable ahead, without committing to a single one."
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/p4/manifest.jsonl")
    ap.add_argument("--out", default="data/p4/vlm_cache")
    ap.add_argument("--model", default="/mmu_mllm_hdd_3/liuzihan08/vla/models/Qwen3-VL-4B-Instruct")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0, help="this process handles lines where idx %% num_shards == shard")
    ap.add_argument("--layer", type=int, default=-1, help="which hidden_states layer to cache (-1 = last)")
    ap.add_argument("--prompt-mode", choices=["command", "drivable"], default="command",
                    help="'command' = P4 (nav command before image); 'drivable' = P5 "
                         "command-agnostic (all drivable directions)")
    ap.add_argument("--limit", type=int, default=0, help="smoke: only process first N assigned frames")
    args = ap.parse_args()

    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    os.makedirs(args.out, exist_ok=True)
    entries = [json.loads(l) for l in open(args.manifest)]
    entries = [e for i, e in enumerate(entries) if i % args.num_shards == args.shard]
    if args.limit > 0:
        entries = entries[: args.limit]
    print(f"[shard {args.shard}/{args.num_shards}] frames to do: {len(entries)}")

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True
    )
    model.eval()
    image_token_id = model.config.image_token_id
    merge = getattr(model.config.vision_config, "spatial_merge_size", 2)

    done = skipped = bad = 0
    for e in entries:
        out_npy = os.path.join(args.out, e["scenario"], e["route"], e["frame"] + ".npy")
        if os.path.exists(out_npy):
            skipped += 1
            continue

        # load raw strip, crop front camera
        try:
            img = Image.open(e["src"]).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            bad += 1
            if bad <= 5:
                print(f"  [bad] {e['src']} ({exc})")
            continue
        W, H = img.size
        f0, f1 = e["front_frac"]
        front = img.crop((int(W * f0), 0, int(W * f1), H))

        # command BEFORE image so causal image-token hidden states are command-conditioned
        # (P4); or a command-agnostic 'drivable' prompt for the multimodal P5 target.
        prompt = (
            PROMPT_DRIVABLE if args.prompt_mode == "drivable"
            else PROMPT_TEMPLATE.format(cmd=e["command"])
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": front},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to("cuda:0")

        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states[args.layer][0]  # (seq, D)
        mask = inputs["input_ids"][0] == image_token_id
        img_hs = hs[mask]  # (n_img, D)

        grid = inputs["image_grid_thw"][0].tolist()  # [t, h, w]
        h2, w2 = grid[1] // merge, grid[2] // merge
        if h2 * w2 != img_hs.shape[0]:
            bad += 1
            print(f"  [bad] token/grid mismatch {img_hs.shape[0]} vs {h2}x{w2} for {e['key']}")
            continue
        feat = img_hs.reshape(h2, w2, -1).to(torch.float16).cpu().numpy()

        os.makedirs(os.path.dirname(out_npy), exist_ok=True)
        np.save(out_npy, feat)
        done += 1
        if done <= 3 or done % 2000 == 0:
            print(f"[shard {args.shard}] {done} done  {e['key']}  cmd={e['command']} "
                  f"feat={feat.shape} ({feat.nbytes/1024:.0f}KB) -> {out_npy}")

    print(f"[shard {args.shard}] finished: done={done} skipped={skipped} bad={bad}")


if __name__ == "__main__":
    main()
