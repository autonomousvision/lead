#!/bin/bash
# P5 step 1: extract COMMAND-AGNOSTIC ("drivable") VLM features into a new cache,
# mirroring P4 step 2 but with --prompt-mode drivable (no nav command -> features
# encode all drivable directions, matching the multimodal support target).
# One shard per GPU, resumable (skips existing .npy). Runs in the qwenvl env.
#
#   Full 8-GPU run:   bash scripts/p4/extract_vlm_p5_drivable.sh
#   Subset of GPUs:   GPUS=0,1,2,3 bash scripts/p4/extract_vlm_p5_drivable.sh
#   Smoke (1 GPU):    GPUS=0 bash scripts/p4/extract_vlm_p5_drivable.sh \
#                         --manifest data/p4/manifest_smoke.jsonl --out data/p5/vlm_cache_smoke --limit 4

source /mmu_mllm_hdd_3/liuzihan08/miniconda3/etc/profile.d/conda.sh
conda activate qwenvl

# Node has no internet; the Qwen model is a local dir. Force offline so nothing retries HF.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

MANIFEST=${MANIFEST:-data/p4/manifest.jsonl}
OUT=${OUT:-data/p5/vlm_cache}
MODEL=${MODEL:-/mmu_mllm_hdd_3/liuzihan08/vla/models/Qwen3-VL-4B-Instruct}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}

IFS=',' read -ra GPU_ARR <<< "$GPUS"
N=${#GPU_ARR[@]}
echo "P5 drivable extraction: manifest=$MANIFEST out=$OUT gpus=[$GPUS] shards=$N"

for idx in "${!GPU_ARR[@]}"; do
    g=${GPU_ARR[$idx]}
    CUDA_VISIBLE_DEVICES=$g python scripts/p4/extract_vlm_features.py \
        --manifest "$MANIFEST" \
        --out "$OUT" \
        --model "$MODEL" \
        --prompt-mode drivable \
        --num-shards "$N" \
        --shard "$idx" \
        "$@" &
done
wait
echo "✓ P5 drivable extraction done -> $OUT"
