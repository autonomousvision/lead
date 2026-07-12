#!/bin/bash
# P4b: open-loop ablation -- planner's intent from TFv6 head vs Qwen-VL head (+ a
# zero-intent control), backbone + planner frozen. Multi-GPU via torchrun (DDP
# sharding: each rank evaluates a disjoint slice, sums are all-reduced, rank 0
# prints the table). Runs on the VLM-cache frame subset. Pass extra args through,
# e.g. --limit 64 for a smoke test.
#   Usage: CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/p4/eval_p4b_openloop.sh [--limit 64]

source /mmu_mllm_hdd_3/liuzihan08/miniconda3/etc/profile.d/conda.sh
conda activate lead

# Node has no internet; timm's resnet34 backbone is already cached under
# ~/.cache/huggingface. Force offline so it reads the cache instead of retrying HF.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_DISABLE=1
export NCCL_P2P_LEVEL=NVL
nproc_per_node=$(python -c "import torch; print(torch.cuda.device_count())")
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((10000 + RANDOM % 50000))

torchrun --standalone \
    --nnodes=1 \
    --nproc_per_node=$nproc_per_node \
    --max_restarts=0 \
    scripts/p4/eval_p4b_openloop.py \
    --p2-ckpt outputs/local_training/control_p2_full/model_0030.pth \
    --p4a-ckpt outputs/local_training/vlm_intent_p4a/model_0014.pth \
    --vlm-cache data/p4/vlm_cache \
    --manifest data/p4/manifest.jsonl \
    --batch-size 32 \
    --num-workers 8 \
    "$@"
