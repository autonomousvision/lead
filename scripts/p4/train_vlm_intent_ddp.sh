#!/bin/bash
# P4a: train VLMIntentDecoder (Qwen-VL hidden states -> BEV intent), distill expert route.
# Only trains the lightweight decoder (~13M params, no backbone forward) -> fast.
# Usage: CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/p4/train_vlm_intent_ddp.sh

export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export NCCL_P2P_DISABLE=1
export NCCL_P2P_LEVEL=NVL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nproc_per_node=$(python -c "import torch; print(torch.cuda.device_count())")
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((10000 + RANDOM % 50000))

torchrun --standalone \
    --nnodes=1 \
    --nproc_per_node=$nproc_per_node \
    --max_restarts=0 \
    scripts/p4/train_vlm_intent.py \
    --vlm-cache data/p4/vlm_cache \
    --logdir outputs/local_training/vlm_intent_p4a \
    --batch-size 256 \
    --lr 6e-4 \
    --epochs 15 \
    --num-workers 6
