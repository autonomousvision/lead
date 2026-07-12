#!/bin/bash
# P5a: train VLMIntentDecoder to distill the MULTIMODAL drivable-support field
# (all forward-reachable arms from the hdmap), instead of the single expert route.
# Reuses the P4a pipeline + cached VLM features; only the distillation target changes.
# Usage: CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/p4/train_vlm_intent_p5a_ddp.sh

source /mmu_mllm_hdd_3/liuzihan08/miniconda3/etc/profile.d/conda.sh
conda activate lead

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
    --vlm-cache data/p5/vlm_cache \
    --manifest data/p4/manifest.jsonl \
    --logdir outputs/local_training/vlm_intent_p5a_tversky \
    --multimodal-intent \
    --tversky-weight 0.75 \
    --batch-size 256 \
    --lr 6e-4 \
    --epochs 15 \
    --num-workers 6 \
    "$@"
