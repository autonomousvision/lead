#!/bin/bash
# P1: train the visual-intent head on top of the perception-pretrained backbone.
# Loads pretrain weights (strict=False -> intent_decoder starts fresh, backbone loaded).
# Usage: CUDA_VISIBLE_DEVICES=4,5,6,7 bash scripts/train_intent_ddp.sh

export OMP_NUM_THREADS=$(nproc)
export OPENBLAS_NUM_THREADS=1
export NCCL_P2P_DISABLE=1
export NCCL_P2P_LEVEL=NVL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nproc_per_node=$(python -c "import torch; print(torch.cuda.device_count())")
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((10000 + RANDOM % 50000))

export LEAD_TRAINING_CONFIG="logdir=outputs/local_training/intent_p1 \
load_file=outputs/local_training/pretrain/model_0030.pth \
use_intent_decoder=true \
use_planning_decoder=false"

torchrun --standalone \
    --nnodes=1 \
    --nproc_per_node=$nproc_per_node \
    --max_restarts=0 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --no-python \
    python3 lead/training/train.py
