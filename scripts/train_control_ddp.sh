#!/bin/bash
# P2.1: train the control head (planning decoder) CONDITIONED on the visual-intent field.
# Loads the P1 checkpoint (intent head + backbone); planning/control head starts fresh.
# Only imitation loss for now (collision cost = P2.2). Goal: conditioning must not hurt ADE/FDE.
# Usage: CUDA_VISIBLE_DEVICES=4,5,6,7 bash scripts/train_control_ddp.sh

export OMP_NUM_THREADS=$(nproc)
export OPENBLAS_NUM_THREADS=1
export NCCL_P2P_DISABLE=1
export NCCL_P2P_LEVEL=NVL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nproc_per_node=$(python -c "import torch; print(torch.cuda.device_count())")
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((10000 + RANDOM % 50000))

export LEAD_TRAINING_CONFIG="logdir=outputs/local_training/control_p2 \
load_file=outputs/local_training/intent_p1/model_0030.pth \
use_intent_decoder=true \
use_planning_decoder=true \
use_control_conditioning=true \
batch_size=128 \
epochs=8"

torchrun --standalone \
    --nnodes=1 \
    --nproc_per_node=$nproc_per_node \
    --max_restarts=0 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --no-python \
    python3 lead/training/train.py
