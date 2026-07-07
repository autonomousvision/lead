#!/bin/bash
# P2.2 (full run): intent head + intent-conditioned control head + differentiable
# collision cost. Full 31-epoch training to see the real effect.
#   - batch 256 (per-GPU 32 on 8 GPUs) for high utilization
#   - lr 6e-4  (sqrt-scaled for the 4x batch vs the base recipe's batch 64 / lr 3e-4)
#   - loads the P1 checkpoint (backbone + intent head); control head starts fresh
# Usage: CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/train_control_p2full_ddp.sh

export OMP_NUM_THREADS=$(nproc)
export OPENBLAS_NUM_THREADS=1
export NCCL_P2P_DISABLE=1
export NCCL_P2P_LEVEL=NVL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nproc_per_node=$(python -c "import torch; print(torch.cuda.device_count())")
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((10000 + RANDOM % 50000))

export LEAD_TRAINING_CONFIG="logdir=outputs/local_training/control_p2_full \
load_file=outputs/local_training/intent_p1/model_0030.pth \
use_intent_decoder=true \
use_planning_decoder=true \
use_control_conditioning=true \
use_collision_cost=true \
batch_size=128 \
lr=3e-4 \
epochs=31"

torchrun --standalone \
    --nnodes=1 \
    --nproc_per_node=$nproc_per_node \
    --max_restarts=0 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --no-python \
    python3 lead/training/train.py
