#!/bin/bash
# P5b-S: single-mode baseline. Same recipe as train_p5b_M_ddp.sh but the frozen intent
# head is the SINGLE-MODE P4a decoder (command-conditioned, data/p4/vlm_cache). Paired
# with M so the only difference is single-mode vs multimodal intent -> clean S-vs-M.
# Usage: CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/p4/train_p5b_S_ddp.sh

source /mmu_mllm_hdd_3/liuzihan08/miniconda3/etc/profile.d/conda.sh
conda activate lead

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export OMP_NUM_THREADS=$(nproc)
export OPENBLAS_NUM_THREADS=1
export NCCL_P2P_DISABLE=1
export NCCL_P2P_LEVEL=NVL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nproc_per_node=$(python -c "import torch; print(torch.cuda.device_count())")
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((10000 + RANDOM % 50000))

export LEAD_TRAINING_CONFIG="logdir=outputs/local_training/p5b_S \
load_file=outputs/local_training/pretrain/model_0030.pth \
image_encoder_pretrained=false \
use_vlm_intent=true \
vlm_intent_ckpt=outputs/local_training/vlm_intent_p4a/model_0014.pth \
vlm_cache_dir=data/p4/vlm_cache \
vlm_manifest=data/p4/manifest.jsonl \
use_intent_decoder=false \
use_planning_decoder=true \
use_control_conditioning=true \
use_collision_cost=true \
batch_size=128 \
lr=3e-4 \
epochs=20"

torchrun --standalone \
    --nnodes=1 \
    --nproc_per_node=$nproc_per_node \
    --max_restarts=0 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --no-python \
    python3 lead/training/train.py
