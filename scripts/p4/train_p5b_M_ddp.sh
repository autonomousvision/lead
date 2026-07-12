#!/bin/bash
# P5b-M: finetune backbone+planner on top of a FROZEN MULTIMODAL VLM intent head
# (P5a-Tversky). Starts from the pretrained backbone (same as P1/P2 lineage), trains
# on the ~96k VLM-cache subset. Pair with train_p5b_S_ddp.sh (single-mode P4a head)
# for the clean S-vs-M closed-loop comparison.
# Usage: CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/p4/train_p5b_M_ddp.sh

source /mmu_mllm_hdd_3/liuzihan08/miniconda3/etc/profile.d/conda.sh
conda activate lead

# Node has no internet; timm's resnet34 backbone is cached locally. Force offline.
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

export LEAD_TRAINING_CONFIG="logdir=outputs/local_training/p5b_M \
load_file=outputs/local_training/pretrain/model_0030.pth \
image_encoder_pretrained=false \
use_vlm_intent=true \
vlm_intent_ckpt=outputs/local_training/vlm_intent_p5a_tversky/model_0014.pth \
vlm_cache_dir=data/p5/vlm_cache \
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
