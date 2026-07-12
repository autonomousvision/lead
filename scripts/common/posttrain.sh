#!/bin/bash

# Run from the repo root, regardless of the invoking directory.
cd "$(dirname "$(realpath "${BASH_SOURCE:-$0}")")/../.."

export OMP_NUM_THREADS=$(nproc)
export OPENBLAS_NUM_THREADS=1 # Shuts off numpy multithreading, to avoid threads spawning other threads.
export NCCL_P2P_DISABLE=1 # https://github.com/huggingface/accelerate/issues/314
export NCCL_P2P_LEVEL=NVL # https://github.com/huggingface/accelerate/issues/314
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export LEAD_CONFIG="training.experiment.logdir=outputs/local_training/posttrain \
training.experiment.load_file=outputs/local_training/pretrain/model_0030.pth \
agent.transfuser.use_planning_decoder=true"

# Lightning's DDPStrategy spawns one worker per visible GPU itself; no torchrun needed.
python3 src/lead/training/train.py
