#!/usr/bin/bash
# Training body for a single Slurm job. Invoked by the `train` function in
# scripts/slurm/slurm_experiment_init.sh — do not sbatch this file directly.
set -e

which python
which python3

pwd
export WANDB__SERVICE_WAIT=300
export PYTHONUNBUFFERED=1

# CUDA debug
nvidia-smi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
	export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
else
	export OMP_NUM_THREADS=$(nproc)
fi
export OPENBLAS_NUM_THREADS=1 # Shuts off numpy multithreading, to avoid threads spawning other threads.

export NCCL_P2P_DISABLE=1 # https://github.com/huggingface/accelerate/issues/314
export NCCL_P2P_LEVEL=NVL # https://github.com/huggingface/accelerate/issues/314
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "LEAD_CONFIG: $LEAD_CONFIG"

# Lightning's DDPStrategy spawns one worker per visible GPU itself; no torchrun needed.
python src/lead/training/train.py
