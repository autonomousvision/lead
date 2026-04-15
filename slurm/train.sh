#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=a100-galvani
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=long.nguyen@student.uni-tuebingen.de
#SBATCH --mem=256gb

# Initialize Conda
set -e

which python
which python3

# print info about current job
if [ -n "$SLURM_JOB_ID" ]; then
	scontrol show job $SLURM_JOB_ID
fi

pwd
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":${PYTHONPATH}
export WANDB__SERVICE_WAIT=300
export PYTHONUNBUFFERED=1

# CUDA debug
nvidia-smi

if which sbatch >/dev/null; then
	export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
else
	export OMP_NUM_THREADS=$(nproc)
fi
export OPENBLAS_NUM_THREADS=1 # Shuts off numpy multithreading, to avoid threads spawning other threads.

export NCCL_P2P_DISABLE=1 # https://github.com/huggingface/accelerate/issues/314
export NCCL_P2P_LEVEL=NVL # https://github.com/huggingface/accelerate/issues/314
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nproc_per_node=$(python -c "import torch; print(torch.cuda.device_count())")
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((10000 + RANDOM % 50000))
echo "Using $nproc_per_node GPUs per node. MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "LEAD_TRAINING_CONFIG: $LEAD_TRAINING_CONFIG"
if [ "$nproc_per_node" -le 1 ]; then
    python lead/training/train.py
else
	torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node --max_restarts=0 --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d lead/training/train.py
fi
