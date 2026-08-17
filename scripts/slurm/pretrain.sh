#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --mem=400gb

cd "$(dirname "$(realpath "${BASH_SOURCE:-$0}")")/../.."

export LEAD_OUTPUT_DIR_ROOT=$(dotenv LEAD_OUTPUT_DIR_ROOT)

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 NUMBA_THREADING_LAYER=workqueue
export NCCL_P2P_DISABLE=1
export NCCL_P2P_LEVEL=NVL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LEAD_RUNTIME_TYPE_CHECKING=false

srun --kill-on-bad-exit=1 python3 src/lead/training/train.py \
	training.experiment.output_dir=$LEAD_OUTPUT_DIR_ROOT/local_training/pretrain \
	training.data.read_from_cache_store=true \
	training.optimization.torch_compile_mode=max-autotune
