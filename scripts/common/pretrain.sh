#!/bin/bash

# Run from the repo root, regardless of the invoking directory.
cd "$(dirname "$(realpath "${BASH_SOURCE:-$0}")")/../.."

# Root for generated outputs, from .env
export LEAD_OUTPUT_DIR_ROOT=$(dotenv LEAD_OUTPUT_DIR_ROOT)

# One data loader worker per CPU core already, so the numeric libraries must
# not start threads of their own.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# Keep numba from spawning too many threads, once per data loader worker.
export NUMBA_NUM_THREADS=1 NUMBA_THREADING_LAYER=workqueue
export NCCL_P2P_DISABLE=1 # https://github.com/huggingface/accelerate/issues/314
export NCCL_P2P_LEVEL=NVL # https://github.com/huggingface/accelerate/issues/314
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Beartype and Dynamo cannot run together, so the checks are off here to let
# torch.compile take the model.
export LEAD_RUNTIME_TYPE_CHECKING=false

# Autotuning searches for kernels a long while before the first step, which
# pays off over a full training run; remove it for a faster training start-up.
# Cudagraphs stay disabled: with DDP graph splitting they crash on the second
# step (stale cudagraph outputs).
python3 src/lead/training/train.py \
	training.experiment.output_dir=$LEAD_OUTPUT_DIR_ROOT/local_training/pretrain \
	training.data.read_from_cache_store=true \
	training.optimization.torch_compile_mode=max-autotune
