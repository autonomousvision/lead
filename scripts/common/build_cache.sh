#!/usr/bin/bash

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 NUMBA_THREADING_LAYER=workqueue

python3 -m lead.training.build_cache training.data.force_cache_rebuild=true "$@"
