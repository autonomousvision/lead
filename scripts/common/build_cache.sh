#!/usr/bin/bash

# Build the policy's training cache store for the configured split: every
# cacheable builder, both views. Forwards config overrides verbatim, e.g.:
#   build_cache training.data.py123d_split=normal_view

# There is already one worker process per core, so stop the libraries from
# starting threads of their own.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
# Keep numba from spawning too many threads; it ignores the settings above.
export NUMBA_NUM_THREADS=1 NUMBA_THREADING_LAYER=workqueue

python3 -m lead.training.build_cache training.data.force_cache_rebuild="${FORCE_CACHE_REBUILD:-true}" "$@"
