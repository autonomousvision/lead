#!/usr/bin/bash

source scripts/slurm/slurm_experiment_init.sh

export CHECKPOINT_DIR=outputs/training/001_example/012_postrain32_2/251025_182334
export LEAD_CONFIG="$LEAD_CONFIG"

evaluate_bench2drive
