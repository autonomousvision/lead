#!/usr/bin/bash

source scripts/slurm/slurm_experiment_init.sh

export CHECKPOINT_DIR=outputs/training/001_example/011_postrain32_1/251025_182331
export LEAD_CONFIG="$LEAD_CONFIG"

evaluate_longest6
