#!/usr/bin/bash

source scripts/slurm/slurm_experiment_init.sh

export CHECKPOINT_DIR=outputs/training/001_example/010_postrain32_0/251025_182327
export LEAD_CONFIG="$LEAD_CONFIG"

evaluate_fail2drive
