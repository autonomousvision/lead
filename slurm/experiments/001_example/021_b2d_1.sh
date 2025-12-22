#!/usr/bin/bash

source slurm/init.sh

export CHECKPOINT_DIR=outputs/training/001_example/011_postrain32_1/251025_182331
export LEAD_TRAINING_CONFIG="$LEAD_TRAINING_CONFIG"
export LEAD_CLOSED_LOOP_CONFIG="$LEAD_CLOSED_LOOP_CONFIG"

evaluate_bench2drive220
