#!/usr/bin/bash

source slurm/init.sh

export CHECKPOINT_DIR=outputs/training/001_example/012_postrain32_2/251025_182334
export LEAD_TRAINING_CONFIG="$LEAD_TRAINING_CONFIG"
export LEAD_CLOSED_LOOP_CONFIG="$LEAD_CLOSED_LOOP_CONFIG custom_weather=ClearSunrise"

evaluate_longest6
