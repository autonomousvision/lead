#!/usr/bin/bash

source slurm/init.sh

export LEAD_TRAINING_CONFIG="$LEAD_TRAINING_CONFIG model_type=plant"

train --cpus-per-task=64 --partition=L40Sday --time=01:00:00 --gres=gpu:1
