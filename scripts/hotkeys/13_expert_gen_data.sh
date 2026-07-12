#!/bin/bash

# Run from the repo root, regardless of the invoking directory.
cd "$(dirname "$(realpath "${BASH_SOURCE:-$0}")")/../.."

reset_carla_world

# Create log directory if it doesn't exist
rm -rf outputs/expert_evaluation/
mkdir -p outputs/expert_evaluation/logs
rm -rf $(dotenv PY123D_DATA_ROOT)/logs
mkdir -p $(dotenv PY123D_DATA_ROOT)/logs

python -u -m lead \
    --expert \
    --routes src/lead/routes/data_routes/50x38_Town12/CrossingBicycleFlow/830_2.xml \
    2>&1 | tee outputs/expert_evaluation/logs/out.log
