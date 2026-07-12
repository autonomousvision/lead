#!/bin/bash

# Run from the repo root, regardless of the invoking directory.
cd "$(dirname "$(realpath "${BASH_SOURCE:-$0}")")/../.."

reset_carla_world

export LEAD_CONFIG="agent.transfuser.use_radar_detection=false evaluation.inference.strict_weight_load=false"

python -m lead \
    --checkpoint outputs/training/700_regnety_032/010_postrain32_0/250913_153308 \
    --routes src/lead/routes/benchmark_routes/longest6/00.xml
