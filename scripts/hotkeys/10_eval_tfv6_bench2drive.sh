#!/bin/bash

# Run from the repo root, regardless of the invoking directory.
cd "$(dirname "$(realpath "${BASH_SOURCE:-$0}")")/../.."

reset_carla_world

export LEAD_CONFIG="evaluation.produce_demo_image=true \
    evaluation.produce_demo_video=true \
    evaluation.produce_debug_image=true \
    evaluation.produce_debug_video=true \
    evaluation.produce_input_image=true \
    evaluation.produce_input_video=true"

python -m lead \
    --checkpoint outputs/checkpoints/tfv6_regnety032 \
    --routes src/lead/routes/benchmark_routes/bench2drive/2143.xml \
    --bench2drive
