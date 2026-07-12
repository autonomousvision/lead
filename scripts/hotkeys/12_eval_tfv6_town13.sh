#!/bin/bash

# Run from the repo root, regardless of the invoking directory.
cd "$(dirname "$(realpath "${BASH_SOURCE:-$0}")")/../.."

reset_carla_world

python -m lead \
    --checkpoint outputs/checkpoints/tfv6_resnet34 \
    --routes src/lead/routes/benchmark_routes/Town13/0.xml \
    --timeout 180
