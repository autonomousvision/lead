#!/bin/bash

# Run from the repo root, regardless of the invoking directory.
cd "$(dirname "$(realpath "${BASH_SOURCE:-$0}")")/../.."

cd 3rd_party/carla_route_generator

# Call the Python script with all arguments passed to this shell script
python3 scripts/window.py "$@"
