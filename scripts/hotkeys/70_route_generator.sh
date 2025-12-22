#!/bin/bash

cd 3rd_party/carla_route_generator

# Call the Python script with all arguments passed to this shell script
python3 scripts/window.py "$@"
