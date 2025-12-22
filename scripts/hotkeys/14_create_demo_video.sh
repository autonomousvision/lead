#!/bin/bash

# Script to create demo videos from simulator outputs
# Layout: Front camera on top (full width), demo_1 and demo_2 side-by-side below
# [       front      ]
# [cinema][bev]

# Check if directory argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <path_to_simulator_outputs>"
    echo "Example: $0 outputs/demos/Town15_1/simulator_outputs"
    exit 1
fi

SIMULATOR_OUTPUT_DIR="$1"
DEMO1_DIR="${SIMULATOR_OUTPUT_DIR}/demo_1"
DEMO2_DIR="${SIMULATOR_OUTPUT_DIR}/demo_2"
INPUT_CAMERA_DIR="${SIMULATOR_OUTPUT_DIR}/input_camera"
OUTPUT_VIDEO="${SIMULATOR_OUTPUT_DIR}/combined_demo.mp4"

# Check if required directories exist
if [ ! -d "$DEMO1_DIR" ]; then
    echo "Error: demo_1 directory not found at $DEMO1_DIR"
    exit 1
fi

if [ ! -d "$DEMO2_DIR" ]; then
    echo "Error: demo_2 directory not found at $DEMO2_DIR"
    exit 1
fi

echo "Creating video at 20fps (this will be fast!)..."

# Create video with layout:
# - Left: demo_1
# - Right: demo_2
ffmpeg -framerate 20 -pattern_type glob -i "${DEMO1_DIR}/*.png" \
       -framerate 20 -pattern_type glob -i "${DEMO2_DIR}/*.png" \
       -filter_complex "[0:v][1:v]hstack=inputs=2[v]" \
       -map "[v]" \
       -c:v libx264 -pix_fmt yuv420p -crf 28 -preset medium \
       -y "$OUTPUT_VIDEO"

if [ $? -eq 0 ]; then
    echo "Video created successfully: $OUTPUT_VIDEO"
    echo "Done!"
else
    echo "Error creating video"
    exit 1
fi
