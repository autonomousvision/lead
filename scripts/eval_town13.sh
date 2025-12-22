#!/bin/bash

# Checkpoints
export BENCHMARK_ROUTE_ID=0
export CHECKPOINT_DIR=outputs/checkpoints/tfv6_resnet34/
export ROUTES=data/benchmark_routes/Town13/$BENCHMARK_ROUTE_ID.xml

# Set environment variables
export EVALUATION_OUTPUT_DIR=outputs/local_evaluation/$BENCHMARK_ROUTE_ID/
export PYTHONPATH=3rd_party/leaderboard:$PYTHONPATH
export PYTHONPATH=3rd_party/scenario_runner:$PYTHONPATH
export SCENARIO_RUNNER_ROOT=3rd_party/scenario_runner
export LEADERBOARD_ROOT=3rd_party/leaderboard
export IS_BENCH2DRIVE=0
export PLANNER_TYPE=only_traj
export SAVE_PATH=$EVALUATION_OUTPUT_DIR/
export PYTHONUNBUFFERED=1

set -x
set +e

# Recreate output folders
rm -rf $EVALUATION_OUTPUT_DIR/
mkdir -p $EVALUATION_OUTPUT_DIR

CUDA_VISIBLE_DEVICES=0 python3 3rd_party/leaderboard/leaderboard/leaderboard_evaluator.py \
    --routes=$ROUTES \
    --track=SENSORS \
    --checkpoint=$EVALUATION_OUTPUT_DIR/checkpoint_endpoint.json \
    --agent=lead/inference/sensor_agent.py \
    --agent-config=$CHECKPOINT_DIR \
    --debug=0 \
    --record=None \
    --resume=False \
    --port=2000 \
    --traffic-manager-port=8000 \
    --timeout=120 \
    --debug-checkpoint=$EVALUATION_OUTPUT_DIR/debug_checkpoint/debug_checkpoint_endpoint.txt \
    --traffic-manager-seed=0 \
    --repetitions=1
