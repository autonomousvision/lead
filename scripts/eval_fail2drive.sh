#!/bin/bash

# Checkpoints
export CHECKPOINT_DIR=outputs/checkpoints/tfv6_resnet34
export ROUTES=data/benchmark_routes/fail2drive/Base_Animals_0075.xml

# Set environment variables
export BENCHMARK_ROUTE_ID=$(basename $ROUTES .xml)
export EVALUATION_OUTPUT_DIR=outputs/local_evaluation/$BENCHMARK_ROUTE_ID/
export PYTHONPATH=3rd_party/fail2drive/leaderboard:$PYTHONPATH
export PYTHONPATH=3rd_party/fail2drive/scenario_runner:$PYTHONPATH
export SCENARIO_RUNNER_ROOT=3rd_party/fail2drive/scenario_runner
export LEADERBOARD_ROOT=3rd_party/fail2drive/leaderboard
export IS_BENCH2DRIVE=0
export PLANNER_TYPE=only_traj
export SAVE_PATH=$EVALUATION_OUTPUT_DIR/
export PYTHONUNBUFFERED=1

set -x
set +e

# Recreate output folders
rm -rf $EVALUATION_OUTPUT_DIR/
mkdir -p $EVALUATION_OUTPUT_DIR

# Reset CARLA World
python3 scripts/reset_carla_world.py

CUDA_VISIBLE_DEVICES=0 python3 3rd_party/fail2drive/leaderboard/leaderboard/leaderboard_evaluator.py \
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
    --timeout=60 \
    --debug-checkpoint=$EVALUATION_OUTPUT_DIR/debug_checkpoint/debug_checkpoint_endpoint.txt \
    --traffic-manager-seed=0 \
    --repetitions=1
