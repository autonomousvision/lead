#!/bin/bash

# Run from the repo root, regardless of the invoking directory.
cd "$(dirname "$(realpath "${BASH_SOURCE:-$0}")")/../.."

# Root for generated outputs, from .env
export LEAD_OUTPUT_DIR_ROOT=$(dotenv LEAD_OUTPUT_DIR_ROOT)

# Checkpoints
export CHECKPOINT_DIR=$LEAD_OUTPUT_DIR_ROOT/checkpoints/noradar_resnet34
export ROUTES=src/lead/routes/benchmark_routes/bench2drive/23687.xml

# Set environment variables
export BENCHMARK_ROUTE_ID=$(basename $ROUTES .xml) # Last part of the route file name, e.g., 0 for 0.xml
export EVALUATION_OUTPUT_DIR=$LEAD_OUTPUT_DIR_ROOT/local_evaluation/$BENCHMARK_ROUTE_ID/
export PYTHONPATH=3rd_party/leaderboard/bench2drive/leaderboard:$PYTHONPATH
export PYTHONPATH=3rd_party/leaderboard/bench2drive/scenario_runner:$PYTHONPATH
export SCENARIO_RUNNER_ROOT=3rd_party/leaderboard/bench2drive/scenario_runner
export LEADERBOARD_ROOT=3rd_party/leaderboard/bench2drive/leaderboard
export IS_BENCH2DRIVE=1
export PLANNER_TYPE=only_traj
export SAVE_PATH=$EVALUATION_OUTPUT_DIR/
export PYTHONUNBUFFERED=1

set -x
set +e

# Recreate output folders
rm -rf $EVALUATION_OUTPUT_DIR/
mkdir -p $EVALUATION_OUTPUT_DIR

# Reset CARLA World
reset_carla_world

CUDA_VISIBLE_DEVICES=0 python3 3rd_party/leaderboard/bench2drive/leaderboard/leaderboard/leaderboard_evaluator.py \
    --routes=$ROUTES \
    --track=SENSORS \
    --checkpoint=$EVALUATION_OUTPUT_DIR/checkpoint_endpoint.json \
    --agent=src/lead/evaluation/agents/transfuser/transfuser_agent.py \
    --agent-config=$CHECKPOINT_DIR \
    --debug=0 \
    --record=None \
    --resume=False \
    --port=2000 \
    --traffic-manager-port=8000 \
    --timeout=60 \
    --traffic-manager-seed=0 \
    --repetitions=1
