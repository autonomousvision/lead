#!/bin/bash

# Run from the repo root, regardless of the invoking directory.
cd "$(dirname "$(realpath "${BASH_SOURCE:-$0}")")/../.."

# Enter data here
export ROUTES=src/lead/routes/data_routes/leaderboard1/BlockedIntersection/Town06_13.xml
export LEAD_LOG_LEVEL="DEBUG"

# Set standard environment variables
export SCENARIO_NAME=$(basename $(dirname $ROUTES))
export ROUTE_NUMBER=$(basename $ROUTES .xml)
export PYTHONPATH=3rd_party/CARLA/standard_0915/PythonAPI/carla:$PYTHONPATH
export PYTHONPATH=3rd_party/leaderboard/expert/leaderboard:$PYTHONPATH
export PYTHONPATH=3rd_party/leaderboard/expert/scenario_runner:$PYTHONPATH
export DEBUG_CHALLENGE=0
export TEAM_CONFIG=$ROUTES
export DATAGEN=1

# Set paths for saving data and results
export SAVE_PATH=data/expert_debug/data/$SCENARIO_NAME
export CHECKPOINT_ENDPOINT=data/expert_debug/results/${ROUTE_NUMBER}_result.json

# Clean previous data
rm -rf data/expert_debug/buckets
rm -rf data/expert_debug/data
rm -rf data/expert_debug/results

# Reset CARLA World
reset_carla_world

# Start the evaluation
python -u 3rd_party/leaderboard/expert/leaderboard/leaderboard/leaderboard_evaluator_local.py \
    --port=2000 \
    --traffic-manager-port=8000 \
    --traffic-manager-seed=0 \
    --routes=$ROUTES \
    --repetitions=1 \
    --track=MAP \
    --checkpoint=${CHECKPOINT_ENDPOINT} \
    --agent=src/lead/expert/expert_agent.py \
    --agent-config=$ROUTES \
    --debug=0 \
    --resume=1 \
    --timeout=600 &

PYTHON_PID=$!
wait $PYTHON_PID
