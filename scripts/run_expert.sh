#!/bin/bash

# Function to cleanup on exit
cleanup() {
    echo "Caught signal, cleaning up..."
    if [ ! -z "$PYTHON_PID" ]; then
        kill -9 $PYTHON_PID 2>/dev/null
        pkill -9 -P $PYTHON_PID 2>/dev/null
    fi
    pkill -9 -f "leaderboard_evaluator_local.py" 2>/dev/null
    exit 130
}

# Trap SIGINT and SIGTERM
trap cleanup INT TERM

# Enter data here
export ROUTE_DIR=data/data_routes/leaderboard1/
export SCENARIO_NAME=BlockedIntersection
export ROUTE_NUMBER=Town06_13

# Set standard environment variables
export PYTHONPATH=3rd_party/CARLA_0915/PythonAPI/carla:$PYTHONPATH
export PYTHONPATH=3rd_party/leaderboard_autopilot:$PYTHONPATH
export PYTHONPATH=3rd_party/scenario_runner_autopilot:$PYTHONPATH
export DEBUG_CHALLENGE=0
export ROUTES=$ROUTE_DIR/$SCENARIO_NAME/$ROUTE_NUMBER.xml
export TEAM_CONFIG=$ROUTES
export DATAGEN=1

# Set paths for saving data and results
export SAVE_PATH=data/expert_debug/data/$SCENARIO_NAME
export CHECKPOINT_ENDPOINT=data/expert_debug/results/${ROUTE_NUMBER}_result.json

# Clean previous data
rm -rf data/expert_debug/buckets
rm -rf data/expert_debug/data
rm -rf data/expert_debug/results

# Start the evaluation
python -u 3rd_party/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py \
    --port=2000 \
    --traffic-manager-port=8000 \
    --traffic-manager-seed=0 \
    --routes=$ROUTES \
    --repetitions=1 \
    --track=MAP \
    --checkpoint=${CHECKPOINT_ENDPOINT} \
    --agent=lead/expert/expert.py \
    --agent-config=$ROUTES \
    --debug=0 \
    --resume=1 \
    --timeout=600 &

PYTHON_PID=$!
wait $PYTHON_PID
