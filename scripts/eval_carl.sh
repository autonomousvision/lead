#!/bin/bash

python3 scripts/reset_carla_world.py

# Produce videos
export LEAD_CLOSED_LOOP_CONFIG="produce_demo_video=true"

# Checkpoints and route
export CHECKPOINT_DIR=outputs/checkpoints/CaRL
export ROUTES=data/benchmark_routes/bench2drive/24240.xml

# Environment
export BENCHMARK_ROUTE_ID="$(basename "${ROUTES}" .xml)"
export EVALUATION_OUTPUT_DIR="outputs/local_evaluation/${BENCHMARK_ROUTE_ID}"
export CARLA_ROOT=3rd_party/CARLA_0915
export SCENARIO_RUNNER_ROOT="3rd_party/scenario_runner"
export LEADERBOARD_ROOT="3rd_party/leaderboard"
export PYTHONPATH=3rd_party/leaderboard:$PYTHONPATH
export PYTHONPATH=3rd_party/scenario_runner:$PYTHONPATH
export PYTHONPATH=${CARLA_ROOT}/PythonAPI/carla:${SCENARIO_RUNNER_ROOT}:${LEADERBOARD_ROOT}

# Agent runtime env vars
export SAVE_PATH="${EVALUATION_OUTPUT_DIR}/"
export DEBUG_ENV_AGENT=1
export RECORD=0
export SAVE_PNG=0
export UPSCALE_FACTOR=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HIGH_FREQ_INFERENCE=0
export SAMPLE_TYPE=mean
export CPP=0
export CPP_PORT=5555
export CARLA_RPC_PORT=${CARLA_RPC_PORT:-2000}
export CARLA_TM_PORT=${CARLA_TM_PORT:-8000}

set -x
set -e

# Recreate output folder
rm -rf "${EVALUATION_OUTPUT_DIR}"
mkdir -p "${EVALUATION_OUTPUT_DIR}" "${SAVE_PATH}"

python "${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py" \
  --routes="${ROUTES}" \
  --track=MAP \
  --checkpoint="${EVALUATION_OUTPUT_DIR}/checkpoint_endpoint.json" \
  --agent="lead/carl/carl_agent.py" \
  --agent-config="${CHECKPOINT_DIR}" \
  --debug=0 \
  --record= \
  --resume=1 \
  --port="${CARLA_RPC_PORT}" \
  --traffic-manager-port="${CARLA_TM_PORT}" \
  --timeout=900 \
  --traffic-manager-seed=0 \
  --repetitions=1
