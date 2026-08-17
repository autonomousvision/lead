#!/bin/bash
cd "$(dirname "$(realpath "${BASH_SOURCE:-$0}")")/../.."

_lead_output_dir_root=$(dotenv LEAD_OUTPUT_DIR_ROOT)
_checkpoint_dir=$_lead_output_dir_root/checkpoints/tfv6_resnet34/
_routes=src/lead/routes/benchmark_routes/longest6/00.xml

export BENCHMARK_ROUTE_ID=$(basename $_routes .xml)
_evaluation_output_dir=$_lead_output_dir_root/local_evaluation/$BENCHMARK_ROUTE_ID/
export PYTHONPATH=3rd_party/leaderboard/standard/leaderboard:$PYTHONPATH
export PYTHONPATH=3rd_party/leaderboard/standard/scenario_runner:$PYTHONPATH
export SCENARIO_RUNNER_ROOT=3rd_party/leaderboard/standard/scenario_runner
export SAVE_PATH=$_evaluation_output_dir/
export PYTHONUNBUFFERED=1

set -x
set +e

rm -rf $_evaluation_output_dir/
mkdir -p $_evaluation_output_dir

reset_carla_world

CUDA_VISIBLE_DEVICES=0 python3 3rd_party/leaderboard/standard/leaderboard/leaderboard/leaderboard_evaluator.py \
    --routes=$_routes \
    --track=SENSORS \
    --checkpoint=$_evaluation_output_dir/checkpoint_endpoint.json \
    --agent=src/lead/evaluation/agents/transfuser/transfuser_agent.py \
    --agent-config=$_checkpoint_dir \
    --debug=0 \
    --record=None \
    --resume=False \
    --port=2000 \
    --traffic-manager-port=8000 \
    --timeout=60 \
    --traffic-manager-seed=0 \
    --repetitions=1
