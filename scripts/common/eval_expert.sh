#!/bin/bash
cd "$(dirname "$(realpath "${BASH_SOURCE:-$0}")")/../.."

_routes=src/lead/routes/data_routes/leaderboard1/BlockedIntersection/Town06_13.xml
export LEAD_LOG_LEVEL="DEBUG"

_scenario_name=$(basename $(dirname $_routes))
_route_number=$(basename $_routes .xml)
_checkpoint_endpoint=data/expert_debug/results/${_route_number}_result.json
export PYTHONPATH=3rd_party/CARLA/standard_0915/PythonAPI/carla:$PYTHONPATH
export PYTHONPATH=3rd_party/leaderboard/expert/leaderboard:$PYTHONPATH
export PYTHONPATH=3rd_party/leaderboard/expert/scenario_runner:$PYTHONPATH
export DATAGEN=1
export SAVE_PATH=data/expert_debug/data/$_scenario_name

rm -rf data/expert_debug/buckets
rm -rf data/expert_debug/data
rm -rf data/expert_debug/results

reset_carla_world

python -u 3rd_party/leaderboard/expert/leaderboard/leaderboard/leaderboard_evaluator_local.py \
    --port=2000 \
    --traffic-manager-port=8000 \
    --traffic-manager-seed=0 \
    --routes=$_routes \
    --repetitions=1 \
    --track=MAP \
    --checkpoint=${_checkpoint_endpoint} \
    --agent=src/lead/expert/expert_agent.py \
    --agent-config=$_routes \
    --debug=0 \
    --resume=1 \
    --timeout=600
