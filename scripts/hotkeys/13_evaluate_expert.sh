#!/bin/bash
export SCENARIO_RUNNER_ROOT=3rd_party/scenario_runner_autopilot
export LEADERBOARD_ROOT=3rd_party/leaderboard_autopilot

# Enter data here
export ROUTE_DIR=data/routes/
export SCENARIO_NAME=paper
export ROUTE_NUMBER=example1 # 20920 #2606 # 1825 #1833
export EXPERT_CONFIG="eval_expert=true"

rm -rf data/carla_today/results/data/garage_v2_2025_06_23/data
rm  data/carla_today/results/data/garage_v2_2025_06_23/visualization/**/*.jpeg
rm -f data/carla_today/3rd_person/*

# carla
export PYTHONPATH=3rd_party/CARLA_0916/PythonAPI/carla:$PYTHONPATH
export PYTHONPATH=3rd_party/leaderboard_autopilot:$PYTHONPATH
export PYTHONPATH=3rd_party/scenario_runner_autopilot:$PYTHONPATH
export REPETITIONS=1
export DEBUG_CHALLENGE=0
export DEBUG_CHALLENGE_PDM_LITE=0
export TEAM_AGENT=data_collection/data_agent_0915.py
export CHALLENGE_TRACK_CODENAME=MAP
export ROUTES=$ROUTE_DIR/$SCENARIO_NAME/$ROUTE_NUMBER.xml
export TOWN=Town12
export REPETITION=0
export TM_SEED=0

export CHECKPOINT_ENDPOINT=data/carla_today/results/data/garage_v2_2025_06_23/results/$SCENARIO_NAME/$ROUTE_NUMBER_result.json
export TEAM_CONFIG=$ROUTES
export RESUME=1
export DATAGEN=1
export SAVE_PATH=data/carla_today/results/data/garage_v2_2025_06_23/data/$SCENARIO_NAME

echo "Start python"

export FREE_STREAMING_PORT=2001
export FREE_WORLD_PORT=2000
export TM_PORT=8000

echo "FREE_STREAMING_PORT: $FREE_STREAMING_PORT"
echo "FREE_WORLD_PORT: $FREE_WORLD_PORT"
echo "TM_PORT: $TM_PORT"

eval "$(../miniconda3/bin/conda shell.bash hook)"
if [ -z "$CONDA_INTERPRETER" ]; then
    export CONDA_INTERPRETER="lead" # Check if CONDA_INTERPRETER is not set, then set it to lead
fi
source activate "$CONDA_INTERPRETER"
which python3

rm $CHECKPOINT_ENDPOINT

python 3rd_party/leaderboard/leaderboard/leaderboard_evaluator_local.py --port=${FREE_WORLD_PORT}         --traffic-manager-port=${TM_PORT} --traffic-manager-seed=${TM_SEED} --routes=${ROUTES} --repetitions=${REPETITIONS}             --track=${CHALLENGE_TRACK_CODENAME} --checkpoint=${CHECKPOINT_ENDPOINT} --agent=${TEAM_AGENT}                 --agent-config=${TEAM_CONFIG} --debug=0 --resume=${RESUME} --timeout=600

# python3 scripts/tools/data/carla_leaderboard2/008_visualize_local_new_pdm_lite.py
