#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1080ti:1
#SBATCH --cpus-per-task=2
#SBATCH --partition=day
#SBATCH --mem=64gb

shopt -s globstar
set -e

# Set up interpreter
eval "$(../miniconda3/bin/conda shell.bash hook)"
if [ -z "$CONDA_INTERPRETER" ]; then
    export CONDA_INTERPRETER="lead"
fi
source activate "$CONDA_INTERPRETER"
which python3


# Set environment variables
export PYTHONPATH=3rd_party/leaderboard:$PYTHONPATH
export PYTHONPATH=3rd_party/scenario_runner:$PYTHONPATH
export SCENARIO_RUNNER_ROOT=3rd_party/scenario_runner
export LEADERBOARD_ROOT=3rd_party/leaderboard
export IS_BENCH2DRIVE=0
export PLANNER_TYPE=only_traj
export SAVE_PATH=$EVALUATION_OUTPUT_DIR/
export PYTHONUNBUFFERED=1

export PORT=2000
export TM_PORT=8000
if ! nvidia-smi | grep -q "CarlaUE4-Linux-Shipping"; then
    export PORT=$(91_random_free_port.sh)
    export TM_PORT=$(91_random_free_port.sh)
    bash $CARLA_ROOT/CarlaUE4.sh --world-port=$PORT -nosound -graphicsadapter=0 -RenderOffScreen &
    sleep 180
    nvidia-smi
fi

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
    --port=$PORT \
    --traffic-manager-port=$TM_PORT \
    --timeout=120 \
    --debug-checkpoint=$EVALUATION_OUTPUT_DIR/debug_checkpoint/debug_checkpoint_endpoint.txt \
    --traffic-manager-seed=0 \
    --repetitions=1
