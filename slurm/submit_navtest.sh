#!/usr/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=long.nguyen@student.uni-tuebingen.de
#SBATCH --mem=32gb

TRAIN_TEST_SPLIT=navtest

export NAVSIM_DEVKIT_ROOT="${LEAD_PROJECT_ROOT}/3rd_party/navsim_workspace/navsimv1.1"
export HYDRA_FULL_ERROR=1


eval "$(../miniconda3/bin/conda shell.bash hook)"
if [ -z "$CONDA_INTERPRETER" ]; then
    export CONDA_INTERPRETER="navsimv1.1" # Check if CONDA_INTERPRETER is not set, then set it to navsimv1.1
fi
source activate "$CONDA_INTERPRETER"
which python

export TEAM_NAME="TFv6"
export AUTHORS="TFv6"
export EMAIL="TFv6"
export INSTITUTION="TFv6"
export COUNTRY="TFv6"

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_create_submission_pickle.py \
train_test_split=$TRAIN_TEST_SPLIT \
agent=carla_transfuser_agent \
agent.checkpoint_path=$CHECKPOINT \
experiment_name=$EXPERIMENT_NAME \
team_name=$TEAM_NAME \
output_dir=$EVALUATION_OUTPUT_DIR \
authors=$AUTHORS \
email=$EMAIL \
institution=$INSTITUTION \
country=$COUNTRY \
sensor_blobs_path=3rd_party/navsim_workspace/dataset/sensor_blobs/test
