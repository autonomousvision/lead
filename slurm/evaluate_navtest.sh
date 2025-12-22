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

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
train_test_split=$TRAIN_TEST_SPLIT \
agent=carla_transfuser_agent \
worker=single_machine_thread_pool \
agent.checkpoint_path=$CHECKPOINT \
experiment_name=$EXPERIMENT_NAME \
output_dir=$EVALUATION_OUTPUT_DIR \
metric_cache_path=3rd_party/navsim_workspace/exp/metric_cache
