#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=1-00:00
#SBATCH --cpus-per-task=1
#SBATCH --partition=cpu-galvani
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=long.nguyen@student.uni-tuebingen.de
#SBATCH --mem=4G

shopt -s globstar

# Set up interpreter
eval "$(conda shell.bash hook)"
if [ -z "$CONDA_INTERPRETER" ]; then
    export CONDA_INTERPRETER="lead" # Check if CONDA_INTERPRETER is not set, then set it to lead
fi
source activate "$CONDA_INTERPRETER"
which python3

set -e

# Generate for each split a evaluation script
set -x
python3 $LEAD_PROJECT_ROOT/slurm/evaluation/evaluate_scripts_generator.py \
--base_checkpoint_endpoint $EVALUATION_OUTPUT_DIR \
--route_folder $LEAD_PROJECT_ROOT/data/benchmark_routes/$EVALUATION_DATASET \
--team_agent lead/expert/expert.py $SCRIPT_GENERATOR_PARAMETERS \
--track MAP \
--expert

# Run the evaluation scripts parallely.
python3 $LEAD_PROJECT_ROOT/slurm/evaluation/evaluate.py \
--slurm_dir $EVALUATION_OUTPUT_DIR/scripts \
--job_name $EXPERIMENT_RUN_ID $EVALUATION_PARAMETERS
