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
set -e

# Set up interpreter
eval "$(../miniconda3/bin/conda shell.bash hook)"
if [ -z "$CONDA_INTERPRETER" ]; then
    export CONDA_INTERPRETER="lead" # Check if CONDA_INTERPRETER is not set, then set it to lead
fi
source activate "$CONDA_INTERPRETER"
which python3

# Generate for each split a evaluation script
set -x
python3 scripts/tools/evaluation/evaluate_scripts_generator_expert.py \
--base_checkpoint_endpoint $EVALUATION_OUTPUT_DIR \
--route_folder data/routes/$EVALUATION_DATASET \
--team_agent data_collection/data_agent_0915.py $SCRIPT_GENERATOR_PARAMETERS

# Run the evaluation scripts parallely.
python3 scripts/tools/evaluation/evaluate.py \
--slurm_dir $EVALUATION_OUTPUT_DIR/scripts \
--job_name $EXPERIMENT_RUN_ID $EVALUATION_PARAMETERS

evaluation_exit_code=$?
if [ $evaluation_exit_code -ne 0 ]; then
    python3 scripts/tools/evaluation/send_alert.py --title "Evaluation of $EXPERIMENT_RUN_ID failed" --message "The evaluation of $EXPERIMENT_RUN_ID failed with exit code $evaluation_exit_code."
fi
