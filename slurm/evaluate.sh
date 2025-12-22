shopt -s globstar
set -e

# Set up interpreter
eval "$(../miniconda3/bin/conda shell.bash hook)"
if [ -z "$CONDA_INTERPRETER" ]; then
    export CONDA_INTERPRETER="lead"
fi
source activate "$CONDA_INTERPRETER"
which python3

# Generate for each split a evaluation script
set -x
python3 slurm/evaluation/evaluate_scripts_generator.py \
--base_checkpoint_endpoint $EVALUATION_OUTPUT_DIR \
--route_folder data/routes/$EVALUATION_DATASET \
--team_agent lead/inference/sensor_agent.py $SCRIPT_GENERATOR_PARAMETERS

# Run the evaluation scripts parallely.
python3 slurm/evaluation/evaluate.py \
--slurm_dir $EVALUATION_OUTPUT_DIR/scripts \
--job_name $EXPERIMENT_RUN_ID $EVALUATION_PARAMETERS
