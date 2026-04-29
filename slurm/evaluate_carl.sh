shopt -s globstar
set -e

# Generate for each split a evaluation script
set -x
python3 slurm/evaluation/evaluate_scripts_generator.py \
--base_checkpoint_endpoint $EVALUATION_OUTPUT_DIR \
--route_folder data/benchmark_routes/$EVALUATION_DATASET \
--team_agent lead/carl/carl_agent.py \
--track MAP $SCRIPT_GENERATOR_PARAMETERS

# Run the evaluation scripts parallelly.
python3 slurm/evaluation/evaluate.py \
--slurm_dir $EVALUATION_OUTPUT_DIR/scripts \
--job_name $EXPERIMENT_RUN_ID $EVALUATION_PARAMETERS
