#!/usr/bin/env bash
# Evaluation body launched (via screen) by the `evaluate` function in
# scripts/slurm/slurm_experiment_init.sh. Conda is already active in the inherited environment —
# slurm_experiment_init.sh activates it before `evaluate` spawns this script's screen session.
set -e
shopt -s globstar

which python3

# Generate for each split a evaluation script
set -x
python3 scripts/slurm/evaluation/evaluate_scripts_generator.py \
--base_checkpoint_endpoint $EVALUATION_OUTPUT_DIR \
--route_folder src/lead/routes/benchmark_routes/$EVALUATION_DATASET \
--team_agent src/lead/evaluation/transfuser_agent.py $SCRIPT_GENERATOR_PARAMETERS

# Run the evaluation scripts parallelly.
python3 scripts/slurm/evaluation/evaluate.py \
--slurm_dir $EVALUATION_OUTPUT_DIR/scripts \
--job_name $EXPERIMENT_RUN_ID $EVALUATION_PARAMETERS
