#!/usr/bin/bash
# Slurm bootstrap for training/evaluation experiment scripts (scripts/slurm/experiments/**).
# Source it after any #SBATCH directives:
#   source scripts/slurm/slurm_experiment_init.sh
# Builds on scripts/slurm/slurm_init.sh: TRAINING_OUTPUT_DIR/EVALUATION_OUTPUT_DIR are just
# named aliases for its generic, path-mirrored OUTPUT_DIR (outputs/<script's own path>/<date>).

# The evaluate_* drivers below must keep running on the login node (they launch a screen
# session that orchestrates its own per-route sbatch jobs), not become a batch job themselves,
# so defer slurm_init.sh's automatic resubmit-on-source; `train` calls `submit` explicitly once
# it's ready instead.
LEAD_SLURM_MANUAL_SUBMIT=1
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/slurm_init.sh"
unset LEAD_SLURM_MANUAL_SUBMIT

############################# Init. Create variables when script gets sourced. #############################

# `train` resubmits this whole script to Slurm, which reruns it end-to-end on the compute node,
# so any variable built by appending to itself (export X="$X foo") must start clean each time,
# or it would accumulate a duplicate of everything appended before the resubmit.
unset LEAD_CONFIG

# Help to identify the run. Don't change this. Uses LEAD_SCRIPT (an absolute path resolved by
# slurm_init.sh before it cd'd to the repo root), not $0, so this keeps working even when $0 was
# a path relative to some other directory the script was invoked from.
EXPERIMENT_NAME=$(basename "$(dirname "$LEAD_SCRIPT")") # Directory name is the experiment name
export EXPERIMENT_NAME
export EXPERIMENT_RUN_ID=${EXPERIMENT_NAME}_${SCRIPT_NAME}_${SLURM_JOB_DATE} # Experiment ID

# if Experiment ID has more than 64 characters, error
if [ ${#EXPERIMENT_RUN_ID} -gt 64 ]; then
	echo "Experiment ID too long: ${EXPERIMENT_RUN_ID}"
	exit 1
fi

# Run's outputs will be directed too this directory.
export EVALUATION_OUTPUT_DIR="$OUTPUT_DIR"
export TRAINING_OUTPUT_DIR="$OUTPUT_DIR"
EXPERIMENT_SEED=$(basename "$LEAD_SCRIPT" ".sh" | awk -F'_' '{print $NF}') # Last part of the script name is the seed
export EXPERIMENT_SEED

echo "EXPERIMENT_RUN_ID: ${EXPERIMENT_RUN_ID}"
echo "TRAINING_OUTPUT_DIR: ${TRAINING_OUTPUT_DIR}"
echo "EVALUATION_OUTPUT_DIR: ${EVALUATION_OUTPUT_DIR}"

############################# Training #############################

# A function that creates environment variables for resuming training in-place from a previous
# run's output directory (same run identity, same directory keeps getting written to).
# Usage: resume <last_training_output_dir>
function resume() {
	LAST_TRAINING_OUTPUT_DIR=$1

	export TRAINING_OUTPUT_DIR="$LAST_TRAINING_OUTPUT_DIR"
	SLURM_JOB_DATE=$(basename "$LAST_TRAINING_OUTPUT_DIR") # The run's date, so IDs match the original run
	export SLURM_JOB_DATE
	export EXPERIMENT_RUN_ID=${EXPERIMENT_NAME}_${SCRIPT_NAME}_${SLURM_JOB_DATE}

	# Find the latest model checkpoint
	MODEL_FILE=$(find "$LAST_TRAINING_OUTPUT_DIR" -name "model_*.pth" | sort | tail -n 1)
	MODEL_EPOCH=$(basename "$MODEL_FILE" | grep -oP 'model_\K\d+' | awk '{print $1 + 0}')
	WANDB_ID=$EXPERIMENT_RUN_ID

	# Export training parameters
	export LEAD_CONFIG="$LEAD_CONFIG training.experiment.load_file=$MODEL_FILE"
	export LEAD_CONFIG="$LEAD_CONFIG training.experiment.continue_failed_training=true"
	export LEAD_CONFIG="$LEAD_CONFIG training.experiment.wandb_id=$WANDB_ID"
	export LEAD_CONFIG="$LEAD_CONFIG training.experiment.wandb_resume=allow"
	export LEAD_CONFIG="$LEAD_CONFIG training.experiment.logdir=$TRAINING_OUTPUT_DIR"

	# Output for confirmation
	echo "Resuming training with the following variables:"
	echo "EXPERIMENT_RUN_ID: $EXPERIMENT_RUN_ID"
	echo "TRAINING_OUTPUT_DIR: $TRAINING_OUTPUT_DIR"
	echo "MODEL_FILE: $MODEL_FILE"
	echo "CONTINUE_FAILED_TRAINING_AT_EPOCH: $MODEL_EPOCH"
	echo "WANDB_ID: $WANDB_ID"
	ls "$TRAINING_OUTPUT_DIR"
}

# A function that create environment variables for fine-tuning from the checkpoint file
# Usage: posttrain <checkpoint_file>
function posttrain() {
	model_file=$1
	if [[ "$model_file" == *.pth ]]; then
		# It ends with .pth, proceed as is
		:
	else
		ls "$model_file"
		# Does not end with .pth, assume it's a directory and find the largest model file
		model_file=$(find "$model_file" -name "model_*.pth" | sort -V | tail -n 1)
	fi

	# Export training parameters
	export LEAD_CONFIG="$LEAD_CONFIG training.experiment.load_file=$model_file"
	export LEAD_CONFIG="$LEAD_CONFIG training.experiment.continue_failed_training=false"

	# Output for confirmation
	echo "Fine-tuning with the following model file:"
	echo "MODEL_FILE: $model_file"
}

# Train job. Override LEAD_CONFIG to set up the parameters.
# On the submit side (no SLURM_JOB_ID, sbatch available), resubmits this same script to Slurm,
# carrying its #SBATCH directives, so it reruns end-to-end on a compute node. On the job side
# (or when sbatch isn't available at all), it runs the training body directly.
function train() {
	export LEAD_CONFIG="$LEAD_CONFIG training.experiment.logdir=$TRAINING_OUTPUT_DIR"
	export LEAD_CONFIG="$LEAD_CONFIG training.experiment.seed=$EXPERIMENT_SEED"
	export LEAD_CONFIG="$LEAD_CONFIG training.experiment.description=$EXPERIMENT_RUN_ID"
	export LEAD_CONFIG="$LEAD_CONFIG training.experiment.wandb_id=$EXPERIMENT_RUN_ID"
	export LEAD_CONFIG="$LEAD_CONFIG training.experiment.id=$EXPERIMENT_RUN_ID"
	echo "$TRAINING_OUTPUT_DIR"
	submit "$TRAINING_OUTPUT_DIR" "$EXPERIMENT_RUN_ID"

	if [[ -n "${SLURM_JOB_ID:-}" ]]; then
		bash "$LEAD_ROOT/scripts/slurm/train.sh"
	else
		# Local run without Slurm: still capture logs to a file.
		bash "$LEAD_ROOT/scripts/slurm/train.sh" \
			> >(tee "$TRAINING_OUTPUT_DIR/stdout_local.log") \
			2> >(tee "$TRAINING_OUTPUT_DIR/stderr_local.log" >&2)
	fi
}
############################# CARLA Evaluation #############################

# Evaluate on shorter routes of bench2drive
# Usage: evaluate <checkpoint_dir>
function evaluate() {
	if [[ -z "$EVALUATION_DATASET" ]]; then
		echo "Error: EVALUATION_DATASET is not set."
		exit 1
	fi
	mkdir -p "$EVALUATION_OUTPUT_DIR"
	ln -s $CHECKPOINT_DIR/model_0030.pth "$EVALUATION_OUTPUT_DIR/model_0030_0.pth"
	ln -s $CHECKPOINT_DIR/config.json "$EVALUATION_OUTPUT_DIR/config.json"
	export SCRIPT_GENERATOR_PARAMETERS="$SCRIPT_GENERATOR_PARAMETERS --checkpoint_endpoint $CHECKPOINT_DIR"
	export SCRIPT_GENERATOR_PARAMETERS="$SCRIPT_GENERATOR_PARAMETERS --team_config $CHECKPOINT_DIR"
	ls "$CHECKPOINT_DIR"
	echo "Starting evaluation $EXPERIMENT_RUN_ID"
	# Check if CHECKPOINT_DIR is set and model file exists
	if [[ -z "$CHECKPOINT_DIR" || (! -f "$CHECKPOINT_DIR/model_0030.pth") && (! -f "$CHECKPOINT_DIR/model_0030_0.pth") ]]; then
		echo "Error: CHECKPOINT_DIR is not set or $CHECKPOINT_DIR/model_0030.pth or $CHECKPOINT_DIR/model_0030_0.pth does not exist."
		exit 1
	fi
	echo "Evaluating $CHECKPOINT_DIR on $EVALUATION_DATASET"
	export EVALUATION_STDOUT=$EVALUATION_OUTPUT_DIR/stdout_${SLURM_JOB_DATE}.txt
	export EVALUATION_STDERR=$EVALUATION_OUTPUT_DIR/stderr_${SLURM_JOB_DATE}.txt
	echo "$EVALUATION_STDOUT"
	echo "$EVALUATION_STDERR"
	screen -dmS "$EXPERIMENT_RUN_ID" bash -c "scripts/slurm/evaluate.sh > $EVALUATION_STDOUT 2> $EVALUATION_STDERR"
}

# Evaluate on shorter routes of bench2drive
# Usage: evaluate_bench2drive <checkpoint_dir>
function evaluate_bench2drive() {
	# Set up default dataset for evaluation.
	export EVALUATION_DATASET=bench2drive
	export USE_PREEMPTABLE_PARTITION=1
	evaluate "$@"
}

# Evaluate on longer routes of Town13
# Longer timeout, privileged partition and higher number of repetitions
# Usage: evaluate_town13 <checkpoint_dir>
function evaluate_town13() {
	export EVALUATION_DATASET=Town13
	export SCRIPT_GENERATOR_PARAMETERS="$SCRIPT_GENERATOR_PARAMETERS --slurm_timeout 3-00:00:00"
	evaluate "$@"
}

# Evaluate on medium routes of longest6
# Longer timeout, privileged partition and higher number of repetitions
# Usage: evaluate_longest6 <checkpoint_dir>
function evaluate_longest6() {
	export EVALUATION_DATASET=longest6
	export SCRIPT_GENERATOR_PARAMETERS="$SCRIPT_GENERATOR_PARAMETERS --slurm_timeout 0-10:00:00"
	evaluate "$@"
}

# Evaluate on Fail2Drive benchmark (uses custom CARLA simulator with novel assets)
# Usage: evaluate_fail2drive <checkpoint_dir>
function evaluate_fail2drive() {
	export EVALUATION_DATASET=fail2drive
	export CARLA_ROOT=$LEAD_ROOT/3rd_party/CARLA/fail2drive_0915
	export SCRIPT_GENERATOR_PARAMETERS="$SCRIPT_GENERATOR_PARAMETERS --slurm_timeout 0-04:00:00"
	evaluate "$@"
}
