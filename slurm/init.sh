#!/usr/bin/bash

set -e
shopt -s globstar

# Help to identify the run. Don't change this.
EXPERIMENT_NAME=$(basename "$(dirname "$(realpath "$0")")") # Directory name is the experiment name
export EXPERIMENT_NAME
SCRIPT_NAME=$(basename "$0" ".sh") # Script name
export SCRIPT_NAME
SLURM_JOB_DATE=$(date +"%y%m%d_%H%M%S") # Current date
export SLURM_JOB_DATE
export EXPERIMENT_RUN_ID=${EXPERIMENT_NAME}_${SCRIPT_NAME}_${SLURM_JOB_DATE}  # Experiment ID
export EXPERIMENT_RUN_DIR=${EXPERIMENT_NAME}/${SCRIPT_NAME}/${SLURM_JOB_DATE} # Experiment directory

# if Experiment ID has more than 64 characters, error
if [ ${#EXPERIMENT_RUN_ID} -gt 64 ]; then
	echo "Experiment ID too long: ${EXPERIMENT_RUN_ID}"
	exit 1
fi

# Run's outputs will be directed too this directory.
export EVALUATION_OUTPUT_DIR=outputs/evaluation/$EXPERIMENT_RUN_DIR
export TRAINING_OUTPUT_DIR=training/$EXPERIMENT_RUN_DIR
EXPERIMENT_SEED=$(basename "$0" ".sh" | awk -F'_' '{print $NF}') # Last part of the script name is the seed
export EXPERIMENT_SEED

# A function that create environment variables for resuming training from the checkpoint directory
# Usage: resume <checkpoint_dir>
function resume() {
	LAST_TRAINING_OUTPUT_DIR=$1

	# Extract parts of the path
	EXPERIMENT_NAME=$(basename "$(dirname "$(dirname "$LAST_TRAINING_OUTPUT_DIR")")") # Top-level directory as experiment name
	export EXPERIMENT_NAME
	SCRIPT_NAME=$(basename "$(dirname "$LAST_TRAINING_OUTPUT_DIR")") # Second-level directory as script name
	export SCRIPT_NAME
	SLURM_JOB_DATE=$(basename "$LAST_TRAINING_OUTPUT_DIR") # Final directory as the date
	export SLURM_JOB_DATE

	# Construct necessary variables
	export EXPERIMENT_RUN_ID=${EXPERIMENT_NAME}_${SCRIPT_NAME}_${SLURM_JOB_DATE}
	export EXPERIMENT_RUN_DIR=${EXPERIMENT_NAME}/${SCRIPT_NAME}/${SLURM_JOB_DATE}
	export TRAINING_OUTPUT_DIR=training/$EXPERIMENT_RUN_DIR

	# Find the latest model checkpoint
	MODEL_FILE=$(find "$LAST_TRAINING_OUTPUT_DIR" -name "model_*.pth" | sort | tail -n 1)
	MODEL_EPOCH=$(basename "$MODEL_FILE" | grep -oP 'model_\K\d+' | awk '{print $1 + 0}')
	WANDB_ID=${EXPERIMENT_NAME}_${SCRIPT_NAME}_${SLURM_JOB_DATE}

	# Export training parameters
	export LEAD_TRAINING_CONFIG="$LEAD_TRAINING_CONFIG load_file=$MODEL_FILE"
	export LEAD_TRAINING_CONFIG="$LEAD_TRAINING_CONFIG continue_epoch=true"
	export LEAD_TRAINING_CONFIG="$LEAD_TRAINING_CONFIG wandb_id=$WANDB_ID"
	export LEAD_TRAINING_CONFIG="$LEAD_TRAINING_CONFIG wandb_resume=allow"
	export LEAD_TRAINING_CONFIG="$LEAD_TRAINING_CONFIG logdir=$TRAINING_OUTPUT_DIR"

	# Output for confirmation
	echo "Resuming training with the following variables:"
	echo "EXPERIMENT_RUN_ID: $EXPERIMENT_RUN_ID"
	echo "TRAINING_OUTPUT_DIR: $TRAINING_OUTPUT_DIR"
	echo "MODEL_FILE: $MODEL_FILE"
	echo "CONTINUE_EPOCH: $MODEL_EPOCH"
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
	export LEAD_TRAINING_CONFIG="$LEAD_TRAINING_CONFIG load_file=$model_file"
	export LEAD_TRAINING_CONFIG="$LEAD_TRAINING_CONFIG continue_epoch=false"

	# Output for confirmation
	echo "Fine-tuning with the following model file:"
	echo "MODEL_FILE: $model_file"
}

# Train job
# Override LEAD_TRAINING_CONFIG to set up the parameters
function train() {
	mkdir -p "$TRAINING_OUTPUT_DIR"
	export LEAD_TRAINING_CONFIG="$LEAD_TRAINING_CONFIG logdir=$TRAINING_OUTPUT_DIR"
	export LEAD_TRAINING_CONFIG="$LEAD_TRAINING_CONFIG seed=$EXPERIMENT_SEED"
	export LEAD_TRAINING_CONFIG="$LEAD_TRAINING_CONFIG description=$EXPERIMENT_RUN_ID"
	export LEAD_TRAINING_CONFIG="$LEAD_TRAINING_CONFIG wandb_id=$EXPERIMENT_RUN_ID"
	export LEAD_TRAINING_CONFIG="$LEAD_TRAINING_CONFIG id=$EXPERIMENT_RUN_ID"
	# Submit the job
	echo "$TRAINING_OUTPUT_DIR"
	if [[ -z "$SLURM_JOB_ID" && $(which sbatch) ]]; then
		output_file=$TRAINING_OUTPUT_DIR/stdout_${SLURM_JOB_DATE}.txt
		error_file=$TRAINING_OUTPUT_DIR/stderr_${SLURM_JOB_DATE}.txt
		echo "${output_file}"
		echo "${error_file}"

		sbatch --output "${output_file}" "--error" "${error_file}" "--job-name" "${EXPERIMENT_RUN_ID}" "$@" slurm/train.sh
	else
		bash slurm/train.sh
	fi
}

# Evaluate on shorter routes of bench2drive220
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
	screen -dmS "$EXPERIMENT_RUN_ID" bash -c "slurm/evaluate.sh > $EVALUATION_STDOUT 2> $EVALUATION_STDERR"
}

# Evaluate on shorter routes of bench2drive220
# Usage: evaluate_bench2drive220 <checkpoint_dir>
function evaluate_bench2drive220() {
	# Set up default dataset for evaluation.
	export EVALUATION_DATASET=bench2drive220
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

# Evaluate on navtest
function evaluate_navtest() {
	sbatch --job-name $EXPERIMENT_RUN_ID --output $EVALUATION_OUTPUT_DIR/stdout.txt --error $EVALUATION_OUTPUT_DIR/stderr.txt $@ slurm/evaluate_navtest.sh
}

# Evaluate on warmup_two_stage
function evaluate_warmup() {
	sbatch --job-name $EXPERIMENT_RUN_ID --output $EVALUATION_OUTPUT_DIR/stdout.txt --error $EVALUATION_OUTPUT_DIR/stderr.txt $@ slurm/evaluate_warmup_two_stage.sh
}

# Evaluate on navhard_two_stage
function evaluate_navhard() {
	sbatch --job-name $EXPERIMENT_RUN_ID --output $EVALUATION_OUTPUT_DIR/stdout.txt --error $EVALUATION_OUTPUT_DIR/stderr.txt $@ slurm/evaluate_navhard_two_stage.sh
}
