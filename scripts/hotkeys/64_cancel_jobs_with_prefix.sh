#!/bin/bash

# Check if an argument is provided
if [ -z "$1" ]; then
	echo "Error: No argument provided. Please provide a job name or pattern."
	exit 1
fi

# Find and cancel SLURM jobs
echo "Canceling SLURM jobs matching '$1'..."
squeue -u $USER --format="%.18i %.50j %.8u %.2t %.10M %.6D %.10R" | grep "$1" | awk '{print $1}' | xargs -r scancel

# Find and kill screen sessions
echo "Killing screen sessions matching '$1'..."
screen -ls | grep "$1" | awk '{print $1}' | cut -d. -f1 | xargs -r -I {} screen -X -S {} quit

echo "Done."
