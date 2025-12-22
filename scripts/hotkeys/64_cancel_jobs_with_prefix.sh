#!/bin/bash

# Check if an argument is provided
if [ -z "$1" ]; then
	echo "Error: No argument provided. Please provide a job name or pattern."
	exit 1
fi

# Find and cancel jobs
squeue -u $USER --format="%.18i %.50j %.8u %.2t %.10M %.6D %.10R" | grep "$1" | awk '{print $1}' | xargs -r scancel
