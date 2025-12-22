 #!/usr/bin/bash

source slurm/init.sh

export LEAD_TRAINING_CONFIG="$LEAD_TRAINING_CONFIG image_architecture=regnety_032 lidar_architecture=regnety_032"

train --cpus-per-task=32 --partition=a100-galvani --time=3-00:00:00 --gres=gpu:4
