#!/usr/bin/bash

source slurm/init.sh

export LEAD_TRAINING_CONFIG="$LEAD_TRAINING_CONFIG image_architecture=regnety_032 lidar_architecture=regnety_032"
export LEAD_TRAINING_CONFIG="$LEAD_TRAINING_CONFIG use_planning_decoder=true"
posttrain outputs/training/001_example/000_pretrain1_0/251018_092144

train --cpus-per-task=64 --partition=L40Sday --time=4-00:00:00 --gres=gpu:4
