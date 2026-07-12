#!/usr/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --time=4-00:00:00
#SBATCH --mem=256gb

source scripts/slurm/slurm_experiment_init.sh

# Partition defaults to SLURM_PARTITION from .env; override here since posttraining wants L40S.
export SLURM_PARTITION=L40Sday

export LEAD_CONFIG="$LEAD_CONFIG agent.transfuser.image_architecture=regnety_032 agent.transfuser.lidar_architecture=regnety_032"
export LEAD_CONFIG="$LEAD_CONFIG agent.transfuser.use_planning_decoder=true"
posttrain outputs/training/001_example/000_pretrain1_0/251018_092144

train
