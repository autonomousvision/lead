#!/usr/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=3-00:00:00
#SBATCH --mem=256gb

source scripts/slurm/slurm_experiment_init.sh

# Partition defaults to SLURM_PARTITION from .env; override here since pretraining wants a100s.
export SLURM_PARTITION=a100-galvani

export LEAD_CONFIG="$LEAD_CONFIG policy.transfuser.image_architecture=regnety_032 policy.transfuser.lidar_architecture=regnety_032"

train
