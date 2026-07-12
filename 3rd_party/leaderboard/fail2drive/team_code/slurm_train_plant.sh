#!/bin/bash
#SBATCH --job-name=plant_000_0
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=3-00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=36
#SBATCH --output=/mnt/lustre/work/geiger/bjaeger25/garage_2_cleanup/results/logs/plant_000_0_%a_%A.out  # File to which STDOUT will be written
#SBATCH --error=/mnt/lustre/work/geiger/bjaeger25/garage_2_cleanup/results/logs/plant_000_0_%a_%A.err   # File to which STDERR will be written
#SBATCH --partition=2080-galvani

# IMPORTANT: Start this script from within team_code folder, otherwise it will not work

# print info about current job
scontrol show job $SLURM_JOB_ID

pwd
export CARLA_ROOT=/mnt/lustre/work/geiger/bjaeger25/CARLA_0_9_15
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":${PYTHONPATH}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mnt/lustre/work/geiger/bjaeger25/miniconda3/lib

export OMP_NUM_THREADS=36  # Limits pytorch to spawn at most num cpus cores threads
export OPENBLAS_NUM_THREADS=1  # Shuts off numpy multithreading, to avoid threads spawning other threads.
torchrun --nnodes=1 --nproc_per_node=4 --max_restarts=1 --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d \
    train.py --id plant_000_0 --use_disk_cache 1 --seed 0 --epochs 47 --batch_size 512 --lr 1e-4 --setting all \
    --root_dir /mnt/qb/work/geiger/bjaeger25/data/garage_2_cleanup/results/data/garage_v0_2024_10_24/data \
    --logdir /mnt/lustre/work/geiger/bjaeger25/garage_2_cleanup/results \
    --use_controller_input_prediction 0 --use_wp_gru 1 --continue_epoch 0 --cpu_cores 36 --num_repetitions 1 \
    --schedule_reduce_epoch_01 45 --weight_decay 0.1 --use_grad_clip 1 --use_optim_groups 1 --use_plant 1