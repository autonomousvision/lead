#!/bin/bash
#SBATCH --job-name=tf_010_0
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=3-00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --output=/mnt/lustre/work/geiger/bjaeger25/garage_2_cleanup/results/logs/tfpp_010_0_%a_%A.out  # File to which STDOUT will be written
#SBATCH --error=/mnt/lustre/work/geiger/bjaeger25/garage_2_cleanup/results/logs/tfpp_010_0_%a_%A.err   # File to which STDERR will be written
#SBATCH --partition=a100-galvani

# IMPORTANT: Start this script from within team_code folder, otherwise it will not work

# print info about current job
scontrol show job $SLURM_JOB_ID

pwd
export CARLA_ROOT=/mnt/lustre/work/geiger/bjaeger25/CARLA_0_9_15
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":${PYTHONPATH}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mnt/lustre/work/geiger/bjaeger25/miniconda3/lib

export OMP_NUM_THREADS=32  # Limits pytorch to spawn at most num cpus cores threads
export OPENBLAS_NUM_THREADS=1  # Shuts off numpy multithreading, to avoid threads spawning other threads.
torchrun --nnodes=1 --nproc_per_node=4 --max_restarts=0 --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d \
    train.py --id tfpp_010_0 --use_disk_cache 1 --crop_image 1 --seed 0 --epochs 31 --batch_size 16 --lr 3e-4 --setting all \
    --root_dir /mnt/lustre/work/geiger/bjaeger25/garage_2_cleanup/results/data/garage_v1_2024_11_07/data \
    --logdir /mnt/lustre/work/geiger/bjaeger25/garage_2_cleanup/results \
    --use_controller_input_prediction 1 --continue_epoch 0 --cpu_cores 32 --num_repetitions 1 --use_cosine_schedule 1 --cosine_t0 1 \
    --image_architecture regnety_032 --lidar_architecture regnety_032