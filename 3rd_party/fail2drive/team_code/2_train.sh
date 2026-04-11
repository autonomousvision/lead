#!/bin/bash

export CARLA_ROOT=/mnt/bernhard/carla_0_9_15
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":${PYTHONPATH}

export OMP_NUM_THREADS=8  # Limits pytorch to spawn at most num cpus cores threads
export OPENBLAS_NUM_THREADS=1  # Shuts off numpy multithreading, to avoid threads spawning other threads.
CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --nproc_per_node=1 --max_restarts=1 --rdzv_id=1234 --rdzv_backend=c10d \
    train.py --id tfpp_001_0 --crop_image 1 --seed 0 --epochs 31 --batch_size 32 --lr 3e-4 --setting 13_withheld \
    --root_dir /mnt/bernhard/data/garage_v0_2024_10_22/data \
    --logdir /mnt/bernhard/code/garage_2_cleanup/results \
    --use_controller_input_prediction 1 --use_wp_gru 0 --use_discrete_command 1 --use_tp 1 --tp_attention 0 --continue_epoch 1 --cpu_cores 8 --num_repetitions 1 \
    --image_architecture hf-hub:apple/mobileclip_s0_timm --lidar_architecture hf-hub:apple/mobileclip_s0_timm --cosine_t0 2
