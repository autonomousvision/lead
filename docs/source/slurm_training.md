# Training

> **Note**: This is a completely optional feature. The SLURM integration is designed for users with access to HPC clusters who want to scale their experiments efficiently. All functionality can also be run locally without SLURM.

## Overview

We have a complete example pipeline for training and evaluating model at [slurm/experiments/001_example](https://github.com/autonomousvision/lead/tree/main/slurm/experiments/001_example).

## Pre-training

Start pre-training with

```bash
bash slurm/experiments/001_example/000_pretrain1_0.sh
```

This will create a pre-training session at `outputs/training/001_example/000_pretrain1_0/<year><month><day>_<hour><minute><second>`, setting the training seed as `0`.

## Post-training

After pre-training, we start the post-training with training seed set to `2`, as indicated by the last digit of the script

```bash
bash slurm/experiments/001_example/012_postrain32_2.sh
```

Its content explained:

```bash
#!/usr/bin/bash

source slurm/init.sh

export LEAD_TRAINING_CONFIG="$LEAD_TRAINING_CONFIG image_architecture=regnety_032 lidar_architecture=regnety_032"   # Same as in pre_training
export LEAD_TRAINING_CONFIG="$LEAD_TRAINING_CONFIG use_planning_decoder=true"                                       # Add a planner on top
posttrain outputs/training/001_example/000_pretrain1_0/251018_092144                                                # Specify where the pre-trained model is at

train --cpus-per-task=64 --partition=L40Sday --time=4-00:00:00 --gres=gpu:4
```

## Resume crashed training

As can be seen in [slurm/init.sh](https://github.com/autonomousvision/lead/blob/main/slurm/init.sh), when a training crashes, we can restart it easily by adding a simple line to  [slurm/experiments/001_example/012_postrain32_2.sh](https://github.com/autonomousvision/lead/blob/main/slurm/experiments/001_example/012_postrain32_2.sh)

```bash
#!/usr/bin/bash

source slurm/init.sh

export LEAD_TRAINING_CONFIG="$LEAD_TRAINING_CONFIG image_architecture=regnety_032 lidar_architecture=regnety_032"   # Same as in pre_training
export LEAD_TRAINING_CONFIG="$LEAD_TRAINING_CONFIG use_planning_decoder=true"                                       # Add a planner on top
posttrain outputs/training/001_example/000_pretrain1_0/251018_092144                                                # Specify where the pre-trained model is at
resume outputs/training/001_example/012_postrain32_2/251018_092144                                                  # Specify the training directory of the post-training session

train --cpus-per-task=64 --partition=L40Sday --time=4-00:00:00 --gres=gpu:4
```

Now, you can restart the training as many times as needed without any other change
