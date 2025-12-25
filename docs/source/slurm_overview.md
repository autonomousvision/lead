# Overview

> **Note**: This is a completely optional feature. The SLURM integration is designed for users with access to HPC clusters who want to scale their experiments efficiently. All functionality can also be run locally without SLURM.

Accompanying LEAD is a minimal SLURM-wrapper that we found useful for our research. This wrapper is however highly opinionated so we keep it optional.

## Why a wrapper?

Several needs motivate our system. Most importantly, we want to have minimal mental overhead when:
- run experiments, evaluations, restart multiple times for multiple seeds.
- remembering names of an experiment. This wrapper gives unified names for: SLURM Jobs, WandB experiments, output directories, etc.
- run multiple trainings and evaluations in parallel.
- run experiments on multiple clusters on parallel where partition names are different.

## Principle of the wrapper

Each experiment (pre-training, post-training, evaluation, etc.) corresponds to an individual bash script.

## Example

Say we want to start a pre-training from scratch, as in [slurm/experiments/001_example/000_pretrain1_0.sh](https://github.com/autonomousvision/lead/blob/main/slurm/experiments/001_example/000_pretrain1_0.sh)

```bash
 #!/usr/bin/bash

source slurm/init.sh

export LEAD_TRAINING_CONFIG="$LEAD_TRAINING_CONFIG image_architecture=regnety_032 lidar_architecture=regnety_032"

train --cpus-per-task=32 --partition=a100-galvani --time=3-00:00:00 --gres=gpu:4
```

The script is named with following convention `slurm/experiments/<number_id>_<experiment_name>/<another_number_id>_<experiment_step>_<seed>.sh`.

The first line runs [slurm/init.sh](https://github.com/autonomousvision/lead/blob/main/slurm/init.sh), which in turns create environment variables and define bash functions.

The second line defines environment variables for the Python training script. The same `image_architecture` and `lidar_architecture` options can be found in [lead/training/config_training.py](https://github.com/autonomousvision/lead/blob/main/lead/training/config_training.py).

The third line start the training, after defined the SLURM parameters. The function `train` can be found in [slurm/init.sh](https://github.com/autonomousvision/lead/blob/main/slurm/init.sh).

Simply by running this scripts, you can start a training which has an output directory at

```bash
outputs/training/001_example/000_pretrain1_0/<year><month><day>_<hour><minute><second>
```
