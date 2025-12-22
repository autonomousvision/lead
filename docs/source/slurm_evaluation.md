# Evaluation

> **Note**: This is a completely optional feature. The SLURM integration is designed for users with access to HPC clusters who want to scale their experiments efficiently. All functionality can also be run locally without SLURM.

## Overview

Again, we have a complete example pipeline for training and evaluating model at `slurm/experiments/001_example`.

We recommend training three different seeds and evaluate each seed once on Bench2Drive and Longest6 v2.

If you really want to be sure, each seed can be evaluated three times.

## Example

Take a look at `slurm/experiments/001_example/020_b2d_0.sh` and its content

```bash
#!/usr/bin/bash

source slurm/init.sh

export CHECKPOINT_DIR=outputs/training/001_example/010_postrain32_0/251025_182327  # Point to where the model at
export LEAD_TRAINING_CONFIG="$LEAD_TRAINING_CONFIG"                                # Override training parameter at test time, if needed (rarely)
export LEAD_CLOSED_LOOP_CONFIG="$LEAD_CLOSED_LOOP_CONFIG"                          # Override closed loop parameter at test time

evaluate_bench2drive220
```

Here we point the evaluation to the checkpoint directory. Then we have the option to override the training configuration at test time, which is rarely needed.
Then we have another option to change the closed loop configuration.

Finally, `evaluate_bench2drive220` starts a `screen` session on the node, which on its own will start SLURM job for evaluate each route individually.

Live parameters of the evaluations can be found at `slurm/configs`.
