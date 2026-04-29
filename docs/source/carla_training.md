# Training

This guide covers local model training for CARLA Leaderboard.

```{note}
This tutorial is specific to training policies for CARLA Leaderboard. For cross-dataset training (NAVSIM, Waymo), see [Cross-dataset training](cross_dataset_training.md).
For large-scale training on HPC clusters, refer to the [SLURM Training Guide](slurm_training.md).
```

## Prerequisites

The following steps need to be performed once per dataset.

### Prepare Data

This step assumes you either

1. Fellow the [data collection tutorial](data_collection.md) to collect data locally

2. Or downloaded the data from  [Hugging Face](https://huggingface.co/datasets/ln2697/lead).

Eitherway, your data structure should match:

```{code-block}
:emphasize-lines: 2,3,4

data/carla_leaderboard2
├── data
│   └── BlockedIntersection
│       └── 999_Rep-1_Town06_13_route0_12_22_22_34_45
```

### Build Data Buckets

Buckets group training samples by characteristics (scenarios, towns, weather, road curvature, etc.) to enable curriculum learning and balanced batch sampling.

Default bucket collections:
- **Pre-training**: [lead/data_buckets/full_pretrain_bucket_collection.py](https://github.com/kesai-labs/lead/blob/main/lead/data_buckets/full_pretrain_bucket_collection.py) samples uniformly across all available data
- **Post-training**: [lead/data_buckets/full_posttrain_bucket_collection.py](https://github.com/kesai-labs/lead/blob/main/lead/data_buckets/full_posttrain_bucket_collection.py) filters out the initial and final samples of each sequence (which may contain initialization artifacts) and samples uniformly from the remaining data

Buckets are built once, stored on disk, and automatically reused in subsequent runs.

Build pre-training buckets:

```bash
python3 scripts/build_buckets_pretrain.py
```

Build post-training buckets:

```bash
python3 scripts/build_buckets_posttrain.py
```

Expected output structure:

```{code-block}
:emphasize-lines: 2,3,4

data/carla_leaderboard2
├── buckets
│   ├── full_posttrain_buckets_8_8_8_5.gz
│   └── full_pretrain_buckets.gz
├── data
│   └── BlockedIntersection
│       └── 999_Rep-1_Town06_13_route0_12_22_22_34_45
```

Bucket files use relative paths and are portable across machines.

```{note}
Buckets contain only the data samples available at build time. Rebuild buckets after adding or removing routes.
```

```{note}
If you encounter hard-to-debug errors after major code changes, try deleting and rebuilding the buckets.
```

### Build Persistent Data Cache

Raw sensor data (images, LiDAR, RADAR) requires preprocessing—decompression, format conversion, and temporal alignment. The training cache stores preprocessed, compressed data on disk, eliminating redundant computation and accelerating data loading. Once built, the cache is reused across all training runs.

Two cache types:

- **`persistent_cache`**: Stored alongside the dataset, reused across all training sessions. See [PersistentCache](https://github.com/kesai-labs/lead/blob/main/lead/data_loader/training_cache.py)
- **`training_session_cache`**: Temporary cache on local SSD during cluster jobs, implemented with [diskcache](https://pypi.org/project/diskcache/). During the first few epochs, data is loaded from shared storage and cached on the job's local SSD for faster subsequent access. This implementation is specific to our organization's SLURM cluster setup.

Build the persistent cache:

```bash
python3 scripts/build_cache.py
```

Expected output structure:

```{code-block}
:emphasize-lines: 5,6,7

data/carla_leaderboard2
├── buckets
│   ├── full_posttrain_buckets_8_8_8_5.gz
│   └── full_pretrain_buckets.gz
├── cache
│   └── BlockedIntersection
│       └── 999_Rep-1_Town06_13_route0_12_22_22_34_45
├── data
│   └── BlockedIntersection
│       └── 999_Rep-1_Town06_13_route0_12_22_22_34_45
```

Training session cache is loaded in the first few epochs of the training from shared disk and stored on training session's disk. This implementation is specific to SLURM clusters of our organization.

```{note}
Rebuild the cache after modifying the data pipeline (e.g., adding new semantic classes or changing preprocessing steps).
```

## Perception Pre-training

Following standard TransFuser-like training procedures, training occurs in two phases: first, train only the perception backbone, then train the complete model end-to-end.

```bash
python3 lead/training/train.py logdir=outputs/local_training/pretrain
```

Training takes approximately 1-2 minutes if trained with only one route and produces:

```
outputs/local_training/pretrain
├── config.json
├── events.out.tfevents.1764250874.local.105366.0
├── gradient_steps_skipped_0030.txt
├── model_0030.pth
├── optimizer_0030.pth
├── scaler_0030.pth
└── scheduler_0030.pth
```

The training script generates WandB/TensorBoard logs and visualization images at `outputs/training_viz`. Control logging frequency with `log_scalars_frequency` and `log_images_frequency` in [lead/training/config_training.py](https://github.com/kesai-labs/lead/blob/main/lead/training/config_training.py).

Image logging runs at least once per epoch and can be expensive. Disable it by setting `visualize_training=false` in the config.

View training logs with [TensorBoard](https://www.tensorflow.org/tensorboard):

```bash
tensorboard --logdir outputs/local_training/pretrain
```

[WandB](https://wandb.ai) logging is also supported. Enable it by setting `log_wandb=true` in the training config.

![](../assets/wandb.png)

## Planning Post-training

```{note}
During post-training, the epoch count resets to 0. The optimizer state is reinitialized because the planner was not included in the pre-training checkpoint.
```

After pre-training completes, continue with post-training to add the planner and train the complete model end-to-end:

```bash
python3 lead/training/train.py logdir=outputs/local_training/posttrain load_file=outputs/local_training/pretrain/model_0030.pth use_planning_decoder=true
```

## Resume Failed Training

To continue from a failed training run, set `continue_failed_training=true` in the training config.

## Distributed Training

The pipeline supports [Torch DDP](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html). See example scripts:
- [scripts/pretrain_ddp.sh](https://github.com/kesai-labs/lead/blob/main/scripts/pretrain_ddp.sh)
- [scripts/posttrain_ddp.sh](https://github.com/kesai-labs/lead/blob/main/scripts/posttrain_ddp.sh)

## Common Issues

### CARLA Server Running in Background

This error may occur:

```bash
RuntimeError: cuDNN error: CUDNN_STATUS_INTERNAL_ERROR
```

This typically indicates a CARLA server is consuming VRAM in the background. Kill all CARLA processes:

```bash
bash scripts/clean_carla.sh
```

### Unknown CARLA Root Path: \<carla_root>

Register your CARLA dataset configuration in the `target_dataset` function in [lead/training/config_training.py](https://github.com/kesai-labs/lead/blob/main/lead/training/config_training.py). Map your CARLA root path to the appropriate dataset configuration, which specifies the expected sensor setup and data format.

**Why this design?** This allows running training experiments while simultaneously collecting data with different sensor configurations. The `target_dataset` in [lead/expert/config_expert.py](https://github.com/kesai-labs/lead/blob/main/lead/expert/config_expert.py) controls which sensors are mounted on the expert vehicle during data collection.
