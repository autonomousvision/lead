# Training

This guide covers training models on CARLA data for CARLA Leaderboard.

> **Note**: For training on cross-datasets (NavSim, Waymo), see [Cross-Dataset Training](cross_dataset_training.md).

## Prerequisites

Each of the following steps has to be done only once.

### 1. Prepare Data

We will upload a dataset soon. Stay tuned! In the mean time, follow this [tutorial](https://ln2697.github.io/lead/docs/data_collection.html) to collect data.

If data collected locally, run

```bash
cp -r data/expert_debug data/carla_leaderboard2
```

### 2. Build Data Buckets

Buckets group training samples by characteristics (e.g., scenarios, towns, weather, scenarios, road curvature, etc.) to enable curriculum learning and balanced batch sampling.

By default we use [full_pretrain_bucket_collection](https://github.com/autonomousvision/lead/blob/main/lead/training/data_loader/buckets/full_pretrain_bucket_collection.py) for pre-training and [full_posttrain_bucket_collection](https://github.com/autonomousvision/lead/blob/main/lead/training/data_loader/buckets/full_posttrain_bucket_collection.py) for post-training, e.g., we train uniformly on all samples.

Buckets are built once and stored on disk in the dataset directory. In subsequents runs they are reused automatically. This is neccessary to save time.

To build pretrain bucket, run

```bash
python3 scripts/build_buckets_pretrain.py
```

To build posttrain bucket, run

```bash
python3 scripts/build_buckets_posttrain.py
```

If everything is ok, this should be the output

```html
data/carla_leaderboard2
├── buckets
│   ├── full_posttrain_buckets_8_8_8_5.gz
│   └── full_pretrain_buckets.gz
├── data
│   └── BlockedIntersection
│       └── 999_Rep-1_Town06_13_route0_12_22_22_34_45
└── results
    └── Town06_13_result.json
```

The bucket files can be used on other computers since file paths in each bucket is relative.

**Note:** A bucket contains all and only the paths of data samples that are available at bucket building time. If you later add or delete routes, you need to rebuild the buckets.

### 3. Build Persistent Data Cache

Raw sensor data (images, LiDAR, RADAR, etc.) requires significant preprocessing before training - decompression, format conversion, and perturbation alignment. The training cache stores preprocessed and compressed data to disk, eliminating redundant computation and dramatically speeding up data loading. Once built, the cache is reused across training runs, reducing the data loading bottleneck.

Two types of cache are used:
- **`persistent_cache`**: Stored alongside the dataset, reused across all training sessions. See implementation at [PersistentCache](https://github.com/autonomousvision/lead/blob/main/lead/training/data_loader/carla_dataset_utils.py).
- **`training_session_cache`**: Temporary cache on local SSD of a cluster job. We use [diskcache](https://pypi.org/project/diskcache/) for this purpose.

To build cache, run

```bash
python3 scripts/build_cache.py
```

If everything is ok, this should be the output

```html
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
└── results
    └── Town06_13_result.json
```

**Note:** After changing something in pipeline (e.g., add new semantic class), you might need to check whether the cache needs to be rebuilt.

**Note:** After building data cache, the pipeline only needs the meta files in `data/carla_leaderboard2/data`, everything else can be deleted.

## Model Training

Following standard procedures on CARLA, we train the model in two phases, first only the perception backbone is trained, only after that we train everything jointly..

### 1. Perception pre-training

```bash
bash scripts/pretrain.sh
```

The training will takes around 1-2 minutes and produces following structure

```html
outputs/local_training/pretrain
├── clipper_0030.pth
├── config.json
├── events.out.tfevents.1764250874.local.105366.0
├── gradient_steps_skipped_0030.txt
├── model_0030.pth
├── optimizer_0030.pth
├── scaler_0030.pth
└── scheduler_0030.pth
```

To debug training, the script also regulary produces WandB/TensorBoard logs and images at `outputs/training_viz`. The frequency can be controlled with `log_scalars_frequency` and `log_images_frequency`.

The image logging could be quite expensive, it runs at least once per epoch. To turn if off completely, set `visualize_training=false` in training config.

To observe the training logging with TensorBoard, run

```bash
tensorboard --logdir outputs/local_training/pretrain
```

We also support WandB, to turn it on, set `log_wandb=true` in training config.

### 2. Post-training

> **Note**: The epoch count will be reset back to 0.

After pre-training, we continue with the post-training where we put the planner on top of the model
and train the whole model end-to-end.

```bash
bash scripts/posttrain.sh
```

### [Optional] Resume failed training

To continue a failed training, set `continue_failed_training=true`.

### [Optional] Distributed Training

The pipeline supports [Torch DDP](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html). An example:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 --max_restarts=0 --rdzv_backend=c10d python3 lead/training/train.py
```

## Common issues

### CARLA server running

A common error might happen with following error message
```bash
RuntimeError: cuDNN error: CUDNN_STATUS_INTERNAL_ERROR
```

It might come from CARLA server running in background and eating vram. To kill CARLA, run

```bash
bash scripts/clean_carla.sh
```
