# Training

TransFuser trains in two phases: perception first, then planning on top of the
pretrained weights.

## Build the cache

Cacheable part outputs (lidar raster, planning and detection targets) are
precomputed once, for both views. Sensors are never cached; they read from the logs
in milliseconds but would dominate storage.

```console
user@host:~/lead$ bash scripts/common/build_cache.sh
```

Config overrides are forwarded verbatim:

```console
user@host:~/lead$ bash scripts/common/build_cache.sh training.data.force_cache_rebuild=true
```

Training never builds the cache. Enabling `read_from_cache_store` without a built
store fails at startup.

The store's manifest also carries a `cache_finger_print`: the config values that
decide what a cached tensor holds (BEV geometry, sensor preprocessing, label
knobs). Building or reading a store whose fingerprint no longer matches the
current config fails the same way, instead of silently serving samples built
under a different config — rebuild it with `force_cache_rebuild=true`.

## Pretrain

Perception only: with `use_planning_decoder=false` the planning losses are zeroed.

```console
user@host:~/lead$ bash scripts/common/pretrain.sh
```

Or without the wrapper script:

```console
user@host:~/lead$ python -m lead.training.train \
      training.data.read_from_cache_store=true \
      training.experiment.output_dir=/path/to/outputs/pretrain
```

## Post-train

Same data, planning decoder enabled, starting from the pretrained weights:

```console
user@host:~/lead$ bash scripts/common/posttrain.sh
```

```console
user@host:~/lead$ python -m lead.training.train \
      training.data.read_from_cache_store=true \
      policy.transfuser.use_planning_decoder=true \
      training.experiment.initial_weights_file=/path/to/outputs/pretrain/model_0030.pth \
      training.experiment.output_dir=/path/to/outputs/posttrain
```

`initial_weights_file` also loads the `config.yaml` stored next to it, so the run
inherits the pretraining config; your own overrides still win.

## Override the config

Any field of the tree, as `section.path.key=value`. Highest priority first: CLI
arguments, the `LEAD_CONFIG` environment dotlist, the checkpoint's `config.yaml`,
class defaults.

```console
user@host:~/lead$ python -m lead.training.train training.optimization.batch_size=32 debug_mode=true   # on the command line
user@host:~/lead$ export LEAD_CONFIG="training.optimization.learning_rate=1e-4 training.experiment.seed=1"   # or as an environment dotlist, space separated
```

| key                                               | what it does                              |
| :------------------------------------------------ | :---------------------------------------- |
| `training.data.read_from_cache_store`             | serve cacheable builders from the store   |
| `training.data.force_cache_rebuild`               | recompute stored outputs on a cache build |
| `training.data.use_sensor_perturbation`           | draw the shifted rig with probability     |
| `training.data.towns`                             | train on these towns only                 |
| `training.data.py123d_log_names`                  | train on these logs only                  |
| `training.data.max_num_scenes`                    | cap the number of scenes                  |
| `training.data.num_chunks`, `.chunk_index`        | shard the scene list across runs          |
| `training.data.shuffle_scenes`                    | shuffle the scene order                   |
| `training.optimization.batch_size`                | batch size, split across GPUs             |
| `training.optimization.learning_rate`             | AdamW learning rate                       |
| `training.experiment.output_dir`                  | where checkpoints and `config.yaml` land  |
| `training.experiment.initial_weights_file`        | weights to start from                     |
| `training.experiment.resume_from_last_checkpoint` | continue an interrupted run               |
| `policy.transfuser.use_planning_decoder`          | perception phase off, planning on         |
| `policy.target`                                   | which policy to train                     |

The keys above are the same whichever policy trains, but the policy owns the whole read side:
`build_scene_filter()` combines them with the temporal window and modality requirements of its
config contract, and its `build_scene_loader()` wraps that in the loader it hands to its dataset. So
one object decides which scenes exist and how each is read, and the dataset only turns what comes
back into tensors.

```console
user@host:~/lead$ python -m lead.training.train \
      training.data.towns=[Town03,Town05] \
      training.data.max_num_scenes=5000 \
      training.data.shuffle_scenes=true    # two towns, a tenth of the scenes, shuffled — no code change
```
