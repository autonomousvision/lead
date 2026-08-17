# Training

TransFuser trains in two phases: perception first, then planning on top of the
pretrained weights.

## Build the cache

Cacheable data (lidar raster, planning and detection targets) are
precomputed once, for both normal and perturbated sensor views. The rest will be read directly from the py123d logs:

```console
user@host:~/lead$ bash scripts/common/build_cache.sh
```

Training never builds the cache. Enabling `read_from_cache_store` without a built
store fails at startup.

The store's manifest also carries a `cache_finger_print`: the config values that
decide what a cached tensor holds (BEV geometry, sensor preprocessing, label
knobs). Building or reading a store whose fingerprint no longer matches the
current config fails the same way, instead of silently serving samples built
under a different config — rebuild it with `force_cache_rebuild=true`.

## Training

Perception pre-training

```console
user@host:~/lead$ bash scripts/common/pretrain.sh
```

From the pre-trained checkpoint we enable the planning decoder and start the post-training:

```console
user@host:~/lead$ bash scripts/common/posttrain.sh
```

For training on multiple nodes or GPUs, see the SLURM scripts for [pre-training](../scripts/slurm/pretrain.sh) and [post-training](../scripts/slurm/posttrain.sh).
