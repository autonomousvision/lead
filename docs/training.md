# Training

Training runs in two phases. Perception pre-training:

```bash
bash scripts/common/pretrain.sh
```

Load the pre-trained perception backbone and start planning post-training — set
`policy.transfuser.use_planning_decoder=true` and `training.experiment.load_file`
to the pre-training checkpoint you want to build on:

```bash
bash scripts/common/posttrain.sh
```

`training.is_pretraining` is derived from `use_planning_decoder`, so this flag is
what actually switches phases; `logdir`/`load_file` just point at the right run.

## Configuring a run

Both scripts call `python3 src/lead/training/train.py` with no CLI flags — all
overrides go through the `LEAD_CONFIG` environment variable, a space-separated
dotlist against `LeadConfig`:

```bash
export LEAD_CONFIG="training.experiment.logdir=outputs/local_training/pretrain \
training.optimization.batch_size=32 \
training.optimization.epochs=50"
```

CLI dotlist args (when invoking `train.py` directly) take priority over
`LEAD_CONFIG`, which takes priority over a loaded checkpoint's stored config.
See `src/lead/config/training/` for the full set of training/optimization/
experiment knobs (learning rate, mixed precision, `torch.compile`, checkpoint
retention, WandB logging cadence, etc.).

## Checkpoints and resuming

Checkpoints are written to `logdir` as `model_{epoch:04d}.pth`. To resume or to
chain pre-training into post-training, point `training.experiment.load_file` at
one of these files — `initialize_config` merges the checkpoint's saved
`config.yaml` underneath your current overrides. Set
`training.experiment.continue_failed_training=true` to load with `strict=False`
when resuming a run that died mid-epoch.

## Multi-GPU

Lightning's `DDPStrategy` spawns one worker per visible GPU itself, so no
`torchrun` wrapper is needed — `CUDA_VISIBLE_DEVICES` controls which GPUs are
used.
