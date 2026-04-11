# Known Issues

Known issues in the evaluation pipeline and models.

## Multi-GPU Training Can Slightly Degrade Performance

Training on 4 GPUs sometimes yields marginally lower closed-loop performance than single-GPU training. The effect is small and doesn't change qualitative conclusions, but appears consistently in certain runs.

## Static Graph Can Degrade Performance

We avoid [static_graph](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) in the pipeline due to observed performance issues.

## CARLA Waypoint PID Controller Needs Better Tuning

A well-tuned controller (e.g., MPC) can significantly improve performance. Preliminary experiments showed ~5-7 DS improvement on Bench2Drive for TFv5, though these numbers are approximate since controller tuning wasn't the focus.

## CARLA 0.9.16 Has Goal-Point Issues

CARLA 0.9.16 currently has problems with the goal-point pipeline that degrade policy behavior. We don't recommend evaluating models on this version.

## Torch version

The training of CARLA agents work best with Torch `2.5.x`. Newer version might results into training divergence.
