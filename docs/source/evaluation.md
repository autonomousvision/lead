# Evaluation

This guide covers basic local evaluation for getting started. For large-scale evaluation (running many routes parallely), see the [SLURM Evaluation Guide](slurm_evaluation.md).

## Overview

We provided in the [Quick Start](https://github.com/autonomousvision/lead?tab=readme-ov-file#quick-start) tutorial how to evaluate a trained policy 
on a single Bench2Drive route. Currently we support scripts to evaluate the three most popular benchmarks locally:

* [Bench2Drive script](https://github.com/autonomousvision/lead/blob/main/scripts/eval_bench2drive.sh) - 220 routes
* [Longest6 v2 script](https://github.com/autonomousvision/lead/blob/main/scripts/eval_longest6.sh) - 36 routes
* [Town13 script](https://github.com/autonomousvision/lead/blob/main/scripts/eval_town13.sh) - 20 routes

## Running Evaluations

**1. Start CARLA Server**

Before running any evaluation, start the CARLA server:

```bash
bash scripts/start_carla.sh
```

**2. Customize Checkpoint and Route**

Each evaluation script contains these key variables you can modify:

```bash
export BENCHMARK_ROUTE_ID=23687  # Route ID to evaluate
export CHECKPOINT_DIR=outputs/checkpoints/tfv6_resnet34/  # Path to model checkpoint
```

For Bench2Drive, route IDs range from 0-219. For Longest6, use 00-35. For Town13, use 0-19.

**3. Configuration Options**

Evaluation behavior is controlled by [config_closed_loop.py](https://github.com/autonomousvision/lead/blob/main/lead/inference/config_closed_loop.py). Key settings:

* `produce_demo_video` - Generate bird's-eye view visualization videos
* `produce_debug_video` - Generate detailed debug videos with sensor data
* `produce_demo_image` - Save individual demo frames
* `produce_debug_image` - Save individual debug frames

Turn off video generation for faster evaluation:

```python
# In your environment or config override
produce_demo_video = False
produce_debug_video = False
```

The evaluation configuration can be changed for each progress individually with the environment variable `LEAD_CLOSED_LOOP_CONFIG`.

**4. Output Structure**

Each evaluation creates:

```
outputs/local_evaluation/<route_id>/
├── checkpoint_endpoint.json      # Metrics and results
├── metric_info.json              # Detailed evaluation metrics
├── demo_images/                  # Bird's-eye view frames
├── debug_images/                 # Debug visualization frames
└── debug_checkpoint/             # Debug checkpoints
```

If video generation is enabled:
```
outputs/local_evaluation/
├── <route_id>_demo.mp4          # Bird's-eye view video
└── <route_id>_debug.mp4         # Debug video with sensor data
```

## Summarize Leaderboard 2.0 Results for Longest6 v2 and Town13

After completing all routes in a benchmark, aggregate results using the result parser:

```bash
python3 scripts/tools/result_parser.py \
    --xml data/benchmark_routes/bench2drive220.xml \
    --results outputs/local_evaluation/
```

This generates a summary CSV with:
- Driving score
- Route completion percentage
- Infraction breakdown (collisions, traffic violations, etc.)
- Per-kilometer statistics

## Summarize Bench2Drive Results

Bench2Drive provides some more metrics beyond the official Leaderboard 2.0 metrics of Longest6 v2 and Town13. See [official guide](https://github.com/Thinklab-SJTU/Bench2Drive?tab=readme-ov-file#eval-tools).

The tools for Bench2Drive in our repo can be found [here](https://github.com/autonomousvision/lead/blob/main/3rd_party/Bench2Drive/tools/).

## Best Practices

**Turn off production of videos and images for Longest6 v2 and Town13**: With enough compute (16-32 GTX 1080ti), evaluation can take up to 1 day for 3 seeds on Longest6 v2 and up to 2 days for 3 seeds on Town13. About 90% of the routes will be finished within a few hours.

**Restart CARLA between routes**: Running multiple routes on the same CARLA instance can lead to rendering bugs (see image, taken from Bench2Drive paper).

![](../assets/buggy_rendering.png)

```bash
bash scripts/clean_carla.sh  # Kill CARLA processes
bash scripts/start_carla.sh  # Restart fresh instance
```

**Memory management**: By default, the pipeline loads all three checkpoint seeds as an ensemble. If memory is limited, rename two of the checkpoint files so only one seed loads.

**Use correct leaderboard and scenario_runner**: Longest6 v2 and Town13 should be evaluated on the normal leaderboard setup. Bench2Drive must be evaluated on [code of their repo](https://github.com/autonomousvision/lead/tree/main/3rd_party/Bench2Drive), otherwise the results are not valid.

**Evaluation variance**: CARLA is highly stochastic, even with fixed seeds. Results can vary significantly between runs due to traffic randomness and other non-deterministic factors. Our recommended evaluation protocol:

- Minimum (standard practice): Train 3 models with different seeds, evaluate each once → 3 evaluation runs total
- Optimal (for publications): Train 3 models with different seeds, evaluate each 3 times → 9 evaluation runs total

We use the minimum protocol in our group.