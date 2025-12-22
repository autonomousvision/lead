# Collect Data

> **Note**: This is a completely optional feature. The SLURM integration is designed for users with access to HPC clusters who want to scale their experiments efficiently. All functionality can also be run locally without SLURM.

## Quick Start

Adapt setting in `slurm/data_collection/collect_data.py`. In particular, you might be interested in to changing those:
- `repetitions`: For more diversity, each route can be ran multiple times with different seeds.
- `partitions`: Specific for your cluster.
- `dataset_name`: Change this to your need.

This script can be parametrized at running time by following config files:
- `slurm/configs/max_num_parallel_jobs_collect_data.txt` controls how many jobs are spawned parallel.
- `slurm/configs/max_sleep.txt` controls the time between starting CARLA server and starting python client.

Log into a login-node of your cluster, run

```bash
python3 slurm/data_collection/collect_data.py
```

If everything is configured correctly, then you should see those outputs produced after a few seconds

```html
data/carla_leaderboard2
├── data     # Sensor data will be stored here
├── results  # Redundant results jsons will be stored here
├── scripts  # SLURM/Bash scripts will be stored here
├── stderr   # stderr SLURM logs
└── stdout   # stdout SLURM logs
```

The data collections can run up to 2 days on 90 GPUs for 9000 routes. We recommend to run the script in `screen` or `tmux`.

## Monitor Progress

To see if there are any issues with data collection, runs

```bash
python3 slurm/data_collection/print_collect_data_progress.py
```

You need to adapt the variable `root`. In general, everything below 10% failure rate is normal.

## Delete Failed Routes

Run this script

```bash
python3 slurm/data_collection/delete_failed_routes.py
```
