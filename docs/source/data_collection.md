# Collect Data

This guide covers data collection on CARLA using LEAD.

> **Note**: This tutorial is designed for local development and debugging. For large-scale data-collection on HPC clusters, please refer to the [SLURM Guide](slurm_data_collection.md).

## Start Data Collection Locally

1. Start CARLA

```bash
bash scripts/start_carla.sh
```

2. Run expert

```bash
bash scripts/run_expert.sh
```

## Inspect Collected Data

Inspect data by using the notebook [notebooks/inspect_expert_output.ipynb](https://github.com/autonomousvision/lead/blob/leaderboard_2/notebooks/inspect_expert_output.ipynb).

## About the Expert

As can be seen in [lead/expert/expert.py](lead/expert/expert.py), LEAD:

- Proposes a shortest path by searching lane graph with A*.
- Augment this shortest path to avoid collision with static obstacles.
- Propose an initial target speed with IDM.
- Augment this target speed to avoid collision with dynamic hazards.

The expert's current state is still far from optimal. We recommend to play around and improve the current code base for more realistic expert's behaviors.
