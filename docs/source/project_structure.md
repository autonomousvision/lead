# Project Structure

This page provides a detailed overview of the LEAD project's directory structure and organization.

## Main Python Package (`lead/`)

### `lead/common/`
Core utilities used across the project:
- Configuration management
- Logging and visualization utilities
- Geometric transformations and coordinate systems
- Data preprocessing helpers

### `lead/data_buckets/`
Manages data organization into buckets for efficient training:
- Bucket creation and validation
- Data sampling strategies
- Dataset statistics and metadata

### `lead/data_loader/`
PyTorch data loading pipeline:
- Dataset implementations for different benchmarks
- Data augmentation strategies
- Efficient batch loading and caching

### `lead/expert/`
Expert driver implementation:
- Privileged ground-truth planning
- Data collection orchestration

### `lead/inference/`
Closed-loop evaluation pipeline:
- Model inference wrapper
- Sensor data preprocessing
- Metric computation
- Visualization and logging

### `lead/webapp/`
Interactive web dashboard for analyzing driving failures:
- Video playback of driving runs
- Infraction highlighting and annotation

### `lead/tfv6/`
TransFuser V6 model architecture:
- Vision encoder networks (ResNet, RegNet)
- LiDAR/Radar feature extraction
- Multi-modal fusion transformer
- Trajectory prediction heads
- Control output layers

### `lead/training/`
Training pipeline and utilities:
- PyTorch Lightning training modules
- Loss functions and metrics
- Optimizer and scheduler configurations
- Checkpointing and logging
- Distributed training support

## Top-Level Structure

```
lead/                        # Main Python package
├── common/                  # Shared utilities and helpers
├── data_buckets/            # Data bucket management
├── data_loader/             # PyTorch data loading
├── expert/                  # Expert driver implementation
├── inference/               # Closed-loop evaluation
├── webapp/                  # Interactive infraction viewer
├── tfv6/                    # TransFuser V6 model architecture
└── training/                # Training pipeline

3rd_party/                   # Third-party dependencies
├── alpamayo/                # NVIDIA Alpamayo model
├── Bench2Drive/             # Bench2Drive benchmark tools
├── CARLA_0915/              # CARLA simulator
├── carla_route_generator/   # Route generation tools
├── leaderboard/             # CARLA Leaderboard 2.0
├── leaderboard_autopilot/   # Autopilot-specific leaderboard
├── navsim_workspace/        # NAVSIM evaluation
├── scenario_runner/         # CARLA scenario execution
└── scenario_runner_autopilot/  # Autopilot scenario runner

data/                        # Route definitions and debug data
├── benchmark_routes/        # Official benchmark routes
├── carla_leaderboard2/      # Sensor data
├── data_routes/             # Training data collection routes
├── debug_routes/            # Debugging routes
└── expert_debug/            # Expert driver debug data

docs/                        # Documentation source
├── source/                  # Sphinx documentation files
├── build/                   # Built documentation (generated)
└── assets/                  # Images and media

outputs/                     # Training and evaluation outputs
├── checkpoints/             # Model checkpoints
├── evaluation/              # SLURM evaluation results
├── local_evaluation/        # Local evaluation results
├── local_training/          # Local training outputs
├── training/                # SLURM training outputs
├── training_viz/            # Training visualizations
└── visualization/           # General visualizations

scripts/                     # Utility scripts
├── generate_data.py            # Data generation script
├── build_buckets_pretrain.py   # Pre-training data buckets
├── build_buckets_posttrain.py  # Post-training data buckets
├── build_cache.py              # Data cache generation
├── setup_carla.sh              # CARLA installation
├── start_carla.sh              # CARLA startup
└── ...                         # More utility scripts

slurm/                       # SLURM job scripts
├── init.sh                  # Bash constructor for a SLURM job
├── experiments/             # Jobs description
└── evaluation/              # Evaluation helper scripts

notebooks/                   # Jupyter notebooks
├── carla_offline_inference.ipynb
├── data_format.ipynb
├── inspect_expert_output.ipynb
└── ...

tests/                       # Unit and integration tests
```
