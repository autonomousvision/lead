# Project Structure

## `lead/` — main Python package

```
lead/
├── carl_agent/              # CaRL RL expert agent (BEV-based RL policy). Independent of the rest of the repo.
│   ├── carl_agent.py        # Agent entry point, drives the RL policy in CARLA
│   ├── config.py            # Hyperparameters for the CaRL model
│   ├── model.py             # RL policy network
│   ├── nav_planner.py       # Navigation/route planner for CaRL
│   ├── distributions.py     # Action sampling distributions (Gaussian, Beta, Uniform+Beta)
│   ├── rl_utils.py          # Traffic light preprocessing and coordinate transforms
│   └── bev/                 # BEV map rendering and observation utilities
│
├── common/                  # Shared utilities across agents, evaluation, training and LEAD expert
│   ├── base_agent.py        # Abstract agent base class
│   ├── config_base.py       # Base config dataclass
│   ├── constants.py         # Project-wide constants and enums
│   ├── common_utils.py      # General-purpose helpers (serialization, math, etc.)
│   ├── logging_config.py    # Central logging configuration (setup_logging())
│   ├── kalman_filter.py     # Kalman filter for tracking CARLA actors
│   ├── pid_controller.py    # Lateral and longitudinal PID controllers
│   ├── ransac.py            # RANSAC ground plane removal from point clouds
│   ├── route_planner.py     # Waypoint management, trajectory interpolation, coordinate transforms
│   ├── sensor_setup.py      # Sensor configs (cameras, lidar, radar)
│   └── weathers.py          # Weather-to-visibility mapping
│
├── data_buckets/            # Dataset split definitions (train/val/test buckets)
│   ├── abstract_bucket_collection.py  # ABC for bucket collections with serialization and loading
│   ├── bucket.py            # Single data bucket holding paths and metadata
│   ├── failed_bucket_collection.py    # Collection of routes that failed during data collection
│   ├── full_pretrain_bucket_collection.py   # Pretraining split bucket collection
│   ├── full_posttrain_bucket_collection.py  # Posttraining split bucket collection
│   ├── navsim_bucket_collection.py    # NavSim dataset bucket collection
│   ├── waymo_bucket_collection.py     # Waymo dataset bucket collection
│   ├── town13_heldout_*_bucket_collection.py  # Town13 held-out splits (pre/post)
│   └── route_filtering.py   # Route-level filtering logic (validity checks, deduplication)
│
├── data_loader/             # PyTorch datasets for each data source
│   ├── carla_dataset.py     # CARLA Leaderboard 2.0 dataset
│   ├── carla_dataset_utils.py  # CARLA data loading helpers
│   ├── navsim_dataset.py    # NavSim dataset
│   ├── navsim_dataset_utils.py  # NavSim data loading helpers
│   ├── waymo_e2e_dataset.py # Waymo Open Dataset end-to-end dataset
│   └── training_cache.py    # Compressed on-disk cache wrapper for training samples
│
├── expert/                  # Expert/privileged agent for data collection
│   ├── expert.py            # Main expert agent — driving logic and meta-information handling
│   ├── expert_base.py       # Base class with shared properties/methods for expert variants
│   ├── expert_py123d.py     # Expert variant that logs data in Py123D Arrow format
│   ├── expert_py123d_utils.py  # CARLA-to-ISO8855 coordinate conversion utilities
│   ├── expert_data.py       # Data collection functionalities (saving frames, labels)
│   ├── expert_utils.py      # Shared expert helper functions
│   ├── config_expert.py     # Expert configuration dataclass
│   ├── kinematic_bicycle_model.py  # Kinematic bicycle model for vehicle dynamics
│   ├── privileged_route_planner.py # Route planning with traffic light/stop sign distances, lane changes
│   ├── scenario_sorter.py   # Sorts active scenarios by euclidean distance to ego
│   └── hdmap/               # HD map utilities (BEV rendering, traffic lights, stop signs)
│
├── inference/               # Inference pipelines (closed-loop and open-loop)
│   ├── closed_loop_inference.py   # CARLA closed-loop evaluation with PID control
│   ├── open_loop_inference.py     # Open-loop evaluation (NavSim / Bench2Drive)
│   ├── sensor_agent.py            # Sensor-based agent wrapper for CARLA leaderboard protocol
│   ├── inference_utils.py         # Geometry helpers (polygons, coordinate transforms)
│   ├── infraction_recorder.py     # Records and serializes driving infractions
│   ├── video_recorder.py          # Video recording and processing for evaluation runs
│   ├── config_closed_loop.py      # Closed-loop evaluation config
│   └── config_open_loop.py        # Open-loop evaluation config
│
├── tfv6/                    # TransFuser v6 model architecture
│   ├── tfv6.py              # Top-level model orchestrating backbone + decoders
│   ├── transfuser_backbone.py   # Image + LiDAR fusion backbone
│   ├── transfuser_utils.py      # Backbone utility layers and functions
│   ├── bev_decoder.py           # BEV semantic segmentation head
│   ├── perspective_decoder.py   # Perspective view output head
│   ├── planning_decoder.py      # Waypoint, path and target speed planning head
│   ├── tfv5_planning_decoder.py # Legacy TransFuser v5 planning decoder
│   ├── center_net_decoder.py    # CenterNet-based 2D bounding box detector
│   └── radar_detector.py        # 2D radar object detector
│
├── training/                # Training loop and supporting code
│   ├── train.py             # Main training entry point (DDP-aware)
│   ├── config_training.py   # Training configuration dataclass
│   ├── logger.py            # WandB metric and media logging
│   ├── rfs.py               # Waymo Open Dataset Rater Feedback Score metric
│   ├── mixed_training_utils.py  # Multi-dataset sample scheduling (abstract + implementations)
│   └── training_utils.py    # Seed, checkpointing, data loader construction, misc helpers
│
├── visualization/           # Visualization utilities
│   ├── visualizer.py        # High-level visualizer composing images and overlays
│   └── viz_utils.py         # Low-level drawing primitives (bboxes, trajectories, text)
│
├── infraction_webapp/       # Flask app for browsing infraction videos
│   ├── app.py               # Flask server entry point
│   └── templates/ static/   # HTML templates and static assets
│
└── leaderboard_wrapper.py   # Thin wrapper adapting LEAD to CARLA leaderboard agent protocol
```

## `3rd_party/` — external dependencies (submodules)

```
3rd_party/
├── leaderboard/             # CARLA Leaderboard 2.0 evaluation framework (modify minimally)
├── leaderboard_autopilot/   # Autopilot variant of leaderboard (heavily modified for LEAD expert)
├── scenario_runner/         # CARLA scenario runner — scenario definitions live here
├── scenario_runner_autopilot/  # Autopilot variant of scenario runner
├── Bench2Drive/             # Bench2Drive benchmark (bundles its own leaderboard + scenario runner)
├── CARLA_0915/              # CARLA simulator build (version 0.9.15)
├── CARLA_0916/              # CARLA simulator build (version 0.9.16)
├── carla_route_generator/   # Tool for generating CARLA evaluation routes
├── navsim_workspace/        # NavSim evaluation workspace
│   ├── navsimv1.1/          # NavSim v1.1 (navtest splits)
│   ├── navsimv2.2/          # NavSim v2.2 (navhard splits)
│   ├── dataset/             # Cached dataset for NavSim
│   └── exp/                 # NavSim experiment outputs
├── vscode_pydata_viewer/    # VS Code extension for viewing data files
└── typings/                 # Python type stubs for CARLA API
```

## `scripts/` — runnable scripts and utilities

```
scripts/
├── main.sh                  # Primary training launch script
├── pretrain_ddp.sh          # DDP pretraining launch
├── posttrain_ddp.sh         # DDP posttraining launch
├── build_cache.py           # Build compressed training cache from raw data
├── build_buckets_pretrain.py   # Generate pretraining bucket collections
├── build_buckets_posttrain.py  # Generate posttraining bucket collections
│
├── eval_bench2drive.sh      # Evaluate on Bench2Drive benchmark
├── eval_carl.sh             # Evaluate CaRL RL agent
├── eval_longest6.sh         # Evaluate on Longest6 routes
├── eval_town13.sh           # Evaluate on Town13 routes
├── eval_expert.sh           # Evaluate expert agent
├── eval_expert_123d.sh      # Evaluate Py123D expert variant
│
├── start_carla.sh           # Start CARLA server
├── setup_carla.sh           # Install/configure CARLA
├── clean_carla.sh           # Kill stale CARLA processes
├── reset_carla_world.py     # Reset CARLA world state (reload map)
│
├── 123d_viser.py            # Start Viser viewer for 123D scene visualization
├── download_one_checkpoint.sh  # Download a single model checkpoint
├── download_one_route.sh    # Download a single route's data
├── random_free_port.sh      # Find and print a random free TCP port
├── unzip_routes.sh          # Batch-unzip route archives
│
├── data_tools/              # Data inspection and management scripts
│   ├── 005–017_*            # Visualization, cleanup, and route management scripts
│   ├── build_data_cache.*   # Build data cache (script + launcher)
│   ├── visualize_buckets.*  # Visualize bucket statistics
│   └── visualize_data.*     # Visualize raw training data samples
│
├── hotkeys/                 # Shell hotkey scripts (registered via 01_setup_hotkeys.sh)
│   ├── 01–06_*              # Hotkey setup, pull, push, submodules, teardown
│   ├── 10–19_*              # Eval and expert data generation hotkeys
│   ├── 20–25_*              # Download scripts (mlcloud, tcml, a100 nodes)
│   ├── 30–34_*              # Zip/unzip data scripts
│   ├── 40–44_*              # CARLA management (start, clean, load town, reset)
│   ├── 50–52_*              # Docs build, webapp, route generator
│   ├── 61–68_*              # SLURM utilities (squeue, scancel, allocate GPU, screen)
│   ├── 70_sync_launch_json.py  # Sync VS Code launch.json with current config
│   ├── 80_start_viser.py    # Start Viser 3D viewer
│   └── 90–92_*              # Activate environment roots (lead, navsim v1.1, v2.2)
│
└── tools/                   # Misc eval/deployment tools
    ├── evaluation/          # Evaluation helper scripts
    ├── proxy_simulator/     # Proxy simulator for offline evaluation
    ├── routes_duplication/  # Route file duplication utilities
    ├── route_bridge.*       # Bridge routes between leaderboard formats
    ├── split_routes_to_individual.py  # Split multi-route XML into individual route files
    ├── result_parser.py     # Parse and aggregate evaluation result JSONs
    └── Dockerfile.master / make_docker.sh / run_docker.sh  # Docker build and run support
```

## `slurm/` — SLURM job submission and experiment management

```
slurm/
├── config_slurm.py          # Reads cluster config from configs/ directory
├── init.sh                  # Environment initialization sourced by SLURM jobs
│
├── train.sh                 # Submit training job to SLURM
├── evaluate.sh              # Submit evaluation job (generic, benchmark-agnostic)
├── evaluate_navtest.sh      # Submit NavTest evaluation job
├── evaluate_navhard_two_stage.sh  # Submit NavHard evaluation (two-stage pipeline)
├── evaluate_expert.sh       # Submit expert data collection job
├── evaluate_carl.sh         # Submit CaRL RL agent evaluation job
├── submit_navtest.sh        # Thin wrapper invoking evaluate_navtest.sh
├── submit_navhard.sh        # Thin wrapper invoking evaluate_navhard_two_stage.sh
│
├── configs/                 # Tunable cluster parameters (one value per .txt file)
│   ├── max_num_parallel_jobs_*.txt   # Max parallel SLURM jobs per benchmark
│   ├── max_num_attempts_*.txt        # Max retry attempts per benchmark
│   ├── max_sleep.txt                 # Max sleep between job polling cycles
│   └── wandb_log_frequency_*.txt     # WandB logging interval per benchmark
│
├── evaluation/              # Python logic for managing eval jobs
│   ├── evaluate.py                   # Core evaluation orchestrator (submit, monitor, retry)
│   ├── evaluate_scripts_generator.py # Generates per-route shell scripts for SLURM
│   ├── evaluate_utils.py             # Shared evaluation helpers (JSON parsing, status checks)
│   ├── evaluate_wandb_logger.py      # Logs aggregated eval results to WandB
│   └── merge_route_json.py           # Merges per-route result JSONs into one summary
│
├── data_collection/         # SLURM data collection job management
│   ├── collect_data.py               # Orchestrate parallel data collection jobs
│   ├── delete_failed_routes.py       # Remove failed route directories from disk
│   └── print_collect_data_progress.py # Print data collection progress stats
│
└── experiments/             # One directory per experiment run
    ├── 001_example/
    ├── ...
    └── NNN_<name>/          # Each contains job configs and result logs
```

### Experiment directory convention

Each `slurm/experiments/NNN_<name>/` directory holds the config and outputs for one experiment.

## Other top-level directories

```
tests/                  # Unit tests (mirrors lead/ subpackage structure)
notebooks/              # Jupyter notebooks for data inspection and debugging
website/                # Project website (leaderboard results, static HTML)
docs/                   # Sphinx documentation source (Makefile + source/)
```
