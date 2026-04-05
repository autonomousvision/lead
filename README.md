<p align="center">
  <img src="docs/assets/logo.png" alt="LEAD">
</p>

<h2 align="center">
  <b>Minimizing Learner–Expert Asymmetry in End-to-End Driving</b>
</h2>

<p align="center">
  <a href="https://ln2697.github.io/lead">Website</a>&nbsp;&nbsp;·&nbsp;&nbsp;<a href="https://ln2697.github.io/lead/docs">Docs</a>&nbsp;&nbsp;·&nbsp;&nbsp;<a href="https://huggingface.co/datasets/ln2697/lead_carla">Dataset</a>&nbsp;&nbsp;·&nbsp;&nbsp;<a href="https://huggingface.co/ln2697/tfv6">Model</a>&nbsp;&nbsp;·&nbsp;&nbsp;<a href="https://huggingface.co/ln2697/tfv6_navsim">NAVSIM Model</a>&nbsp;&nbsp;·&nbsp;&nbsp;<a href="https://ln2697.github.io/assets/pdf/Nguyen26.EA.SUPP.pdf">Supplementary</a>&nbsp;&nbsp;·&nbsp;&nbsp;<a href="https://arxiv.org/abs/2512.20563">Paper</a>
</p>

<p align="center">
  An open-source end-to-end driving stack for CARLA, achieving state-of-the-art<br>closed-loop performance across all major Leaderboard 2.0 benchmarks.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Bench2Drive-95.2 🏆-0F6E56?style=for-the-badge&labelColor=1D9E75" alt="Bench2Drive">
  <img src="https://img.shields.io/badge/Longest6_V2-62 🏆-185FA5?style=for-the-badge&labelColor=378ADD" alt="Longest6 V2">
  <img src="https://img.shields.io/badge/Town13-5.2 🏆-854F0B?style=for-the-badge&labelColor=BA7517" alt="Town13">
</p>

---

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Updates](#updates)
- [Quick Start CARLA Leaderboard](#quick-start-carla-leaderboard)
  - [1. Environment initialization](#1-environment-initialization)
  - [2. Install dependencies](#2-install-dependencies)
  - [3. Download checkpoints](#3-download-checkpoints)
  - [4. Setup VSCode/PyCharm](#4-setup-vscodepycharm)
  - [5. Evaluate model](#5-evaluate-model)
  - [6. Infraction Analysis Webapp](#6-infraction-analysis-webapp)
- [CARLA Training](#carla-training)
- [CARLA Data Collection](#carla-data-collection)
- [CARLA 123D Data Collection](#carla-123d-data-collection)
- [CARLA Benchmarking](#carla-benchmarking)
- [CaRL Agent Evaluation](#carl-agent-evaluation)
- [NAVSIM Training and Evaluation](#navsim-training-and-evaluation)
- [Project Structure](#project-structure)
- [Common Issues](#common-issues)
- [Beyond CARLA: Cross-Benchmark Deployment](#beyond-carla-cross-benchmark-deployment)
- [Further Documentation](#further-documentation)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)
- [License](#license)

---

## Updates

<div align="center">

| Date         | Content                                                                                                                                        |
| :----------- | :--------------------------------------------------------------------------------------------------------------------------------------------- |
| **26.03.21** | Added evaluation support for the RL planner [CaRL](https://github.com/autonomousvision/carl), see [instructions](#carl-agent-evaluation). |
| **26.03.18** | Deactivated Kalman Filter and all post-processing heuristics. See performance report [here](#3-download-checkpoints).   |
| **26.02.25** | LEAD is accepted to **CVPR 2026**!                                                                                                             |
| **26.02.25** | NAVSIM extension released. Code and [instructions](#navsim-training-and-evaluation) available. Supplementary data coming soon.                 |
| **26.02.02** | Preliminary support for [123D](https://github.com/autonomousvision/py123d). See [instructions](#carla-123d-data-collection).                   |
| **26.01.13** | CARLA dataset and training documentation released.                                                                                             |
| **25.12.24** | Initial release — paper, checkpoints, expert driver, and inference code.                                                                       |

</div>

---

## Quick Start CARLA Leaderboard

### 1. Environment initialization

Clone the repository and register the project root:

```bash
git clone https://github.com/kesai-labs/lead.git
cd lead

# Set project root variable
echo -e "export LEAD_PROJECT_ROOT=$(pwd)" >> ~/.bashrc

# Activate project's hook
echo "source $(pwd)/scripts/main.sh" >> ~/.bashrc

# Reload shell config
source ~/.bashrc
```

Verify that `~/.bashrc` reflects these paths correctly.

### 2. Install dependencies

We use Miniconda, conda-lock, and uv:

```bash
# (Optional, needed in some cases) Accept terms and services
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create conda environment
pip install conda-lock && conda-lock install -n lead conda-lock.yml

# Activate conda environment
conda activate lead

# Install dependencies
pip install uv && uv pip install -r requirements.txt && uv pip install -e .

# Install other tools needed for development
conda install -c conda-forge ffmpeg parallel tree gcc zip unzip git-lfs

# Optional: activate git hooks
pre-commit install
```

Set up CARLA:

```bash
# Download and setup CARLA at 3rd_party/CARLA_0915
bash scripts/setup_carla.sh

# Or symlink your pre-installed CARLA
ln -s /your/carla/path 3rd_party/CARLA_0915
```

### 3. Download checkpoints

Pre-trained checkpoints are hosted on HuggingFace. Following are the results from the paper. To reproduce these results, enable the Kalman filter, stop-sign, and creeping heuristics by setting `sensor_agent_creeping=True use_kalman_filter=True slower_for_stop_sign=True` in [config_closed_loop](lead/inference/config_closed_loop.py).

<div align="center">

| Variant                | Bench2Drive | Longest6 v2 |  Town13  |                                 Checkpoint                                  |
| :--------------------- | :---------: | :---------: | :------: | :-------------------------------------------------------------------------: |
| Full TransFuser V6     |   **95**    |   **62**    | **5.24** |    [Link](https://huggingface.co/ln2697/tfv6/tree/main/tfv6_regnety032)     |
| ResNet34 (60M params)  |     94      |     57      |   5.01   |     [Link](https://huggingface.co/ln2697/tfv6/tree/main/tfv6_resnet34)      |
| &ensp; + Rear camera   |     95      |     53      |   TBD    |   [Link](https://huggingface.co/ln2697/tfv6/tree/main/4cameras_resnet34)    |
| &ensp; − Radar         |     94      |     52      |   TBD    |    [Link](https://huggingface.co/ln2697/tfv6/tree/main/noradar_resnet34)    |
| &ensp; Vision only     |     91      |     43      |   TBD    |  [Link](https://huggingface.co/ln2697/tfv6/tree/main/visiononly_resnet34)   |
| &ensp; Town13 held out |     93      |     52      |   3.52   | [Link](https://huggingface.co/ln2697/tfv6/tree/main/town13heldout_resnet34) |

</div>

<b>Without these heuristics</b>, the performance changes as follows, with the biggest boost observed on Town13.

<div align="center">

| Variant            | Bench2Drive  | Longest6 v2 |   Town13   |
| :----------------- | :----------: | :---------: | :--------: |
| Full E2E TransFuser V6 | 95 &rarr; 94 | 62 &rarr; 62 | 5 &rarr; 10 |

</div>

Download the checkpoints:

```bash
# Download one checkpoint for testing
bash scripts/download_one_checkpoint.sh

# Download all checkpoints
git clone https://huggingface.co/ln2697/tfv6 outputs/checkpoints
cd outputs/checkpoints
git lfs pull
```

### 4. Setup VSCode/PyCharm

**VSCode** — install recommended extensions when prompted. Debugging works out of the box.

![VSCode recommended extensions prompt](docs/assets/vscode.png)

**PyCharm** — add the CARLA Python API `3rd_party/CARLA_0915/PythonAPI/carla` to your interpreter paths via `Settings → Python → Interpreter → Show All → Show Interpreter Paths`.

![PyCharm interpreter paths setting](docs/assets/pycharm.png)

### 5. Evaluate model

Verify your setup with a single route:

```bash
# Start driving environment
bash scripts/start_carla.sh

# Turn on images and videos output
export LEAD_CLOSED_LOOP_CONFIG="produce_demo_image=true \
  produce_demo_video=true \
  produce_debug_image=true \
  produce_debug_video=true \
  produce_input_image=true \
  produce_input_video"

# Run policy on one route
python lead/leaderboard_wrapper.py \
  --checkpoint outputs/checkpoints/tfv6_resnet34 \
  --routes data/benchmark_routes/bench2drive/23687.xml \
  --bench2drive
```

Driving logs are saved to `outputs/local_evaluation/<route_id>/`:

<div align="center">

| Output                     | Description                    |
| :------------------------- | :----------------------------- |
| `*_debug.mp4`              | Debug visualization video      |
| `*_demo.mp4`               | Demo video                     |
| `*_grid.mp4`               | Grid visualization video       |
| `*_input.mp4`              | Raw input video                |
| `alpasim_metric_log.json`  | AlpaSim metric log             |
| `checkpoint_endpoint.json` | Checkpoint endpoint metadata   |
| `infractions.json`         | Detected infractions           |
| `metric_info.json`         | Evaluation metrics             |
| `debug_images/`            | Per-frame debug visualizations |
| `demo_images/`             | Per-frame demo images          |
| `grid_images/`             | Per-frame grid visualizations  |
| `input_images/`            | Per-frame raw inputs           |
| `input_log/`               | Input log data                 |

</div>

### 6. Infraction Analysis Webapp

Launch the interactive infraction dashboard to analyze driving failures — especially useful for Longest6 or Town13 where iterating over evaluation logs is time-consuming:

```bash
python lead/infraction_webapp/app.py
```

Navigate to http://localhost:5000 and point it at `outputs/local_evaluation`.

> [!TIP]
> The app supports browser bookmarking to jump directly to a specific timestamp.

---

## CARLA Training

Download the dataset from HuggingFace:

```bash
# Download all routes
git clone https://huggingface.co/datasets/ln2697/lead_carla data/carla_leaderboard2/zip
cd data/carla_leaderboard2/zip
git lfs pull

# Or download a single route for testing
bash scripts/download_one_route.sh

# Unzip the routes
bash scripts/unzip_routes.sh

# Build data cache
python scripts/build_cache.py
```

**Perception pretraining.** Logs and checkpoints are saved to `outputs/local_training/pretrain`:

```bash
# Single GPU
python3 lead/training/train.py \
  logdir=outputs/local_training/pretrain

# Distributed Data Parallel
bash scripts/pretrain_ddp.sh
```

**Planning post-training.** Logs and checkpoints are saved to `outputs/local_training/posttrain`:

```bash
# Single GPU
python3 lead/training/train.py \
  logdir=outputs/local_training/posttrain \
  load_file=outputs/local_training/pretrain/model_0030.pth \
  use_planning_decoder=true

# Distributed Data Parallel
bash scripts/posttrain_ddp.sh
```

> [!TIP]
> 1. For distributed training on SLURM, see the [SLURM training docs](https://ln2697.github.io/lead/docs/slurm_training.html).
> 2. For a complete workflow (pretrain → posttrain → eval), see this [example](slurm/experiments/001_example).
> 3. For detailed documentation, see the [training guide](https://ln2697.github.io/lead/docs/carla_training.html).

---

## CARLA Data Collection

With CARLA running, collect data for a single route via **Python** (recommended for debugging):

```bash
python lead/leaderboard_wrapper.py \
  --expert \
  --routes data/data_routes/lead/noScenarios/short_route.xml
```

Or via **bash** (recommended for flexibility):

```bash
bash scripts/eval_expert.sh
```

Collected data is saved to `outputs/expert_evaluation/` with the following structure:

<div align="center">

| Directory                | Content                                     |
| :----------------------- | :------------------------------------------ |
| `bboxes/`                | 3D bounding boxes per frame                 |
| `depth/`                 | Compressed and quantized depth maps         |
| `depth_perturbated/`     | Depth from perturbated ego state            |
| `hdmap/`                 | Ego-centric rasterized HD map               |
| `hdmap_perturbated/`     | HD map aligned to perturbated ego pose      |
| `lidar/`                 | LiDAR point clouds                          |
| `metas/`                 | Per-frame metadata and ego state            |
| `radar/`                 | Radar detections                            |
| `radar_perturbated/`     | Radar from perturbated ego state            |
| `rgb/`                   | RGB images                                  |
| `rgb_perturbated/`       | RGB from perturbated ego state              |
| `semantics/`             | Semantic segmentation maps                  |
| `semantics_perturbated/` | Semantics from perturbated ego state        |
| `results.json`           | Route-level summary and evaluation metadata |

</div>

> [!TIP]
> 1. To configure camera/lidar/radar calibration, see [config_base.py](lead/common/config_base.py) and [config_expert.py](lead/expert/config_expert.py).
> 2. For large-scale collection on SLURM, see the [data collection docs](https://ln2697.github.io/lead/docs/data_collection.html).
> 3. The [Jupyter notebooks](notebooks) provide visualization examples.

---

## CARLA 123D Data Collection

With CARLA running, collect data in [123D](https://github.com/autonomousvision/py123d) format via **Python**:

```bash
export LEAD_EXPERT_CONFIG="target_dataset=6 \
  py123d_data_format=true \
  use_radars=false \
  lidar_stack_size=2 \
  save_only_non_ground_lidar=false \
  save_lidar_only_inside_bev=false"

python -u $LEAD_PROJECT_ROOT/lead/leaderboard_wrapper.py \
    --expert \
    --py123d \
    --routes data/data_routes/50x38_Town12/ParkingCrossingPedestrian/3250_1.xml
```

Or via **bash**:

```bash
bash scripts/eval_expert_123d.sh
```

Output in 123D format is saved to `data/carla_leaderboard2_py123d/`:

<div align="center">

| Directory            | Content                                |
| :------------------- | :------------------------------------- |
| `logs/train/*.arrow` | Per-route driving logs in Arrow format |
| `logs/train/*.json`  | Per-route metadata                     |
| `maps/carla/*.arrow` | Map data in Arrow format               |

</div>

To visualize collected scenes in 3D with [Viser](https://viser.studio/):

```bash
python scripts/123d_viser.py
```

> [!TIP]
> This feature is experimental. Change `PY123D_DATA_ROOT` in `scripts/main.sh` to set the output directory.

---

## CARLA Benchmarking

With CARLA running, evaluate on any benchmark via **Python**:

```bash
python lead/leaderboard_wrapper.py \
  --checkpoint outputs/checkpoints/tfv6_resnet34 \
  --routes <ROUTE_FILE> \
  [--bench2drive]
```

<div align="center">

| Benchmark   | Route file                                    | Extra flag      |
| :---------- | :-------------------------------------------- | :-------------- |
| Bench2Drive | `data/benchmark_routes/bench2drive/23687.xml` | `--bench2drive` |
| Longest6 v2 | `data/benchmark_routes/longest6/00.xml`       | —               |
| Town13      | `data/benchmark_routes/Town13/0.xml`          | —               |

</div>

Or via **bash**:

```bash
bash scripts/eval_bench2drive.sh   # Bench2Drive
bash scripts/eval_longest6.sh      # Longest6 v2
bash scripts/eval_town13.sh        # Town13
```

Results are saved to `outputs/local_evaluation/` with videos, infractions, and metrics.

> [!TIP]
> 1. See the [evaluation docs](https://ln2697.github.io/lead/docs/evaluation.html) for details.
> 2. For distributed evaluation, see the [SLURM evaluation docs](https://ln2697.github.io/lead/docs/slurm_evaluation.html).
> 3. Our SLURM wrapper supports WandB for reproducible benchmarking.

---

## CaRL Agent Evaluation

With CARLA running, evaluate the CaRL agent via **Python**:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python lead/leaderboard_wrapper.py \
  --checkpoint outputs/checkpoints/CaRL \
  --routes data/benchmark_routes/bench2drive/24240.xml \
  --carl-agent \
  --bench2drive \
  --timeout 900
```

Or via **bash**:

```bash
bash scripts/eval_carl.sh
```

The results are in `outputs/local_evaluation/<route_id>/`.

> [!TIP]
> 1. With small code changes, you can also integrate CaRL into LEAD's expert-driving pipeline as a hybrid expert policy.
> 2. For large scale evaluation on SLURM, see [this directory](slurm/experiments/003_evaluate_carl).

---

## NAVSIM Training and Evaluation

**Setup.** Install `navtrain` and `navtest` splits following [navsimv1.1/docs/install.md](3rd_party/navsim_workspace/navsimv1.1/docs/install.md), then install the `navhard` split following [navsimv2.2/docs/install.md](3rd_party/navsim_workspace/navsimv2.2/docs/install.md).

**Training.** Run perception pretraining ([script](slurm/experiments/002_navsim_example/000_pretrain1_0.sh)) followed by planning post-training ([script](slurm/experiments/002_navsim_example/010_postrain32_0.sh)). We use one seed for pretraining and three seeds for post-training to estimate performance variance.

**Evaluation.** Run evaluation on [navtest](slurm/experiments/002_navsim_example/020_navtest_0.sh) and [navhard](slurm/experiments/002_navsim_example/030_navhard_0.sh).

---

## Project Structure

The project is organized into the following top-level directories. See the [full documentation](https://ln2697.github.io/lead/docs/project_structure.html) for a detailed breakdown.

<div align="center">

| Directory    | Purpose                                                               |
| :----------- | :-------------------------------------------------------------------- |
| `lead/`      | Main package — model architecture, training, inference, expert driver |
| `3rd_party/` | Third-party dependencies (CARLA, benchmarks, evaluation tools)        |
| `data/`      | Route definitions. Sensor data will be stored here, too.              |
| `scripts/`   | Utility scripts for data processing, training, and evaluation         |
| `outputs/`   | Checkpoints, evaluation results, and visualizations                   |
| `notebooks/` | Jupyter notebooks for data inspection and analysis                    |
| `slurm/`     | SLURM job scripts for large-scale experiments                         |

</div>

---

## Common Issues

| Symptom                                        | Fix                                                            |
| :--------------------------------------------- | :------------------------------------------------------------- |
| Stale or corrupted data errors                 | Delete and rebuild the training cache / buckets                |
| Simulator hangs or is unresponsive             | Restart the CARLA simulator                                    |
| Route or evaluation failures                   | Restart the leaderboard                                        |
| Need to reset the map without restarting CARL  | Run `scripts/reset_carla_world.py` (much faster on large maps) |

---

## Beyond CARLA: Cross-Benchmark Deployment

The LEAD pipeline and TFv6 models serve as reference implementations across multiple E2E driving platforms:

<div align="center">

| Platform                                                                           | Model           | Highlight                                                         |
| :--------------------------------------------------------------------------------- | :-------------- | :---------------------------------------------------------------- |
| [Waymo E2E Driving Challenge](https://waymo.com/open/challenges/2025/e2e-driving/) | DiffusionLTF    | **2nd place** in the inaugural vision-based E2E driving challenge |
| [NAVSIM v1](https://huggingface.co/spaces/AGC2024-P/e2e-driving-navtest)           | LTFv6           | +3 PDMS over Latent TransFuser baseline on `navtest`              |
| [NAVSIM v2](https://huggingface.co/spaces/AGC2025/e2e-driving-navhard)             | LTFv6           | +6 EPMDS over Latent TransFuser baseline on `navhard`             |
| [NVIDIA AlpaSim](https://github.com/NVlabs/alpasim)                                | TransFuserModel | Official baseline policy for closed-loop simulation               |

</div>

---

## Further Documentation

For a deeper dive, visit the [full documentation site](https://ln2697.github.io/lead/docs):

<p align="center">
<a href="https://ln2697.github.io/lead/docs/data_collection.html">Data Collection</a>&nbsp;&nbsp;·&nbsp;&nbsp;<a href="https://ln2697.github.io/lead/docs/carla_training.html">Training</a>&nbsp;&nbsp;·&nbsp;&nbsp;<a href="https://ln2697.github.io/lead/docs/evaluation.html">Evaluation</a>.
</p>

The documentation will be updated regularly.

---

## Acknowledgements

This project builds on the shoulders of excellent open-source work. Special thanks to [carla_garage](https://github.com/autonomousvision/carla_garage) for the foundational codebase.

<p align="center">
  <a href="https://github.com/OpenDriveLab/DriveLM/blob/DriveLM-CARLA/pdm_lite/docs/report.pdf">PDM-Lite</a>&nbsp;&nbsp;·&nbsp;&nbsp;<a href="https://github.com/carla-simulator/leaderboard">Leaderboard</a>&nbsp;&nbsp;·&nbsp;&nbsp;<a href="https://github.com/carla-simulator/scenario_runner">Scenario Runner</a>&nbsp;&nbsp;·&nbsp;&nbsp;<a href="https://github.com/autonomousvision/navsim">NAVSIM</a>&nbsp;&nbsp;·&nbsp;&nbsp;<a href="https://github.com/waymo-research/waymo-open-dataset">Waymo Open Dataset</a>
  <br>
  <a href="https://github.com/RenzKa/simlingo">SimLingo</a>&nbsp;&nbsp;·&nbsp;&nbsp;<a href="https://github.com/autonomousvision/plant2">PlanT2</a>&nbsp;&nbsp;·&nbsp;&nbsp;<a href="https://github.com/autonomousvision/Bench2Drive-Leaderboard">Bench2Drive Leaderboard</a>&nbsp;&nbsp;·&nbsp;&nbsp;<a href="https://github.com/Thinklab-SJTU/Bench2Drive/">Bench2Drive</a>&nbsp;&nbsp;·&nbsp;&nbsp;<a href="https://github.com/autonomousvision/CaRL">CaRL</a>
</p>

Long Nguyen led development of the project. Kashyap Chitta, Bernhard Jaeger, and Andreas Geiger contributed through technical discussion and advisory feedback. Daniel Dauner provided guidance with NAVSIM.

---

## Citation

If you find this work useful, please consider giving this repository a star and citing our paper:

```bibtex
@inproceedings{Nguyen2026CVPR,
	author = {Long Nguyen and Micha Fauth and Bernhard Jaeger and Daniel Dauner and Maximilian Igl and Andreas Geiger and Kashyap Chitta},
	title = {LEAD: Minimizing Learner-Expert Asymmetry in End-to-End Driving},
	booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
	year = {2026},
}
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
