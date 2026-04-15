<p align="center">
  <img src="https://avatars.githubusercontent.com/u/269669339?s=200&v=4" alt="KE:SAI" height="60">&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://www.iapb.org/wp-content/uploads/2020/09/The-Eberhard-Karls-University-of-Tubingen.png" alt="University of Tübingen" height="60">&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://research.nvidia.com/labs/gear/images/media/nvidia_hu_9c090f760c3ff52d.png" alt="NVIDIA" height="60">
</p>

<h2 align="center">
  <b>LEAD: Minimizing Learner–Expert Asymmetry in End-to-End Driving</b>
</h2>

<p align="center">
  <a href="https://ln2697.github.io/lead">Website</a>&nbsp;&nbsp;·&nbsp;&nbsp;<a href="https://ln2697.github.io/lead/docs">Docs</a>&nbsp;&nbsp;·&nbsp;&nbsp;<a href="https://huggingface.co/datasets/ln2697/lead_carla">Dataset</a>&nbsp;&nbsp;·&nbsp;&nbsp;<a href="https://huggingface.co/ln2697/tfv6">Model</a>&nbsp;&nbsp;·&nbsp;&nbsp;<a href="https://huggingface.co/ln2697/tfv6_navsim">NAVSIM Model</a>&nbsp;&nbsp;·&nbsp;&nbsp;<a href="https://ln2697.github.io/assets/pdf/Nguyen26.EA.SUPP.pdf">Supplementary</a>&nbsp;&nbsp;·&nbsp;&nbsp;<a href="https://arxiv.org/abs/2512.20563">Paper</a>
</p>

<p align="center">
  An open-source end-to-end driving stack for CARLA.</br>
  ▶ State-of-the-art performance on all major Leaderboard 2.0 benchmarks ◀
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Bench2Drive-95 DS 🏆-00484B?style=for-the-badge&labelColor=006064" alt="Bench2Drive">
  <img src="https://img.shields.io/badge/Longest6_V2-62 DS 🏆-141A5F?style=for-the-badge&labelColor=1A237E" alt="Longest6 V2">
  <img src="https://img.shields.io/badge/Town13-10 DS 🏆-BF8E1E?style=for-the-badge&labelColor=FFBE28" alt="Town13">
  <img src="https://img.shields.io/badge/Fail2Drive-75 HM 🏆-4A148C?style=for-the-badge&labelColor=6A1B9A" alt="Fail2Drive">
</p>


<p align="center">
  <a href="https://ln2697.github.io">Long Nguyen</a><sup>1,3</sup>&nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="https://de.linkedin.com/in/micha-fauth-b4492a22b">Micha Fauth</a><sup>1</sup>&nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="https://kait0.github.io">Bernhard Jaeger</a><sup>1,3</sup>&nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="https://danieldauner.github.io">Daniel Dauner</a><sup>1,3</sup>
  <br>
  <a href="https://maximilianigl.com">Maximilian Igl</a><sup>2</sup>&nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="http://www.cvlibs.net">Andreas Geiger</a><sup>1,3</sup>&nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="https://kashyap7x.github.io">Kashyap Chitta</a><sup>2</sup>
</p>

<p align="center">
  <sup>1</sup>University of Tübingen, Tübingen AI Center&nbsp;&nbsp;·&nbsp;&nbsp;<sup>2</sup>NVIDIA Research&nbsp;&nbsp;·&nbsp;&nbsp;<sup>3</sup>KE:SAI
</p>

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Updates](#updates)
- [1. Quick Start](#1-quick-start)
  - [1.1. Environment initialization](#11-environment-initialization)
  - [1.2. Install dependencies](#12-install-dependencies)
  - [1.3. Download checkpoints](#13-download-checkpoints)
  - [1.4. Setup VSCode/PyCharm](#14-setup-vscodepycharm)
  - [1.5. Evaluate model](#15-evaluate-model)
  - [1.6. Infraction Analysis Webapp](#16-infraction-analysis-webapp)
- [2. CARLA Research Cycle](#2-carla-research-cycle)
  - [2.1. (Optional) Extending Data Routes](#21-optional-extending-data-routes)
  - [2.2. Collecting Data](#22-collecting-data)
  - [2.3. Training](#23-training)
  - [2.4. Benchmarking](#24-benchmarking)
- [3. Extensions](#3-extensions)
  - [3.1. Fail2Drive Evaluation](#31-fail2drive-evaluation)
  - [3.2. CaRL Agent Evaluation](#32-carl-agent-evaluation)
  - [3.3. NAVSIM Training and Evaluation](#33-navsim-training-and-evaluation)
  - [3.4. CARLA 123D Data Collection](#34-carla-123d-data-collection)
- [4. Project Structure](#4-project-structure)
- [5. Common Issues](#5-common-issues)
- [Beyond CARLA: Cross-Benchmark Deployment](#beyond-carla-cross-benchmark-deployment)
- [Further Documentation](#further-documentation)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

## Updates

<div align="center">

| Date         | Content                                                                                                                                 |
| :----------- | :-------------------------------------------------------------------------------------------------------------------------------------- |
| **26.04.14** | <b>WARNING: </b> The parameter `transfuser_token_dim`'s default value should be `256`, as used in the paper, and not `64`.          |
| **26.04.13** | <b>WARNING: </b> In some rare cases, we notice training instability. See [instructions](#5-common-issues), if you face similar problem. |
| **26.04.11** | Added Fail2Drive benchmark support, see [instructions](#31-fail2drive-evaluation).                                                      |
| **26.03.21** | Added evaluation support for the reinforcement-learning planner CaRL, see [instructions](#32-carl-agent-evaluation).                    |
| **26.03.18** | Deactivated Kalman Filter and all post-processing heuristics. See performance report [here](#13-download-checkpoints).                  |
| **26.02.25** | LEAD is accepted to **CVPR 2026** 🎉                                                                                                     |
| **26.02.25** | NAVSIM extension released. Code and [instructions](#33-navsim-training-and-evaluation) available. Supplementary data coming soon.       |
| **26.02.02** | Preliminary support for 123D. See [instructions](#34-carla-123d-data-collection).                                                       |
| **26.01.13** | CARLA dataset and training documentation released.                                                                                      |
| **25.12.24** | Initial release — paper, checkpoints, expert driver, and inference code.                                                                |

</div>

## 1. Quick Start

Get LEAD running locally: from cloning the repo and installing dependencies, to downloading a pretrained checkpoint and driving a CARLA Leaderboard 2.0 route end-to-end. We tested the instructions on the following configurations:

<div align="center">

| OS           | GPU               | CUDA | Driver | Inference | Training |
| ------------ | ----------------- | ---- | :----: | :-------: | :------: |
| Ubuntu 22.04 | L40S       | 13.0 |  580   |     ✅     |    ✅     |
| Ubuntu 22.04 | A100       | 13.0 |  580   |     ❌     |    ✅     |
| Ubuntu 24.04 | RTX 5090   | 13.1 |  590   |     ✅     |    ❌     |
| Ubuntu 22.04 | RTX A4000  | 13.0 |  580   |     ✅     |    ❌     |
| Ubuntu 22.04 | GTX 1080ti | 13.0 |  580   |     ✅     |    ❌     |

</div>

### 1.1. Environment initialization

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

### 1.2. Install dependencies

We use Miniconda as container and uv for Python dependencies. Runtime and dev dependencies are declared entirely in [pyproject.toml](pyproject.toml).

```bash
# (Optional, needed in some cases) Accept terms and services
conda tos accept --override-channels --channel \
  https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel \
  https://repo.anaconda.com/pkgs/r

# Create a conda environment
conda create -n lead python=3.10 -y
conda activate lead

# Install system-level tools
conda install -c conda-forge \
  ffmpeg parallel tree gcc zip unzip git-lfs uv

# Tell uv to use conda environment
mkdir -p $CONDA_PREFIX/etc/conda/activate.d \
  $CONDA_PREFIX/etc/conda/deactivate.d
echo 'export VIRTUAL_ENV=$CONDA_PREFIX' > \
  $CONDA_PREFIX/etc/conda/activate.d/uv.sh
echo 'unset VIRTUAL_ENV' > \
  $CONDA_PREFIX/etc/conda/deactivate.d/uv.sh

# Reactivate conda environment
conda activate lead

# Install dependencies
uv sync --active --extra dev

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

> [!TIP]
> uv cheatsheet:
> ```bash
> # Add dependency
> uv add --active <pkg>
>
> # Add dev dependency
> uv add --active --optional dev <pkg>
>
> # Update uv.lock
> uv lock
> ```

### 1.3. Download checkpoints

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

| Variant                | Bench2Drive  | Longest6 v2  |   Town13    |
| :--------------------- | :----------: | :----------: | :---------: |
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

### 1.4. Setup VSCode/PyCharm

<div align="center">

<table>
  <tr>
    <th width="50%">VSCode</th>
    <th width="50%">PyCharm</th>
  </tr>
  <tr>
    <td>Install the recommended extensions when prompted. Debugging works out of the box.</td>
    <td>Add the CARLA Python API <code>3rd_party/CARLA_0915/PythonAPI/carla</code> to your interpreter paths via <code>Settings → Python → Interpreter → Show All → Show Interpreter Paths</code>.</td>
  </tr>
  <tr>
    <td><img src="docs/assets/vscode.png" alt="VSCode recommended extensions prompt"></td>
    <td><img src="docs/assets/pycharm.png" alt="PyCharm interpreter paths setting"></td>
  </tr>
</table>

</div>

### 1.5. Evaluate model

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

> [!TIP]
> If run into OOM issue, there are few options:
> 1. Run those commands block the later 2 seeds from being loaded into memory for ensembling:
> ```bash
> mv outputs/checkpoints/tfv6_resnet34/model_0030_1.pth \
> outputs/checkpoints/tfv6_resnet34/_model_0030_1.pth
>
> mv outputs/checkpoints/tfv6_resnet34/model_0030_2.pth \
> outputs/checkpoints/tfv6_resnet34/_model_0030_2.pth
> ```
>
> 2. For local computer, start CARLA with `-quality-level=Poor`. This reduces
> the rendering quality, however will introduce distribution shift and
> should not be used for official evaluation.

### 1.6. Infraction Analysis Webapp

Launch the interactive infraction dashboard to analyze driving failures — especially useful for Longest6 or Town13 where iterating over evaluation logs is time-consuming:

```bash
python lead/infraction_webapp/app.py
```

Navigate to http://localhost:5000 and point it at `outputs/local_evaluation`.

> [!TIP]
> The app supports browser bookmarking to jump directly to a specific timestamp.

## 2. CARLA Research Cycle

The primary focus of this repository is solving the [original CARLA Leaderboard 2.0](https://leaderboard.carla.org/get_started_v2_0/). This section walks through the full research loop — collecting expert demonstrations, training a TFv6 policy, and benchmarking it closed-loop.

### 2.1. (Optional) Extending Data Routes

Before collecting, you may want to enlarge or diversify the route set under `data/data_routes/`. This step is optional — the shipped routes are enough to reproduce the paper results. However, if you want to improve the performance of the model, in particular for Longest6 v2 or Fail2Drive, introduce more routes is the easiest way to achieve this.

Two ways to do it:

- **Automatic** — sample routes programmatically from a CARLA town (e.g. random start/goal pairs with scenario annotations). Useful when you want large-scale coverage without hand-authoring.
- **Manual** — use the bundled [carla_route_generator](3rd_party/carla_route_generator), a GUI tool for clicking waypoints on a map and exporting routes. Launch it via the hotkey script:

```bash
cd 3rd_party/carla_route_generator
python3 scripts/window.py
```

Generated XML files can be dropped directly into `data/data_routes/` and picked up by the expert during data collection.

> [!TIP]
> Out of the box, [carla_route_generator](3rd_party/carla_route_generator) is purely mouse-driven (left-click to add/remove waypoints, right-click for scenarios, wheel to pan/zoom). Annotating hundreds of routes this way is slow. There are few tricks to accelerate the process:
> 1. We strongly recommend extending [scripts/window.py](3rd_party/carla_route_generator/scripts/window.py) with Qt `QShortcut` / `keyPressEvent` bindings for the actions you repeat most — e.g. add new route. Even one or two of keys cuts manual annotation time substantially.
> 2. Only annotate route manually, add scenarios automatically via Python.

### 2.2. Collecting Data

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

On a SLURM Cluster of 92 GTX 1080ti, the data collection is often finished after 2 days.

> [!TIP]
> 1. To configure camera/lidar/radar calibration, see [config_base.py](lead/common/config_base.py) and [config_expert.py](lead/expert/config_expert.py).
> 2. For large-scale collection on SLURM, see the [data collection docs](https://ln2697.github.io/lead/docs/data_collection.html).
> 3. The [Jupyter notebooks](notebooks) provide visualization examples.

### 2.3. Training

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

### 2.4. Benchmarking

With CARLA running, evaluate on any benchmark via **Python**:

```bash
python lead/leaderboard_wrapper.py \
  --checkpoint outputs/checkpoints/tfv6_resnet34 \
  --routes <ROUTE_FILE> \
  [--bench2drive]
```

<div align="center">

| Benchmark   | Route file                                               | Extra flag      |
| :---------- | :------------------------------------------------------- | :-------------- |
| Bench2Drive | `data/benchmark_routes/bench2drive/23687.xml`            | `--bench2drive` |
| Longest6 v2 | `data/benchmark_routes/longest6/00.xml`                  | —               |
| Town13      | `data/benchmark_routes/Town13/0.xml`                     | —               |
| Fail2Drive  | `data/benchmark_routes/fail2drive/Base_Animals_0075.xml` | `--fail2drive`  |

</div>

Or via **bash**:

```bash
bash scripts/eval_bench2drive.sh   # Bench2Drive
bash scripts/eval_longest6.sh      # Longest6 v2
bash scripts/eval_town13.sh        # Town13
bash scripts/eval_fail2drive.sh    # Fail2Drive (requires CARLA_F2D)
```

Results are saved to `outputs/local_evaluation/` with videos, infractions, and metrics.

> [!TIP]
> 1. See the [evaluation docs](https://ln2697.github.io/lead/docs/evaluation.html) for details.
> 2. For distributed evaluation, see the [SLURM evaluation docs](https://ln2697.github.io/lead/docs/slurm_evaluation.html).
> 3. Our SLURM wrapper supports WandB for reproducible benchmarking.

## 3. Extensions

Beyond the core Leaderboard 2.0 workflow, LEAD also supports additional benchmarks (Fail2Drive, NAVSIM), an alternative RL policy (CaRL), and an alternative data format (123D). Each extension plugs into the code base with minimal changes.

### 3.1. Fail2Drive Evaluation

[Fail2Drive](https://github.com/autonomousvision/fail2drive) is a CARLA Leaderboard 2 benchmark for testing closed-loop generalization on unseen long-tail scenarios.

**Setup.** Download the Fail2Drive simulator (custom CARLA build with novel assets):

```bash
mkdir -p 3rd_party/CARLA_F2D
curl -L https://hf.co/datasets/SimonGer/Fail2Drive/resolve/main/fail2drive_simulator.tar.gz \
  | tar -xz -C 3rd_party/CARLA_F2D
```

**Evaluate.** With CARLA_F2D running, evaluate on a single route:

```bash
# Start the Fail2Drive CARLA simulator
bash 3rd_party/CARLA_F2D/CarlaUE4.sh

# Evaluate model on one route
LEAD_CLOSED_LOOP_CONFIG="sensor_agent_creeping=True \
  use_kalman_filter=True \
  slower_for_stop_sign=True" \
python lead/leaderboard_wrapper.py \
  --checkpoint outputs/checkpoints/tfv6_regnety \
  --routes data/benchmark_routes/fail2drive/Base_Animals_0075.xml \
  --fail2drive
```

> [!TIP]
> For SLURM evaluation, use `evaluate_fail2drive` from `slurm/init.sh`. See existing experiment scripts for usage patterns.

**Results.**

<div align="center">

<table>
  <thead>
    <tr>
      <th rowspan="2">Method</th>
      <th colspan="1">Bench2Drive</th>
      <th colspan="3">Fail2Drive In-Distribution</th>
      <th colspan="3">Fail2Drive Generalization</th>
    </tr>
    <tr>
      <th>DS ↑</th>
      <th>DS ↑</th>
      <th>SR(%) ↑</th>
      <th>HM ↑</th>
      <th>DS ↑</th>
      <th>SR(%) ↑</th>
      <th>HM ↑</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>TCP</td>
      <td align="center">59.9</td>
      <td align="center">24.7</td>
      <td align="center">39.1</td>
      <td align="center">30.3</td>
      <td align="center">24.5 <sub>(-0.8%)</sub></td>
      <td align="center">31.4 <sub>(-19.7%)</sub></td>
      <td align="center">27.5 <sub>(-9.1%)</sub></td>
    </tr>
    <tr>
      <td>UniAD</td>
      <td align="center">45.8</td>
      <td align="center">47.5</td>
      <td align="center">36.3</td>
      <td align="center">41.2</td>
      <td align="center">44.0 <sub>(-7.4%)</sub></td>
      <td align="center">27.6 <sub>(-24.0%)</sub></td>
      <td align="center">33.9 <sub>(-17.6%)</sub></td>
    </tr>
    <tr>
      <td>Orion</td>
      <td align="center">77.8</td>
      <td align="center">53.0</td>
      <td align="center">52.0</td>
      <td align="center">52.5</td>
      <td align="center">51.2 <sub>(-3.4%)</sub></td>
      <td align="center">46.0 <sub>(-11.5%)</sub></td>
      <td align="center">48.5 <sub>(-7.7%)</sub></td>
    </tr>
    <tr>
      <td>HiP-AD</td>
      <td align="center">86.8</td>
      <td align="center">74.1</td>
      <td align="center">70.7</td>
      <td align="center">72.4</td>
      <td align="center">67.1 <sub>(-9.4%)</sub></td>
      <td align="center">56.7 <sub>(-19.8%)</sub></td>
      <td align="center">61.5 <sub>(-15.1%)</sub></td>
    </tr>
    <tr>
      <td>SimLingo</td>
      <td align="center">85.1</td>
      <td align="center">82.6</td>
      <td align="center">79.3</td>
      <td align="center">80.9</td>
      <td align="center">71.7 <sub>(-13.2%)</sub></td>
      <td align="center">55.0 <sub>(-30.6%)</sub></td>
      <td align="center">62.2 <sub>(-23.1%)</sub></td>
    </tr>
    <tr>
      <td>TFv5</td>
      <td align="center">84.2</td>
      <td align="center">83.3</td>
      <td align="center">78.5</td>
      <td align="center">80.8</td>
      <td align="center">75.4 <sub>(-9.5%)</sub></td>
      <td align="center">61.1 <sub>(-22.2%)</sub></td>
      <td align="center">67.5 <sub>(-16.5%)</sub></td>
    </tr>
    <tr>
      <td><b>TFv6 (Ours)</b></td>
      <td align="center"><b>95.2</b></td>
      <td align="center"><b>90.2</b></td>
      <td align="center"><b>93.3</b></td>
      <td align="center"><b>91.7</b></td>
      <td align="center"><b>79.5</b> <sub>(-11.9%)</sub></td>
      <td align="center"><b>70.7</b> <sub>(-24.2%)</sub></td>
      <td align="center"><b>74.8</b> <sub>(-18.4%)</sub></td>
    </tr>
  </tbody>
</table>

</div>

### 3.2. CaRL Agent Evaluation

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

### 3.3. NAVSIM Training and Evaluation

**Setup.** Install `navtrain` and `navtest` splits following [navsimv1.1/docs/install.md](3rd_party/navsim_workspace/navsimv1.1/docs/install.md), then install the `navhard` split following [navsimv2.2/docs/install.md](3rd_party/navsim_workspace/navsimv2.2/docs/install.md).

**Training.** Run perception pretraining ([script](slurm/experiments/002_navsim_example/000_pretrain1_0.sh)) followed by planning post-training ([script](slurm/experiments/002_navsim_example/010_postrain32_0.sh)). We use one seed for pretraining and three seeds for post-training to estimate performance variance.

**Evaluation.** Run evaluation on [navtest](slurm/experiments/002_navsim_example/020_navtest_0.sh) and [navhard](slurm/experiments/002_navsim_example/030_navhard_0.sh).

### 3.4. CARLA 123D Data Collection

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

## 4. Project Structure

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

## 5. Common Issues

| Issue                                             | Fix                                                                                                                                                                                                                                    |
| :------------------------------------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Stale or corrupted data errors                    | Delete and rebuild the training cache / buckets                                                                                                                                                                                        |
| Simulator hangs or is unresponsive                | Restart the CARLA simulator                                                                                                                                                                                                            |
| Route or evaluation failures                      | Restart the leaderboard                                                                                                                                                                                                                |
| Training instability after PyTorch version update | No fix for now. We tried to upgrade Torch several times but failed to achieve stable training on newer Torch versions.                                                                                                                 |
| OOM in evaluation                                 | Use larger GPU. In our [submit_job](slurm/evaluation/evaluate_utils.py) utility function, we first attempt to use a smaller GPU partition (1080ti/2080ti). After some failures, we switch automatically to a partition with more VRAM. |
| Training instability in general                   | Turn off mixed-precision training and train in 32bit precision.                                                                                                                                                                        |

If you face training instability, as reported in issue [#67](https://github.com/kesai-labs/lead/issues/67), try following solutions:
1. Turn off mixed-precision training in [config](lead/training/config_training.py).
2. Purge the Conda environment and reinstall from scratch.
3. Try the latest working commit at `a41d11616`.

## Beyond CARLA: Cross-Benchmark Deployment

The LEAD pipeline and TFv6 models serve as reference implementations across multiple E2E driving platforms:

<div align="center">

| Platform                                                                                         | Model            | Highlight                                                         |
| :----------------------------------------------------------------------------------------------- | :--------------- | :---------------------------------------------------------------- |
| [Waymo E2E Driving Challenge](https://waymo.com/open/challenges/2025/e2e-driving/)               | DiffusionLTF     | **2nd place** in the inaugural vision-based E2E driving challenge |
| [NAVSIM v1 Huggingface Leaderboard](https://huggingface.co/spaces/AGC2024-P/e2e-driving-navtest) | LTFv6            | +3 PDMS over Latent TransFuser baseline on `navtest`              |
| [NAVSIM v2 Huggingface Leaderboard](https://huggingface.co/spaces/AGC2025/e2e-driving-navhard)   | LTFv6            | +6 EPMDS over Latent TransFuser baseline on `navhard`             |
| [NVIDIA AlpaSim](https://github.com/NVlabs/alpasim)                                              | TransFuserDriver | Official baseline policy for closed-loop simulation               |

</div>

## Further Documentation

For a deeper dive, visit the [full documentation site](https://ln2697.github.io/lead/docs):

<p align="center">
<a href="https://ln2697.github.io/lead/docs/data_collection.html">Data Collection</a>&nbsp;&nbsp;·&nbsp;&nbsp;<a href="https://ln2697.github.io/lead/docs/carla_training.html">Training</a>&nbsp;&nbsp;·&nbsp;&nbsp;<a href="https://ln2697.github.io/lead/docs/evaluation.html">Evaluation</a>.
</p>

The documentation will be updated regularly.

## Acknowledgements

This project builds on the shoulders of excellent open-source work. Special thanks to [carla_garage](https://github.com/autonomousvision/carla_garage) for the foundational codebase.

<p align="center">
  <a href="https://github.com/OpenDriveLab/DriveLM/blob/DriveLM-CARLA/pdm_lite/docs/report.pdf">PDM-Lite</a>&nbsp;&nbsp;·&nbsp;&nbsp;<a href="https://github.com/carla-simulator/leaderboard">Leaderboard</a>&nbsp;&nbsp;·&nbsp;&nbsp;<a href="https://github.com/carla-simulator/scenario_runner">Scenario Runner</a>&nbsp;&nbsp;·&nbsp;&nbsp;<a href="https://github.com/autonomousvision/navsim">NAVSIM</a>&nbsp;&nbsp;·&nbsp;&nbsp;<a href="https://github.com/waymo-research/waymo-open-dataset">Waymo Open Dataset</a>
  <br>
  <a href="https://github.com/RenzKa/simlingo">SimLingo</a>&nbsp;&nbsp;·&nbsp;&nbsp;<a href="https://github.com/autonomousvision/plant2">PlanT2</a>&nbsp;&nbsp;·&nbsp;&nbsp;<a href="https://github.com/autonomousvision/Bench2Drive-Leaderboard">Bench2Drive Leaderboard</a>&nbsp;&nbsp;·&nbsp;&nbsp;<a href="https://github.com/Thinklab-SJTU/Bench2Drive/">Bench2Drive</a>&nbsp;&nbsp;·&nbsp;&nbsp;<a href="https://github.com/autonomousvision/CaRL">CaRL</a>&nbsp;&nbsp;·&nbsp;&nbsp;<a href="https://github.com/autonomousvision/fail2drive">Fail2Drive</a>
</p>

Long Nguyen led development of the project. Kashyap Chitta, Bernhard Jaeger, and Andreas Geiger contributed through technical discussion and advisory feedback. Daniel Dauner provided guidance with NAVSIM.

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
