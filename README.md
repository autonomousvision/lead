<h1>LEAD: Minimizing Learner-Expert Asymmetry in End-to-End Driving</h1>

LEAD is an end-to-end driving stack for the CARLA Leaderboard 2.0, covering the data generation pipeline, a TransFuser-style driving policy, and popular driving benchmarks like Bench2Drive and Fail2Drive. The repository builds on py123d and focuses only on CARLA research; the [CVPR 2026 branch](https://github.com/kesai-labs/lead/tree/cvpr2026) holds the paper version.

**Latest update `v1.1.0`** (2026.07.15):

- Added [basic documentation](https://github.com/kesai-labs/lead/docs/index.md).
- Introduce abstraction for training and evaluation.
- Bugs fixes and verification pipeline.

<details>
<summary>Older changelog entries</summary>

| Version | Date       | Content                                                                                                                                       |
| :------ | :--------- | :-------------------------------------------------------------------------------------------------------------------------------------------- |
| v1.0.0  | 2026.07.12 | <ul><li>Integrated py123d as first-class data format.</li><li>Modernised code base.</li><li>Expert run time increased by up to 10x.</li></ul> |

</details>

> \[!WARNING\]
> Repo is under active development. Interfaces may change without notice.

## Setup for development

Clone the repository:

```bash
git clone https://github.com/kesai-labs/lead.git
cd lead
```

Install dependencies. Any conda-compatible environment manager works (e.g.
[micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html),
[miniconda](https://docs.conda.io/projects/miniconda/en/latest/), or
[conda](https://docs.conda.io/projects/conda/en/latest/index.html)):

```bash
# Create an environment
micromamba create -n lead python=3.10 -y
micromamba activate lead

# Install system tools
micromamba install -c conda-forge ffmpeg parallel tree gcc zip unzip git-lfs uv rclone -y

# Install project
UV_PROJECT_ENVIRONMENT=$CONDA_PREFIX uv sync --extra dev --reinstall-package lead

# Reload environment to add shell scripts to PATH
micromamba deactivate && micromamba activate lead
```

After this, edit `.env` yourself if needed. After that, install CARLA:

```bash
# Download and setup CARLA 0.9.16
bash scripts/common/setup_carla.sh
```

Run tests and lints:

```bash
pre-commit install

# Unit tests with dynamic type checking enabled
LEAD_RUNTIME_TYPE_CHECKING=true pytest tests/unittests

# Run pre-commit with static type checking
pre-commit run --all-files
```

See [setup](docs/setup.md) for more details.

## Data collection

The expert drives CARLA and writes a py123d dataset:

```bash
python -m lead --expert --routes lead/src/lead/routes/data_routes/lead/Accident/route_001761.xml
```

See [data collection](docs/data_generation.md) for details.

The logs are ordinary py123d logs, so py123d can read them directly:

```python
from py123d.api.scene.arrow.arrow_scene_builder import ArrowSceneBuilder
from py123d.api.scene.scene_filter import SceneFilter
from py123d.common.execution.thread_pool_executor import ThreadPoolExecutor

scenes = ArrowSceneBuilder(
    logs_root="/path/to/lead-data/logs",
    maps_root="/path/to/lead-data/maps",
).get_scenes(SceneFilter(future_num_iterations=8), ThreadPoolExecutor())

scene = scenes[0]
ego = scene.get_ego_state_se3_at_iteration(0)  # py123d.datatypes.EgoStateSE3
```

See [data access](docs/data_access.md) for further documentation on how to read the data.

## Training

```bash
bash scripts/common/pretrain.sh   # perception pre-training
bash scripts/common/posttrain.sh  # planning post-training
```

See [training](docs/training.md) for details.

## Evaluation

```bash
python -m lead --checkpoint <ckpt> --routes src/lead/routes/benchmark_routes/bench2drive/1711.xml
```

See [evaluation](docs/eval.md) for more information on benchmarking.

## Other resources

- [Extended documentation](docs/index.md).
- [Py123D documentation](https://kesai.eu/py123d/)

## Citations

```
@inproceedings{Nguyen2026CVPR,
	author = {Long Nguyen and Micha Fauth and Bernhard Jaeger and Daniel Dauner and Maximilian Igl and Andreas Geiger and Kashyap Chitta},
	title = {LEAD: Minimizing Learner-Expert Asymmetry in End-to-End Driving},
	booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
	year = {2026},
}
```
