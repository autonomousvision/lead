<h1>LEAD: Minimizing Learner-Expert Asymmetry in End-to-End Driving</h1>

LEAD is an end-to-end driving stack for the CARLA Leaderboard 2.0, covering the data generation pipeline, a TransFuser-style driving policy, and popular driving benchmarks like Bench2Drive and Fail2Drive. We build upon py123d and focus only on CARLA research.

**Latest update `v1.2.0`** (2026.07.20):

- Fix GNSS localization for CARLA 0.9.16, which projects GNSS with a transverse Mercator.
- Reduce failure rate on Slurm when collecting data.

<details>
<summary>Older changelog entries</summary>

| Version | Date       | Content                                                                                                                                                                                                   |
| :------ | :--------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| v1.1.0  | 2026.07.15 | <ul><li>Added [basic documentation](https://github.com/kesai-labs/lead/docs/index.md).</li><li>Introduce abstraction for training and evaluation.</li><li>Bugs fixes and verification pipeline.</li></ul> |
| v1.0.0  | 2026.07.12 | <ul><li>Integrated py123d as first-class data format.</li><li>Modernised code base.</li><li>Expert run time increased by up to 10x.</li></ul>                                                             |

</details>

> \[!NOTE\]
> This branch is a rewrite of the [CVPR 2026 branch](https://github.com/kesai-labs/lead/tree/cvpr2026) and is under active development; interfaces may change without notice. Its datasets and checkpoints will be released soon and will not be compatible with the CVPR 2026 ones. To reproduce the paper, use the CVPR 2026 branch.

## Setup for development

Clone the repository:

```bash
git clone https://github.com/kesai-labs/lead.git
cd lead
```

Install dependencies. Any conda-compatible environment manager works, we recommend [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html):

```bash
# Create an environment
micromamba create -n lead python=3.10 -y
micromamba activate lead

# Install system tools
micromamba install -c conda-forge ffmpeg gcc git-lfs uv -y

# Tell uv once where to install deps to
export UV_PROJECT_ENVIRONMENT=$CONDA_PREFIX

# Install project
uv sync --extra dev --reinstall-package lead

# Reload environment to add shell scripts to PATH
micromamba deactivate && micromamba activate lead
```

Install CARLA if closed-loop evaluation or customized data are required. This step is optional if you only need the standard dataset:

```bash
# Download and setup CARLA 0.9.16
bash scripts/common/setup_carla.sh
```

See [setup](docs/setup.md) for more details.

## Data collection

To collect one single route for testing:

```bash
python -m lead --expert --routes lead/src/lead/routes/data_routes/lead/Accident/route_001761.xml
```

See [data collection](docs/data_generation.md) for details.

## Read data

Collected data can be read by py123d directly:

```python
from py123d.api.scene.arrow.arrow_scene_builder import ArrowSceneBuilder
from py123d.api.scene.scene_filter import SceneFilter
from py123d.common.execution.thread_pool_executor import ThreadPoolExecutor

# Build scenes
scenes = ArrowSceneBuilder(
    logs_root="</path/to/lead-data>/logs",
    maps_root="</path/to/lead-data>/maps",
).get_scenes(SceneFilter(future_num_iterations=8), ThreadPoolExecutor())

# Access first scene
scene = scenes[0]

# Access data of first frame
ego = scene.get_ego_state_se3_at_iteration(0)  # py123d.datatypes.EgoStateSE3
```

See [data access](docs/data_access.md) for further documentation on the data.

## Citation

```
@inproceedings{Nguyen2026CVPR,
	author = {Long Nguyen and Micha Fauth and Bernhard Jaeger and Daniel Dauner and Maximilian Igl and Andreas Geiger and Kashyap Chitta},
	title = {LEAD: Minimizing Learner-Expert Asymmetry in End-to-End Driving},
	booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
	year = {2026},
}

@article{Dauner2026ARXIV,
  title={123D: Unifying Multi-Modal Autonomous Driving Data at Scale},
  author={Dauner, Daniel and Charraut, Valentin and Berle, Bastian and Li, Tianyu and Nguyen, Long and Wang, Jiabao and Jing, Changhui and Igl, Maximilian and Caesar, Holger and Ivanovic, Boris and Geiger, Andreas and Chitta, Kashyap},
  journal={arXiv preprint arXiv:2605.08084},
  year={2026}
}
```
