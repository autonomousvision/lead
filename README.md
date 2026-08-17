<h2 align="center">LEAD: Minimizing Learner-Expert Asymmetry in End-to-End Driving</h2>

<div align="center">

[![Unit Tests](https://github.com/kesai-labs/lead/actions/workflows/ci.yml/badge.svg)](https://github.com/kesai-labs/lead/actions/workflows/ci.yml)
[![E2E Test](https://github.com/kesai-labs/lead/actions/workflows/ci_e2e.yml/badge.svg)](https://github.com/kesai-labs/lead/actions/workflows/ci_e2e.yml)
[![Python 3.10 - 3.12](https://img.shields.io/badge/Python-3.10%20--%203.12-3776ab)](https://www.python.org/downloads/)
[![PyTorch 2.8](https://img.shields.io/badge/PyTorch-2.8-ee4c2c)](https://pytorch.org/)
[![License MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![CARLA 0.9.15 | 0.9.16](https://img.shields.io/badge/CARLA-0.9.15%20%7C%200.9.16-e8710a)](https://carla.org/)

</div>

<p align="center">
  <a href="https://ln2697.github.io/lead/"><b>Website</b></a> ·
  <a href="https://kesai.eu/blog/2026-06-26-lead/"><b>Blog</b></a> ·
  <a href="https://arxiv.org/abs/2512.20563"><b>Paper</b></a> ·
  <a href="https://huggingface.co/datasets/ln2697/lead-123d"><b>Dataset</b></a> ·
  <a href="https://huggingface.co/ln2697/transfuser-carla-123d"><b>Checkpoints</b></a> ·
  <a href="https://github.com/kesai-labs/lead/releases/latest"><b>Latest Release</b></a>
</p>

This repository distills years of end-to-end driving research — and the lessons learned along the way — into a complete and opinionated infrastructure for imitation learning research in the CARLA simulator.

The codebase will be maintained in the coming years to support KE:SAI's open science mission, so you can expect future improvements, mostly in the form of less code, reduced complexity and improved documentation.

A short list of highlights, with comparisons against our previous [cvpr2026](https://github.com/kesai-labs/lead/tree/cvpr2026) pipeline, which builds on [carla_garage](https://github.com/autonomousvision/carla_garage) and the original [transfuser](https://github.com/autonomousvision/transfuser) codebase:

- **Fast data generator**: runs at ~10 steps/s on a consumer GPU, up to 10× faster ⚡.
- **Standardized format**: the data are readable, filterable, and visualizable with [py123d](https://github.com/kesai-labs/py123d) 🌟.
- **Efficient storage**: the full dataset fits in ~1 TB, with up to 1000× fewer files.
- **Rapid training**: TransFuser trains in under 24 h on 4×H100, more than 3× faster ⚡.
- **SoTA performance**: best driving scores on all available Leaderboard 2.0 benchmarks at release.

> \[!NOTE\]
> Branch `cvpr2026` is where we host the code, data and checkpoints for the arXiv paper. The datasets and checkpoints of the `main` branch are not compatible with the datasets and checkpoints of the `cvpr2026` branch; in fact, those two branches are completely independent. To reproduce the paper, use the `cvpr2026` branch. We recommend that future research build on the `main` branch.

Release notes:

- [v1.5.0 on Aug 20, 2026](https://github.com/kesai-labs/lead/releases/tag/v1.5.0): Released more checkpoints and [documentation](docs/speed.md) to tune the training pipeline's efficiency.
- [v1.4.0 on Aug 9, 2026](https://github.com/kesai-labs/lead/releases/tag/v1.4.0): Initial release of dataset in new format and pre-trained checkpoints.

### 🛠️ Setup for development

Grab the code:

```console
user@host:~$ git clone https://github.com/kesai-labs/lead.git
user@host:~$ cd lead
```

Set up the environment. Any conda-compatible manager works:

```console
user@host:~/lead$ micromamba create -n lead python=3.10 -y             # Create container
user@host:~/lead$ micromamba activate lead
user@host:~/lead$ pip install uv                                       # Install uv
user@host:~/lead$ uv pip install -e "." --reinstall-package lead       # Install lead and its dependencies
user@host:~/lead$ micromamba deactivate && micromamba activate lead    # Add scripts/cli to system's PATH
```

Install CARLA; version 0.9.16 is required for data collection. Evaluation works on 0.9.15, too.

```console
user@host:~/lead$ bash scripts/common/setup_carla.sh               # CARLA 0.9.16
user@host:~/lead$ bash scripts/common/setup_carla_fail2drive.sh    # Optional: Fail2Drive CARLA 0.9.15
```

If you get stuck, take a look at our [pipeline](.github/workflows/ci_e2e.yml). We dedicate a machine with GPU for testing purpose.

### 🔥 Get the data

We provide a dataset of 8,930 routes across 43 scenario types on all 12 CARLA town maps. Compared to previous TransFuser-lineage codebases, the recordings are denser (higher frequency), cover the full 360° surround view, and span more sensor modalities. See the [latest release](https://github.com/kesai-labs/lead/releases/latest) for information on how to obtain the data.

<div align="center">

| Modality                          | Frequency | Format                                            |
| :-------------------------------- | :-------- | :------------------------------------------------ |
| RGB                               | 4 Hz      | JPEG, quality adapted to weather and daytime      |
| Depth                             | 4 Hz      | PNG, 8-bit linear quantization saturating at 50 m |
| Semantic segmentation             | 4 Hz      | PNG, CARLA class ids                              |
| Instance segmentation             | 4 Hz      | PNG, CARLA actor ids                              |
| Lidar                             | 20 Hz     | LAZ-compressed point cloud                        |
| Radar                             | 20 Hz     | Raw points with radial velocity                   |
| Ego states, boxes, traffic lights | 20 Hz     | Arrow tables                                      |

</div>

See [data access](docs/data_access.md) for the full stream table and the on-disk layout.

In case you want to collect the data yourself with customized sensor configurations, the expert is reasonably fast; on a modest GTX 1080 Ti and 2 CPU cores it runs at around 10 steps per second. We plan to extend the data generation pipeline with a learning-based expert in the future. To generate data on a single route:

```console
user@host:~/lead$ python -m lead --expert --routes src/lead/routes/data_routes/lead/Accident/route_001761.xml
```

Collecting the full set of routes takes less than a day on a cluster with 64 GTX 1080 Ti GPUs. See [data collection](docs/data_generation.md) for details, including how to scale the collection on SLURM.

### 💡 Read and visualize data

The logs are regular 123D data with no CARLA dependency; the entire `py123d` API works on them out of the box ✨

```python
>>> from py123d.api.scene.arrow.arrow_scene_builder import ArrowSceneBuilder
>>> from py123d.api.scene.scene_filter import SceneFilter
>>> from py123d.common.execution.thread_pool_executor import ThreadPoolExecutor
>>>
>>> scenes = ArrowSceneBuilder(
...    logs_root="data/lead/123D/logs",
...    maps_root="data/lead/123D/maps",
... ).get_scenes(
...     SceneFilter(split_names=["normal_view"]),
...     ThreadPoolExecutor(),
... )

>>> scenes[0].get_camera_at_iteration(0, "pcam_f0").image.shape
(384, 384, 3)
```

See [data access](docs/data_access.md) for documentation on the data layout, and the [notebook](notebooks/data_access.ipynb) for a worked example. To inspect data, we provide two options: you can either point the standard `py123d-viser` tool at a log and look around:

```console
user@host:~/lead$ py123d-viser 'scene_filter.split_names=[normal_view]'                          # every log in normal_view
user@host:~/lead$ scripts/cli/viser data/lead/123D/logs/normal_view/<scenario_type>/<log_name>   # open a single log
```

Or you can visualize the input features and output labels of TransFuser, using only low-level drawing libraries such as `cv2` and `matplotlib`; see this [notebook](notebooks/data_visualization.ipynb) for more details.

### 💪 Training

Before training, you can optionally run cache building, which precomputes expensive features and inputs:

```console
user@host:~/lead$ bash scripts/common/build_cache.sh   # Build ~60GB training cache
```

Its manifest stores a `cache_finger_print` of the config. If the fingerprint changes, training detects the stale cache, fails, and requests a rebuild. After the cache is built, which should take at most two hours on a modern computer, training can start:

```console
user@host:~/lead$ python -m lead.training.train training.data.read_from_cache_store=true # Start training
```

See [training](docs/training.md) for more details on training. To tune the pipeline for training throughput, see the [faster training guide](docs/speed.md).

### 🚀 Evaluation

Use your own checkpoints from `outputs/`, or download our trained ones: see the [latest release](https://github.com/kesai-labs/lead/releases/latest) for information on how to obtain them.

Evaluation drives against a live simulator, so start CARLA in a second terminal:

```console
user@host:~/lead$ scripts/cli/start_carla
```

And start the evaluation on one route:

```console
# Bench2Drive
user@host:~/lead$ python -m lead --checkpoint <checkpoint dir> --routes src/lead/routes/benchmark_routes/bench2drive/23687.xml --bench2drive

# Longest6 v2
user@host:~/lead$ python -m lead --checkpoint <checkpoint dir> --routes src/lead/routes/benchmark_routes/longest6/00.xml

# Town13
user@host:~/lead$ python -m lead --checkpoint <checkpoint dir> --routes src/lead/routes/benchmark_routes/Town13/0.xml
```

Routes for Bench2Drive, Town13, longest6, and Fail2Drive ship under `src/lead/routes/benchmark_routes/`. See [evaluation](docs/eval.md) for more information on how to scale evaluation on SLURM.

### 📖 Citation

If our work is useful to you, please cite it and leave a star ⭐ on the repository:

```bibtex
@inproceedings{Nguyen2026CVPR,
  author    = {Long Nguyen and Micha Fauth and Bernhard Jaeger and Daniel Dauner and Maximilian Igl and Andreas Geiger and Kashyap Chitta},
  title     = {LEAD: Minimizing Learner-Expert Asymmetry in End-to-End Driving},
  booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026},
}

@article{Dauner2026ARXIV,
  author  = {Dauner, Daniel and Charraut, Valentin and Berle, Bastian and Li, Tianyu and Nguyen, Long and Wang, Jiabao and Jing, Changhui and Igl, Maximilian and Caesar, Holger and Ivanovic, Boris and Geiger, Andreas and Chitta, Kashyap},
  title   = {123D: Unifying Multi-Modal Autonomous Driving Data at Scale},
  journal = {arXiv preprint arXiv:2605.08084},
  year    = {2026},
}
```
