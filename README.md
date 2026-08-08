<h1 align="center">LEAD: Minimizing Learner-Expert Asymmetry in End-to-End Driving</h1>

<div align="center">

[![Unit Tests](https://github.com/kesai-labs/lead/actions/workflows/ci.yml/badge.svg)](https://github.com/kesai-labs/lead/actions/workflows/ci.yml)
[![E2E Test](https://github.com/kesai-labs/lead/actions/workflows/ci_e2e.yml/badge.svg)](https://github.com/kesai-labs/lead/actions/workflows/ci_e2e.yml)
[![Python 3.10 | 3.11 | 3.12](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-3776ab)](https://www.python.org/downloads/)
[![PyTorch 2.13](https://img.shields.io/badge/pytorch-2.13-ee4c2c)](https://pytorch.org/)
[![License MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Format Py123D](https://img.shields.io/badge/format-Py123D-8a2be2)](https://github.com/kesai-labs/py123d)
[![Docs](https://img.shields.io/badge/docs-index-informational)](docs/index.md)

</div>

<p align="center">
  <a href="https://ln2697.github.io/lead/"><b>Website</b></a> ·
  <a href="https://kesai.eu/blog/2026-06-26-lead/"><b>Blog</b></a> ·
  <a href="https://arxiv.org/abs/2512.20563"><b>Paper</b></a>
</p>

LEAD grew out of years of hands-on research in end-to-end driving, and it provides a complete and opinionated infrastructure for driving research in the CARLA simulator: driving expert, data in the standardized [Py123D](https://github.com/kesai-labs/py123d) format, efficient dataloaders optimized for high throughput training, a PyTorch Lightning trainer with minimal boilerplate, and popular closed-loop benchmarks such as Bench2Drive or Fail2Drive.

The codebase follows modern software-engineering principles and will be maintained in the next years. As long as your policy implements the API contracts, the rest of the stack, from cache building to closed-loop evaluation in CARLA, does not need to know anything about it. To keep quality high, we lean heavily on linting, static type checking, and runtime type checking.

A handful of Python commands is all you need to walk through a whole E2E driving stack 🚀:

```console
user@host:~/lead$ python -m lead --expert --routes ConstructionObstacle/route_001761.xml
[INFO] Wrote data in 3m 12s

user@host:~/lead$ python -m lead.training.build_cache
[INFO] 61GB cache built at data/lead/123D/transfuser_cache

user@host:~/lead$ python -m lead.training.train
[INFO] Devices: 8 x DDPStrategy | precision bf16-mixed | compile True
[INFO] Batch: 256 global, 32 per device | 19104 steps over 30 epochs

user@host:~/lead$ python -m lead --checkpoint checkpoints/transfuser --bench2drive bench2drive/ParkingCutIn_1711.xml
[INFO] Finished evaluation. See output video: output/bench2drive/1711.mp4
```

Highlights at a glance ⚡:

- **Fast rule-based expert**: produces data at ~10 steps/s on a consumer GPU.
- **73h of multimodal driving dataset**: hosted on Hugging Face in compact compression form.
- **Standardized format**: readable, filterable, and visualizable with the standard `py123d` package.
- **High-throughput training**: flexible feature cache, asynchronous dataloading, finetuned decoding pipeline.
- **Policy-agnostic stack**: implement one API contract and cache building, training, and closed-loop evaluation work unchanged.
- **Closed-loop benchmarks out of the box**: Bench2Drive and Fail2Drive, with per-route videos and infraction reports.

> \[!NOTE\]
> This branch is a rewrite of the [cvpr2026 branch](https://github.com/kesai-labs/lead/tree/cvpr2026). We are still in active development, so you can expect future improvements. To avoid confusion: the datasets and checkpoints of the `main` branch are not compatible with the datasets and checkpoints of the `cvpr2026` branch; in fact, those two branches are completely independent. To reproduce the paper, use the `cvpr2026` branch.

### 🛠️ Setup for development

Grab the code:

```console
user@host:~$ git clone https://github.com/kesai-labs/lead.git
user@host:~$ cd lead
```

Set up the environment. Any conda-compatible manager works:

```console
user@host:~/lead$ micromamba create -n lead python=3.10 -y             # fresh environment
user@host:~/lead$ micromamba activate lead
user@host:~/lead$ pip install uv
user@host:~/lead$ uv pip install -e "." --reinstall-package lead       # LEAD and its dependencies
user@host:~/lead$ micromamba deactivate && micromamba activate lead    # CLI helpers on PATH
```

Bring in the CARLA simulator. The script pulls the 0.9.16 release and imports the additional maps:

```console
user@host:~/lead$ bash scripts/common/setup_carla.sh               # into 3rd_party/CARLA/standard_0916
user@host:~/lead$ bash scripts/common/setup_carla.sh /opt/carla    # or into a target of your choice
```

If you get stuck, look at our [E2E workflow](.github/workflows/ci_e2e.yml). We dedicate a machine with GPU to test our pipelines carefully. This workflow installs python packages and CARLA, collects data, runs example notebooks and trains a wide matrix of model configurations.

### 🛢 Get the data

We provide a dataset hosted on Hugging Face: 8,930 routes across 43 scenario types on all 12 CARLA town maps.

```console
user@host:~/lead$ pip install "huggingface-hub[hf_xet]"
user@host:~/lead$ hf download ln2697/lead-123d --repo-type dataset --local-dir data/lead/123D   # 1.1 TB
```

The download is resumable. Since the main purpose of the dataset is policy learning, we also deliver perturbed sensor views, which are not strictly required: add `--include 'logs/normal_view/*' 'maps/*' 'config.yaml'` to skip them, and the loader falls back to the nominal rig wherever a perturbed view is missing.

<div align="center">

| Modality                                       | Frequency | Format                                            |
| :--------------------------------------------- | :-------- | :------------------------------------------------ |
| RGB (6 cameras at 384 x 384)                   | 4 Hz      | JPEG, quality adapted to weather and daytime      |
| Depth (6 cameras at 384 x 384)                 | 4 Hz      | PNG, 8-bit linear quantization saturating at 50 m |
| Semantic segmentation (6 cameras at 384 x 384) | 4 Hz      | PNG, CARLA class ids                              |
| Instance segmentation (6 cameras at 384 x 384) | 4 Hz      | PNG, CARLA actor ids                              |
| Lidar (2 roof sensors, merged)                 | 20 Hz     | LAZ-compressed point cloud                        |
| Radar (4 sensors, merged)                      | 20 Hz     | Raw points with radial velocity                   |
| Ego states, boxes, traffic lights              | 20 Hz     | Arrow tables                                      |

</div>

See [data access](docs/data_access.md) for the full stream table and the on-disk layout.

Or collect the data yourself with customized configurations. The expert is rule-based and has no learning-based model in the loop. It runs at around 10 steps per second on a consumer GPU. We plan to extend it with an RL expert in the future. To let the expert drive a single route:

```console
user@host:~/lead$ python -m lead --expert --routes src/lead/routes/data_routes/lead/Accident/route_001761.xml
```

Collecting the full set of routes takes less than a day on a cluster 64 GTX 1080 Ti GPUs. See [data collection](docs/data_generation.md) for details, including how to scale the collection on SLURM.

### 🎨 Read and visualize data

The logs are plain Py123D, so the standard `py123d` package can read them without any imports from `lead`:

```python
>>> from py123d.api.scene.arrow.arrow_scene_builder import ArrowSceneBuilder
>>> from py123d.api.scene.scene_filter import SceneFilter
>>> from py123d.common.execution.thread_pool_executor import ThreadPoolExecutor
>>>
>>> scenes = ArrowSceneBuilder(
...    logs_root="data/lead/123D/logs",
...    maps_root="data/lead/123D/maps",
... ).get_scenes(
...     SceneFilter(
...         split_names=["normal_view"],
...         future_num_iterations=40,  # 2 s at the 20 Hz tick rate
...         required_scene_modalities=["camera:all@initial"],  # drop partial rigs
...     ),
...     ThreadPoolExecutor(),
... )
>>> scenes[0]
<py123d.api.scene.arrow.arrow_scene_api.ArrowSceneAPI object>
>>> scenes[0].get_ego_state_se3_at_iteration(0)
<py123d.datatypes.vehicle_state.ego_state.EgoStateSE3 object>
>>> scenes[0].get_box_detections_se3_at_iteration(0)
<py123d.datatypes.detections.box_detections.BoxDetectionsSE3 object>
```

See [data access](docs/data_access.md) for documentation on the data layout, and the [notebook](notebooks/data_access.ipynb) for a worked example. To inspect data, we provide two options: you can either point the standard Py123D `viser` tool at a log and look around:

```console
user@host:~/lead$ py123d-viser 'scene_filter.split_names=[normal_view]'                          # every log in normal_view
user@host:~/lead$ scripts/cli/viser data/lead/123D/logs/normal_view/<scenario_type>/<log_name>   # open a single log
```

Or you can visualize the input features and output labels of TransFuser, using only low-level drawing libraries such as `cv2` and `matplotlib`; see this [notebook](notebooks/data_visualization.ipynb) for more details.

### 💪 Training

Before training, you can optionally run cache building, which precomputes expensive features and inputs:

```console
user@host:~/lead$ bash scripts/common/build_cache.sh   # re-runnable, cached samples are skipped
```

Its manifest stores a `cache_finger_print` of the config. If the fingerprint changes, training detects the stale cache, fails, and requests a rebuild. After the cache is built, which should take at most two hours on a modern computer, training can start:

```console
user@host:~/lead$ python -m lead.training.train training.data.read_from_cache_store=true # omit to compute targets live
```

Checkpoints land under `outputs/`. See [training](docs/training.md) for the phases and config overrides, and [architecture](docs/architecture.md) for how a sample is assembled.

### 🏁 Closed-loop evaluation

Use your own checkpoints from `outputs/`, or download our trained ones from Hugging Face:

```console
user@host:~/lead$ hf download ln2697/transfuser-carla-123d --local-dir checkpoints
```

Evaluation drives against a live simulator, so start CARLA in a second terminal (the helper is on `PATH` after setup):

```console
user@host:~/lead$ scripts/cli/start_carla
```

Then point `python -m lead` at a checkpoint directory and a benchmark route. The policy is resolved from the checkpoint's `config.yaml`, so any policy implementing the API contract is evaluated the same way:

```console
# Bench2Drive
user@host:~/lead$ python -m lead --checkpoint checkpoints/transfuser --routes src/lead/routes/benchmark_routes/bench2drive/23687.xml --bench2drive

# Longest6 v2
user@host:~/lead$ python -m lead --checkpoint checkpoints/transfuser --routes src/lead/routes/benchmark_routes/longest6/00.xml

# Town13
user@host:~/lead$ python -m lead --checkpoint checkpoints/transfuser --routes src/lead/routes/benchmark_routes/Town13/0.xml
```

Routes for Bench2Drive, Town13, longest6, and Fail2Drive ship under `src/lead/routes/benchmark_routes/`. Fail2Drive needs the `--fail2drive` flag and its own simulator build under `3rd_party/CARLA/fail2drive_0915`, as it is not compatible with the standard CARLA release. Each run writes a per-route video and infraction report under `outputs/local_evaluation/<route_id>/`; aggregate multiple routes into benchmark scores with `scripts/common/result_parser.py` (`f2d_result_parser.py` for Fail2Drive). The `scripts/common/eval_*.sh` scripts show fully parameterized single-route runs for every benchmark.

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
