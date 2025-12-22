<h2 align="center">
<b> LEAD: Minimizing Learner–Expert Asymmetry in End-to-End Driving </b>
</h2>

<p align="center">
  <a href="https://ln2697.github.io/lead"><strong>Project Page</strong></a> ·
  <a href="https://ln2697.github.io/lead/docs"><strong>Documentation</strong></a> ·
  <a href="https://huggingface.co/ln2697/TFv6"><strong>Weights</strong></a> ·
  <a href="https://arxiv.org/abs/XXXX.XXXXX"><strong>Paper</strong></a> ·
  <a href="https://ln2697.github.io/assets/pdf/Nguyen2026LEADSUPP.pdf"><strong>Supplemental</strong></a>
</p>

<p align="center">This repository contains the official code implementation, trained models, data, and experimental setup for LEAD and TFv6, a state-of-the-art expert-student policy pair in CARLA.</p>

## Main features
- Lean pipeline: Python first, pure PyTorch, minimal dependencies.
- Cross-dataset training: Support for NAVSIM, Waymo Perception and Waymo E2E.
- Data-centric infrastructure:
  - Notebooks for data visualization and debugging.
  - BearType & JaxTyping for type & tensor shape safety.
  - Loggings for reproducibility and observability.

The repository follows a data-first design and is intended to be easily extended with custom models.

## TODOs

Coming soon:
- [ ] Datasets
- [ ] Pixi support
- [ ] CARLA 0.9.16 support

## Updates

- `2025/12/23` Arxiv Paper Release

- `2025/12/23` Code Release

## Setup project

To follow those steps, we assume you have [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install) installed. This guide was tested on Ubuntu 24.04.

**1. Clone project**

```bash
# 1. Clone the repository
git clone git@github.com:autonomousvision/lead.git
cd lead
```

**2. Setup environment variables**

If you are on Bash

```bash
echo "export LEAD_PROJECT_ROOT=$(pwd)" >> ~/.bashrc
echo "source $(pwd)/scripts/main.sh" >> ~/.bashrc
source ~/.bashrc
```

Alternatively on Zsh

```bash
echo "export LEAD_PROJECT_ROOT=$(pwd)" >> ~/.zshrc
echo "source $(pwd)/scripts/main.sh" >> ~/.zshrc
source ~/.zshrc
```

**3. Create python environment**

```bash
# Install conda-lock
pip install conda-lock

# Create conda environment
conda-lock install -n lead conda-lock.yml

# Activate conda environment
conda activate lead

# Install dependencies
pip install -r requirements.txt
```

**4. Setup CARLA**

Install CARLA 0.9.15 at `3rd_party/CARLA_0915`

```bash
bash scripts/setup_carla.sh
```

Or if you have CARLA 0.9.15 locally, link it to

```bash
ln -s /your/carla/path $LEAD_PROJECT_ROOT/3rd_party/CARLA_0915
```

**5. Setup pre-commit hooks**

```bash
pre-commit install
```

**6. Install further dependencies**

```bash
conda install conda-forge::ffmpeg
conda install conda-forge::parallel
conda install conda-forge::tree
```

## Quick Start

Verify your setup by evaluating expert and model. To follow those steps we assume you have [git lfs](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage) installed. This guide was tested with RTX-2080ti and CUDA 13.0.

**1. Download model checkpoints**

We provide pre-trained checkpoints on [HuggingFace](https://huggingface.co/ln2697/TFv6) for reproducibility. We report Driving Score metric here, where higher means better.

<div align="center">

| Checkpoint                                                                                    | Description               | Bench2Drive | Longest6 v2 |
| --------------------------------------------------------------------------------------------- | ------------------------- | :---------: | :---------: |
| [tfv6_regnety032](https://huggingface.co/ln2697/TFv6/tree/main/tfv6_regnety032)               | TFv6                      |  **95.2**   |   **62**    |
| [tfv6_resnet34](https://huggingface.co/ln2697/TFv6/tree/main/tfv6_resnet34)                   | ResNet34 Backbone         |    94.7     |     57      |
| [4cameras_resnet34](https://huggingface.co/ln2697/TFv6/tree/main/4cameras_resnet34)           | Additional rear camera    |    95.1     |     53      |
| [noradar_resnet34](https://huggingface.co/ln2697/TFv6/tree/main/noradar_resnet34)             | No radar sensor           |    94.7     |     52      |
| [visiononly_resnet34](https://huggingface.co/ln2697/TFv6/tree/main/visiononly_resnet34)       | Vision-only driving model |    91.6     |     43      |
| [town13heldout_resnet34](https://huggingface.co/ln2697/TFv6/tree/main/town13heldout_resnet34) | Generalization evaluation |    93.1     |     52      |

</div>

You can either download all checkpoints at once

```bash
git clone https://huggingface.co/ln2697/TFv6 outputs/checkpoints
cd outputs/checkpoints
git lfs pull
```

Or you can download only one checkpoint

```bash
mkdir -p outputs/checkpoints/tfv6_resnet34
wget https://huggingface.co/ln2697/TFv6/resolve/main/tfv6_resnet34/config.json -O outputs/checkpoints/tfv6_resnet34/config.json
wget https://huggingface.co/ln2697/TFv6/resolve/main/tfv6_resnet34/model_0030_0.pth -O outputs/checkpoints/tfv6_resnet34/model_0030_0.pth
```

**2. Run model evaluation**

See evaluation configuration at `lead/inference/config_closed_loop.py`. You might want to turn `produce_demo_video` and `produce_debug_video` off to save compute and evaluate faster.

Start CARLA
```bash
bash scripts/start_carla.sh
```

Evaluate one Bench2Drive test route
```bash
bash scripts/eval_bench2drive.sh
```

Results will be saved to `outputs/local_evaluation`

```html
outputs/local_evaluation
├── 23687
│   ├── checkpoint_endpoint.json
│   ├── debug_images
│   ├── demo_images
│   └── metric_info.json
├── 23687_debug.mp4
└── 23687_demo.mp4
```

**3. Run expert evaluation**

Assume CARLA server running

```bash
bash scripts/run_expert.sh
```

Data collected at `data/expert_debug`

```html
data/expert_debug
├── data
│   └── BlockedIntersection
│       └── 999_Rep-1_Town06_13_route0_12_22_22_34_45
│           ├── bboxes
│           ├── depth
│           ├── depth_perturbated
│           ├── hdmap
│           ├── hdmap_perturbated
│           ├── lidar
│           ├── metas
│           ├── radar
│           ├── radar_perturbated
│           ├── results.json
│           ├── rgb
│           ├── rgb_perturbated
│           ├── semantics
│           └── semantics_perturbated
└── results
    └── Town06_13_result.json
```

## Documentation and Further Resources

For detailed training, data-collection, and large-scale instructions, see the [full documentation](https://ln2697.github.io/lead/docs). In particular, we provide
- [Tutorial Notebooks](https://ln2697.github.io/lead/docs/jupyter_notebooks.html)
- [Cross-dataset Training](https://ln2697.github.io/lead/docs/cross_dataset_training.html)

The predecessor repository [carla_garage](https://github.com/autonomousvision/carla_garage) provides many useful documentations, in particular:
- [CARLA Coordinate Systems](https://github.com/autonomousvision/carla_garage/blob/leaderboard_2/docs/coordinate_systems.md)
- [History of TransFuser](https://github.com/autonomousvision/carla_garage/blob/leaderboard_2/docs/history.md)
- [Common Mistakes in Benchmarking Autonomous Driving](https://github.com/autonomousvision/carla_garage/blob/leaderboard_2/docs/common_mistakes_in_benchmarking_ad.md)

Other helpful repositories
* [SimLingo](https://github.com/RenzKa/simlingo), [PlanT2](https://github.com/autonomousvision/plant2), [Bench2Drive Leaderboard](https://github.com/autonomousvision/Bench2Drive-Leaderboard), [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive/), [CaRL](https://github.com/autonomousvision/CaRL), [carla_route_generator](https://github.com/autonomousvision/carla_route_generator)

E2E self-driving research
* [Why study self-driving?](https://emergeresearch.substack.com/p/why-study-self-driving?triedRedirect=true)
* [End-to-end Autonomous Driving: Challenges and Frontiers](https://arxiv.org/abs/2306.16927)

## Acknowledgements

Special thanks to [carla_garage](https://github.com/autonomousvision/carla_garage) for the foundational codebase. We also thank the creators of the numerous open-source projects.

* [PDM-Lite](https://github.com/OpenDriveLab/DriveLM/blob/DriveLM-CARLA/pdm_lite/docs/report.pdf), [leaderboard](https://github.com/carla-simulator/leaderboard), [scenario_runner](https://github.com/carla-simulator/scenario_runner), [NAVSIM](https://github.com/autonomousvision/navsim), [Waymo Open Dataset](https://github.com/waymo-research/waymo-open-dataset)

This project was primarily developed by Long Nguyen, who led and implemented the core experiments. Kashyap Chitta, Bernhard Jaeger, and Andreas Geiger contributed through technical discussion and advisory feedback.

## Citation

If you find this work useful, please consider starring the repository ⭐ and citing:

```bibtex
@article{nguyen2026lead,
  title={LEAD: Minimizing Learner-Expert Asymmetry in End-to-End Driving},
  author={Nguyen, Long and Fauth, Micha and Jaeger, Bernhard and Dauner, Daniel and Igl, Maximilian and Geiger, Andreas and Chitta, Kashyap},
  journal={arXiv preprint},
  year={2026}
}
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
