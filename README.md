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

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue.svg" alt="Python 3.10">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-orange.svg" alt="PyTorch 2.0+">
  <img src="https://img.shields.io/badge/CARLA-0.9.15-green.svg" alt="CARLA 0.9.15">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
</p>

<p align="center">Official implementation of LEAD and TFv6, an expert-student policy pair for autonomous driving research in CARLA. Includes a complete pipeline for data collection, training, and closed-loop evaluation.</p>

<p align="center">
  <img src="docs/assets/banner.webp" alt="LEAD Banner" width="80%">
</p>

## Main Features

LEAD is a model-agnostic entry point to end-to-end driving research with CARLA simulator. Built with a simple, modular structure that
supports fast and parallel experiments.

- **Lean pipeline**: Pure PyTorch with minimal dependencies and an iteration-focused implementation.
- **Cross-dataset training**: Training and evaluation support for NAVSIM and Waymo datasets.
- **Data-centric infrastructure**:
  - Interactive notebooks for data exploration, visualization and debugging.
  - Type and tensor shape safety with BearType & JaxTyping.
  - Compact datasets with lower storage overhead (72h of driving fits in ~200GB).

## Table of Contents

- [Roadmap](#roadmap)
- [Updates](#updates)
- [Setup Project](#setup-project)
- [Quick Start](#quick-start)
- [Bench2Drive Results](#bench2drive-results)
- [Documentation and Resources](#documentation-and-resources)
- [External Resources](#external-resources)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)
- [License](#license)

## Roadmap

- [x] ✅ Checkpoints and inference code
- [x] 🚧 Documentation, training pipeline and expert code
- [ ] Full dataset release on HuggingFace
- [ ] Cross-dataset training tools and documentation

Status: Active development. Core code and checkpoints are released; remaining components coming soon.

## Updates

- `2025/12/23` Arxiv Paper Release

- `2025/12/23` Code Release

## Setup Project

> ⏱️ **15 minutes**

**1. Clone project**

```bash
git clone https://github.com/autonomousvision/lead.git
cd lead
```

**2. Setup environment variables**

```bash
{
  echo
  echo "export LEAD_PROJECT_ROOT=$(pwd)"
  echo "source $(pwd)/scripts/main.sh"
} >> ~/.bashrc

source ~/.bashrc
```

<details>
<summary>For Zsh</summary>

```bash
{
  echo
  echo "export LEAD_PROJECT_ROOT=$(pwd)"
  echo "source $(pwd)/scripts/main.sh"
} >> ~/.zshrc

source ~/.zshrc
```

</details>

**3. Create python environment with [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install)**

```bash
pip install conda-lock
conda-lock install -n lead conda-lock.yml
conda activate lead
pip install -r requirements.txt
pip install -e . # Install project
```

**4. Setup CARLA**

Install CARLA 0.9.15 at `3rd_party/CARLA_0915`

```bash
bash scripts/setup_carla.sh
```

<details>
<summary>Or softlink existing CARLA</summary>

```bash
ln -s /your/carla/path $LEAD_PROJECT_ROOT/3rd_party/CARLA_0915
```

</details>

**5. Further setup**

```bash
pre-commit install # Git hooks
conda install conda-forge::ffmpeg conda-forge::parallel conda-forge::tree # Misc
```

**Note**

We also provide a minimal docker compose setup (not extensively tested yet) [here](docker-README.md).

## Quick Start

> ⏱️ **5 minutes**

**1. Download model checkpoints**

We provide pre-trained checkpoints on [HuggingFace](https://huggingface.co/ln2697/TFv6) for reproducibility.

<div align="center">

| Checkpoint                                                                                    | Description               | Bench2Drive | Longest6 v2 |  Town13  |
| --------------------------------------------------------------------------------------------- | ------------------------- | :---------: | :---------: | :------: |
| [tfv6_regnety032](https://huggingface.co/ln2697/TFv6/tree/main/tfv6_regnety032)               | TFv6                      |  **95.2**   |   **62**    | **5.01** |
| [tfv6_resnet34](https://huggingface.co/ln2697/TFv6/tree/main/tfv6_resnet34)                   | ResNet34 Backbone         |    94.7     |     57      |   3.31   |
| [4cameras_resnet34](https://huggingface.co/ln2697/TFv6/tree/main/4cameras_resnet34)           | Additional rear camera    |    95.1     |     53      |    -     |
| [noradar_resnet34](https://huggingface.co/ln2697/TFv6/tree/main/noradar_resnet34)             | No radar sensor           |    94.7     |     52      |    -     |
| [visiononly_resnet34](https://huggingface.co/ln2697/TFv6/tree/main/visiononly_resnet34)       | Vision-only driving model |    91.6     |     43      |    -     |
| [town13heldout_resnet34](https://huggingface.co/ln2697/TFv6/tree/main/town13heldout_resnet34) | Generalization evaluation |    93.1     |     52      |   2.65   |

</div>

To download one checkpoint:

```bash
mkdir -p outputs/checkpoints/tfv6_resnet34
wget https://huggingface.co/ln2697/TFv6/resolve/main/tfv6_resnet34/config.json -O outputs/checkpoints/tfv6_resnet34/config.json
wget https://huggingface.co/ln2697/TFv6/resolve/main/tfv6_resnet34/model_0030_0.pth -O outputs/checkpoints/tfv6_resnet34/model_0030_0.pth
```

<details>
<summary>Alternatively, to download all checkpoints at once with <a href="https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage">git lfs</a>:</summary>

```bash
git clone https://huggingface.co/ln2697/TFv6 outputs/checkpoints
cd outputs/checkpoints
git lfs pull
```

</details>

**2. Run model evaluation**

See evaluation configuration at [config_closed_loop](lead/inference/config_closed_loop.py). Turn off the options `produce_demo_video` and `produce_debug_video` for faster evaluation. By default, the pipeline loads all three seeds of a checkpoint as an ensemble. If memory is a problem,
simply change prefix of two of the three seeds so only the first seed is loaded.

```bash
bash scripts/start_carla.sh # Start CARLA server
bash scripts/eval_bench2drive.sh # Evaluate one Bench2Drive route
bash scripts/clean_carla.sh # Optional: clean CARLA server
```

<details>
<summary>Results will be saved to <code>outputs/local_evaluation</code> with the following structure:</summary>

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

</details>

**3. Run expert evaluation**

Evaluate expert and collect data

```bash
bash scripts/start_carla.sh # Start CARLA if not done already
bash scripts/run_expert.sh # Run expert on one route
```

<details>
<summary>Data collected will be stored at <code>data/expert_debug</code> and should have following structure:</summary>

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

</details>

## Bench2Drive Results

We evaluate TFv6 on the [Bench2Drive](https://github.com/autonomousvision/Bench2Drive-Leaderboard/tree/ab8021b027fa9c4765f9a732355d3b2ae93736a0) benchmark, which consists of 220 routes across multiple towns with challenging weather conditions and traffic scenarios.

<div align="center">

| Method          | Venue    |    DS     |    SR     |   Merge   | Overtake  | EmgBrake  | Give Way  | Traffsign |
| --------------- | -------- | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |
| TF++ (TFv5)     | ICCV23   |   84.21   |   67.27   |   58.75   |   57.77   |   83.33   |   40.00   |   82.11   |
| SimLingo        | CVPR25   |   85.07   |   67.27   |   54.01   |   57.04   |   88.33   | **53.33** |   82.45   |
| R2SE            | -        |   86.28   |   69.54   |   53.33   |   61.25   |   90.00   |   50.00   |   84.21   |
| HiP-AD          | ICCV25   |   86.77   |   69.09   |   50.00   |   84.44   |   83.33   |   40.00   |   72.10   |
| BridgeDrive     | -        |   86.87   |   72.27   |   63.50   |   57.77   |   83.33   |   40.00   |   82.11   |
| DiffRefiner     | AAAI26   |   87.10   |   71.40   |   63.80   |   60.00   |   85.00   |   50.00   |   86.30   |
| **TFv6 (Ours)** | **Ours** | **95.28** | **86.80** | **72.50** | **97.77** | **91.66** |   40.00   | **89.47** |

<em>DS = Driving Score, SR = Success Rate; Metrics follow the CARLA Leaderboard 2.0 protocol. Higher is better.</em>

</div>

## Documentation and Resources

For detailed training, data-collection, and large-scale instructions, see the [full documentation](https://ln2697.github.io/lead/docs). In particular, we provide:
- [Tutorial Notebooks](https://ln2697.github.io/lead/docs/jupyter_notebooks.html)
- [Cross-dataset Training](https://ln2697.github.io/lead/docs/cross_dataset_training.html)
- [Frequently Asked Questions](https://ln2697.github.io/lead/docs/faq)
- [Known Issues](https://ln2697.github.io/lead/docs/known_issues.html)

We maintain custom forks of CARLA evaluation tools with our modifications:
* [scenario_runner_autopilot](https://github.com/ln2697/scenario_runner_autopilot), [leaderboard_autopilot](https://github.com/ln2697/leaderboard_autopilot), [Bench2Drive](https://github.com/ln2697/Bench2Drive), [scenario_runner](https://github.com/ln2697/scenario_runner), [leaderboard](https://github.com/ln2697/leaderboard)

## External Resources

Useful documentations from other repositories:
- [CARLA Coordinate Systems](https://github.com/autonomousvision/carla_garage/blob/leaderboard_2/docs/coordinate_systems.md)
- [History of TransFuser](https://github.com/autonomousvision/carla_garage/blob/leaderboard_2/docs/history.md)
- [Common Issues with CARLA](https://github.com/Thinklab-SJTU/Bench2Drive?tab=readme-ov-file#deal-with-carla)
- [Some Useful Commands of CARLA](https://github.com/autonomousvision/CaRL/tree/main/CARLA#local-debugging)
- [About Longest6 v2 Benchmark](https://github.com/autonomousvision/CaRL/tree/main/CARLA#longest6-v2)
- [About Town13 Benchmark](https://github.com/autonomousvision/carla_garage?tab=readme-ov-file#carla-leaderboard-20-validation-routes)
- [Random Scenario Generation](https://github.com/autonomousvision/CaRL/tree/main/CARLA#scenario-generation)
- [Manual Scenario Labeling](https://github.com/autonomousvision/carla_route_generator)

Other helpful repositories:
* [SimLingo](https://github.com/RenzKa/simlingo), [PlanT2](https://github.com/autonomousvision/plant2), [Bench2Drive Leaderboard](https://github.com/autonomousvision/Bench2Drive-Leaderboard), [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive/), [CaRL](https://github.com/autonomousvision/CaRL)

E2E self-driving research:
* [Why study self-driving?](https://emergeresearch.substack.com/p/why-study-self-driving?triedRedirect=true)
* [End-to-end Autonomous Driving: Challenges and Frontiers](https://arxiv.org/abs/2306.16927)
* [Common Mistakes in Benchmarking Autonomous Driving](https://github.com/autonomousvision/carla_garage/blob/leaderboard_2/docs/common_mistakes_in_benchmarking_ad.md)

## Acknowledgements

Special thanks to [carla_garage](https://github.com/autonomousvision/carla_garage) for the foundational codebase. We also thank the creators of the numerous open-source projects.

* [PDM-Lite](https://github.com/OpenDriveLab/DriveLM/blob/DriveLM-CARLA/pdm_lite/docs/report.pdf), [leaderboard](https://github.com/carla-simulator/leaderboard), [scenario_runner](https://github.com/carla-simulator/scenario_runner), [NAVSIM](https://github.com/autonomousvision/navsim), [Waymo Open Dataset](https://github.com/waymo-research/waymo-open-dataset)

Long Nguyen led development of the project. Kashyap Chitta, Bernhard Jaeger, and Andreas Geiger contributed through technical discussion and advisory feedback.

## Citation

If you find this work useful, please consider give this repository a star ⭐. Also cite our work if you use it in your research:

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
