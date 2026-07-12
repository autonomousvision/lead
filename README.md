# LEAD

**Minimizing Learner-Expert Asymmetry in End-to-End Driving.**

LEAD is an end-to-end driving stack for the CARLA Leaderboard 2.0: a privileged expert that
collects data, a TransFuser-style policy that learns from it, and a batteries-included
harness to evaluate that policy on the standard CARLA benchmarks (Bench2Drive, Longest6,
Town13, Fail2Drive).

`main` is a full rewrite of the CVPR 2026 codebase, built on
[py123d](https://github.com/kesai-labs/py123d) in place of the old bespoke data layer — same
driving performance, substantially less code, and much faster data collection, loading, and
training. It is CARLA-only by design, and it is where ongoing development happens.

**To reproduce the paper, use the `cvpr2026` branch**, which is frozen at the submitted state
and still has everything the paper needs — including what `main` has since dropped.

> \[!WARNING\]
> **Repo is under active development.** Interfaces, configs, and checkpoints may change
> without notice, and it has not yet been validated end-to-end at the paper's numbers.

<div align="center">

| Date         | Content                                                                            |
| :----------- | :--------------------------------------------------------------------------------- |
| **26.xx.xx** | Deep clean: streamlined the codebase down to the core CARLA Leaderboard 2.0 stack. |

</div>

## 1. Quick Start

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
UV_PROJECT_ENVIRONMENT=$CONDA_PREFIX uv sync --extra dev

# Reload environment to add shell scripts to PATH
micromamba deactivate && micromamba activate lead
```

After this, edit `.env` yourself and continue with setting up CARLA

```bash
# Download and setup CARLA 0.9.16
bash scripts/common/setup_carla.sh
```

Run tests and lints

```bash
pre-commit install

pytest tests/unittests

pre-commit run --all-files
```

## 2. Configuration

Everything is one config tree (`src/lead/config/`), overridable from three places, highest
priority last: **profile** → **`LEAD_CONFIG`** → **CLI dotlist**.

```bash
# Env var (this is what the shell/Slurm scripts use)
export LEAD_CONFIG="agent.transfuser.image_architecture=regnety_032 training.experiment.seed=1"
```

Profiles (`src/lead/config_profiles/`) are named deltas on the defaults — pick one with
`expert.config_profile=leaderboard2_3cameras` or `agent.config_profile=transfuser`. See
[src/lead/config_profiles/README.md](src/lead/config_profiles/README.md).

Machine-local settings (CARLA path, dataset root, Slurm partition, parallelism) live in `.env`,
copied from [.env.example](.env.example). It is re-read on every access, so edits apply to
already-running jobs.

## 3. Data collection

The expert drives the data routes and writes a py123d dataset to `PY123D_DATA_ROOT` (`.env`).

**Single route**, for debugging the expert — starts CARLA yourself first (`start_carla`):

```bash
python -m lead --expert --routes src/lead/routes/data_routes/lead/noScenarios/short_route.xml
```

**Full dataset**, on Slurm. This submits one job per route file in
`src/lead/routes/data_routes/`, retries failures up to `MAX_NUM_ATTEMPTS_COLLECT_DATA`, and
keeps `MAX_NUM_PARALLEL_JOBS_COLLECT_DATA` jobs in flight (both in `.env`):

```bash
python scripts/slurm/data_collection/collect_data.py

# progress
python scripts/slurm/data_collection/print_collect_data_progress.py
```

Pass `--route_folder` to collect a subset instead.

Afterwards, prune routes the expert failed:

```bash
python scripts/slurm/data_quality/delete_failed_routes_py123d.py
```

## 4. Training

Two stages: **pretrain** the perception backbone, then **posttrain** with the planning decoder.
Both read the collected py123d dataset and are configured entirely through `LEAD_CONFIG`.
Training is Lightning; DDP across all visible GPUs is handled for you.

**Locally**, single node:

```bash
bash scripts/common/pretrain.sh    # → outputs/local_training/pretrain
bash scripts/common/posttrain.sh   # → outputs/local_training/posttrain
```

`posttrain.sh` loads `pretrain/model_0030.pth` and sets
`agent.transfuser.use_planning_decoder=true`. Edit the `LEAD_CONFIG` line at the top of either
script to change the run.

**On Slurm**, write an experiment script under `scripts/slurm/experiments/<experiment>/` and run
it — it resubmits itself as a batch job, so no `sbatch` needed:

```bash
bash scripts/slurm/experiments/001_example/000_pretrain1_0.sh
```

```bash
#!/usr/bin/bash
#SBATCH --gres=gpu:4
#SBATCH --time=3-00:00:00

source scripts/slurm/slurm_experiment_init.sh

export LEAD_CONFIG="$LEAD_CONFIG agent.transfuser.image_architecture=regnety_032"

train                              # or: posttrain <ckpt>; train
```

The trailing `_0` in the filename is the seed. Outputs, checkpoints (`model_XXXX.pth`), and the
run's `config.json` land in `outputs/training/<experiment>/<script>/<date>/`, mirrored to W&B.
`resume <last_output_dir>` continues an interrupted run in place.

## 5. Evaluation

A checkpoint directory is what you point evaluation at: it must contain `model_0030.pth` and
`config.json` (i.e. a training output dir).

**Single route**, locally, with CARLA already running:

```bash
python -m lead \
    --checkpoint outputs/checkpoints/tfv6_resnet34 \
    --routes src/lead/routes/benchmark_routes/Town13/0.xml

# Bench2Drive and Fail2Drive use their own leaderboard forks
python -m lead --checkpoint <ckpt> --routes src/lead/routes/benchmark_routes/bench2drive/23687.xml --bench2drive
python -m lead --checkpoint <ckpt> --routes src/lead/routes/benchmark_routes/fail2drive/0.xml --fail2drive
```

Results go to `outputs/local_evaluation/<route_id>/`. There are also ready-made single-route
scripts with all env vars pinned: `scripts/common/eval_{bench2drive,longest6,town13,fail2drive,expert}.sh`.

**Full benchmark**, on Slurm — same experiment-script pattern as training, one job per route:

```bash
#!/usr/bin/bash
source scripts/slurm/slurm_experiment_init.sh

export CHECKPOINT_DIR=outputs/training/001_example/010_postrain32_0/251025_182327

evaluate_bench2drive     # or evaluate_longest6 / evaluate_town13 / evaluate_fail2drive
```

Benchmarks and their route sets:

| Benchmark     | Routes                                         | Notes                                                      |
| :------------ | :--------------------------------------------- | :--------------------------------------------------------- |
| `bench2drive` | `src/lead/routes/benchmark_routes/bench2drive` | Short scenario routes                                      |
| `longest6`    | `src/lead/routes/benchmark_routes/longest6`    | Medium routes                                              |
| `Town13`      | `src/lead/routes/benchmark_routes/Town13`      | Long routes, official Leaderboard 2.0 validation town      |
| `fail2drive`  | `src/lead/routes/benchmark_routes/fail2drive`  | Needs the separate `3rd_party/CARLA/fail2drive_0915` build |

Aggregate the per-route JSONs into driving score / infractions:

```bash
python scripts/common/result_parser.py \
    --xml src/lead/routes/benchmark_routes/Town13.xml \
    --results outputs/evaluation/<experiment>/<script>/<date>

# fail2drive has its own parser
python scripts/common/f2d_result_parser.py --results <dir>
```

To inspect *why* a route failed, the webapp renders infractions with the recorded sensor data —
see [src/lead/webapp/README.md](src/lead/webapp/README.md).

## 6. Layout

```
src/lead/lead/          expert (perception, planning, dynamics, recorders)
src/lead/policy/        the learned policy (TransFuser)
src/lead/training/      Lightning training loop, dataloaders
src/lead/evaluation/    leaderboard agent wrapping the policy for inference
src/lead/config/        config tree; config_profiles/ holds named deltas
src/lead/routes/        data_routes/ (collection) and benchmark_routes/ (evaluation)
scripts/common/         single-machine scripts
scripts/slurm/          cluster drivers: collection, training, evaluation, experiments
3rd_party/              CARLA + the leaderboard/scenario_runner forks
```
