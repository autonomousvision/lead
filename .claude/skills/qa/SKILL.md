---
name: qa
description: Answer questions about the project by reading documentation sources in priority order. Use when the user asks "how does X work", "where is X", "what does X do", or any project-related question.
user-invocable: true
---

# Project Q&A

Answer the user's question about this project. Use the general lookup order first, then the topic-specific maps below for deeper dives.

## General lookup order

Read these in order:

1. `.claude/docs/project_structure.md` — directory layout and component overview.
2. `README.md` — high-level project description, setup, and usage.
3. `DEVELOPING.md` — development workflow, testing, and contribution guide.
4. `docs/source/` — detailed Sphinx documentation pages.

## Topic: Data generation / collection

- [slurm/data_collection/collect_data.py](slurm/data_collection/collect_data.py) — SLURM orchestrator launching parallel CARLA data collection jobs.
- [slurm/data_collection/delete_failed_routes.py](slurm/data_collection/delete_failed_routes.py) — cleans up failed route directories.
- [3rd_party/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py](3rd_party/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py) — runs CARLA scenarios per route.
- [lead/leaderboard_wrapper.py](lead/leaderboard_wrapper.py) — adapts LEAD to the CARLA leaderboard agent protocol.
- [lead/expert/expert.py](lead/expert/expert.py) — main expert agent driving logic.
- [lead/expert/expert_base.py](lead/expert/expert_base.py) — shared base class for expert variants.
- [lead/expert/expert_data.py](lead/expert/expert_data.py) — saves frames and labels during collection.
- [lead/expert/config_expert.py](lead/expert/config_expert.py) — expert configuration dataclass.
- [lead/expert/privileged_route_planner.py](lead/expert/privileged_route_planner.py) — privileged route/lane planning.

## Topic: Data preparation

- [lead/data_loader/carla_dataset.py](lead/data_loader/carla_dataset.py) — CARLA Leaderboard 2.0 PyTorch dataset.
- [lead/data_loader/carla_dataset_utils.py](lead/data_loader/carla_dataset_utils.py) — CARLA loading and preprocessing helpers.
- [lead/data_loader/training_cache.py](lead/data_loader/training_cache.py) — compressed on-disk training cache.
- [lead/data_loader/navsim_dataset.py](lead/data_loader/navsim_dataset.py) — NavSim PyTorch dataset.
- [lead/data_loader/waymo_e2e_dataset.py](lead/data_loader/waymo_e2e_dataset.py) — Waymo Open Dataset end-to-end dataset.
- [lead/data_buckets/abstract_bucket_collection.py](lead/data_buckets/abstract_bucket_collection.py) — ABC for bucket collections (serialization, loading).
- [lead/data_buckets/bucket.py](lead/data_buckets/bucket.py) — single bucket holding paths and metadata.
- [lead/data_buckets/route_filtering.py](lead/data_buckets/route_filtering.py) — route validity and deduplication.
- [scripts/build_cache.py](scripts/build_cache.py) — entry point to build the training cache.
- [scripts/build_buckets_pretrain.py](scripts/build_buckets_pretrain.py) — generate pretraining bucket collections.

## Topic: Training

- [lead/training/train.py](lead/training/train.py) — main DDP-aware training entry point (Trainer class).
- [lead/training/config_training.py](lead/training/config_training.py) — training hyperparameters dataclass.
- [lead/training/training_utils.py](lead/training/training_utils.py) — seeding, checkpointing, data loader construction.
- [lead/training/mixed_training_utils.py](lead/training/mixed_training_utils.py) — multi-dataset sample scheduling.
- [lead/training/logger.py](lead/training/logger.py) — WandB metric and media logging.
- [scripts/pretrain_ddp.sh](scripts/pretrain_ddp.sh) — DDP pretraining launcher.
- [scripts/posttrain_ddp.sh](scripts/posttrain_ddp.sh) — DDP posttraining launcher.
- [slurm/experiments/](slurm/experiments/) — per-experiment SLURM scripts (see `/prepare-slurm-scripts`).

## Topic: Model architecture (TransFuser v6)

- [lead/tfv6/tfv6.py](lead/tfv6/tfv6.py) — top-level model orchestrating backbone and decoder heads, loss aggregation.
- [lead/tfv6/transfuser_backbone.py](lead/tfv6/transfuser_backbone.py) — image + LiDAR fusion backbone.
- [lead/tfv6/transfuser_utils.py](lead/tfv6/transfuser_utils.py) — backbone utility layers and functions.
- [lead/tfv6/planning_decoder.py](lead/tfv6/planning_decoder.py) — waypoint, path and target speed planning head.
- [lead/tfv6/bev_decoder.py](lead/tfv6/bev_decoder.py) — BEV semantic segmentation head.
- [lead/tfv6/perspective_decoder.py](lead/tfv6/perspective_decoder.py) — perspective view output head.
- [lead/tfv6/center_net_decoder.py](lead/tfv6/center_net_decoder.py) — CenterNet 2D bounding box detector.
- [lead/tfv6/radar_detector.py](lead/tfv6/radar_detector.py) — 2D radar object detector.

## Topic: Evaluation (bench2drive, longest6, town13)

- [slurm/evaluation/evaluate.py](slurm/evaluation/evaluate.py) — core orchestrator: submit, monitor, retry SLURM eval jobs.
- [slurm/evaluation/evaluate_scripts_generator.py](slurm/evaluation/evaluate_scripts_generator.py) — generates per-route shell scripts.
- [slurm/evaluation/evaluate_utils.py](slurm/evaluation/evaluate_utils.py) — JSON parsing, status checks.
- [slurm/evaluation/merge_route_json.py](slurm/evaluation/merge_route_json.py) — merges per-route results into one summary.
- [slurm/evaluation/evaluate_wandb_logger.py](slurm/evaluation/evaluate_wandb_logger.py) — logs aggregated eval results to WandB.
- [lead/inference/closed_loop_inference.py](lead/inference/closed_loop_inference.py) — CARLA closed-loop evaluation with PID control.
- [lead/inference/open_loop_inference.py](lead/inference/open_loop_inference.py) — open-loop evaluation (NavSim / Bench2Drive).
- [lead/inference/sensor_agent.py](lead/inference/sensor_agent.py) — sensor-based agent wrapper for the CARLA leaderboard protocol.
- [lead/inference/infraction_recorder.py](lead/inference/infraction_recorder.py) — records and serializes driving infractions.
- [lead/inference/config_closed_loop.py](lead/inference/config_closed_loop.py) — closed-loop eval config.

## Rules

1. **Read in order.** Start with general docs, then dive into topic-specific files.
2. **Be specific.** Point the user to the exact file, section, or line.
3. **Don't guess.** If none of the sources answer the question, say so and suggest where else to look.
