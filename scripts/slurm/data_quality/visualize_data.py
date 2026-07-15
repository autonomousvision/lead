#!/usr/bin/env python3
# SBATCH --ntasks=1
# SBATCH --nodes=1
# SBATCH --time=1-00:00:00
# SBATCH --gres=gpu:0
# SBATCH --cpus-per-task=24
# SBATCH --mem=128gb

"""Render py123d expert logs frame by frame with the ground-truth visualizer.

Frames are split into successful/failed subfolders by route outcome, grouped
within each per ``--group-by``, and rendered across worker processes.
"""

import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait

import torch
from PIL import Image
from slurm.slurm_init import slurm_init
from tqdm import tqdm

from lead.config import LeadConfig, load_lead_config
from lead.policy.transfuser.dataloader.dataset import TransfuserDataset
from lead.policy.transfuser.visualization.ground_truth_visualizer import (
    GroundTruthVisualizer,
)

RESIZE_FACTOR = 0.75


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--random",
        action="store_true",
        help="Draw samples in random order instead of sequentially.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Stop after rendering this many frames per view (default: all).",
    )
    parser.add_argument(
        "--group-by",
        choices=["route", "scenario_type", "scenario_type_dir", "flat"],
        default="scenario_type_dir",
        help=(
            "How to organize output images: one folder per route, one folder per "
            "currently active scenario type, one folder per the route's scenario "
            "directory, or a single flat folder (default: scenario_type_dir)."
        ),
    )
    return parser.parse_args()


def _is_failed_record(record: dict) -> bool:
    """A route counts as failed only for driving mistakes (collision, infraction).

    Software failures (agent/simulation crashes, setup errors) are excluded:
    those runs have no recorded data to visualize in the first place.
    """
    score_explained_by_excused_infractions = record["num_infractions"] == len(
        record["infractions"]["min_speed_infractions"],
    ) + len(record["infractions"]["outside_route_lanes"])
    return (
        record["scores"]["score_composed"] < 100.0
        and not score_explained_by_excused_infractions
    )


def _collect_failed_timestamps(lead_config: LeadConfig) -> set[str]:
    """Route timestamps (from leaderboard results) whose run counts as failed."""
    data_root = lead_config.training.data.py123d_data_root
    results_root = os.path.join(data_root, "results")
    failed_timestamps: set[str] = set()
    if not os.path.isdir(results_root):
        return failed_timestamps

    for scenario in os.listdir(results_root):
        scenario_dir = os.path.join(results_root, scenario)
        if not os.path.isdir(scenario_dir):
            continue
        for file_name in os.listdir(scenario_dir):
            if not file_name.endswith(".json"):
                continue
            with open(os.path.join(scenario_dir, file_name), encoding="utf-8") as f:
                records = json.load(f)["_checkpoint"]["records"]
            if records and _is_failed_record(records[0]):
                failed_timestamps.add(records[0]["timestamp"])
    return failed_timestamps


def _render_and_save_frame(
    lead_config: LeadConfig,
    sample: dict,
    output_dir: str,
    frame_index: int,
) -> None:
    """Render one frame and save it to disk; runs in a worker process."""
    image = GroundTruthVisualizer(lead_config, data=sample).visualize()
    img = Image.fromarray(image)
    img = img.resize((int(img.width * RESIZE_FACTOR), int(img.height * RESIZE_FACTOR)))
    img.save(
        os.path.join(output_dir, f"rgb_{str(frame_index).zfill(6)}.jpeg"),
        quality=80,
        optimize=True,
    )


def render_frames(
    lead_config: LeadConfig,
    view_name: str,
    group_by: str,
    random_order: bool = False,
    max_samples: int | None = None,
) -> None:
    output_root = os.path.join(
        lead_config.training.data.py123d_data_root,
        "transfuser_visualization",
    )
    print(f"Visualizing {view_name} -> {output_root}")

    # Split the job's assigned cores between loading (I/O + decode) and
    # rendering (CPU-bound drawing) so the two pools don't oversubscribe it.
    num_cpus = lead_config.training.assigned_cpu_cores
    render_workers = max(1, num_cpus // 2)
    loader_workers = max(1, num_cpus - render_workers)
    max_pending_frames = render_workers * 4

    failed_timestamps = _collect_failed_timestamps(lead_config)
    dataset = TransfuserDataset(lead_config)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=random_order,
        num_workers=loader_workers,
    )
    num_samples = min(max_samples, len(dataloader)) if max_samples else len(dataloader)

    frame_counters: dict[str, int] = defaultdict(int)
    status_by_log: dict[str, str] = {}
    made_dirs: set[str] = set()

    with (
        ProcessPoolExecutor(max_workers=render_workers) as executor,
        tqdm(total=num_samples, desc=f"Rendering {view_name}") as pbar,
    ):
        pending: set[Future] = set()
        for i, sample in enumerate(dataloader):
            if i >= num_samples:
                break

            log_name = sample["route_number"][0]
            status = status_by_log.setdefault(
                log_name,
                "failed"
                if any(log_name.endswith(ts) for ts in failed_timestamps)
                else "success",
            )
            if group_by == "route":
                group = log_name
            elif group_by == "scenario_type":
                group = sample["scenario_type"][0]
            elif group_by == "scenario_type_dir":
                group = sample["scenario_type_dir"][0]
            else:  # flat
                group = ""
            output_dir = os.path.join(output_root, status, view_name, group)
            if output_dir not in made_dirs:
                os.makedirs(output_dir, exist_ok=True)
                made_dirs.add(output_dir)

            frame_index = frame_counters[output_dir]
            frame_counters[output_dir] += 1

            pending.add(
                executor.submit(
                    _render_and_save_frame,
                    lead_config,
                    sample,
                    output_dir,
                    frame_index,
                ),
            )
            if len(pending) >= max_pending_frames:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    future.result()
                    pbar.update(1)

        for future in pending:
            future.result()
            pbar.update(1)


def visualize_normal_view(
    group_by: str,
    random_order: bool = False,
    max_samples: int | None = None,
) -> None:
    lead_config = load_lead_config()
    lead_config.training.data.use_sensor_perburtation = False
    lead_config.policy.transfuser.use_planning_decoder = False
    lead_config.training.experiment.visualize_dataset = True
    render_frames(
        lead_config,
        lead_config.training.data.py123d_split,
        group_by,
        random_order,
        max_samples,
    )


def visualize_perturbated_view(
    group_by: str,
    random_order: bool = False,
    max_samples: int | None = None,
) -> None:
    lead_config = load_lead_config(
        loaded_config={"training": {"data": {"use_sensor_perburtation_prob": 1.0}}},
    )
    lead_config.policy.transfuser.use_planning_decoder = False
    lead_config.training.experiment.visualize_dataset = True
    render_frames(
        lead_config,
        lead_config.expert.data_collection.py123d_perturbated_split,
        group_by,
        random_order,
        max_samples,
    )


if __name__ == "__main__":
    slurm_init()
    os.environ["OMP_NUM_THREAD"] = "1"
    args = parse_args()
    visualize_normal_view(args.group_by, args.random, args.max_samples)
    visualize_perturbated_view(args.group_by, args.random, args.max_samples)
