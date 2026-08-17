#!/usr/bin/env python3
import argparse
import json
import logging
import os
import subprocess
import time
from collections.abc import Iterable
from pathlib import Path

from lead.common import runtime_variables
from lead.common.logging_setup import setup_logging

LOG = logging.getLogger(__name__)

LEADERBOARD_FLAGS = {
    "standard": "",
    "bench2drive": "--bench2drive",
    "fail2drive": "--fail2drive",
}
CARLA_BOOT_ATTEMPTS = 5
CARLA_BOOT_TIMEOUT = 60
POLL_INTERVAL_SECONDS = 30

JOB_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --output={output_dir}/logs/{route_id}.log
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --gres={gres}
#SBATCH --time={time_limit}

cd {code_dir}
export PATH={code_dir}/scripts/cli:$PATH
export PYTHONUNBUFFERED=1

trap '[ -n "$CARLA_PID" ] && clean_carla "$CARLA_PID"' EXIT

# CARLA dies within seconds when one of its ports is taken, so every attempt
# re-rolls the ports and a collision heals inside the job instead of by resubmit.
CARLA_READY=0
for attempt in $(seq 1 {boot_attempts}); do
	PORT=$(random_free_port)
	TM_PORT=$(random_free_port)
	STREAMING_PORT=$(random_free_port)
	echo "Attempt $attempt: PORT=$PORT TM_PORT=$TM_PORT STREAMING_PORT=$STREAMING_PORT"

	# setsid makes CARLA its own process group, so killing it takes down the
	# engine that CarlaUE4.sh starts as a child and not just the wrapper.
	setsid start_carla "$PORT" "$STREAMING_PORT" &
	CARLA_PID=$!

	for i in $(seq 1 {boot_timeout}); do
		sleep 1
		if ! kill -0 "$CARLA_PID" 2>/dev/null; then
			echo "CARLA died after $i s"
			break
		fi
		if test_carla_connection "$PORT"; then
			echo "CARLA serving on port $PORT after $i s"
			CARLA_READY=1
			break
		fi
	done

	if [ "$CARLA_READY" -eq 1 ]; then
		break
	fi
	clean_carla "$CARLA_PID"
	CARLA_PID=""
done

if [ "$CARLA_READY" -ne 1 ]; then
	echo "CARLA did not come up after {boot_attempts} attempts" >&2
	exit 1
fi

python3 -m lead \\
	--checkpoint {checkpoint} \\
	--routes {route_file} {leaderboard_flag} \\
	--port "$PORT" \\
	--traffic-manager-port "$TM_PORT" \\
	--output-dir {route_output_dir}
"""


def result_path(output_dir: Path, route_id: str) -> Path:
    """The leaderboard result file the given route writes.

    Args:
        output_dir: The evaluation run's output directory.
        route_id: Route identifier, the route XML's stem.

    Returns:
        Path of the route's checkpoint_endpoint.json.
    """
    return output_dir / "routes" / route_id / "checkpoint_endpoint.json"


def route_needs_rerun(result: Path) -> bool:
    """Whether a route still has to run, or run again after a crash.

    A leaderboard failure with a nonzero route score is driving behaviour and
    counts as finished; a missing, unreadable or zero-score result is a crash.

    Args:
        result: The route's checkpoint_endpoint.json path.

    Returns:
        True when the route must be (re)submitted.
    """
    if not result.exists():
        return True
    try:
        checkpoint = json.loads(result.read_text(encoding="utf-8"))["_checkpoint"]
        records = checkpoint["records"]
    except (OSError, ValueError, KeyError):
        return True
    return not records or any(
        record["status"] != "Failed - TickRuntime"
        and (
            record["status"].startswith("Failed")
            or record["scores"]["score_route"] < 1e-11
        )
        for record in records
    )


def write_job_script(
    route_file: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> Path:
    """Write the job script evaluating one route.

    Args:
        route_file: The route XML to evaluate.
        output_dir: The evaluation run's output directory.
        args: Parsed command line arguments.

    Returns:
        Path of the written job script.
    """
    route_id = route_file.stem
    script = output_dir / "scripts" / f"{route_id}.sh"
    script.parent.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    script.write_text(
        JOB_TEMPLATE.format(
            job_name=f"{args.job_name}_{route_id}",
            partition=args.partition,
            output_dir=output_dir,
            route_id=route_id,
            cpus=args.cpus,
            mem=args.mem,
            gres=args.gres,
            time_limit=args.time_limit,
            code_dir=runtime_variables.project_root(),
            boot_attempts=CARLA_BOOT_ATTEMPTS,
            boot_timeout=CARLA_BOOT_TIMEOUT,
            checkpoint=Path(args.checkpoint).absolute(),
            route_file=route_file.absolute(),
            leaderboard_flag=LEADERBOARD_FLAGS[args.leaderboard],
            route_output_dir=output_dir / "routes" / route_id,
        ),
        encoding="utf-8",
    )
    return script


def submit(script: Path) -> str:
    """Submit one route job.

    Args:
        script: The route's generated job script.

    Returns:
        The Slurm job id.
    """
    output = subprocess.check_output(["sbatch", str(script)], text=True)
    return output.strip().rsplit(" ", maxsplit=1)[-1]


def queued_job_ids(job_ids: Iterable[str]) -> set[str]:
    """Which of the given jobs are still pending or running.

    Args:
        job_ids: Slurm job ids to check.

    Returns:
        The subset still in the queue, or all of them when squeue is
        unreachable, so a transient failure triggers no wave of resubmissions.
    """
    result = subprocess.run(
        ["squeue", "--noheader", "--format=%i", "--user", os.environ["USER"]],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        LOG.warning(f"squeue failed: {result.stderr.strip()}")
        return set(job_ids)
    return set(job_ids) & set(result.stdout.split())


def fill_pool(
    pending: list[tuple[Path, int]],
    running: dict[str, tuple[Path, int]],
    max_parallel_jobs: int,
) -> None:
    """Submit waiting routes until the pool is full.

    Args:
        pending: Job scripts waiting for a slot, with their attempt number.
        running: Jobs in flight by Slurm job id; extended in place.
        max_parallel_jobs: Maximum number of jobs in flight.
    """
    while pending and len(running) < max_parallel_jobs:
        script, attempt = pending.pop(0)
        job_id = submit(script)
        running[job_id] = (script, attempt)
        LOG.info(f"Submitted {script.stem} as job {job_id} (attempt {attempt}).")


def parse_args() -> argparse.Namespace:
    """Parse the command line arguments.

    Returns:
        The parsed arguments.
    """
    benchmark_routes = (
        runtime_variables.project_root() / "src/lead/routes/benchmark_routes"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint directory holding config.yaml and the model weights.",
    )
    parser.add_argument(
        "--route-dir",
        type=Path,
        default=benchmark_routes / "longest6",
        help="Directory of route XML files, one job per file.",
    )
    parser.add_argument(
        "--leaderboard",
        choices=sorted(LEADERBOARD_FLAGS),
        default="standard",
        help="Leaderboard the routes belong to.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory; defaults to <LEAD_OUTPUT_DIR_ROOT>/slurm_evaluation/<route dir>.",
    )
    parser.add_argument("--partition", required=True, help="Slurm partition.")
    parser.add_argument("--job-name", default="eval", help="Slurm job name prefix.")
    parser.add_argument(
        "--max-parallel-jobs",
        type=int,
        default=8,
        help="Number of route jobs in flight at a time.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Submissions per route before giving up on it.",
    )
    parser.add_argument("--cpus", type=int, default=4, help="CPUs per route job.")
    parser.add_argument("--mem", default="32gb", help="Memory per route job.")
    parser.add_argument("--gres", default="gpu:1", help="Generic resources per job.")
    parser.add_argument(
        "--time-limit",
        default="0-04:00:00",
        help="Slurm time limit per route job.",
    )
    return parser.parse_args()


def main() -> None:
    """Submit one job per route and babysit them until all routes are done."""
    setup_logging()
    args = parse_args()

    route_files = sorted(args.route_dir.glob("**/*.xml"))
    if not route_files:
        raise SystemExit(f"No route XMLs under {args.route_dir}")
    output_dir = args.output_dir or (
        runtime_variables.output_dir_root() / "slurm_evaluation" / args.route_dir.name
    )
    output_dir = output_dir.absolute()

    pending: list[tuple[Path, int]] = []
    for route_file in route_files:
        if route_needs_rerun(result_path(output_dir, route_file.stem)):
            pending.append((write_job_script(route_file, output_dir, args), 1))
        else:
            LOG.info(f"Route {route_file.stem} already finished. Skipping.")
    LOG.info(f"{len(pending)}/{len(route_files)} routes to evaluate in {output_dir}.")

    running: dict[str, tuple[Path, int]] = {}
    given_up: list[str] = []
    fill_pool(pending, running, args.max_parallel_jobs)
    while running:
        time.sleep(POLL_INTERVAL_SECONDS)
        alive = queued_job_ids(running)
        for job_id in [job_id for job_id in running if job_id not in alive]:
            script, attempt = running.pop(job_id)
            route_id = script.stem
            if not route_needs_rerun(result_path(output_dir, route_id)):
                LOG.info(f"Route {route_id} finished.")
            elif attempt < args.max_attempts:
                LOG.info(f"Route {route_id} crashed. Retrying.")
                pending.append((script, attempt + 1))
            else:
                LOG.warning(f"Route {route_id} crashed {attempt} times. Giving up.")
                given_up.append(route_id)
        fill_pool(pending, running, args.max_parallel_jobs)

    LOG.info(
        f"Evaluated {len(route_files) - len(given_up)}/{len(route_files)} routes."
        f" Results in {output_dir}/routes.",
    )
    if given_up:
        LOG.warning(f"Routes without a result: {', '.join(given_up)}")


if __name__ == "__main__":
    main()
