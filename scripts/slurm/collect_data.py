"""Submit a Slurm job array that drives the expert agent to collect one route each.

Resubmits jobs that fail or get preempted, up to a per-job attempt cap, and
skips routes whose result file already shows a completed run.
"""

import argparse
import glob
import json
import logging
import os
import random
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

from lead.common import runtime
from lead.common.dotenv import read_dotenv, read_dotenv_int
from lead.common.logging import setup_logging
from lead.config import load_lead_config, yaml_filtered

LOG = logging.getLogger(__name__)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return self.streams[0].isatty()


def refresh_job_ports(job_file: str) -> None:
    """Rewrite a job script to re-check ports at run time so a resubmitted job
    does not retry ports that collided with another process."""
    text = Path(job_file).read_text(encoding="utf-8")
    text = re.sub(
        r"export FREE_WORLD_PORT=.*",
        "export FREE_WORLD_PORT=$(random_free_port)",
        text,
    )
    text = re.sub(
        r"export FREE_STREAMING_PORT=.*",
        "export FREE_STREAMING_PORT=$(random_free_port)",
        text,
    )
    text = re.sub(
        r"export TM_PORT=.*",
        "export TM_PORT=$(random_free_port)",
        text,
    )
    Path(job_file).write_text(text, encoding="utf-8")


def make_bash(
    data_save_root: str,
    code_dir: str,
    route_file_number: str,
    agent_name: str,
    route_file: str,
    ckeckpoint_endpoint: str,
    save_pth: str,
    seed: int,
    carla_root: str,
    town: str,
    repetition: int,
    scenario_name: str,
    jobname: str,
    timeout: str,
) -> str:
    os.makedirs(f"{data_save_root}/stderr", exist_ok=True)
    os.makedirs(f"{data_save_root}/stdout", exist_ok=True)
    os.makedirs(f"{data_save_root}/scripts", exist_ok=True)
    jobfile = f"{data_save_root}/scripts/{route_file_number}_Rep{repetition}.sh"
    # Read fresh from .env so edits apply to jobs not yet submitted. SLURM parses
    # the #SBATCH directives before the shell runs, so they must be literal text
    # in the generated script.
    partition_name = read_dotenv("COLLECT_DATA_PARTITION")
    carla_boot_timeout = read_dotenv_int("COLLECT_DATA_CARLA_BOOT_TIMEOUT")
    carla_boot_attempts = read_dotenv_int("COLLECT_DATA_CARLA_BOOT_ATTEMPTS")
    cpus_per_task = read_dotenv_int("COLLECT_DATA_CPUS_PER_TASK")
    mem = read_dotenv("COLLECT_DATA_MEM")
    gres = read_dotenv("COLLECT_DATA_GRES")
    eval_timeout = read_dotenv_int("COLLECT_DATA_TIMEOUT")
    # create folder
    Path(jobfile).parent.mkdir(parents=True, exist_ok=True)

    template = f"""#!/bin/bash
#SBATCH --job-name={jobname}_{route_file_number}
#SBATCH --partition={partition_name}
#SBATCH -o {data_save_root}/stdout/{route_file_number}.log
#SBATCH -e {data_save_root}/stderr/{route_file_number}.log
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}
#SBATCH --time={timeout}
#SBATCH --gres={gres}

echo "SLURMD_NODENAME: $SLURMD_NODENAME"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
scontrol show job $SLURM_JOB_ID
which python
which python3

sleep 2

cd {code_dir}
pwd

export SCENARIO_RUNNER_ROOT={code_dir}/3rd_party/leaderboard/expert/scenario_runner
export LEADERBOARD_ROOT={code_dir}/3rd_party/leaderboard/expert/leaderboard

# carla
export CARLA_ROOT={carla_root}
export CARLA_SERVER={carla_root}/CarlaUE4.sh
export PYTHONPATH={carla_root}/PythonAPI/carla:$PYTHONPATH
export PYTHONPATH=3rd_party/leaderboard/expert/leaderboard:$PYTHONPATH
export PYTHONPATH=3rd_party/leaderboard/expert/scenario_runner:$PYTHONPATH
export REPETITIONS=1
export DEBUG_CHALLENGE=0
export TEAM_AGENT={agent_name}
export CHALLENGE_TRACK_CODENAME=MAP
export ROUTES={route_file}
export TOWN={town}
export REPETITION={repetition}
export TM_SEED={seed}
export SCENARIO_NAME={scenario_name}

export CHECKPOINT_ENDPOINT={ckeckpoint_endpoint}
export TEAM_CONFIG={route_file}
export RESUME=1
export DATAGEN=1
export SAVE_PATH={save_pth}

echo "Start python"
nvidia-smi

export FREE_WORLD_PORT=$(random_free_port)
export FREE_STREAMING_PORT=$(random_free_port)
export TM_PORT=$(random_free_port)
echo "CARLA_ROOT: $CARLA_ROOT"

trap '[ -n "$CARLA_PID" ] && kill -9 -- "-$CARLA_PID" 2>/dev/null' EXIT

# Start CARLA and wait until it answers RPC calls. If a port is taken, CARLA
# dies within seconds; the next attempt re-rolls random ports, so collisions
# with other processes heal within the job instead of relying on a resubmit.
CARLA_READY=0
for attempt in $(seq 1 {carla_boot_attempts}); do
    if [ "$attempt" -gt 1 ]; then
        export FREE_WORLD_PORT=$(random_free_port)
        export FREE_STREAMING_PORT=$(random_free_port)
        export TM_PORT=$(random_free_port)
    fi
    echo "Attempt ${{attempt}}: FREE_WORLD_PORT=$FREE_WORLD_PORT FREE_STREAMING_PORT=$FREE_STREAMING_PORT TM_PORT=$TM_PORT"

    echo "-- memory before CARLA launch --"
    free -h
    cat /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory.current 2>/dev/null
    cat /sys/fs/cgroup/memory/memory.limit_in_bytes /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null

    # Restrict Vulkan to the NVIDIA GPU: with an AMD iGPU and Mesa's software
    # lavapipe ICD also installed, -graphicsadapter=0 can otherwise select the
    # wrong device and CARLA renders on the CPU.
    VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
    __NV_PRIME_RENDER_OFFLOAD=1 \
    bash {carla_root}/CarlaUE4.sh \
        --world-port=$FREE_WORLD_PORT \
        -RenderOffScreen \
        -nosound \
        -graphicsadapter=0 \
        -carla-streaming-port=$FREE_STREAMING_PORT &
    CARLA_PID=$!

    for i in $(seq 1 {carla_boot_timeout}); do
        sleep 1
        if ! kill -0 $CARLA_PID 2>/dev/null; then
            echo "CARLA died after ${{i}}s (port taken or crash)"
            nvidia-smi
            dmesg -T 2>/dev/null | tail -n 20
            sleep 5
            break
        fi
        if test_carla_connection $FREE_WORLD_PORT; then
            echo "CARLA serving on port $FREE_WORLD_PORT after ${{i}}s"
            CARLA_READY=1
            break
        fi
    done

    if [ "$CARLA_READY" -eq 1 ]; then
        break
    fi
    kill -9 -- "-$CARLA_PID" 2>/dev/null
    wait $CARLA_PID 2>/dev/null
    CARLA_PID=""
done

if [ "$CARLA_READY" -ne 1 ]; then
    echo "CARLA failed to start after {carla_boot_attempts} attempts"
    exit 1
fi

nvidia-smi

which python3

echo "Starting expert at route {route_file} in town {town} with scenario {scenario_name}."

python 3rd_party/leaderboard/expert/leaderboard/leaderboard/leaderboard_evaluator_local.py \
    --port=${{FREE_WORLD_PORT}} \
    --traffic-manager-port=${{TM_PORT}} \
    --traffic-manager-seed=${{TM_SEED}} \
    --routes=${{ROUTES}} \
    --repetitions=${{REPETITIONS}} \
    --track=${{CHALLENGE_TRACK_CODENAME}} \
    --checkpoint=${{CHECKPOINT_ENDPOINT}} \
    --agent=${{TEAM_AGENT}} \
    --agent-config=${{TEAM_CONFIG}} \
    --debug=0 \
    --resume=${{RESUME}} \
    --timeout={eval_timeout}
"""

    with open(jobfile, "w", encoding="utf-8") as f:
        f.write(template)
    return jobfile


def get_running_jobs(jobname: str, user_name: str) -> tuple[int, list[str], list[str]]:
    output = subprocess.check_output(
        ["squeue", "--noheader", "--format=%i %j", "-u", user_name],
        text=True,
    )
    jobs = [
        line.split()
        for line in output.splitlines()
        if line.split()[-1].startswith(f"{jobname}_")
    ]
    #  job name is e.g. "collect_4170_0"; the route file number is "4170_0"
    routefile_number_list = ["_".join(name.split("_")[-2:]) for _, name in jobs]
    jobid_list = [jobid for jobid, _ in jobs]
    return len(jobs), routefile_number_list, jobid_list


def is_job_queued(jobid: str) -> bool:
    result = subprocess.run(
        ["squeue", "--noheader", "-j", jobid],
        capture_output=True,
        text=True,
        check=False,
    )
    return bool(result.stdout.strip())


def submit_job(job_file: str) -> str:
    output = subprocess.check_output(["sbatch", job_file], text=True)
    return output.strip().rsplit(" ", maxsplit=1)[-1]


def wait_for_jobs_to_finish(
    data_save_root: str,
    jobname: str,
    user_name: str,
    max_n_parallel_jobs: int,
) -> None:
    currently_running_jobs, _, _ = get_running_jobs(jobname, user_name)
    LOG.info(f"{currently_running_jobs}/{max_n_parallel_jobs} jobs are running...")
    while currently_running_jobs >= max_n_parallel_jobs:
        time.sleep(5)
        currently_running_jobs, _, _ = get_running_jobs(jobname, user_name)


RERUN_STATUSES = (
    "Started",
    "Failed",
    "Failed - Agent couldn't be set up",
    "Failed - Simulation crashed",
    "Failed - Agent crashed",
)


def route_needs_rerun(result_file: str) -> bool:
    """A route must run (again) unless its result file shows a completed run
    with a nonzero route score."""
    if not os.path.exists(result_file):
        return True
    with open(result_file, encoding="utf-8") as f_result:
        checkpoint = json.load(f_result)["_checkpoint"]
    progress = checkpoint["progress"]
    if len(progress) < 2 or progress[0] < progress[1]:
        return True
    return any(
        record["status"] in RERUN_STATUSES
        or record["scores"]["score_route"] <= 0.00000000001
        for record in checkpoint["records"]
    )


def arg_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect dataset")
    parser.add_argument(
        "--route_folder",
        type=str,
        default=str(runtime.project_root() / "src/lead/routes/data_routes"),
        help="Folder containing route files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parse()
    repetitions = 1
    repetition_start = 0
    shuffle_routes = True
    job_name = "collect"
    username = os.environ["USER"]
    code_root = str(runtime.project_root())
    carla_root = read_dotenv("CARLA_ROOT")
    max_route_per_scenario_type = 5  # -1 means no limit

    agent = f"{code_root}/src/lead/expert/expert_agent.py"

    scenario_white_lists = []  # Empty list = all scenarios allowed
    scenario_blacklist = ["YieldToEmergencyVehicle"]  # Scenarios to exclude
    town_white_list = [
        "Town12",
        "Town13",
        "Town15",
    ]  # Empty list = all towns allowed, e.g. ["Town12", "Town13"]
    data_save_directory = read_dotenv("PY123D_DATA_ROOT")

    os.makedirs(data_save_directory, exist_ok=True)
    expert_config = load_lead_config().expert
    expert_config_dict = yaml_filtered(expert_config.to_dict())
    with open(f"{data_save_directory}/config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(expert_config_dict, f, sort_keys=False)
    sys.stdout = Tee(
        sys.stdout,
        open(f"{data_save_directory}/stdout.log", "a", encoding="utf-8"),
    )
    sys.stderr = Tee(
        sys.stderr,
        open(f"{data_save_directory}/stderr.log", "a", encoding="utf-8"),
    )
    setup_logging()

    git_branch = subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=code_root,
        text=True,
    ).strip()
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=code_root,
        text=True,
    ).strip()
    git_status = subprocess.check_output(
        ["git", "status", "--short"],
        cwd=code_root,
        text=True,
    )
    git_diff = subprocess.check_output(["git", "diff"], cwd=code_root, text=True)
    with open(f"{data_save_directory}/git_status.txt", "w", encoding="utf-8") as f:
        f.write(f"Git branch: {git_branch}\n")
        f.write(f"Git commit: {git_commit}\n")
        f.write(f"Git status:\n{git_status}\n")
        f.write(f"Git diff:\n{git_diff}\n")
    LOG.info(
        f"Git branch: {git_branch}, commit: {git_commit}. Full status/diff written to {data_save_directory}/git_status.txt",
    )

    # Maps must exist before the job array starts: jobs converting them
    # concurrently race on the shared per-town map files.
    LOG.info("Converting 123D maps (already-converted towns are skipped)...")
    subprocess.run(
        [
            sys.executable,
            str(runtime.project_root() / "scripts/common/convert_py123d_maps.py"),
            "--dataset",
            expert_config.data_collection.py123d_dataset,
        ],
        cwd=code_root,
        check=True,
    )

    route_folder = args.route_folder
    LOG.info("Start looking for routes...")
    routes = glob.glob(f"{route_folder}/**/*.xml", recursive=True)
    if shuffle_routes:
        random.seed(42)
        random.shuffle(routes)
    LOG.info(f"Found {len(routes)} routes in total.")
    if len(scenario_white_lists) > 0:
        routes = [
            route
            for route in routes
            if any(scenario in route.split("/") for scenario in scenario_white_lists)
        ]

    if len(scenario_blacklist) > 0:
        routes = [
            route
            for route in routes
            if not any(scenario in route.split("/") for scenario in scenario_blacklist)
        ]
        LOG.info(f"Applied scenario blacklist. Total routes: {len(routes)}")

    # Scenario type is the parent directory name (e.g. .../Accident/1054_0.xml),
    # avoiding an XML parse per file, which is slow on network disks.
    if max_route_per_scenario_type > 0:
        scenario_type_counts = {}
        filtered_routes = []
        for route in routes:
            scenario_type = os.path.basename(os.path.dirname(route)) or "noScenarios"
            count = scenario_type_counts.get(scenario_type, 0)
            if count < max_route_per_scenario_type:
                filtered_routes.append(route)
                scenario_type_counts[scenario_type] = count + 1
        routes = filtered_routes
        LOG.info(
            f"Applied max_route_per_scenario_type={max_route_per_scenario_type}. Total routes: {len(routes)}",
        )

    job_number = 1
    meta_jobs = {}

    # shuffle routes
    random.seed(42)
    random.shuffle(routes)
    seed_counter = (
        1000000 * repetition_start - 1
    )  # for the traffic manager, which is incremented so that we get different traffic each time

    num_routes = len(routes)
    for repetition in range(repetition_start, repetitions):
        for route in routes:
            seed_counter += 1

            try:
                tree = ET.parse(route)  # 'route' is the XML filepath
                root = tree.getroot()
                route_elem = root.find("route")
                assert route_elem is not None
                town = route_elem.attrib["town"]
            except Exception as e:
                LOG.error(f"Error parsing town from route {route}: {e}")
                raise e
            scenario_elem = root.find("route/scenarios/scenario")
            scenario_type = (
                scenario_elem.attrib["type"]
                if scenario_elem is not None
                else "noScenarios"
            )

            if len(town_white_list) > 0 and town not in town_white_list:
                LOG.info(f"Ignoring route in town: {town}")
                continue

            if (
                len(scenario_white_lists) > 0
                and scenario_type not in scenario_white_lists
            ):
                LOG.info(f"Ignoring route with scenario type: {scenario_type}")
                continue

            if len(scenario_blacklist) > 0 and scenario_type in scenario_blacklist:
                LOG.info(
                    f"Ignoring blacklisted route with scenario type: {scenario_type}",
                )
                continue

            routefile_number = route.split("/")[-1].split(".")[
                0
            ]  # this is the number in the xml file name, e.g. 22_0.xml
            # Same prefix as the 123D log name (the run timestamp is only known at
            # run time and lives inside the file as records[0]["timestamp"]).
            ckpt_endpoint = f"{data_save_directory}/results/{scenario_type}/{town}_Rep{repetition}_{routefile_number}_result.json"

            # Passed through as SAVE_PATH: the expert uses it only as a
            # datagen-enabled marker, so the directory is never created.
            save_path = f"{data_save_directory}/data/{scenario_type}"

            if not route_needs_rerun(ckpt_endpoint):
                LOG.info(
                    f"Job {routefile_number} already exists and is finished. Skipping...",
                )
            else:
                wait_for_jobs_to_finish(
                    data_save_directory,
                    job_name,
                    username,
                    read_dotenv_int("MAX_NUM_PARALLEL_JOBS_COLLECT_DATA"),
                )

                # Generate the job script right before submitting so it picks up the
                # latest .env values (partition and SLURM resource requests).
                job_file = make_bash(
                    data_save_directory,
                    code_root,
                    routefile_number,
                    agent,
                    route,
                    ckpt_endpoint,
                    save_path,
                    seed_counter,
                    carla_root,
                    town,
                    repetition,
                    scenario_type,
                    job_name,
                    timeout="0-01:00:00",
                )

                LOG.info(
                    f"Submitting job {job_number}/{num_routes}: {job_name}_{routefile_number}. ",
                )
                time.sleep(1)
                jobid = submit_job(job_file)
                LOG.info(f"Jobid: {jobid}")
                meta_jobs[jobid] = (
                    False,
                    job_file,
                    ckpt_endpoint,
                    1,
                )  # job_finished, job_file, result_file, attempts
            job_number += 1

    time.sleep(1)
    while True:
        num_running_jobs, _, _ = get_running_jobs(job_name, username)
        LOG.info(f"{num_running_jobs} jobs are running... Job: {job_name}")
        time.sleep(5)

        # resubmit unfinished jobs
        max_attempts = read_dotenv_int("MAX_NUM_ATTEMPTS_COLLECT_DATA")
        for jobid in list(meta_jobs.keys()):
            job_finished, job_file, result_file, attempts = meta_jobs[jobid]
            if job_finished or is_job_queued(jobid):
                continue
            if not route_needs_rerun(result_file):
                LOG.info(f"Finished job {job_file}")
                meta_jobs[jobid] = (True, None, None, attempts)
                continue
            if attempts >= max_attempts:
                continue

            routefile_number = Path(job_file).stem
            LOG.info(
                f"Resubmit job {routefile_number} (previous id: {jobid}). Waiting for jobs to finish...",
            )
            wait_for_jobs_to_finish(
                data_save_directory,
                job_name,
                username,
                read_dotenv_int("MAX_NUM_PARALLEL_JOBS_COLLECT_DATA"),
            )

            # SLURM appends to the same stdout/stderr logs (--open-mode=append),
            # so every attempt's output stays available.
            refresh_job_ports(job_file)
            new_jobid = submit_job(job_file)
            meta_jobs[new_jobid] = (False, job_file, result_file, attempts + 1)
            meta_jobs[jobid] = (True, None, None, attempts)
            LOG.info(f"resubmitted job {routefile_number}. (new id: {new_jobid})")

        time.sleep(10)

        if num_running_jobs == 0:
            break
