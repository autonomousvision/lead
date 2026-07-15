import argparse
import glob
import json
import logging
import os
import random
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
    max_sleep = read_dotenv_int("COLLECT_DATA_MAX_SLEEP")
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

export FREE_STREAMING_PORT=$(random_free_port)
export FREE_WORLD_PORT=$(random_free_port)
export TM_PORT=$(random_free_port)

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

echo "FREE_STREAMING_PORT: $FREE_STREAMING_PORT"
echo "FREE_WORLD_PORT: $FREE_WORLD_PORT"
echo "TM_PORT: $TM_PORT"
echo "CARLA_ROOT: $CARLA_ROOT"

bash {carla_root}/CarlaUE4.sh \
    --world-port=$FREE_WORLD_PORT \
    -RenderOffScreen \
    -nosound \
    -graphicsadapter=0 \
    -carla-streaming-port=$FREE_STREAMING_PORT &

sleep {max_sleep}

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
    job_list = (
        subprocess.check_output(
            (
                f"SQUEUE_FORMAT2='jobid:10,username:{len(username)},name:130' squeue --sort V | grep {user_name} | \
                    grep {jobname} || true"
            ),
            shell=True,
        )
        .decode("utf-8")
        .splitlines()
    )
    currently_num_running_jobs = len(job_list)
    #  line is sth like "4767364   gwb791 eval_julian_4170_0   "
    routefile_number_list = [
        line.split("_")[-2] + "_" + line.split("_")[-1].strip() for line in job_list
    ]
    pid_list = [line.split(" ")[0] for line in job_list]
    return currently_num_running_jobs, routefile_number_list, pid_list


def get_last_line_from_file(
    filepath: str,
) -> str:  # used to check log files for errors
    try:
        with open(filepath, "rb", encoding="utf-8") as f:
            try:
                f.seek(-2, os.SEEK_END)
                while f.read(1) != b"\n":
                    f.seek(-2, os.SEEK_CUR)
            except OSError:
                f.seek(0)
            last_line = f.readline().decode()
    except:
        last_line = ""
    return last_line


def cancel_jobs_with_err_in_log(logroot: str, jobname: str, user_name: str) -> None:
    # check if the log file contains certain error messages, then terminate the job
    LOG.info("Checking logs for errors...")
    _, routefile_number_list, pid_list = get_running_jobs(jobname, user_name)
    for i, rf_num in enumerate(routefile_number_list):
        logfile_path = os.path.join(logroot, f"run_files/logs/qsub_out{rf_num}.log")
        last_line = get_last_line_from_file(logfile_path)
        terminate = False
        if "Actor" in last_line and "not found!" in last_line:
            terminate = True
        if "Watchdog exception - Timeout" in last_line:
            terminate = True
        if "Engine crash handling finished; re-raising signal 11" in last_line:
            terminate = True
        if terminate:
            LOG.info(
                f"Terminating route {rf_num} with pid {pid_list[i]} due to error in logfile.",
            )
            subprocess.check_output(f"scancel {pid_list[i]}", shell=True)


def wait_for_jobs_to_finish(
    logroot: str,
    jobname: str,
    user_name: str,
    max_n_parallel_jobs: int,
) -> None:
    currently_running_jobs, _, _ = get_running_jobs(jobname, user_name)
    LOG.info(f"{currently_running_jobs}/{max_n_parallel_jobs} jobs are running...")
    counter = 0
    while currently_running_jobs >= max_n_parallel_jobs:
        if counter == 0:
            cancel_jobs_with_err_in_log(logroot, jobname, user_name)
        time.sleep(5)
        currently_running_jobs, _, _ = get_running_jobs(jobname, user_name)
        counter = (counter + 1) % 4


def get_num_jobs(job_name: str, username: str) -> tuple[int, int]:
    len_usrn = len(username)
    num_running_jobs = int(
        subprocess.check_output(
            f"SQUEUE_FORMAT2='username:{len_usrn},name:130' squeue --sort V | grep {username} | grep {job_name} | wc -l",
            shell=True,
        )
        .decode("utf-8")
        .replace("\n", ""),
    )
    max_num_parallel_jobs = read_dotenv_int("MAX_NUM_PARALLEL_JOBS_COLLECT_DATA")

    return num_running_jobs, max_num_parallel_jobs


def is_job_done(result_file: str) -> bool:
    need_run_again = False
    if os.path.exists(result_file):
        with open(result_file, encoding="utf-8") as f_result:
            evaluation_data = json.load(f_result)
        progress = evaluation_data["_checkpoint"]["progress"]
        if len(progress) < 2 or progress[0] < progress[1]:
            need_run_again = True
        else:
            for record in evaluation_data["_checkpoint"]["records"]:
                if record["scores"]["score_route"] <= 0.00000000001:
                    need_run_again = True
                if record["status"] == "Failed - Agent couldn't be set up":
                    need_run_again = True
                if record["status"] == "Failed":
                    need_run_again = True
                if record["status"] == "Failed - Simulation crashed":
                    need_run_again = True
                if record["status"] == "Failed - Agent crashed":
                    need_run_again = True
                if record["status"] == "Started":
                    need_run_again = True

        if not need_run_again:
            # delete old job
            LOG.info(f"Finished job {result_file}")
    else:
        need_run_again = True
    return not need_run_again


def arg_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect dataset")
    parser.add_argument(
        "--route_folder",
        type=str,
        default="src/lead/routes/data_routes",
        help="Folder containing route files",
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
    max_route_per_scenario_type = 10  # -1 means no limit

    agent = f"{code_root}/src/lead/expert/expert_agent.py"

    scenario_white_lists = []  # Empty list = all scenarios allowed
    scenario_blacklist = ["YieldToEmergencyVehicle"]  # Scenarios to exclude
    town_white_list = [
        "Town12",
        "Town13",
        "Town15",
    ]  # Empty list = all towns allowed, e.g. ["Town12", "Town13"]
    data_save_directory = os.path.join(code_root, read_dotenv("PY123D_DATA_ROOT"))
    log_root = f"{data_save_directory}/slurm"

    os.makedirs(data_save_directory, exist_ok=True)
    expert_config_dict = yaml_filtered(load_lead_config().expert.to_dict())
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

    route_folder = os.path.join(code_root, args.route_folder)
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

    port_offset = 0
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

            if is_job_done(ckpt_endpoint):
                LOG.info(
                    f"Job {routefile_number} already exists and is finished. Skipping...",
                )
            else:
                # Wait until submitting new jobs that the #jobs are at below max
                num_running_jobs, max_num_parallel_jobs = get_num_jobs(
                    job_name=job_name,
                    username=username,
                )
                LOG.info(
                    f"{num_running_jobs}/{max_num_parallel_jobs} jobs are running...",
                )
                while num_running_jobs >= max_num_parallel_jobs:
                    num_running_jobs, max_num_parallel_jobs = get_num_jobs(
                        job_name=job_name,
                        username=username,
                    )
                    time.sleep(0.05)

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
                jobid = (
                    subprocess.check_output(f"sbatch {job_file}", shell=True)
                    .decode("utf-8")
                    .strip()
                    .rsplit(" ", maxsplit=1)[-1]
                )
                LOG.info(f"Jobid: {jobid}")
                meta_jobs[jobid] = (
                    False,
                    job_file,
                    ckpt_endpoint,
                    1,
                )  # job_finished, job_file, result_file, attempts
            job_number += 1

    time.sleep(1)
    training_finished = False
    while not training_finished:
        num_running_jobs, _, _ = get_running_jobs(job_name, username)
        LOG.info(f"{num_running_jobs} jobs are running... Job: {job_name}")
        cancel_jobs_with_err_in_log(log_root, job_name, username)
        time.sleep(20)

        # resubmit unfinished jobs
        for k in list(meta_jobs.keys()):
            job_finished, job_file, result_file, attempts = meta_jobs[k]
            need_to_resubmit = False
            max_attempts = read_dotenv_int("MAX_NUM_ATTEMPTS_COLLECT_DATA")
            if not job_finished and attempts < max_attempts:
                # check whether job is running
                if (
                    int(
                        subprocess.check_output(
                            f"squeue | grep {k} | wc -l",
                            shell=True,
                        )
                        .decode("utf-8")
                        .strip(),
                    )
                    == 0
                ):
                    # check whether result file is finished?
                    if os.path.exists(result_file):
                        with open(result_file, encoding="utf-8") as f_result:
                            evaluation_data = json.load(f_result)
                        progress = evaluation_data["_checkpoint"]["progress"]
                        if len(progress) < 2 or progress[0] < progress[1]:
                            need_to_resubmit = True
                        else:
                            for record in evaluation_data["_checkpoint"]["records"]:
                                if record["scores"]["score_route"] <= 0.00000000001:
                                    need_to_resubmit = True
                                if (
                                    record["status"]
                                    == "Failed - Agent couldn't be set up"
                                ):
                                    need_to_resubmit = True
                                if record["status"] == "Failed":
                                    need_to_resubmit = True
                                if record["status"] == "Failed - Simulation crashed":
                                    need_to_resubmit = True
                                if record["status"] == "Failed - Agent crashed":
                                    need_to_resubmit = True

                        if not need_to_resubmit:
                            # delete old job
                            LOG.info(f"Finished job {job_file}")
                            meta_jobs[k] = (True, None, None, attempts)

                    else:
                        need_to_resubmit = True

            if need_to_resubmit:
                # rename old error files to still access it
                routefile_number = Path(job_file).stem
                LOG.info(
                    f"Resubmit job {routefile_number} (previous id: {k}). Waiting for jobs to finish...",
                )

                max_num_parallel_jobs = read_dotenv_int(
                    "MAX_NUM_PARALLEL_JOBS_COLLECT_DATA",
                )
                wait_for_jobs_to_finish(
                    log_root,
                    job_name,
                    username,
                    max_num_parallel_jobs,
                )

                time_now_log = time.time()
                os.system(
                    f'mkdir -p "{log_root}/run_files/logs_{routefile_number}_{time_now_log}"',
                )
                os.system(
                    f"cp {log_root}/run_files/logs/qsub_err{routefile_number}.log {log_root}/ \
                          run_files/logs_{routefile_number}_{time_now_log}",
                )
                os.system(
                    f"cp {log_root}/run_files/logs/qsub_out{routefile_number}.log {log_root}/ \
                          run_files/logs_{routefile_number}_{time_now_log}",
                )

                jobid = (
                    subprocess.check_output(f"sbatch {job_file}", shell=True)
                    .decode("utf-8")
                    .strip()
                    .rsplit(" ", maxsplit=1)[-1]
                )
                meta_jobs[jobid] = (False, job_file, result_file, attempts + 1)
                meta_jobs[k] = (True, None, None, attempts)
                LOG.info(f"resubmitted job {routefile_number}. (new id: {jobid})")

        time.sleep(10)

        if num_running_jobs == 0:
            training_finished = True
