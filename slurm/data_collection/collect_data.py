import argparse
import glob
import json
import logging
import os
import random
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from lead.common.logging_config import setup_logging

setup_logging()
logging.getLogger().setLevel(logging.INFO)
LOG = logging.getLogger(__name__)


def make_bash(
    data_save_root,
    code_dir,
    route_file_number,
    agent_name,
    route_file,
    ckeckpoint_endpoint,
    save_pth,
    seed,
    carla_root,
    town,
    repetition,
    scenario_name,
    jobname,
    partition_name,
    timeout="0-01:00",
):
    os.makedirs(f"{data_save_root}/stderr", exist_ok=True)
    os.makedirs(f"{data_save_root}/stdout", exist_ok=True)
    os.makedirs(f"{data_save_root}/scripts", exist_ok=True)
    jobfile = f"{data_save_root}/scripts/{route_file_number}_Rep{repetition}.sh"
    with open("slurm/configs/max_sleep.txt", encoding="utf-8") as f:
        max_sleep = int(f.read())
    # create folder
    Path(jobfile).parent.mkdir(parents=True, exist_ok=True)
    template = f"""#!/bin/bash
#SBATCH --job-name={jobname}_{route_file_number}
#SBATCH --partition={partition_name}
#SBATCH -o {data_save_root}/stdout/{route_file_number}.log
#SBATCH -e {data_save_root}/stderr/{route_file_number}.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40gb
#SBATCH --time={timeout}
#SBATCH --gres=gpu:1080ti:1

echo "SLURMD_NODENAME: $SLURMD_NODENAME"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
scontrol show job $SLURM_JOB_ID

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "Job started: $dt"

echo "Current branch:"
git branch
echo "Current commit:"
git log -1
echo "Current hash:"
git rev-parse HEAD


export FREE_STREAMING_PORT=$(91_random_free_port.sh)
export FREE_WORLD_PORT=$(91_random_free_port.sh)
export TM_PORT=$(91_random_free_port.sh)

sleep 2

echo "start python"
pwd

export SCENARIO_RUNNER_ROOT={code_dir}/3rd_party/scenario_runner_autopilot
export LEADERBOARD_ROOT={code_dir}/3rd_party/leaderboard_autopilot

# carla
export CARLA_ROOT={carla_root}
export CARLA_SERVER={carla_root}/CarlaUE4.sh
export PYTHONPATH={carla_root}/PythonAPI/carla:$PYTHONPATH
export PYTHONPATH=3rd_party/leaderboard_autopilot:$PYTHONPATH
export PYTHONPATH=3rd_party/scenario_runner_autopilot:$PYTHONPATH
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

bash {carla_root}/CarlaUE4.sh \
    --world-port=$FREE_WORLD_PORT \
    -RenderOffScreen \
    -nosound \
    -graphicsadapter=0 \
    -carla-streaming-port=$FREE_STREAMING_PORT &

sleep {max_sleep}

eval "$(../miniconda3/bin/conda shell.bash hook)"
if [ -z "$CONDA_INTERPRETER" ]; then
    export CONDA_INTERPRETER="lead" # Check if CONDA_INTERPRETER is not set, then set it to lead
fi
source activate "$CONDA_INTERPRETER"
which python3

python 3rd_party/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py \
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
    --timeout=600
"""

    with open(jobfile, "w", encoding="utf-8") as f:
        f.write(template)
    return jobfile


def get_running_jobs(jobname, user_name):
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
    routefile_number_list = [line.split("_")[-2] + "_" + line.split("_")[-1].strip() for line in job_list]
    pid_list = [line.split(" ")[0] for line in job_list]
    return currently_num_running_jobs, routefile_number_list, pid_list


def get_last_line_from_file(filepath):  # this is used to check log files for errors
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


def cancel_jobs_with_err_in_log(logroot, jobname, user_name):
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
            LOG.info(f"Terminating route {rf_num} with pid {pid_list[i]} due to error in logfile.")
            subprocess.check_output(f"scancel {pid_list[i]}", shell=True)


def wait_for_jobs_to_finish(logroot, jobname, user_name, max_n_parallel_jobs):
    currently_running_jobs, _, _ = get_running_jobs(jobname, user_name)
    LOG.info(f"{currently_running_jobs}/{max_n_parallel_jobs} jobs are running...")
    counter = 0
    while currently_running_jobs >= max_n_parallel_jobs:
        if counter == 0:
            cancel_jobs_with_err_in_log(logroot, jobname, user_name)
        time.sleep(5)
        currently_running_jobs, _, _ = get_running_jobs(jobname, user_name)
        counter = (counter + 1) % 4


def get_num_jobs(job_name, username):
    len_usrn = len(username)
    num_running_jobs = int(
        subprocess.check_output(
            f"SQUEUE_FORMAT2='username:{len_usrn},name:130' squeue --sort V | grep {username} | grep {job_name} | wc -l",
            shell=True,
        )
        .decode("utf-8")
        .replace("\n", "")
    )
    with open("slurm/configs/max_num_parallel_jobs_collect_data.txt", encoding="utf-8") as f:
        max_num_parallel_jobs = int(f.read())

    return num_running_jobs, max_num_parallel_jobs


def is_job_done(result_file):
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
            LOG.info(f"Finished job {job_file}")
    else:
        need_run_again = True
    return not need_run_again


def arg_parse():
    parser = argparse.ArgumentParser(description="Collect dataset")
    parser.add_argument("--root_folder", type=str, default="data/", help="Root folder for data")
    parser.add_argument("--route_folder", type=str, default="data/data_routes", help="Folder containing route files")
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parse()
    repetitions = 1
    repetition_start = 0
    partitions = ["day"]
    job_name = "collect"
    username = os.environ["USER"]
    code_root = os.getcwd()
    carla_root = os.getcwd() + "/3rd_party/CARLA_0915"
    agent = f"{code_root}/lead/expert/expert.py"
    dataset_name = "carla_leaderboard2"
    scenario_white_lists = []  # No white list, allow all
    max_route_per_scenario_type = 5  # -1 means no limit
    root_folder = args.root_folder  # With ending slash
    os.makedirs(root_folder, exist_ok=True)
    data_save_directory = root_folder + dataset_name
    log_root = f"{data_save_directory}/slurm"

    route_folder = args.route_folder
    routes = glob.glob(f"{route_folder}/**/*.xml", recursive=True)
    if len(scenario_white_lists) > 0:
        routes = [route for route in routes if any(scenario in route.split("/") for scenario in scenario_white_lists)]

    # Apply max_route_per_scenario_type constraint
    if max_route_per_scenario_type > 0:
        scenario_type_counts = {}
        filtered_routes = []
        for route in routes:
            try:
                tree = ET.parse(route)
                root = tree.getroot()
                scenario_elem = root.find("route/scenarios/scenario")
                scenario_type = scenario_elem.attrib["type"] if scenario_elem is not None else "noScenarios"

                if scenario_type not in scenario_type_counts:
                    scenario_type_counts[scenario_type] = 0

                if scenario_type_counts[scenario_type] < max_route_per_scenario_type:
                    filtered_routes.append(route)
                    scenario_type_counts[scenario_type] += 1
            except Exception as e:
                LOG.info(f"Warning: Could not parse scenario type from route {route}: {e}")
                # Include route anyway if parsing fails
                filtered_routes.append(route)
        routes = filtered_routes
        LOG.info(f"Applied max_route_per_scenario_type={max_route_per_scenario_type}. Total routes: {len(routes)}")

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
                town = root.find("route").attrib["town"]
            except Exception as e:
                LOG.info(f"Error parsing town from route {route}: {e}")
                raise e
            scenario_elem = root.find("route/scenarios/scenario")
            scenario_type = scenario_elem.attrib["type"] if scenario_elem is not None else "noScenarios"

            if len(scenario_white_lists) > 0 and scenario_type not in scenario_white_lists:
                LOG.info("Ignoring route with scenario type:", scenario_type)
                continue

            routefile_number = route.split("/")[-1].split(".")[0]  # this is the number in the xml file name, e.g. 22_0.xml
            ckpt_endpoint = f"{code_root}/{data_save_directory}/results/{scenario_type}/{routefile_number}_result.json"

            save_path = f"{code_root}/{data_save_directory}/data/{scenario_type}"
            Path(save_path).mkdir(parents=True, exist_ok=True)

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
                random.choice(partitions),
            )

            if is_job_done(ckpt_endpoint):
                LOG.info(f"Job {job_file} already exists and is finished. Skipping...")
            else:
                # Wait until submitting new jobs that the #jobs are at below max
                num_running_jobs, max_num_parallel_jobs = get_num_jobs(job_name=job_name, username=username)
                LOG.info(f"{num_running_jobs}/{max_num_parallel_jobs} jobs are running...")
                while num_running_jobs >= max_num_parallel_jobs:
                    num_running_jobs, max_num_parallel_jobs = get_num_jobs(job_name=job_name, username=username)
                    time.sleep(0.05)

                LOG.info(f"Submitting job {job_number}/{num_routes}: {job_name}_{routefile_number}. ")
                time.sleep(1)
                jobid = (
                    subprocess.check_output(f"sbatch {job_file}", shell=True)
                    .decode("utf-8")
                    .strip()
                    .rsplit(" ", maxsplit=1)[-1]
                )
                LOG.info(f"Jobid: {jobid}")
                meta_jobs[jobid] = (False, job_file, ckpt_endpoint, 0)  # job_finished, job_file, result_file, resubmitted
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
            job_finished, job_file, result_file, resubmitted = meta_jobs[k]
            need_to_resubmit = False
            with open("slurm/configs/max_num_attempts_collect_data.txt", encoding="utf-8") as f:
                max_attempts = int(f.read())
            if not job_finished and resubmitted < max_attempts:
                # check whether job is running
                if int(subprocess.check_output(f"squeue | grep {k} | wc -l", shell=True).decode("utf-8").strip()) == 0:
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
                                if record["status"] == "Failed - Agent couldn't be set up":
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
                            meta_jobs[k] = (True, None, None, 0)

                    else:
                        need_to_resubmit = True

            if need_to_resubmit:
                # rename old error files to still access it
                routefile_number = Path(job_file).stem
                LOG.info(f"Resubmit job {routefile_number} (previous id: {k}). Waiting for jobs to finish...")

                with open("slurm/configs/max_num_parallel_jobs_collect_data.txt", encoding="utf-8") as f:
                    max_num_parallel_jobs = int(f.read())
                wait_for_jobs_to_finish(log_root, job_name, username, max_num_parallel_jobs)

                time_now_log = time.time()
                os.system(f'mkdir -p "{log_root}/run_files/logs_{routefile_number}_{time_now_log}"')
                os.system(
                    f"cp {log_root}/run_files/logs/qsub_err{routefile_number}.log {log_root}/ \
                          run_files/logs_{routefile_number}_{time_now_log}"
                )
                os.system(
                    f"cp {log_root}/run_files/logs/qsub_out{routefile_number}.log {log_root}/ \
                          run_files/logs_{routefile_number}_{time_now_log}"
                )

                jobid = (
                    subprocess.check_output(f"sbatch {job_file}", shell=True)
                    .decode("utf-8")
                    .strip()
                    .rsplit(" ", maxsplit=1)[-1]
                )
                meta_jobs[jobid] = (False, job_file, result_file, resubmitted + 1)
                meta_jobs[k] = (True, None, None, 0)
                LOG.info(f"resubmitted job {routefile_number}. (new id: {jobid})")

        time.sleep(10)

        if num_running_jobs == 0:
            training_finished = True
