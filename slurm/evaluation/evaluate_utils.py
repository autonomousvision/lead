#!/usr/bin/env python
import json
import os
import shutil
import subprocess
import traceback
from pathlib import Path

from merge_route_json import merge_route_json


def is_on_slurm():
    return shutil.which("sbatch") is not None


def did_record_crash(record):
    """Determine if a record crashed because of the software problem or the agent somehow did not move at all."""
    if record["status"] == "Failed - TickRuntime":
        return False  # Bench2Drive specific error when the agent is stucked. No software problem. https://github.com/Thinklab-SJTU/Bench2Drive/blob/8f9f1da103ea7153a17c5b1dd046548a9bcd5af2/leaderboard/leaderboard/scenarios/scenario_manager.py#L182
    if int(record["scores"]["score_route"] < 0.00000000001):
        return True
    if record["status"] == "Failed - Agent couldn't be set up":
        return True
    elif record["status"] == "Failed":
        return True
    elif record["status"] == "Failed - Simulation crashed":
        return True
    elif record["status"] == "Failed - Agent crashed":
        return True
    return False


def did_route_crash(record):
    """Determine if a record crashed because of the software problem or the agent somehow did not move at all."""
    try:
        records = record["_checkpoint"]["records"]
        return len(records) == 0 or any(did_record_crash(record) for record in records)
    except Exception as e:
        print(f"Error determining if record crashed: {e}")
        print(traceback.format_exc())
        return True


def load_json_file(json_file):
    """Load a JSON file and return the content."""
    json_file = Path(json_file)
    if json_file.exists():
        json_content = None
        try:
            with open(json_file) as log_file:
                data = json.load(log_file)
                assert len(data["_checkpoint"]["records"]) > 0
                json_content = log_file.read()
            return data
        except Exception as e:
            print(f"Error reading log file {json_file}: {e}")
            print(traceback.format_exc())
            print(f"JSON Content: {json_content}")
    return None


def get_max_num_attempts():
    return int(open(f"slurm/configs/max_num_attempts_{os.getenv('EVALUATION_DATASET')}.txt").read())


def get_max_parallel_jobs():
    if not is_on_slurm():
        return 1
    return int(open(f"slurm/configs/max_num_parallel_jobs_{os.getenv('EVALUATION_DATASET')}.txt").read())


def get_num_running_jobs(job_name, username=None):
    if username is None:
        username = os.getenv("USER")
    # On local PC
    if not is_on_slurm():
        try:
            return int(
                subprocess.check_output(
                    (f"ps -u $USER -o pid,command | grep '{job_name}' | wc -l"),
                    shell=True,
                )
                .decode("utf-8")
                .strip()
            )
        except subprocess.CalledProcessError as e:
            print(f"Error getting number of running jobs: {e}")
            print(traceback.format_exc())
            return 0
    # On Slurm
    try:
        return int(
            subprocess.check_output(
                (
                    "SQUEUE_FORMAT2='username:15,name:100,state:10' squeue --sort V | grep"
                    f" {username} | grep {job_name} | grep 'RUNNING' | wc -l"
                ),
                shell=True,
            )
            .decode("utf-8")
            .strip()
        )
    except subprocess.CalledProcessError as e:
        print(f"Error getting number of running jobs: {e}")
        print(traceback.format_exc())
        return 0


def submit_job(job_name, script_path, num_attempt):
    """Submit a job script using sbatch and return the job ID."""
    slurm_print_output_path = Path(script_path).parent.parent
    route_id = Path(script_path).stem

    # Define names and outputs
    job_route_name = f"eval_{job_name}_{route_id}"
    std_err_file = f"{slurm_print_output_path}/stderr/stderr_{route_id}_{num_attempt}.txt"
    std_out_file = f"{slurm_print_output_path}/stdout/stdout_{route_id}_{num_attempt}.txt"
    os.makedirs(f"{slurm_print_output_path}/stderr", exist_ok=True)
    os.makedirs(f"{slurm_print_output_path}/stdout", exist_ok=True)
    print(std_err_file)
    print(std_out_file)

    # Create command
    if not is_on_slurm():
        submit_command = f"bash {script_path} > {std_out_file} 2> {std_err_file}"
    else:
        partition = ""
        if num_attempt > 10:  # set to 10 to avoid memory issues. we dont really need this anymore but i don t want to remove it
            partition = (
                "--partition=a100-preemptable-galvani"  # Use A100 GPUs for more than 2 attempts due to potential memory issues
            )

        submit_command = (
            f"sbatch --job-name {job_route_name} --output {std_out_file} --error {std_err_file} {partition} {script_path}"
        )

    # Submit jobs and return job ID
    try:
        pid = None
        process = None
        if is_on_slurm():
            output = subprocess.check_output(submit_command, shell=True)
            pid = output.decode("utf-8").strip().split()[-1]
        else:
            process = subprocess.Popen(submit_command, shell=True, start_new_session=True)
            pid = process.pid
        return pid, process
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job: {e}")
        print("Check script path, partition, or job name for issues.")
        raise e
    except IndexError as e:
        print("Unexpected output format from sbatch command.")
        raise e


def did_job_fail(json_log_path):
    """Determine if a job needs resubmission based on the JSON log file content."""
    json_content = load_json_file(json_log_path)
    if json_content is not None:
        return did_route_crash(json_content)
    return True


def is_job_running(slurm_job_id, slurm_job_status, job_name):
    """Check if a job is still running in the SLURM queue."""
    if not is_on_slurm():
        try:
            process = slurm_job_status["process"]
            process.wait(timeout=0)  # Wait for the process with zero timeout
            return False  # If wait completes, the process is not running
        except subprocess.TimeoutExpired:
            return True  # Timeout means the process is still running
    try:
        subprocess.check_output(f"squeue | grep {slurm_job_id}", shell=True)
        return True
    except subprocess.CalledProcessError:
        return False


def aggregate_metrics():
    """Aggregate metrics for all routes."""
    eval_output_dir = os.getenv("EVALUATION_OUTPUT_DIR")
    eval_dataset = os.getenv("EVALUATION_DATASET")
    merge_route_json(eval_output_dir + "/eval")
    shutil.copyfile(eval_output_dir + "/eval/merged.json", eval_output_dir + "/merged.json")
    try:
        if "bench2drive" not in eval_dataset:
            os.system(
                f"""python3 scripts/tools/evaluation/result_parser.py --xml data/{eval_dataset}.xml --results {eval_output_dir}/eval"""  # noqa: E501
            )
            shutil.copyfile(
                eval_output_dir + "/eval/results.csv",
                eval_output_dir + "/results.csv",
            )
    except Exception as e:
        print(f"Error aggregating metrics: {e}")
        print(traceback.format_exc())
        print("Could not aggregate metrics.")
