#!/usr/bin/env python

import argparse
import os
import signal
import subprocess
import time
from pathlib import Path

from evaluate_wandb_logger import WandBLogger

from lead.inference.config_closed_loop import ClosedLoopConfig
from slurm.evaluation import evaluate_utils


class SlurmJobPool:
    def __init__(self, args):
        """Monitor and manage submission of SLURM jobs with a fixed job pool size."""
        self.status_of_jobs = {}
        self.script_index = 0
        self.videos_to_delete = []

        # Gather all SLURM self.slurm_scripts in the specified directory
        try:
            self.slurm_scripts = sorted(
                [str(file) for file in Path(args.slurm_dir).glob(pattern="*.sh")], key=lambda x: int(Path(x).stem)
            )
        except:
            self.slurm_scripts = sorted([str(file) for file in Path(args.slurm_dir).glob(pattern="*.sh")])

        # Filter self.slurm_scripts if id_list is not empty
        if len(args.id_list) > 0:
            selected_ids = {str(id) for id in args.id_list}  # Convert id_list to string for comparison
            selected_scripts = []
            for script in self.slurm_scripts:
                stem = str(Path(script).stem)
                if stem in selected_ids:
                    selected_scripts.append(script)
            self.slurm_scripts = selected_scripts

            # Ensure self.slurm_scripts are found
            if len(self.slurm_scripts) == 0:
                print(f"No slurm scripts matched the specified IDs: {args.id_list}")
                exit(1)

        self.wandb_logger = WandBLogger(closed_loop_config=ClosedLoopConfig(), num_routes=len(self.slurm_scripts))

        # Submit and monitor jobs
        self.job_name = args.job_name
        self.user_cancelled = False

    def fill_pool(self):
        """Fill the job pool with new jobs if there's space."""
        while len(self.status_of_jobs) < evaluate_utils.get_max_parallel_jobs() and self.script_index < len(self.slurm_scripts):
            script_path = self.slurm_scripts[self.script_index]
            slurm_job_id, process = evaluate_utils.submit_job(self.job_name, script_path, 1)
            self.status_of_jobs[slurm_job_id] = {
                "script": script_path,
                "attempts": 1,
                "route_id": Path(script_path).stem,
                "route_idx": self.script_index,
                "process": process,
            }
            self.script_index += 1
            time.sleep(1)

    def monitor_jobs(self):
        """Monitor the status of jobs in the pool and submit new jobs."""
        self.fill_pool()
        while not self.user_cancelled and (self.status_of_jobs or self.script_index < len(self.slurm_scripts)):
            for slurm_job_id, slurm_job_status in list(
                self.status_of_jobs.items()
            ):  # Check slurm_job_status of each job in the pool
                carla_route_id = Path(slurm_job_status["script"]).stem
                json_log_path = Path(slurm_job_status["script"]).parent.parent / "eval" / f"{carla_route_id}.json"
                if not evaluate_utils.is_job_running(
                    slurm_job_id, slurm_job_status, self.job_name
                ):  # Check if job is still running or needs resubmission
                    print(f"Route {carla_route_id} is not running anymore.")
                    job_failed = evaluate_utils.did_job_fail(json_log_path)
                    if (
                        job_failed and slurm_job_status["attempts"] < evaluate_utils.get_max_num_attempts()
                    ):  # Retry job if it failed and has attempts left
                        print(f"Retrying route {carla_route_id} (Attempt {slurm_job_status['attempts'] + 1})")
                        new_slurm_job_id, process = evaluate_utils.submit_job(
                            self.job_name, slurm_job_status["script"], slurm_job_status["attempts"] + 1
                        )
                        self.status_of_jobs[new_slurm_job_id] = self.status_of_jobs[slurm_job_id]
                        self.status_of_jobs[new_slurm_job_id]["process"] = process
                        self.status_of_jobs[new_slurm_job_id]["attempts"] = slurm_job_status["attempts"] + 1
                    else:
                        if job_failed:  # Job failed
                            print(f"Job for route {carla_route_id} failed after {slurm_job_status['attempts']} attempts.")
                        else:  # Job succeeded
                            print(f"Job for route {carla_route_id} completed successfully.")
                            self.wandb_logger.log_job(
                                json_log_path,
                                slurm_job_status["route_idx"],
                                evaluate_utils.get_num_running_jobs(job_name=self.job_name),
                                slurm_job_status["attempts"],
                                job_failed,
                                carla_route_id,
                            )
                    del self.status_of_jobs[slurm_job_id]

            # Fill the pool with new jobs if there's space
            self.fill_pool()
            time.sleep(5)
        evaluate_utils.aggregate_metrics()
        self.wandb_logger.log_metrics_artifact()
        self.wandb_logger.finish()

    def signal_handler(self, sig, frame):
        """Handle interrupt signal by cancelling all jobs."""
        print("Received interrupt signal, cancelling jobs...")
        self.user_cancelled = True
        processes = []

        if evaluate_utils.is_on_slurm():
            # Create separate scancel processes
            for job_id in self.status_of_jobs.keys():
                processes.append(subprocess.Popen(["scancel", job_id], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        else:
            # Create separate kill processes
            for job_id in self.status_of_jobs.keys():
                processes.append(
                    subprocess.Popen(["kill", "-9", str(job_id)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                )
            processes.append(subprocess.Popen(["90_clean_carla.sh"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))

        # Wait for all processes to complete
        for process in processes:
            process.wait()

        del self.wandb_logger

        os.exit(0)


def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Submit and monitor SLURM job self.slurm_scripts from a directory.")
    parser.add_argument("--slurm_dir", type=str, help="Directory containing SLURM job self.slurm_scripts to submit.")
    parser.add_argument("--job_name", type=str, help="Prefix to know which jobs belong here.")
    parser.add_argument("--id_list", type=str, nargs="*", default=[], help="List of job IDs to submit. If empty, submit all.")
    args = parser.parse_args()
    slurm_job_pool = SlurmJobPool(args)
    # Register signal handlers
    signal.signal(signal.SIGINT, slurm_job_pool.signal_handler)
    signal.signal(signal.SIGTERM, slurm_job_pool.signal_handler)
    slurm_job_pool.monitor_jobs()


if __name__ == "__main__":
    main()
