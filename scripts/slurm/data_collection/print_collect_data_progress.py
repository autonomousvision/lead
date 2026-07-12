import json
import os
import re
import subprocess

from lead.common.dotenv import read_dotenv


def get_running_job_names():
    output = subprocess.check_output(
        ["squeue", "--noheader", "--format=%j", "-u", os.environ["USER"]],
        text=True,
    )
    return set(output.split())


def is_finished(json_path):
    """A route is finished once the leaderboard evaluator's own progress
    counter says so — the same check collect_data.py's is_job_done() uses to
    decide whether a job needs resubmitting. Anything else (still running,
    or the worker died before writing a full record) is not finished yet."""
    if not os.path.isfile(json_path):
        return False
    with open(json_path, encoding="utf-8") as f:
        checkpoint = json.load(f)["_checkpoint"]
    progress = checkpoint.get("progress", [])
    return len(progress) >= 2 and progress[1] > 0 and progress[0] >= progress[1]


def is_failed(json_path):
    with open(json_path, encoding="utf-8") as f:
        results_route = json.load(f)["_checkpoint"]["records"]
    if len(results_route) == 0:
        print(f"Marked finished but has no records: {json_path}")
        return True
    results_route = results_route[0]

    condition1 = results_route["scores"]["score_composed"] < 100.0 and not (
        results_route["num_infractions"]
        == (
            len(results_route["infractions"]["min_speed_infractions"])
            + len(results_route["infractions"]["outside_route_lanes"])
        )
    )
    condition2 = results_route["status"] == "Failed - Agent couldn't be set up"
    condition3 = results_route["status"] == "Failed"
    condition4 = results_route["status"] == "Failed - Simulation crashed"
    condition5 = results_route["status"] == "Failed - Agent crashed"
    if condition1:
        print(
            f"{json_path} Failed due to score: {results_route['scores']['score_composed']}",
        )
    if condition2:
        print(f"{json_path} Failed due to agent setup issue")
    if condition3:
        print(f"{json_path} Failed due to general failure")
    if condition4:
        print(f"{json_path} Failed due to simulation crash")
    if condition5:
        print(f"{json_path} Failed due to agent crash")

    return condition1 or condition2 or condition3 or condition4 or condition5


root = read_dotenv("PY123D_DATA_ROOT")
results_root = f"{root}/results"
running_job_names = get_running_job_names()
scenario_stats = {}
dataset_attempted = 0
dataset_finished = 0
dataset_success = 0
dataset_running = 0
dataset_crashed = 0
for scenario_name in os.listdir(results_root):
    scenario_path = os.path.join(results_root, scenario_name)
    if not os.path.isdir(scenario_path):
        continue

    attempted = 0
    finished = 0
    failed = 0
    running = 0
    crashed = 0

    for file_name in os.listdir(scenario_path):
        if not file_name.endswith(".json"):
            continue
        json_path = os.path.join(scenario_path, file_name)
        # File name is "{town}_Rep{rep}_{routefile_number}_result.json"; job names
        # and stdout/stderr logs use only the route file number. Files from before
        # the renaming lack the town/rep prefix.
        route_id = file_name.removesuffix("_result.json")
        prefix_match = re.match(r"Town\d+_Rep\d+_(.+)", route_id)
        if prefix_match:
            route_id = prefix_match.group(1)
        attempted += 1
        if not is_finished(json_path):
            if f"collect_{route_id}" in running_job_names:
                running += 1
            else:
                crashed += 1
                print(f"{json_path} Crashed before finishing")
                print(f"  stdout: {root}/stdout/{route_id}.log")
                print(f"  stderr: {root}/stderr/{route_id}.log")
            continue
        finished += 1
        if is_failed(json_path):
            failed += 1
            print(f"  stdout: {root}/stdout/{route_id}.log")
            print(f"  stderr: {root}/stderr/{route_id}.log")

    if attempted > 0:
        success = finished - failed
        scenario_stats[scenario_name] = {
            "attempted": attempted,
            "finished": finished,
            "failed": failed,
            "success": success,
            "running": running,
            "crashed": crashed,
        }
    dataset_attempted += attempted
    dataset_finished += finished
    dataset_success += finished - failed
    dataset_running += running
    dataset_crashed += crashed
for k, v in sorted(scenario_stats.items()):
    success_rate = 100 * v["success"] / v["finished"] if v["finished"] > 0 else 0.0
    print(
        f"{k:<40} Finished: {v['finished']:>2}/{v['attempted']:<2} "
        f"Success: {v['success']:>2}, Failed: {v['failed']:>2}, "
        f"Running: {v['running']:>2}, Crashed: {v['crashed']:>2}, "
        f"Success rate: {success_rate:>5.1f}%",
    )
print(f"Finished routes (regardless of success/failure): {dataset_finished}/{dataset_attempted} attempted")
print(f"  Success: {dataset_success}")
print(f"  Failed: {dataset_finished - dataset_success}")
print(f"  Still running: {dataset_running}")
print(f"  Crashed before finishing: {dataset_crashed}")
