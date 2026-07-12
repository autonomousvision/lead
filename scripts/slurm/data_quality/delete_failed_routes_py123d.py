import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = "data/carla_leaderboard2_py123d"
DATA_DIR = os.path.join(BASE_DIR, "data")
LOGS_DIR = os.path.join(BASE_DIR, "logs", "carla_train")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DRY_RUN = True  # set False to actually delete
IGNORE_EMPTY = True  # treat results with empty records as non-failed


def is_failed(json_path: str) -> bool:
    """Mirror of the criterion from 009_check_successful_routes.py."""
    if not os.path.isfile(json_path):
        print(f"File not found: {json_path}")
        return True

    with open(json_path, encoding="utf-8") as f:
        records = json.load(f)["_checkpoint"]["records"]
    if len(records) == 0:
        return not IGNORE_EMPTY
    record = records[0]

    condition1 = record["scores"]["score_composed"] < 100.0 and not (
        record["num_infractions"]
        == len(record["infractions"]["min_speed_infractions"])
        + len(record["infractions"]["outside_route_lanes"])
    )
    condition2 = record["status"] == "Failed - Agent couldn't be set up"
    condition3 = record["status"] == "Failed"
    condition4 = record["status"] == "Failed - Simulation crashed"
    condition5 = record["status"] == "Failed - Agent crashed"
    if condition1:
        print(f"{json_path} Failed due to score: {record['scores']['score_composed']}")
    if condition2:
        print(f"{json_path} Failed due to agent setup issue")
    if condition3:
        print(f"{json_path} Failed due to general failure")
    if condition4:
        print(f"{json_path} Failed due to simulation crash")
    if condition5:
        print(f"{json_path} Failed due to agent crash")

    return condition1 or condition2 or condition3 or condition4 or condition5


def failed_run_timestamp(json_path: str) -> str | None:
    """Return the per-run timestamp suffix (e.g. "1045_1_route0_04_17_03_20_13")."""
    with open(json_path, encoding="utf-8") as f:
        records = json.load(f)["_checkpoint"]["records"]
    if not records:
        return None
    return records[0].get("timestamp")


log_entries = os.listdir(LOGS_DIR) if os.path.isdir(LOGS_DIR) else []

to_be_deleted: list[str] = []
failed_counts = 0
total_counts = 0
missing_log_counts = 0
missing_data_counts = 0
missing_timestamp_counts = 0

for scenario in os.listdir(RESULTS_DIR):
    scenario_results_dir = os.path.join(RESULTS_DIR, scenario)
    if not os.path.isdir(scenario_results_dir):
        continue
    scenario_data_dir = os.path.join(DATA_DIR, scenario)

    for file_name in os.listdir(scenario_results_dir):
        if not file_name.endswith(".json"):
            continue
        json_path = os.path.join(scenario_results_dir, file_name)
        total_counts += 1

        if not is_failed(json_path):
            continue

        timestamp = failed_run_timestamp(json_path)
        if timestamp is None:
            missing_timestamp_counts += 1
            print(f"[{failed_counts}] No timestamp in {json_path}, only deleting result file.")
            to_be_deleted.append(json_path)
            failed_counts += 1
            continue

        # Find matching log dir + json + data dir by timestamp suffix.
        matched_log_dirs = [
            name for name in log_entries if name.endswith(timestamp)
        ]
        matched_data_dirs = (
            [name for name in os.listdir(scenario_data_dir) if name.endswith(timestamp)]
            if os.path.isdir(scenario_data_dir)
            else []
        )

        print(
            f"[{failed_counts}] Failed route {scenario}/{file_name} (ts={timestamp}) "
            f"-> {len(matched_log_dirs)} log entries, {len(matched_data_dirs)} data dirs",
        )

        to_be_deleted.append(json_path)

        if matched_log_dirs:
            for name in matched_log_dirs:
                path = os.path.join(LOGS_DIR, name)
                to_be_deleted.append(path)
        else:
            missing_log_counts += 1

        if matched_data_dirs:
            for name in matched_data_dirs:
                to_be_deleted.append(os.path.join(scenario_data_dir, name))
        else:
            missing_data_counts += 1

        failed_counts += 1

ratio = failed_counts / total_counts if total_counts else 0.0
print(
    f"Total results: {total_counts}, Failed: {failed_counts}, "
    f"Failed ratio: {ratio:.2%}",
)
print(
    f"Failed without matching log dir: {missing_log_counts}, "
    f"without matching data dir: {missing_data_counts}, "
    f"without timestamp: {missing_timestamp_counts}",
)
print(f"Paths queued for deletion: {len(to_be_deleted)}")

if not DRY_RUN and to_be_deleted:
    def _delete(path: str) -> str:
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
        return path

    with ThreadPoolExecutor(max_workers=128) as ex:
        futures = {ex.submit(_delete, path): path for path in to_be_deleted}
        for f in as_completed(futures):
            print(f"Deleted {f.result()}")
    print(f"Deleted {len(to_be_deleted)} paths for {failed_counts} failed routes.")
