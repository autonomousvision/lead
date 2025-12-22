import json
import os


def newest_version():
    data_dirs = os.listdir("data")
    versions = [d for d in data_dirs if d.startswith("carla_leaderboad2_")]
    if not versions:
        raise ValueError("No versions found in data directory.")
    versions = [version.split("_")[-1].replace("v", "") for version in versions]
    versions = [int(version) for version in versions if version.isdigit()]
    if not versions:
        raise ValueError("No valid versions found in data directory.")
    return max(versions)


def is_failed(json_path):
    if not os.path.isfile(json_path):
        print(f"File not found: {json_path}")
        return True

    with open(json_path, encoding="utf-8") as f:
        results_route = json.load(f)["_checkpoint"]["records"]
    if len(results_route) == 0:
        # print(f"Empty results in {json_path}")
        if ignore_empty:
            return False
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
        print(f"{json_path} Failed due to score: {results_route['scores']['score_composed']}")
    if condition2:
        print(f"{json_path} Failed due to agent setup issue")
    if condition3:
        print(f"{json_path} Failed due to general failure")
    if condition4:
        print(f"{json_path} Failed due to simulation crash")
    if condition5:
        print(f"{json_path} Failed due to agent crash")

    return condition1 or condition2 or condition3 or condition4 or condition5


root = "data/carla_leaderboard2"
results_root = f"{root}/results"
ignore_empty = True
scenario_stats = {}
dataset_total = 0
dataset_success = 0
for scenario_name in os.listdir(results_root):
    scenario_path = os.path.join(results_root, scenario_name)
    if not os.path.isdir(scenario_path):
        continue

    total = 0
    failed = 0

    for file_name in os.listdir(scenario_path):
        if not file_name.endswith(".json"):
            continue
        json_path = os.path.join(scenario_path, file_name)
        total += 1
        if is_failed(json_path):
            failed += 1

    if total > 0:
        success = total - failed
        scenario_stats[scenario_name] = {
            "total": total,
            "failed": failed,
            "success": success,
            "failed_percent": round(failed / total * 100, 2),
            "success_percent": round(success / total * 100, 2),
        }
    dataset_total += total
    dataset_success += total - failed
for k, v in sorted(scenario_stats.items()):
    print(f"{k:<35} Success: {v['success_percent']:>5.1f}% | Failed: {v['failed_percent']:>5.1f}% ({v['failed']}/{v['total']})")
print(f"Total: {dataset_total}. Success: {dataset_success}")

# ----------------------------------------------------------------------
# Count total routes and those with results.json
# ----------------------------------------------------------------------
route_total = 0
route_with_results = 0

for scenario_type in os.listdir(os.path.join(root, "data")):
    scenario_type_path = os.path.join(root, "data", scenario_type)
    if not os.path.isdir(scenario_type_path):
        continue

    for route in os.listdir(scenario_type_path):
        route_path = os.path.join(scenario_type_path, route)
        if not os.path.isdir(route_path):
            continue
        route_total += 1
        if os.path.isfile(os.path.join(route_path, "results.json")):
            route_with_results += 1

print(f"Routes: {route_total}. With results.json: {route_with_results}")
