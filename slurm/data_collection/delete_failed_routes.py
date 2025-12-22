import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

from lead.training.data_loader.buckets.abstract_bucket_collection import route_failed

BASE_DIR = "data/carla_leaderboard2_v3/data"
DRY_RUN = False  # set False to actually delete

to_be_deleted = []
failed_counts = 0
total_counts = 0
for scenario in os.listdir(BASE_DIR):
    scenario_dir = os.path.join(BASE_DIR, scenario)
    if not os.path.isdir(scenario_dir):
        continue
    for route in os.listdir(scenario_dir):
        route_dir = os.path.join(scenario_dir, route)
        if not os.path.isdir(route_dir):
            continue

        if route_failed(route_dir):
            print(f"[{failed_counts}] Route failed: {route_dir}")
            to_be_deleted.append(route_dir)
            failed_counts += 1
        total_counts += 1

print(f"Total routes: {total_counts}, Failed routes: {failed_counts}, Failed ratio: {failed_counts / total_counts:.2%}")

if not DRY_RUN and to_be_deleted:

    def _delete(path):
        shutil.rmtree(path)
        return path

    with ThreadPoolExecutor(max_workers=128) as ex:
        futures = {ex.submit(_delete, path): path for path in to_be_deleted}
        for f in as_completed(futures):
            print(f"Deleted {f.result()}")
    print(f"Deleted {len(to_be_deleted)} routes.")
