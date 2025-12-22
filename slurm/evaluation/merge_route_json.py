#!/usr/bin/env python
import argparse
import glob
import json
import os

import numpy as np


def merge_route_json(folder_path):
    file_paths = glob.glob(f"{folder_path}/*.json")
    merged_records = []
    driving_score = []
    success_num = 0
    for file_path in file_paths:
        if "merged.json" in file_path:
            continue
        try:
            with open(file_path) as file:
                data = json.load(file)
                records = data["_checkpoint"]["records"]
                for rd in records:
                    rd.pop("index")
                    merged_records.append(rd)
                    driving_score.append(rd["scores"]["score_composed"])
                    if rd["status"] == "Completed" or rd["status"] == "Perfect":
                        success_flag = True
                        for k, v in rd["infractions"].items():
                            if len(v) > 0 and k != "min_speed_infractions":
                                success_flag = False
                                break
                        if success_flag:
                            success_num += 1
                            print(rd["route_id"])
        except Exception:
            print(f"-----------------------Warning: {file_path} is not a valid json file.")
    merged_records = sorted(merged_records, key=lambda d: d["route_id"], reverse=True)
    _checkpoint = {"records": merged_records}

    if os.getenv("EVALUATION_DATASET") == "Town13":
        merged_data = {
            "_checkpoint": _checkpoint,
            "driving score": sum(driving_score) / 20,
            "success rate": success_num / 20,
            "eval num": len(driving_score),
            "current_driving_score": np.mean(driving_score),
        }
    elif os.getenv("EVALUATION_DATASET") == "longest6":
        merged_data = {
            "_checkpoint": _checkpoint,
            "driving score": sum(driving_score) / 36,
            "success rate": success_num / 36,
            "eval num": len(driving_score),
            "current_driving_score": np.mean(driving_score),
        }
    elif os.getenv("EVALUATION_DATASET") == "bench2drive220":
        merged_data = {
            "_checkpoint": _checkpoint,
            "driving score": sum(driving_score) / 220,
            "success rate": success_num / 220,
            "eval num": len(driving_score),
            "current_driving_score": np.mean(driving_score),
        }
    elif os.getenv("EVALUATION_DATASET") == "fail2drive210":
        merged_data = {
            "_checkpoint": _checkpoint,
            "driving score": sum(driving_score) / 210,
            "success rate": success_num / 210,
            "eval num": len(driving_score),
            "current_driving_score": np.mean(driving_score),
        }
    else:
        raise ValueError("EVALUATION_DATASET is not set or not supported.")

    with open(os.path.join(folder_path, "merged.json"), "w") as file:
        json.dump(merged_data, file, indent=4)
    print(
        f"driving score: {merged_data['driving score']} success rate: {merged_data['success rate']} eval num: {merged_data['eval num']} current_driving_score: {merged_data['current_driving_score']}"
    )
    return merged_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", help="old foo help")
    args = parser.parse_args()
    merge_route_json(args.folder)
