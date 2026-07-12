#!/usr/bin/env python3
"""Sync args and env from hotkeys 10, 11, 12 into the matching launch.json configurations."""

import json
import os
import re

HOTKEY_TO_CONFIG = {
    "10_eval_tfv6_bench2drive.sh": "Eval Bench2Drive",
    "11_eval_tfv6_longest6.sh":    "Eval Longest6",
    "12_eval_tfv6_town13.sh":      "Eval Town13",
}

PRESERVED_ENV = {"PYTHONDONTWRITEBYTECODE"}

script_dir  = os.path.dirname(os.path.abspath(__file__))
launch_path = os.path.join(script_dir, "../..", ".vscode/launch.json")

with open(launch_path) as f:
    raw = f.read()

# Strip trailing commas (JSONC → JSON)
raw = re.sub(r",\s*([}\]])", r"\1", raw)
launch = json.loads(raw)

for hotkey_file, config_name in HOTKEY_TO_CONFIG.items():
    with open(os.path.join(script_dir, hotkey_file)) as f:
        content = f.read()

    env_vars = {
        m.group(1): m.group(2)
        for m in re.finditer(r'^export\s+(\w+)="([^"]*)"', content, re.MULTILINE)
    }

    joined = content.replace("\\\n", " ")
    m = re.search(r'python\s+\S+__main__\.py\s+(.*)', joined)
    args = m.group(1).split() if m else []

    for config in launch["configurations"]:
        if config["name"] == config_name:
            config["args"] = args
            preserved = {k: v for k, v in config.get("env", {}).items() if k in PRESERVED_ENV}
            config["env"] = {**preserved, **env_vars}
            print(f"  {hotkey_file}  ->  \"{config_name}\"")
            break

with open(launch_path, "w") as f:
    json.dump(launch, f, indent=4)

print(f"Saved {launch_path}")
