#!/usr/bin/env bash
# Verify a training output directory holds what the run should have written.
#
# Usage: check_training_artifacts.sh <output_dir> [--checkpoint <epoch>]
#
# config.yaml is always required. With --checkpoint, the epoch's model,
# optimizer and trainer-state files must exist, and the model file must load
# back into the policy the stored config names — the same load the evaluation
# PolicyRunner performs. Run inside the lead environment (python must resolve
# to it).
set -euo pipefail

output_dir=$1
epoch=""
if [[ "${2:-}" == "--checkpoint" ]]; then
	epoch=$3
fi

required=("$output_dir/config.yaml")
if [[ -n "$epoch" ]]; then
	required+=(
		"$output_dir/model_${epoch}.pth"
		"$output_dir/optimizer_${epoch}.pth"
		"$output_dir/trainer_state_${epoch}.pth"
	)
fi

missing=()
for file in "${required[@]}"; do
	if [[ -f "$file" ]]; then
		echo "ok      $file"
	else
		echo "MISSING $file"
		missing+=("$file")
	fi
done
if [[ "${#missing[@]}" -gt 0 ]]; then
	echo "::error::Missing training artifacts in $output_dir: ${missing[*]}"
	exit 1
fi

if [[ -n "$epoch" ]]; then
	python - "$output_dir" "$epoch" <<'EOF'
"""Load the saved weights back into the policy the stored config names."""

import sys

import torch
import yaml

from lead.api.abstract_policy import build_policy
from lead.config.lead_config import load_lead_config

output_dir, epoch = sys.argv[1], sys.argv[2]
with open(f"{output_dir}/config.yaml") as f:
    stored_config = yaml.safe_load(f)
config = load_lead_config(loaded_config=stored_config, raise_on_unknown_key=False)
policy = build_policy(config)
state_dict = torch.load(
    f"{output_dir}/model_{epoch}.pth",
    map_location="cpu",
    weights_only=True,
)
policy.load_state_dict(state_dict, strict=True)
print(f"load-back ok: model_{epoch}.pth -> {type(policy).__name__}")
EOF
fi
