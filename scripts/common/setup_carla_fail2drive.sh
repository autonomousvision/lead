#!/bin/bash
cd "$(dirname "$(realpath "${BASH_SOURCE:-$0}")")/../.."

_target="${1:-3rd_party/CARLA/fail2drive_0915}"
mkdir -p "$_target"

curl -L https://huggingface.co/datasets/SimonGer/fail2drive/resolve/e8bd082943eb90d517c091d77c5f1bbc521f9391/fail2drive_simulator.tar.gz |
	tar -xz -C "$_target"
