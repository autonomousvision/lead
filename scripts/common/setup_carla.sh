#!/bin/bash
# Run from the repo root, regardless of the invoking directory.
cd "$(dirname "$(realpath "${BASH_SOURCE:-$0}")")/../.."

# Optional install target as first argument (default: in-repo 3rd_party location).
target="${1:-3rd_party/CARLA/standard_0916}"

mkdir -p "$target"

cd "$target"

wget -O CARLA_0916.tar.gz https://tiny.carla.org/carla-0-9-16-linux
tar -xvzf CARLA_0916.tar.gz
cd Import
wget -O AdditionalMaps_0.9.16.tar.gz https://tiny.carla.org/additional-maps-0-9-16-linux
tar -xvzf AdditionalMaps_0.9.16.tar.gz
cd ..
bash ImportAssets.sh
