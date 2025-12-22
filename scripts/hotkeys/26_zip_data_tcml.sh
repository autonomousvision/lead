#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=3-00:00
#SBATCH --cpus-per-task=24
#SBATCH --partition=week
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=long.nguyen@student.uni-tuebingen.de
#SBATCH --mem=64G

eval "$(../miniconda3/bin/conda shell.bash hook)"
if [ -z "$CONDA_INTERPRETER" ]; then
    export CONDA_INTERPRETER="lead" # Check if CONDA_INTERPRETER is not set, then set it to lead
fi
source activate "$CONDA_INTERPRETER"
which python3

root="data/carla_leaderboad2_v12/results/data/sensor_data"
src_dir="$root/data"
dst_dir="$root/zip"

mkdir -p "$dst_dir"

# collect all route directories (src_dir/<scenario>/<route>) into an array
mapfile -d '' -t routes < <(find "$src_dir" -mindepth 2 -maxdepth 2 -type d -print0)

echo "Found ${#routes[@]} routes to zip."
echo "First 10 routes:"
printf '%s\n' "${routes[@]:0:10}"

zip_route() {
  route_dir="$1"
  scenario_name=$(basename "$(dirname "$route_dir")")
  route_name=$(basename "$route_dir")
  out_dir="$dst_dir/$scenario_name"
  mkdir -p "$out_dir"
  zip -r -0 "$out_dir/$route_name.zip" "$route_dir" >/dev/null
}

export -f zip_route
export dst_dir

printf '%s\0' "${routes[@]}" | parallel --will-cite -0 -P 32 zip_route {}
