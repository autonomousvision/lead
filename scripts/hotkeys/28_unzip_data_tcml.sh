#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=L40Sday
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=long.nguyen@student.uni-tuebingen.de
#SBATCH --mem=64G

zip_dir="data/carla_leaderboad2_v6/results/data/sensor_data/zip/"

# collect all zip files
mapfile -d '' -t zips < <(find "$zip_dir" -type f -name "*.zip" -print0)

echo "Found ${#zips[@]} zip files to unzip."
echo "First 10 zips:"
printf '%s\n' "${zips[@]:0:10}"

unzip_route() {
  zip_file="$1"
  unzip -o "$zip_file" -d . >/dev/null
}

export -f unzip_route
export target_dir

printf '%s\0' "${zips[@]}" | parallel --will-cite -0 -P 64 unzip_route {}
