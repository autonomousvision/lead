#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=1-00:00
#SBATCH --cpus-per-task=1
#SBATCH --partition=cpu-galvani
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=long.nguyen@student.uni-tuebingen.de
#SBATCH --mem=16G

zip -r data.zip data/carla_leaderboad2_v10/results/data/garage_v10_2025_07_14/data
