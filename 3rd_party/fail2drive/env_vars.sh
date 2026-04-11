LOCAL="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"
CARLA="$LOCAL/f2d_carla"

conda env config vars set WORK_DIR=$LOCAL -n fail2drive
conda env config vars set CARLA_ROOT=$CARLA -n fail2drive

conda env config vars set LEADERBOARD_ROOT=$LOCAL/leaderboard -n fail2drive
conda env config vars set SCENARIO_RUNNER_ROOT=$LOCAL/scenario_runner -n fail2drive

conda env config vars set PYTHONPATH=$CARLA/PythonAPI/carla:$LOCAL/leaderboard:$LOCAL/scenario_runner -n fail2drive

conda deactivate
conda activate fail2drive
