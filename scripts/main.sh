# CARLA environment variables
export CARLA_VERSION="0915"
export CARLA_ROOT="${LEAD_PROJECT_ROOT}/3rd_party/CARLA_${CARLA_VERSION}"

# Python paths
export PYTHONPATH=${CARLA_ROOT}/PythonAPI:${PYTHONPATH}
export PYTHONPATH=${CARLA_ROOT}/PythonAPI/carla:${PYTHONPATH}
export PYTHONPATH=${CARLA_ROOT}/PythonAPI/carla/dist/carla-${CARLA_VERSION}-py3.10-linux-x86_64.egg:${PYTHONPATH}
export PYTHONPATH=$LEAD_PROJECT_ROOT/3rd_party/leaderboard_autopilot:$PYTHONPATH
export PYTHONPATH=$LEAD_PROJECT_ROOT/3rd_party/scenario_runner_autopilot:$PYTHONPATH

# System paths
export PATH=$LEAD_PROJECT_ROOT:$PATH
export PATH=$LEAD_PROJECT_ROO/scripts/hotkeys:$PATH

# NavSim
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="${LEAD_PROJECT_ROOT}/3rd_party/navsim_workspace/dataset/maps"
export NAVSIM_EXP_ROOT="${LEAD_PROJECT_ROOT}/3rd_party/navsim_workspace/exp"
export NAVSIM_DEVKIT_ROOT="${LEAD_PROJECT_ROOT}/3rd_party/navsim_workspace/navsimv1.1"
export OPENSCENE_DATA_ROOT="${LEAD_PROJECT_ROOT}/3rd_party/navsim_workspace/dataset"

# Git aliases
alias reset='git reset --hard'
alias switch='git switch'
alias pull='git pull'
alias log='git log --oneline'
alias diff='git diff'
alias checkout='git checkout'
alias fetch='git fetch'
alias add='git add .'
alias status='git status'

commit() {
	add
	if [ -z "$1" ]; then
		git commit -m "Update"
	else
		git commit -m "$1"
	fi
}

push() {
	add
	if [ -z "$1" ]; then
		git commit -m "Update"
	else
		git commit -m "$1"
	fi
	git push
}
