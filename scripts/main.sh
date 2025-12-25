# CARLA environment variables
export CARLA_VERSION="0915"
export CARLA_ROOT="${LEAD_PROJECT_ROOT}/3rd_party/CARLA_${CARLA_VERSION}"

# Python paths
export PYTHONPATH=${CARLA_ROOT}/PythonAPI:${PYTHONPATH}
export PYTHONPATH=${CARLA_ROOT}/PythonAPI/carla:${PYTHONPATH}
export PYTHONPATH=${CARLA_ROOT}/PythonAPI/carla/dist/carla-${CARLA_VERSION}-py3.10-linux-x86_64.egg:${PYTHONPATH}
export PYTHONPATH=$LEAD_PROJECT_ROOT/3rd_party/leaderboard_autopilot:$PYTHONPATH
export PYTHONPATH=$LEAD_PROJECT_ROOT/3rd_party/scenario_runner_autopilot:$PYTHONPATH

# NavSim
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="${LEAD_PROJECT_ROOT}/3rd_party/navsim_workspace/dataset/maps"
export NAVSIM_EXP_ROOT="${LEAD_PROJECT_ROOT}/3rd_party/navsim_workspace/exp"
export NAVSIM_DEVKIT_ROOT="${LEAD_PROJECT_ROOT}/3rd_party/navsim_workspace/navsimv1.1"
export OPENSCENE_DATA_ROOT="${LEAD_PROJECT_ROOT}/3rd_party/navsim_workspace/dataset"