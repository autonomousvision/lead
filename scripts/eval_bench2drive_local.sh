#!/bin/bash
# Single-machine multi-GPU Bench2Drive eval driver (no SLURM).
# Splits the 220 routes across GPUs; each GPU runs its own CARLA (Epic quality) + agent.
#
# Usage:
#   bash scripts/eval_bench2drive_local.sh <ckpt_dir> "<gpu list>" [out_tag]
#   e.g. bash scripts/eval_bench2drive_local.sh outputs/checkpoints/my_p2 "0 1 2 3 4 5 6 7" my_p2
#
# Prereqs: env set (LEAD_PROJECT_ROOT, CARLA_ROOT), HF offline, resnet34 weights cached.
set -u
CKPT="${1:?checkpoint dir, e.g. outputs/checkpoints/my_p2}"
GPUS="${2:-0}"
TAG="${3:-run}"
ROUTES_DIR="data/benchmark_routes/bench2drive"
LOGDIR="/tmp/b2d_${TAG}"
mkdir -p "$LOGDIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

mapfile -t ROUTES < <(ls "$ROUTES_DIR"/*.xml | sort)
GPU_ARR=($GPUS); NG=${#GPU_ARR[@]}
echo "routes=${#ROUTES[@]}  gpus=$NG  ckpt=$CKPT"

wait_port() { for _ in $(seq 1 60); do ss -ltn | grep -q ":$1 " && return 0; sleep 3; done; return 1; }

worker() {
  local gpu=$1; shift
  local port=$((2000 + gpu*2)) tm=$((8000 + gpu*2)) strm=$((2000 + gpu*2 + 1))
  # start ONE CARLA for this GPU (Epic quality -> no -quality-level)
  CUDA_VISIBLE_DEVICES=$gpu "$CARLA_ROOT/CarlaUE4.sh" --world-port=$port \
      --carla-streaming-port=$strm -nosound -graphicsadapter=$gpu -RenderOffScreen \
      >"$LOGDIR/carla_gpu$gpu.log" 2>&1 &
  local carla_pid=$!
  if ! wait_port $port; then echo "[gpu$gpu] CARLA failed to bind $port"; kill $carla_pid 2>/dev/null; return 1; fi
  echo "[gpu$gpu] CARLA up on $port"
  for xml in "$@"; do
    local id; id=$(basename "$xml" .xml)
    [ -f "outputs/local_evaluation/$id/checkpoint_endpoint.json" ] && { echo "[gpu$gpu] $id done, skip"; continue; }
    echo "[gpu$gpu] route $id ..."
    CUDA_VISIBLE_DEVICES=$gpu timeout 1500 python -m lead \
        --checkpoint "$CKPT" --routes "$xml" --bench2drive \
        --port $port --traffic-manager-port $tm \
        >"$LOGDIR/route_${id}.log" 2>&1 || echo "[gpu$gpu] $id FAILED (see $LOGDIR/route_${id}.log)"
  done
  kill $carla_pid 2>/dev/null
  echo "[gpu$gpu] finished ${#@} routes"
}

for i in "${!GPU_ARR[@]}"; do
  gpu=${GPU_ARR[$i]}; sub=()
  for j in "${!ROUTES[@]}"; do (( j % NG == i )) && sub+=("${ROUTES[$j]}"); done
  worker "$gpu" "${sub[@]}" &
done
wait
echo "=== all workers done. logs in $LOGDIR ==="
echo "collect results + merge:"
echo "  mkdir -p outputs/b2d_${TAG} && for f in outputs/local_evaluation/*/checkpoint_endpoint.json; do cp \$f outputs/b2d_${TAG}/\$(basename \$(dirname \$f)).json; done"
echo "  python slurm/evaluation/merge_route_json.py -f outputs/b2d_${TAG}"
