#!/bin/bash
# Evaluates a checkpoint on the given Bench2Drive routes (ids under
# src/lead/routes/benchmark_routes/bench2drive/), booting a fresh CARLA server
# per route and checking each route's result files. Run from the repo root.
#
# Usage: evaluate_checkpoint.sh <checkpoint_dir> <route_id...>
set -e
# Activating lead also runs setup.py's activate.d PATH hook, which puts
# scripts/cli (clean_carla, start_carla, random_free_port) on PATH.
eval "$(micromamba shell hook --shell bash)"
micromamba activate lead
# -u only after activation: the activation code may reference unset vars.
set -u

checkpoint_dir=$1
shift

# Record everything the agent can draw: with one route per checkpoint the
# videos carry more signal than extra routes would.
export LEAD_CONFIG="\
evaluation.produce_demo_image=true \
evaluation.produce_demo_video=true \
evaluation.produce_debug_image=true \
evaluation.produce_debug_video=true \
evaluation.produce_input_image=true \
evaluation.produce_input_video=true \
evaluation.produce_grid_image=true \
evaluation.produce_grid_video=true"
# Basename keys the output per checkpoint, so evaluating several seeds in one
# run cannot mix their route results.
output_root="${RUNNER_TEMP:-/tmp}/checkpoint_evaluation/$(basename "$checkpoint_dir")"

CARLA_PORT=$(random_free_port)
TM_PORT=$(random_free_port)
# CARLA also binds CARLA_PORT+1 for streaming; keep TM clear of both.
while [ "$TM_PORT" = "$CARLA_PORT" ] || [ "$TM_PORT" = "$((CARLA_PORT + 1))" ]; do
  TM_PORT=$(random_free_port)
done

clean_carla || true
CARLA_PID=""
trap '[ -n "$CARLA_PID" ] && kill -9 -- "-$CARLA_PID" 2>/dev/null' EXIT

failed=()
for route_id in "$@"; do
  echo "::group::route $route_id"
  # setsid makes $CARLA_PID a process-group id, so the group kill in
  # clean_carla reaches the engine process CarlaUE4.sh spawns.
  setsid start_carla "$CARLA_PORT" > carla_server.log 2>&1 &
  CARLA_PID=$!
  for _ in $(seq 1 90); do
    ss -ltn | grep -q ":$CARLA_PORT " && break
    sleep 2
  done
  ss -ltn | grep -q ":$CARLA_PORT " || { cat carla_server.log; exit 1; }
  sleep 10

  route_output="$output_root/$route_id"
  if python -u -m lead --checkpoint "$checkpoint_dir" \
      --routes="src/lead/routes/benchmark_routes/bench2drive/$route_id.xml" \
      --bench2drive \
      --port="$CARLA_PORT" \
      --traffic-manager-port="$TM_PORT" \
      --output-dir="$route_output"; then
    # Every tier route is one the checkpoint completes at driving score 100;
    # the threshold below 100 absorbs simulation nondeterminism, not driving
    # regressions.
    python - "$route_output" <<'EOF' || failed+=("$route_id")
import json
import sys

route_output = sys.argv[1]
record = json.load(open(f"{route_output}/checkpoint_endpoint.json"))
record = record["_checkpoint"]["global_record"]
status = record["status"]
score = record["scores_mean"]["score_composed"]
print(f"status {status} | driving score {score}")
assert status == "Completed", f"route did not complete: {status}"
assert score >= 80.0, f"driving score {score} below 80"
EOF
  else
    failed+=("$route_id")
  fi
  echo "::endgroup::"

  clean_carla "$CARLA_PID"
  wait "$CARLA_PID" 2>/dev/null || true
  CARLA_PID=""
done

if [ "${#failed[@]}" -gt 0 ]; then
  echo "::error::Failing evaluation routes: ${failed[*]}"
  exit 1
fi
echo "All $# evaluation routes passed."
