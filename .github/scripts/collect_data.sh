#!/bin/bash
# Collects expert driving data for the given routes (paths relative to
# src/lead/routes/data_routes/), booting a fresh CARLA server per route.
# Run from the repo root.
set -e
# Activating lead also runs setup.py's activate.d PATH hook, which puts
# scripts/cli (clean_carla, start_carla, random_free_port) on PATH.
eval "$(micromamba shell hook --shell bash)"
micromamba activate lead
# -u only after activation: the activation code may reference unset vars.
set -u

CARLA_PORT=$(random_free_port)
TM_PORT=$(random_free_port)
# CARLA also binds CARLA_PORT+1 for streaming; keep TM clear of both.
while [ "$TM_PORT" = "$CARLA_PORT" ] || [ "$TM_PORT" = "$((CARLA_PORT + 1))" ]; do
  TM_PORT=$(random_free_port)
done

clean_carla || true
CARLA_PID=""
trap '[ -n "$CARLA_PID" ] && kill -9 -- "-$CARLA_PID" 2>/dev/null' EXIT

for route in "$@"; do
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
  python -u -m lead --expert \
      --routes="src/lead/routes/data_routes/$route" \
      --port="$CARLA_PORT" \
      --traffic-manager-port="$TM_PORT" \
      --timeout=600
  clean_carla "$CARLA_PID"
  wait "$CARLA_PID" 2>/dev/null || true
  CARLA_PID=""
done
