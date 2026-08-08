#!/bin/bash
# Smoke tier for dev branches: two routes from the fastest scenario types the
# released checkpoint completes at driving score 100.
routes=(
	2390 # VanillaNonSignalizedTurn
	24340 # ControlLoss
)
exec bash "$(dirname "$0")/evaluate_checkpoint.sh" "$1" "${routes[@]}"
