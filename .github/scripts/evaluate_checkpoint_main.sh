#!/bin/bash
# Main tier: one route from each of the six fastest scenario types the
# released checkpoint completes at driving score 100 on every route.
routes=(
	2390 # VanillaNonSignalizedTurn
	24340 # ControlLoss
	23901 # InterurbanActorFlow
	2273 # MergerIntoSlowTraffic
	23658 # HighwayExit
	2286 # HighwayCutIn
)
exec bash "$(dirname "$0")/evaluate_checkpoint.sh" "$1" "${routes[@]}"
