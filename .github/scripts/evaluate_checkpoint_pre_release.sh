#!/bin/bash
# Pre-release tier: the main tier plus three more score-100 scenario types and
# a second route for the three fastest.
routes=(
	2390 # VanillaNonSignalizedTurn
	24340 # ControlLoss
	23901 # InterurbanActorFlow
	2273 # MergerIntoSlowTraffic
	23658 # HighwayExit
	2286 # HighwayCutIn
	25358 # StaticCutIn
	17563 # SequentialLaneChange
	17752 # DynamicObjectCrossing
	2397 # VanillaNonSignalizedTurn
	24784 # ControlLoss
	23910 # InterurbanActorFlow
)
exec bash "$(dirname "$0")/evaluate_checkpoint.sh" "$1" "${routes[@]}"
