#!/bin/bash

# Resolve to scripts/hotkeys/ even when invoked through the repo-root symlink.
cd "$(dirname "$(realpath "${BASH_SOURCE:-$0}")")"

prev_group=""
for script in [0-9][0-9]_*; do
	num="${script%%_*}"
	name="${script#*_}"
	name="${name%.*}"
	group="${num:0:1}"
	if [ "$group" != "$prev_group" ] && [ -n "$prev_group" ]; then
		echo ""
	fi
	prev_group="$group"
	printf "%s  %s\n" "$num" "${name//_/ }"
done
