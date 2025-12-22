#!/usr/bin/bash

# Get the PID of the CARLA instance running on GPU 0
pid=$(nvidia-smi -i 0 --query-compute-apps=pid,process_name --format=csv,noheader | grep 'CarlaUE4' | awk -F, '{print $1}' | head -n 1)

# Kill if the PID is not empty
if [ -n "$pid" ]; then
	kill -9 "$pid"
	echo "CARLA instance on GPU 0 (PID: $pid) killed."
else
	echo "No CARLA instance found on GPU 0."
fi
