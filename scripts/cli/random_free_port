#!/bin/bash

while true; do
	# Generate a random port in the range
	PORT=$((RANDOM % 50001 + 10000))

	# Check if the port is free using `ss` or `netstat`
	if ! ss -lpn | grep -q ":$PORT " && ! netstat -lpn 2>/dev/null | grep -q ":$PORT "; then
		echo $PORT
		break
	fi
done
