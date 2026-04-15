#!/usr/bin/bash

port=2000
# if there is first argument, use it as port
if [ "$1" != "" ]; then
	port=$1
fi

streaming_port=$((port + 1))
if [ "$2" != "" ]; then
	streaming_port=$2
fi

$CARLA_ROOT/CarlaUE4.sh \
    -quality-level=Poor \
    -world-port=$port \
    -resx=800 \
    -resy=600 \
    -nosound \
    -graphicsadapter=0 \
    -carla-streaming-port=$streaming_port \
    -RenderOffScreen &
