#!/usr/bin/bash

port=2000
# if there is first argument, use it as port
if [ "$1" != "" ]; then
	port=$1
fi

$CARLA_ROOT/CarlaUE4.sh \
    -quality-level=Epic \
    -world-port=$port \
    -resx=800 \
    -resy=600 \
    -nosound \
    -graphicsadapter=0 \
    -carla-streaming-port=2001 \
    -RenderOffScreen &
