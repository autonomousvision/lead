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

export CUDA_VISIBLE_DEVICES=1

$CARLA_ROOT/CarlaUE4.sh \
    --allow-root \
    -quality-level=Low \
    -world-port=$port \
    -resx=640 \
    -resy=480 \
    -nosound \
    -graphicsadapter=0 \
    -carla-streaming-port=$streaming_port \
    -opengl \
    -RenderOffScreen &
