#!/bin/bash

mkdir -p $1

rsync -avz --info=progress2 \
    --include="model_*.pth" \
    --include="config.json" \
    --exclude="*" \
    cluster:$1/ $1/

rsync -avz --info=progress2 cluster:$1/ $1/
