#!/bin/bash

set -x

mkdir -p "$1"

rsync -avz --info=progress2 "cluster:/mnt/lustre/work/geiger/gwb581/$1/" "$1"
