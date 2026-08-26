#!/bin/bash

# Create checkpoint directory
mkdir -p outputs/checkpoints/tfv6_resnet34

# Download config
wget https://huggingface.co/ln2697/tfv6/resolve/main/4cameras_resnet34/config.json \
  -O outputs/checkpoints/tfv6_resnet34/config.json

# Download one checkpoint
wget https://huggingface.co/ln2697/tfv6/resolve/main/4cameras_resnet34/model_0030_0.pth \
  -O outputs/checkpoints/tfv6_resnet34/model_0030_0.pth

wget https://huggingface.co/ln2697/tfv6/resolve/main/4cameras_resnet34/model_0030_1.pth \
  -O outputs/checkpoints/tfv6_resnet34/model_0030_1.pth

wget https://huggingface.co/ln2697/tfv6/resolve/main/4cameras_resnet34/model_0030_2.pth \
  -O outputs/checkpoints/tfv6_resnet34/model_0030_2.pth
