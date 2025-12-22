# Docker Setup for LEAD

This guide explains how to use Docker to run LEAD evaluations without installing dependencies directly on your system.

## Prerequisites

- Docker Engine 29.0 or later
- NVIDIA Docker runtime
- NVIDIA GPU with CUDA support

Install docker as shown [here](https://docs.docker.com/engine/install/ubuntu/).

Follow this [guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) to install NVIDIA Docker runtime.

## Quick Start

### 1. Build the Docker images

```bash
docker compose build
```

This will:
- Build the LEAD evaluation environment
- Install all Python dependencies
- Download and setup CARLA 0.9.15

### 2. Download checkpoints

Download pre-trained checkpoints to `outputs/checkpoints/`:

```bash
mkdir -p outputs/checkpoints/tfv6_resnet34
wget https://huggingface.co/ln2697/TFv6/resolve/main/tfv6_resnet34/config.json -O outputs/checkpoints/tfv6_resnet34/config.json
wget https://huggingface.co/ln2697/TFv6/resolve/main/tfv6_resnet34/model_0030_0.pth -O outputs/checkpoints/tfv6_resnet34/model_0030_0.pth
```

### 3. Run evaluation

#### Evaluate on Bench2Drive:

```bash
# Start CARLA and run Bench2Drive evaluation
docker compose up carla eval-bench2drive
```

#### Evaluate on Longest6:

```bash
# Start CARLA and run Longest6 evaluation
docker compose up carla eval-longest6
```

#### Evaluate on Town13:

```bash
# Start CARLA and run Town13 evaluation
docker compose up carla eval-town13
```

#### Run expert evaluation:

```bash
# Start CARLA and run expert data collection
docker compose up carla eval-expert
```

## Advanced Usage

### Run specific services

```bash
# Start only CARLA
docker compose up carla

# Run Bench2Drive evaluation in background
docker compose up -d carla eval-bench2drive

# Restart CARLA if it becomes unresponsive
docker compose restart carla

# Stop and remove all containers
docker compose down
```
### Maintenance

```bash
# Remove unused images
docker image prune -a

# Remove specific images
docker rmi lead-eval-bench2drive:latest
docker rmi lead-eval-longest6:latest
docker rmi lead-eval-town13:latest
docker rmi lead-eval-expert:latest
docker rmi carlasim/carla:0.9.15
```emove unused images
docker image prune -a

# Remove specific images
docker rmi lead-eval:latest
docker rmi carlasim/carla:0.9.15
```
