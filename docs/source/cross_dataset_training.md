# Cross-dataset Training

LEAD supports cross-dataset training as a side-feature, allowing you to train on datasets beyond CARLA.

## Table of Contents
- [Setup Waymo Data](#setup-waymo-data)
- [Setup NavSim Data](#setup-navsim-data)
- [Training on Waymo](#training-on-waymo)
- [Training on NavSim](#training-on-navsim)

## Setup Waymo Data

Follow this step if you want to use Waymo E2E and Waymo Perception datasets.

### Install Google Cloud CLI

```bash
cd ..
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz
tar -xf google-cloud-cli-linux-x86_64.tar.gz
./google-cloud-sdk/install.sh
gcloud init
```

### Install Additional Python Packages

```bash
pip install waymo-open-dataset-tf-2-12-0==1.6.7
pip install numpy==1.26.0 # Reinstall the correct numpy
```

### Download Waymo Data

```bash
TODO: Add Waymo download instructions
```

## Setup NavSim Data

Follow this step if you want to use the `navtrain` split.

### Download Data

First download data and organize them:

```bash
cd 3rd_party/navsim_workspace/dataset
bash 3rd_party/navsim_workspace/navsimv1.1/download/download_navtrain_parallel.sh
bash 3rd_party/navsim_workspace/navsimv1.1/download/download_test_parallel.sh
bash 3rd_party/navsim_workspace/navsimv2.2/download/download_navhard_two_stage.sh
bash 3rd_party/navsim_workspace/navsimv1.1/download/download_maps.sh
```

### Build Cache

```bash
conda activate navsimv1.1
sbatch scripts/tools/data/navsim/001_navtest_cache.sh
conda activate navsimv2.2
sbatch scripts/tools/data/navsim/002_navhard_cache.sh
```

## Training on Waymo

TODO: Add Waymo training instructions

## Training on NavSim

TODO: Add NavSim training instructions

## Cross-Dataset Evaluation

TODO: Add cross-dataset evaluation instructions
