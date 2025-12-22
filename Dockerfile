# Use NVIDIA CUDA base image with Ubuntu 22.04
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    vim \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    parallel \
    tree \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

ENV PATH="/opt/conda/bin:${PATH}"

# Set working directory
WORKDIR /workspace/lead

# Copy project files
COPY . .

# Install conda-lock and create environment
RUN pip install conda-lock && \
    conda-lock install -n lead conda-lock.yml && \
    echo "source activate lead" >> ~/.bashrc

# Activate conda environment and install requirements
SHELL ["/bin/bash", "--login", "-c"]
RUN conda activate lead && \
    pip install -r requirements.txt

# Setup CARLA (download if not present)
RUN bash scripts/setup_carla.sh || true

# Set environment variables
ENV LEAD_PROJECT_ROOT=/workspace/lead
ENV CARLA_ROOT=/workspace/lead/3rd_party/CARLA_0915
ENV PYTHONPATH="${LEAD_PROJECT_ROOT}:${LEAD_PROJECT_ROOT}/3rd_party/CARLA_0915/PythonAPI/carla:${PYTHONPATH}"
ENV PYTHONUNBUFFERED=1

# Create directories for outputs
RUN mkdir -p outputs/local_evaluation outputs/checkpoints data/expert_debug

# Set default shell to bash with conda
SHELL ["/bin/bash", "--login", "-c"]

# Default command
CMD ["/bin/bash"]
