FROM ubuntu:22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install core dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    python3 \
    python3-pip \
    time \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch (CPU version to keep it lightweight) and standard data libraries
RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install --no-cache-dir numpy pandas scipy
