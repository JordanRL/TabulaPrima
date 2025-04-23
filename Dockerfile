# Dockerfile
ARG BASE_IMAGE_TAG=runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

FROM ${BASE_IMAGE_TAG}

WORKDIR /workspace

# Install git, cmake, and libaio-dev
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    cmake \
    libaio-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone CUTLASS
RUN git clone https://github.com/NVIDIA/cutlass.git /workspace/cutlass
ENV CUTLASS_PATH=/workspace/cutlass
ENV DS_BUILD_CUTLASS=1

# Copy only the dependency file first
COPY pyproject.toml ./

# Install Python dependencies based *only* on the dependency file
# This layer is cached unless pyproject.toml changes
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir .

# Remove the dependency file
RUN rm -rf pyproject.toml

# Clone TabulaPrima
RUN git clone https://github.com/JordanRL/TabulaPrima.git /workspace/TabulaPrima

# Run and start the pod
CMD ["/start.sh"]