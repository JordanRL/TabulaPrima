#!/bin/sh


CUDA_NEWER_BASE="runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"
CUDA_NEWER_TAG="latest-cuda12.8"
CUDA_OLDER_BASE="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
CUDA_OLDER_TAG="latest-cuda12.4"

# Define your image name
IMAGE_NAME="jordanledoux/tabulaprima" # Or use a versioning scheme

# Run the Docker build
#  Assumes Dockerfile is in the repo root.
#  Add --platform linux/amd64 if needed!
echo "Building Docker image: ${IMAGE_NAME}:${CUDA_NEWER_TAG}..."
docker build --platform linux/amd64 -t "${IMAGE_NAME}:${CUDA_NEWER_TAG}" --build-arg BASE_IMAGE_TAG="${CUDA_NEWER_BASE}" .

# Check if build succeeded
if [ $? -ne 0 ]; then
  echo "Docker build failed! Aborting commit." >&2
  exit 1
fi

echo "Building Docker image: ${IMAGE_NAME}:${CUDA_OLDER_TAG}..."
docker build --platform linux/amd64 -t "${IMAGE_NAME}:${CUDA_OLDER_TAG}" --build-arg BASE_IMAGE_TAG="${CUDA_OLDER_BASE}" .

# Check if build succeeded
if [ $? -ne 0 ]; then
  echo "Docker build failed! Aborting commit." >&2
  exit 1
fi

# Push the image (Optional, but usually needed)
#  Assumes you are already logged in via 'docker login'
echo "Pushing Docker image: ${IMAGE_NAME}:${CUDA_NEWER_TAG} ..."
docker push "${IMAGE_NAME}:${CUDA_NEWER_TAG}"

# Check if push succeeded
if [ $? -ne 0 ]; then
  echo "Docker push failed! Aborting commit." >&2
  exit 1
fi

echo "Pushing Docker image: ${IMAGE_NAME}:${CUDA_OLDER_TAG} ..."
docker push "${IMAGE_NAME}:${CUDA_OLDER_TAG}"

# Check if push succeeded
if [ $? -ne 0 ]; then
  echo "Docker push failed! Aborting commit." >&2
  exit 1
fi

echo "Docker build and push successful."