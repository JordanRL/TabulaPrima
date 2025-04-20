# Dockerfile
ARG BASE_IMAGE_TAG=runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

FROM ${BASE_IMAGE_TAG}

WORKDIR /workspace

# Install git
RUN apt-get update && apt-get install -y git --no-install-recommends && rm -rf /var/lib/apt/lists/*

# Copy only the dependency file first
COPY pyproject.toml ./

# Install Python dependencies based *only* on the dependency file
# This layer is cached unless pyproject.toml changes
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir .

# Copy the startup script and make it executable
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set the entrypoint script to run on container start
ENTRYPOINT ["/entrypoint.sh"]

# Define the *default* command to run AFTER the entrypoint finishes.
# If the entrypoint script uses 'exec' at the end, this CMD is ignored.
# If the entrypoint script finishes without 'exec', this CMD runs.
# We can make it just start a bash shell so you can run train.py manually.
CMD ["/start.sh"]