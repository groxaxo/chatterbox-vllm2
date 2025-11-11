# Chatterbox TTS on vLLM - Docker Image
# OpenAI-compatible TTS API optimized for low VRAM GPUs
#
# Build: docker build -t chatterbox-tts-api .
# Run: docker run --gpus all -p 8000:8000 chatterbox-tts-api
#
# Based on CUDA 12.1 runtime for GPU acceleration
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager for fast dependency resolution
RUN pip3 install uv

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install Python dependencies
# This will download model weights from Hugging Face Hub (~1-2GB)
RUN uv sync

# Expose API server port
EXPOSE 8000

# Default environment variables (optimized for low VRAM GPUs)
# Override these in docker-compose.yml or with -e flags
ENV CHATTERBOX_MODEL=multilingual
ENV CHATTERBOX_MAX_BATCH_SIZE=1
ENV CHATTERBOX_MAX_MODEL_LEN=800
ENV CHATTERBOX_HOST=0.0.0.0
ENV CHATTERBOX_PORT=8000

# Health check endpoint
# Allows Docker/K8s to monitor service health
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the OpenAI-compatible API server
CMD ["python3", "api_server.py"]
