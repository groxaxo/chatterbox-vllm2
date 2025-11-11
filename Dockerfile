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

# Install uv package manager
RUN pip3 install uv

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install Python dependencies
RUN uv sync

# Expose API port
EXPOSE 8000

# Set default environment variables (can be overridden by docker-compose or docker run)
ENV CHATTERBOX_MODEL=multilingual
ENV CHATTERBOX_MAX_BATCH_SIZE=1
ENV CHATTERBOX_MAX_MODEL_LEN=800
ENV CHATTERBOX_HOST=0.0.0.0
ENV CHATTERBOX_PORT=8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the API server
CMD ["python3", "api_server.py"]
