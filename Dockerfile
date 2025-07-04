# Multi-stage build for GRPO
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    curl \
    vim \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install the package
RUN pip3 install -e .

# Create necessary directories
RUN mkdir -p /app/models /app/data /app/outputs /app/logs

# Default command
CMD ["python3", "-m", "grpo.cli", "--help"]

# Development image with additional tools
FROM base AS development

# Install development dependencies
RUN pip3 install --no-cache-dir \
    jupyter \
    ipython \
    tensorboard \
    wandb

# Expose ports
EXPOSE 8888 6006

# Training image optimized for GPU
FROM base AS training

# Set up for training
ENV CUDA_VISIBLE_DEVICES=0

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import torch; print(torch.cuda.is_available())"

# Default training command
CMD ["python3", "src/train.py", "--config", "config/grpo_config.yaml"]

# Production inference image
FROM base AS inference

# Optimize for inference
ENV OMP_NUM_THREADS=1 \
    TORCH_NUM_THREADS=1

# Install inference-specific dependencies
RUN pip3 install --no-cache-dir \
    fastapi \
    uvicorn \
    prometheus-client

# Copy inference server
COPY scripts/inference_server.py /app/

# Expose API port
EXPOSE 8000

# Health check for API
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run inference server
CMD ["uvicorn", "inference_server:app", "--host", "0.0.0.0", "--port", "8000"]