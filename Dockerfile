# RecBole Benchmarking Docker Image
# Comparing SASRec, LightGCN, and SGL recommendation models

FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

LABEL maintainer="RecBole Benchmarking Team"
LABEL description="Containerized environment for recommendation system benchmarking"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p dataset results/logs results/metrics results/figures saved

# Make scripts executable
RUN chmod +x scripts/*.sh 2>/dev/null || true

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command: show help
CMD ["python", "main.py", "--help"]

# Example usage:
# Build:  docker build -t recbole-benchmark .
# Run:    docker run --gpus all -v $(pwd)/results:/app/results recbole-benchmark \
#         python main.py --model SASRec --dataset ml-100k
# All:    docker run --gpus all -v $(pwd)/results:/app/results recbole-benchmark \
#         python scripts/run_all_experiments.py
