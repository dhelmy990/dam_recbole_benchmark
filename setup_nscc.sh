#!/bin/bash
# ============================================
# Setup script for NSCC environment
# Run this ONCE before submitting jobs
# ============================================
# Usage: bash setup_nscc.sh
# ============================================

set -e

echo "=========================================="
echo "Setting up RecBole environment for NSCC"
echo "=========================================="

# Load modules
module purge
module load anaconda3
# module load cuda/11.8  # Uncomment if needed

# Create conda environment
ENV_NAME="recbole_bench"

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment $ENV_NAME already exists. Removing..."
    conda env remove -n $ENV_NAME -y
fi

echo ""
echo "Creating conda environment with Python 3.10..."
conda create -n $ENV_NAME python=3.10 -y

# Activate
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

echo ""
echo "Installing PyTorch with CUDA 11.8..."
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

echo ""
echo "Installing RecBole and Ray..."
pip install recbole==1.2.0 "ray[tune]==2.9.0"

echo ""
echo "Installing other dependencies..."
pip install matplotlib seaborn "numpy<2.0" kmeans-pytorch "pyarrow<15.0.0" "setuptools<70"

echo ""
echo "=========================================="
echo "Setup complete!"
echo ""
echo "To submit jobs:"
echo "  qsub submit_nscc.pbs"
echo ""
echo "To check job status:"
echo "  qstat -u \$USER"
echo ""
echo "Results will be in: results/"
echo "=========================================="
