#!/bin/bash
# Setup script for RecBole Benchmarking Environment
# Creates a clean, reproducible environment

set -e

ENV_NAME="recbole_bench"

echo "========================================"
echo "RecBole Benchmarking Environment Setup"
echo "========================================"

# Disable user site-packages to avoid conflicts
export PYTHONNOUSERSITE=1

# Remove existing environment if it exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Removing existing environment..."
    conda env remove -n $ENV_NAME -y
fi

echo ""
echo "Creating fresh environment with Python 3.10..."
conda create -n $ENV_NAME python=3.10 -y

echo ""
echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

echo ""
echo "Installing PyTorch with CUDA 11.8 (stable version)..."
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

echo ""
echo "Installing RecBole and dependencies..."
pip install recbole==1.2.0
pip install "ray[tune]==2.9.0"

echo ""
echo "Downgrading NumPy for RecBole compatibility..."
pip install "numpy<2.0"

echo ""
echo "Installing visualization and utility packages..."
pip install matplotlib seaborn

echo ""
echo "Installing additional dependencies..."
pip install kmeans-pytorch "pyarrow<15.0.0" "setuptools<70"

echo ""
echo "========================================"
echo "Setup complete!"
echo ""
echo "To activate the environment and run experiments:"
echo "  conda activate $ENV_NAME"
echo "  export PYTHONNOUSERSITE=1  # Isolate from user packages"
echo "  python main.py --model SASRec --dataset ml-100k"
echo ""
echo "Or use the wrapper script:"
echo "  ./run.sh main.py --model SASRec --dataset ml-100k"
echo "========================================"
