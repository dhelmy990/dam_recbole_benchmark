#!/bin/bash
# Wrapper script to run experiments with isolated environment
# Usage: ./run.sh [python arguments]
# Example: ./run.sh main.py --model SASRec --dataset ml-100k

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate recbole_bench

# Disable user site-packages to avoid conflicts with ~/.local
export PYTHONNOUSERSITE=1

# Run python with the provided arguments
python "$@"
