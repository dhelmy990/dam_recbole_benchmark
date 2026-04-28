#!/bin/bash
# Master script to run all experiments
# Usage: ./scripts/run_all_experiments.sh

set -e

echo "=========================================="
echo "RecBole Benchmarking Experiment Suite"
echo "=========================================="

# Configuration
SEED=42
N_RUNS=3
OUTPUT_DIR="results"
CONFIG_DIR="configs"

# Ensure dataset directory exists
mkdir -p dataset

echo ""
echo "Step 1: Downloading datasets..."
echo "------------------------------------------"

# RecBole downloads datasets automatically, but we can pre-download
python -c "
from recbole.utils import get_dataset_download_url
print('Datasets will be downloaded automatically by RecBole')
"

echo ""
echo "Step 2: Running experiments..."
echo "------------------------------------------"

python scripts/run_all_experiments.py \
    --models SASRec LightGCN SGL \
    --datasets ml-100k amazon-beauty \
    --config_dir $CONFIG_DIR \
    --output_dir $OUTPUT_DIR \
    --seed $SEED \
    --n_runs $N_RUNS

echo ""
echo "Step 3: Generating final report..."
echo "------------------------------------------"

python scripts/generate_report.py --output_dir $OUTPUT_DIR

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "Results: $OUTPUT_DIR/"
echo "=========================================="
