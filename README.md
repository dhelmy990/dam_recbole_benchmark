# RecBole Benchmarking: Comparing Recommendation Paradigms

A modular benchmarking framework comparing three state-of-the-art recommendation paradigms using RecBole:

| Model | Type | Paper |
|-------|------|-------|
| **SASRec** | Transformer-based | Self-Attentive Sequential Recommendation (ICDM 2018) |
| **LightGCN** | Graph-based | Simplifying and Powering GCN for Recommendation (SIGIR 2020) |
| **SGL** | Contrastive Learning | Self-supervised Graph Learning for Recommendation (SIGIR 2021) |

## Project Structure

```
project/
├── main.py                    # Main entry point
├── configs/
│   ├── base.yaml             # Base configuration
│   ├── models/
│   │   ├── sasrec.yaml       # SASRec hyperparameters
│   │   ├── lightgcn.yaml     # LightGCN hyperparameters
│   │   └── sgl.yaml          # SGL hyperparameters
│   └── datasets/
│       ├── ml-100k.yaml      # MovieLens 100K config
│       └── amazon-beauty.yaml # Amazon Beauty config
├── src/
│   ├── experiments/
│   │   ├── sparsity_analysis.py    # Data sparsity experiments
│   │   └── sensitivity_study.py    # Embedding size sensitivity
│   └── utils/
│       ├── config_loader.py   # YAML config utilities
│       ├── logger.py          # Logging utilities
│       ├── results_handler.py # Results I/O
│       └── visualizer.py      # Matplotlib visualizations
├── scripts/
│   ├── run_all_experiments.py # Orchestration script
│   ├── run_all_experiments.sh # Shell wrapper
│   └── generate_report.py     # Report generation
├── results/                   # Output directory
│   ├── logs/                 # Experiment logs
│   ├── metrics/              # JSON result files
│   └── figures/              # Visualization plots
├── Dockerfile                # Container definition
├── docker-compose.yml        # Multi-service orchestration
└── requirements.txt          # Python dependencies
```

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repository-url>
cd project

# RECOMMENDED: Use the setup script (creates isolated environment)
chmod +x setup_env.sh
./setup_env.sh

# ALTERNATIVE: Manual setup
conda create -n recbole_bench python=3.10 -y
conda activate recbole_bench
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
pip install recbole==1.2.0 "ray[tune]==2.9.0"
pip install matplotlib seaborn "numpy<2.0" kmeans-pytorch "pyarrow<15.0.0"
```

### 2. Run Single Experiment

```bash
# Activate environment and isolate from user packages
conda activate recbole_bench
export PYTHONNOUSERSITE=1

# Train SASRec on ml-100k
python main.py --model SASRec --dataset ml-100k

# Or use the wrapper script
./run.sh main.py --model SASRec --dataset ml-100k

# Train LightGCN on amazon-beauty with custom embedding size
./run.sh main.py --model LightGCN --dataset amazon-beauty --embedding_size 128
```

### 3. Run All Experiments

```bash
# Run complete benchmark suite
python scripts/run_all_experiments.py --n_runs 3

# Or with specific models/datasets
python scripts/run_all_experiments.py \
    --models SASRec LightGCN \
    --datasets ml-100k \
    --n_runs 1
```

## Experiments

### Data Sparsity Analysis

Evaluates model robustness by training on subsampled data (80%, 60%, 40% of training data):

```bash
python -m src.experiments.sparsity_analysis \
    --ratios 1.0 0.8 0.6 0.4 \
    --n_runs 3
```

### Embedding Size Sensitivity

Studies the effect of embedding dimensions (d ∈ {32, 64, 128}):

```bash
python -m src.experiments.sensitivity_study \
    --embedding_sizes 32 64 128 \
    --n_runs 3
```

## Evaluation Metrics

- **Recall@10**: Proportion of relevant items in top-10 recommendations
- **NDCG@10**: Normalized Discounted Cumulative Gain at rank 10

## Docker Usage

### Build and Run

```bash
# Build image
docker build -t recbole-benchmark .

# Run single experiment
docker run --gpus all -v $(pwd)/results:/app/results \
    recbole-benchmark python main.py --model SASRec --dataset ml-100k

# Run all experiments
docker run --gpus all -v $(pwd)/results:/app/results \
    recbole-benchmark python scripts/run_all_experiments.py
```

### Docker Compose

```bash
# Run full benchmark
docker-compose up benchmark

# Run individual model
docker-compose up sasrec-ml100k

# Generate visualizations only
docker-compose up visualize
```

## Results Visualization

Generated plots in `results/figures/`:

- `sparsity_<dataset>_<metric>.png`: Performance vs. data sparsity
- `embedding_<dataset>_<metric>.png`: Performance vs. embedding size
- `model_comparison.png`: Cross-model comparison
- `heatmap_<metric>.png`: Performance heatmap

## Reproducibility

All experiments use:
- Fixed random seed (default: 42)
- YAML-based configuration files
- Logged hyperparameters for each run
- JSON result files with timestamps

To reproduce results:
```bash
python scripts/run_all_experiments.py --seed 42 --n_runs 3
```

## Configuration

### Base Configuration (`configs/base.yaml`)
- Training: 100 epochs, batch size 2048
- Evaluation: 80/10/10 train/valid/test split
- Metrics: Recall@10, NDCG@10
- Early stopping: patience of 10

### Model-Specific Configuration
Each model YAML file contains optimized hyperparameters from the original papers.

## Datasets

| Dataset | Users | Items | Interactions | Domain |
|---------|-------|-------|--------------|--------|
| ml-100k | 943 | 1,682 | ~100,000 | Movies |
| amazon-beauty | ~22K | ~12K | ~200K | E-commerce |
| steam | ~12K | ~5K | ~200K | Video Games |

Datasets are downloaded automatically by RecBole on first run.

## Team Contributions

| Member | Contributions |
|--------|--------------|
| Diego Adriel Helmy | Equal |
| Lam Yik Ting | Equal |

## License

This project is for educational purposes as part of the Data Mining course.

## References

1. Rendle et al. "BPR: Bayesian Personalized Ranking from Implicit Feedback" UAI 2009.
2. He et al. "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation." SIGIR 2020.
3. Wu et al. "Self-supervised Graph Learning for Recommendation." SIGIR 2021.
4. Zhao et al. "RecBole: Towards a Unified, Comprehensive and Efficient Framework for Recommendation Algorithms." CIKM 2021.
