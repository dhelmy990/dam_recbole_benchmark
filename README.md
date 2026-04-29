# Usage
main.ipynb is a simple sequential implementation of the experiments that was run on google colab with an A100 GPU, and can be used as a simple entry point

# RecBole Benchmarking: BPR vs LightGCN vs SGL

Benchmarking three collaborative filtering models of increasing complexity across three structurally diverse recommendation datasets.

| Model | Type | Paper |
|-------|------|-------|
| **BPR** | Matrix Factorisation | Bayesian Personalized Ranking from Implicit Feedback (UAI 2009) |
| **LightGCN** | Graph-based | Simplifying and Powering GCN for Recommendation (SIGIR 2020) |
| **SGL** | Contrastive Learning | Self-supervised Graph Learning for Recommendation (SIGIR 2021) |

## Datasets

| Dataset | Users | Items | Interactions | Domain | Sparsity |
|---------|-------|-------|--------------|--------|----------|
| Last.FM (HetRec 2011) | 1,892 | 17,632 | 92,834 | Music | 99.72% |
| ModCloth | 47,958 | 1,378 | 82,790 | Fashion | 99.87% |
| ML-100K | 943 | 1,682 | 100,000 | Movies | 93.70% |

ML-100K is auto-downloaded by RecBole. Last.FM and ModCloth are hosted on Google Drive and downloaded automatically by the notebook.

## Experiments

1. **Primary comparison** — all 3 models × 3 datasets at embedding size 64
2. **Data sparsity analysis** — training on 100%, 80%, 60%, 40% of data
3. **Embedding size sensitivity** — dimensions {32, 64, 128}

### Evaluation

- **Recall@10** and **NDCG@10** under the uni100 protocol
- 80/10/10 train/valid/test random split, grouped by user
- Explicit ratings (ModCloth, ML-100K) binarised at threshold ≥ 3

## Quick Start

### Environment

```bash
pip install torch==2.5.1 torchvision \
    "recbole==1.2.0" \
    "numpy<2.0" pandas matplotlib seaborn \
    kmeans-pytorch \
    "ray[tune]>=2.0.0" \
    "pyarrow<15.0.0" \
    "scipy<1.13"
```

### Run

The full experiment pipeline is in `main.ipynb`. Open it in Colab (A100 GPU recommended) and run all cells. Total runtime is approximately 7 hours.

The notebook will:
1. Install dependencies and download datasets
2. Run the sparsity analysis (3 models × 3 datasets × 4 ratios)
3. Run the embedding sensitivity study (3 models × 3 datasets × 3 sizes)
4. Generate all figures to `results/figures/`

### Output

Results are saved to `results/`:
- `metrics/` — per-experiment JSON files
- `figures/` — all plots (sparsity curves, embedding bar charts, heatmaps, degree distributions)
- `experiment_summary.json` — aggregate summary

## Reproducibility

All experiments use seed 42. To reproduce:
1. Open `main.ipynb` in Google Colab with A100 runtime
2. Run all cells

## Configuration

### Shared hyperparameters

| Parameter | Value |
|-----------|-------|
| Epochs | 30 |
| Batch size | 2048 |
| Learning rate | 0.001 |
| Early stopping | 3 eval steps |
| Eval mode | uni100 |

### Model-specific

| Parameter | BPR | LightGCN | SGL |
|-----------|-----|----------|-----|
| Embedding size | 64 | 64 | 64 |
| n_layers | — | 3 | 3 |
| reg_weight | — | 1e-5 | 1e-5 |
| ssl_tau | — | — | 0.2 |
| ssl_weight | — | — | 0.1 |
| drop_ratio | — | — | 0.1 |

## Team

| Member | Contributions |
|--------|--------------|
| Diego Adriel Helmy | Equal |
| Lam Yik Ting | Equal |

## References

1. Rendle et al. "BPR: Bayesian Personalized Ranking from Implicit Feedback." UAI 2009.
2. He et al. "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation." SIGIR 2020.
3. Wu et al. "Self-supervised Graph Learning for Recommendation." SIGIR 2021.
4. Zhao et al. "RecBole: Towards a Unified, Comprehensive and Efficient Framework for Recommendation Algorithms." CIKM 2021.
