#!/usr/bin/env python3
"""
Generate comprehensive report from experiment results.

Creates:
1. Summary statistics
2. All visualizations
3. LaTeX tables for the report
4. Training analysis
"""

import os
import sys
import argparse
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.visualizer import ResultsVisualizer
from src.utils.results_handler import ResultsCollector
import pandas as pd


def generate_latex_tables(collector: ResultsCollector, output_dir: str):
    """Generate LaTeX tables for the report."""
    df = collector.load_all_results()

    if len(df) == 0:
        print("No results to generate tables from.")
        return

    # Main results table
    main_results = df.pivot_table(
        values=['recall@10', 'ndcg@10'],
        index='model',
        columns='dataset',
        aggfunc='mean'
    ).round(4)

    latex_main = main_results.to_latex(
        caption='Overall performance comparison of recommendation models.',
        label='tab:main_results',
        float_format='%.4f',
        multicolumn=True,
        multicolumn_format='c'
    )

    with open(os.path.join(output_dir, 'table_main_results.tex'), 'w') as f:
        f.write(latex_main)

    # Sparsity analysis table
    if 'sparsity_ratio' in df.columns:
        sparsity_df = df[df['experiment'] == 'sparsity_analysis'] if 'experiment' in df.columns else df

        for dataset in df['dataset'].unique():
            dataset_df = sparsity_df[sparsity_df['dataset'] == dataset]
            if len(dataset_df) > 0:
                sparsity_table = dataset_df.pivot_table(
                    values='recall@10',
                    index='model',
                    columns='sparsity_ratio',
                    aggfunc='mean'
                ).round(4)

                latex_sparsity = sparsity_table.to_latex(
                    caption=f'Data sparsity analysis on {dataset}.',
                    label=f'tab:sparsity_{dataset.replace("-", "_")}',
                    float_format='%.4f'
                )

                with open(os.path.join(output_dir, f'table_sparsity_{dataset}.tex'), 'w') as f:
                    f.write(latex_sparsity)

    # Embedding sensitivity table
    if 'embedding_size' in df.columns:
        emb_df = df[df['experiment'] == 'sensitivity_study'] if 'experiment' in df.columns else df

        for dataset in df['dataset'].unique():
            dataset_df = emb_df[emb_df['dataset'] == dataset]
            if len(dataset_df) > 0:
                emb_table = dataset_df.pivot_table(
                    values='recall@10',
                    index='model',
                    columns='embedding_size',
                    aggfunc='mean'
                ).round(4)

                latex_emb = emb_table.to_latex(
                    caption=f'Embedding size sensitivity on {dataset}.',
                    label=f'tab:embedding_{dataset.replace("-", "_")}',
                    float_format='%.4f'
                )

                with open(os.path.join(output_dir, f'table_embedding_{dataset}.tex'), 'w') as f:
                    f.write(latex_emb)

    print(f"LaTeX tables saved to: {output_dir}")


def generate_summary_stats(collector: ResultsCollector, output_dir: str):
    """Generate summary statistics."""
    df = collector.load_all_results()

    if len(df) == 0:
        print("No results to summarize.")
        return

    summary = {
        'generated_at': datetime.now().isoformat(),
        'total_experiments': len(df),
        'successful_experiments': len(df[df.get('status', 'success') == 'success']) if 'status' in df.columns else len(df),
        'models': df['model'].unique().tolist(),
        'datasets': df['dataset'].unique().tolist(),
    }

    # Best results per model-dataset
    best_results = []
    for model in df['model'].unique():
        for dataset in df['dataset'].unique():
            subset = df[(df['model'] == model) & (df['dataset'] == dataset)]
            if len(subset) > 0 and 'recall@10' in subset.columns:
                best_idx = subset['recall@10'].idxmax()
                best = subset.loc[best_idx]
                best_results.append({
                    'model': model,
                    'dataset': dataset,
                    'recall@10': float(best.get('recall@10', 0)),
                    'ndcg@10': float(best.get('ndcg@10', 0)),
                    'embedding_size': int(best.get('embedding_size', 64)),
                    'sparsity_ratio': float(best.get('sparsity_ratio', 1.0))
                })

    summary['best_results'] = best_results

    # Overall statistics
    if 'recall@10' in df.columns:
        summary['recall@10_stats'] = {
            'mean': float(df['recall@10'].mean()),
            'std': float(df['recall@10'].std()),
            'min': float(df['recall@10'].min()),
            'max': float(df['recall@10'].max())
        }

    if 'ndcg@10' in df.columns:
        summary['ndcg@10_stats'] = {
            'mean': float(df['ndcg@10'].mean()),
            'std': float(df['ndcg@10'].std()),
            'min': float(df['ndcg@10'].min()),
            'max': float(df['ndcg@10'].max())
        }

    with open(os.path.join(output_dir, 'summary_stats.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Summary statistics saved to: {output_dir}/summary_stats.json")
    return summary


def main():
    parser = argparse.ArgumentParser(description='Generate report from experiment results')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory containing results')
    parser.add_argument('--figures_only', action='store_true',
                        help='Only generate figures')
    parser.add_argument('--tables_only', action='store_true',
                        help='Only generate tables')
    args = parser.parse_args()

    metrics_dir = os.path.join(args.output_dir, 'metrics')
    figures_dir = os.path.join(args.output_dir, 'figures')
    tables_dir = os.path.join(args.output_dir, 'tables')

    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    collector = ResultsCollector(results_dir=metrics_dir)

    print("="*60)
    print("GENERATING EXPERIMENT REPORT")
    print("="*60)

    if not args.tables_only:
        print("\nGenerating visualizations...")
        visualizer = ResultsVisualizer(
            results_dir=metrics_dir,
            output_dir=figures_dir
        )
        visualizer.generate_all_plots()

    if not args.figures_only:
        print("\nGenerating LaTeX tables...")
        generate_latex_tables(collector, tables_dir)

        print("\nGenerating summary statistics...")
        summary = generate_summary_stats(collector, args.output_dir)

        if summary:
            print("\n" + "="*60)
            print("SUMMARY")
            print("="*60)
            print(f"Total experiments: {summary['total_experiments']}")
            print(f"Models: {', '.join(summary['models'])}")
            print(f"Datasets: {', '.join(summary['datasets'])}")

            if 'best_results' in summary:
                print("\nBest Results:")
                for r in summary['best_results']:
                    print(f"  {r['model']} on {r['dataset']}: "
                          f"Recall@10={r['recall@10']:.4f}, NDCG@10={r['ndcg@10']:.4f}")

    print("\n" + "="*60)
    print("REPORT GENERATION COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
