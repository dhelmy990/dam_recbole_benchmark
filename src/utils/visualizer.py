"""Visualization utilities for results analysis."""

import os
import json
import glob
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from collections import defaultdict


class ResultsVisualizer:
    """Generate visualizations for experiment results."""

    def __init__(self, results_dir: str = 'results/metrics', output_dir: str = 'results/figures'):
        self.results_dir = results_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Style configuration
        self.colors = {
            'SASRec': '#2ecc71',   # Green
            'LightGCN': '#3498db', # Blue
            'SGL': '#e74c3c'       # Red
        }
        self.markers = {
            'SASRec': 'o',
            'LightGCN': 's',
            'SGL': '^'
        }

        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'legend.fontsize': 11,
            'figure.figsize': (10, 6)
        })

    def load_results(self) -> pd.DataFrame:
        """Load all results into a DataFrame."""
        results = []
        pattern = os.path.join(self.results_dir, '*.json')

        for filepath in glob.glob(pattern):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    results.append(data)
            except (json.JSONDecodeError, FileNotFoundError):
                continue

        return pd.DataFrame(results)

    def plot_sparsity_analysis(
        self,
        dataset: str,
        metric: str = 'recall@10',
        save: bool = True
    ) -> plt.Figure:
        """
        Plot model performance vs data sparsity.

        Args:
            dataset: Dataset name to filter.
            metric: Metric to plot (recall@10 or ndcg@10).
            save: Whether to save the figure.

        Returns:
            Matplotlib figure.
        """
        df = self.load_results()
        df = df[df['dataset'] == dataset]

        fig, ax = plt.subplots(figsize=(10, 6))

        for model in ['SASRec', 'LightGCN', 'SGL']:
            model_df = df[df['model'] == model].sort_values('sparsity_ratio')

            if len(model_df) > 0:
                # Group by sparsity ratio and compute mean/std
                grouped = model_df.groupby('sparsity_ratio')[metric].agg(['mean', 'std'])

                ax.errorbar(
                    grouped.index,
                    grouped['mean'],
                    yerr=grouped['std'],
                    label=model,
                    color=self.colors.get(model, 'gray'),
                    marker=self.markers.get(model, 'o'),
                    markersize=8,
                    linewidth=2,
                    capsize=5
                )

        ax.set_xlabel('Training Data Ratio (Sparsity)')
        ax.set_ylabel(metric.replace('@', ' @ ').title())
        ax.set_title(f'Data Sparsity Analysis - {dataset}')
        ax.legend(loc='best')
        ax.set_xlim(0.35, 1.05)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))

        plt.tight_layout()

        if save:
            filepath = os.path.join(self.output_dir, f'sparsity_{dataset}_{metric.replace("@", "")}.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved: {filepath}")

        return fig

    def plot_embedding_sensitivity(
        self,
        dataset: str,
        metric: str = 'recall@10',
        save: bool = True
    ) -> plt.Figure:
        """
        Plot model performance vs embedding size.

        Args:
            dataset: Dataset name to filter.
            metric: Metric to plot.
            save: Whether to save the figure.

        Returns:
            Matplotlib figure.
        """
        df = self.load_results()
        df = df[df['dataset'] == dataset]

        fig, ax = plt.subplots(figsize=(10, 6))

        bar_width = 0.25
        embedding_sizes = [32, 64, 128]
        x = np.arange(len(embedding_sizes))

        for i, model in enumerate(['SASRec', 'LightGCN', 'SGL']):
            model_df = df[df['model'] == model]

            means = []
            stds = []
            for emb_size in embedding_sizes:
                size_df = model_df[model_df['embedding_size'] == emb_size]
                if len(size_df) > 0:
                    means.append(size_df[metric].mean())
                    stds.append(size_df[metric].std() if len(size_df) > 1 else 0)
                else:
                    means.append(0)
                    stds.append(0)

            ax.bar(
                x + i * bar_width,
                means,
                bar_width,
                yerr=stds,
                label=model,
                color=self.colors.get(model, 'gray'),
                capsize=4
            )

        ax.set_xlabel('Embedding Size')
        ax.set_ylabel(metric.replace('@', ' @ ').title())
        ax.set_title(f'Embedding Size Sensitivity - {dataset}')
        ax.set_xticks(x + bar_width)
        ax.set_xticklabels(embedding_sizes)
        ax.legend(loc='best')

        plt.tight_layout()

        if save:
            filepath = os.path.join(self.output_dir, f'embedding_{dataset}_{metric.replace("@", "")}.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved: {filepath}")

        return fig

    def plot_model_comparison(
        self,
        metrics: List[str] = ['recall@10', 'ndcg@10'],
        save: bool = True
    ) -> plt.Figure:
        """
        Plot comparison of models across datasets.

        Args:
            metrics: List of metrics to plot.
            save: Whether to save the figure.

        Returns:
            Matplotlib figure.
        """
        df = self.load_results()

        fig, axes = plt.subplots(1, len(metrics), figsize=(14, 6))
        if len(metrics) == 1:
            axes = [axes]

        datasets = df['dataset'].unique()
        models = ['SASRec', 'LightGCN', 'SGL']

        for ax, metric in zip(axes, metrics):
            bar_width = 0.25
            x = np.arange(len(datasets))

            for i, model in enumerate(models):
                means = []
                stds = []
                for dataset in datasets:
                    subset = df[(df['model'] == model) & (df['dataset'] == dataset)]
                    if len(subset) > 0 and metric in subset.columns:
                        means.append(subset[metric].mean())
                        stds.append(subset[metric].std() if len(subset) > 1 else 0)
                    else:
                        means.append(0)
                        stds.append(0)

                ax.bar(
                    x + i * bar_width,
                    means,
                    bar_width,
                    yerr=stds,
                    label=model,
                    color=self.colors.get(model, 'gray'),
                    capsize=4
                )

            ax.set_xlabel('Dataset')
            ax.set_ylabel(metric.replace('@', ' @ ').title())
            ax.set_title(f'Model Comparison: {metric}')
            ax.set_xticks(x + bar_width)
            ax.set_xticklabels(datasets)
            ax.legend(loc='best')

        plt.tight_layout()

        if save:
            filepath = os.path.join(self.output_dir, 'model_comparison.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved: {filepath}")

        return fig

    def plot_training_curves(
        self,
        training_logs: Dict[str, List[float]],
        title: str = 'Training Loss Curves',
        save: bool = True
    ) -> plt.Figure:
        """
        Plot training loss curves for multiple models.

        Args:
            training_logs: Dictionary mapping model names to loss lists.
            title: Plot title.
            save: Whether to save the figure.

        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        for model, losses in training_logs.items():
            epochs = range(1, len(losses) + 1)
            ax.plot(
                epochs,
                losses,
                label=model,
                color=self.colors.get(model, 'gray'),
                linewidth=2
            )

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Training Loss')
        ax.set_title(title)
        ax.legend(loc='best')

        plt.tight_layout()

        if save:
            filepath = os.path.join(self.output_dir, 'training_curves.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved: {filepath}")

        return fig

    def plot_heatmap(
        self,
        metric: str = 'recall@10',
        save: bool = True
    ) -> plt.Figure:
        """
        Plot heatmap of results across models and datasets.

        Args:
            metric: Metric to visualize.
            save: Whether to save the figure.

        Returns:
            Matplotlib figure.
        """
        df = self.load_results()

        # Create pivot table
        pivot = df.pivot_table(
            values=metric,
            index='model',
            columns='dataset',
            aggfunc='mean'
        )

        fig, ax = plt.subplots(figsize=(10, 6))

        im = ax.imshow(pivot.values, cmap='YlGnBu', aspect='auto')

        # Set labels
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticklabels(pivot.index)

        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel(metric.replace('@', ' @ ').title(), rotation=-90, va='bottom')

        # Add text annotations
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                value = pivot.values[i, j]
                if not np.isnan(value):
                    ax.text(j, i, f'{value:.4f}', ha='center', va='center',
                           color='white' if value > pivot.values.mean() else 'black')

        ax.set_title(f'Performance Heatmap: {metric}')
        plt.tight_layout()

        if save:
            filepath = os.path.join(self.output_dir, f'heatmap_{metric.replace("@", "")}.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved: {filepath}")

        return fig

    def generate_all_plots(self) -> None:
        """Generate all standard visualization plots."""
        df = self.load_results()

        if len(df) == 0:
            print("No results found to visualize.")
            return

        datasets = df['dataset'].unique()
        metrics = ['recall@10', 'ndcg@10']

        # Generate plots for each dataset and metric combination
        for dataset in datasets:
            for metric in metrics:
                try:
                    self.plot_sparsity_analysis(dataset, metric)
                except Exception as e:
                    print(f"Could not generate sparsity plot for {dataset}/{metric}: {e}")

                try:
                    self.plot_embedding_sensitivity(dataset, metric)
                except Exception as e:
                    print(f"Could not generate embedding plot for {dataset}/{metric}: {e}")

        # Generate comparison plots
        try:
            self.plot_model_comparison(metrics)
        except Exception as e:
            print(f"Could not generate comparison plot: {e}")

        try:
            for metric in metrics:
                self.plot_heatmap(metric)
        except Exception as e:
            print(f"Could not generate heatmap: {e}")

        print(f"\nAll visualizations saved to: {self.output_dir}")
