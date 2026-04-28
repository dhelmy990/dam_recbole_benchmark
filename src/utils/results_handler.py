"""Results handling and collection utilities."""

import json
import os
import glob
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd


def save_results(results: Dict[str, Any], output_path: str) -> None:
    """
    Save results to a JSON file.

    Args:
        results: Results dictionary.
        output_path: Path to save the JSON file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Add timestamp
    results['timestamp'] = datetime.now().isoformat()

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)


def load_results(results_path: str) -> Dict[str, Any]:
    """
    Load results from a JSON file.

    Args:
        results_path: Path to the JSON results file.

    Returns:
        Results dictionary.
    """
    with open(results_path, 'r') as f:
        return json.load(f)


class ResultsCollector:
    """Collect and aggregate results from multiple experiments."""

    def __init__(self, results_dir: str = 'results/metrics'):
        self.results_dir = results_dir
        self.results: List[Dict[str, Any]] = []

    def load_all_results(self) -> pd.DataFrame:
        """
        Load all results from the results directory.

        Returns:
            DataFrame containing all results.
        """
        self.results = []
        pattern = os.path.join(self.results_dir, '*.json')

        for filepath in glob.glob(pattern):
            try:
                result = load_results(filepath)
                self.results.append(result)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"Warning: Could not load {filepath}: {e}")

        return pd.DataFrame(self.results)

    def filter_results(
        self,
        model: Optional[str] = None,
        dataset: Optional[str] = None,
        embedding_size: Optional[int] = None,
        sparsity_ratio: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Filter results by experiment parameters.

        Args:
            model: Filter by model name.
            dataset: Filter by dataset name.
            embedding_size: Filter by embedding size.
            sparsity_ratio: Filter by sparsity ratio.

        Returns:
            Filtered DataFrame.
        """
        df = self.load_all_results()

        if model:
            df = df[df['model'] == model]
        if dataset:
            df = df[df['dataset'] == dataset]
        if embedding_size:
            df = df[df['embedding_size'] == embedding_size]
        if sparsity_ratio:
            df = df[df['sparsity_ratio'] == sparsity_ratio]

        return df

    def aggregate_by(
        self,
        group_by: List[str],
        metrics: List[str] = ['recall@10', 'ndcg@10']
    ) -> pd.DataFrame:
        """
        Aggregate results by specified columns.

        Args:
            group_by: Columns to group by.
            metrics: Metrics to aggregate.

        Returns:
            Aggregated DataFrame with mean and std.
        """
        df = self.load_all_results()

        agg_dict = {}
        for metric in metrics:
            if metric in df.columns:
                agg_dict[metric] = ['mean', 'std']

        return df.groupby(group_by).agg(agg_dict).round(4)

    def get_best_results(
        self,
        metric: str = 'recall@10',
        group_by: List[str] = ['model', 'dataset']
    ) -> pd.DataFrame:
        """
        Get best results for each group.

        Args:
            metric: Metric to optimize.
            group_by: Columns to group by.

        Returns:
            DataFrame with best results for each group.
        """
        df = self.load_all_results()
        idx = df.groupby(group_by)[metric].idxmax()
        return df.loc[idx]

    def export_to_latex(
        self,
        output_path: str,
        metrics: List[str] = ['recall@10', 'ndcg@10']
    ) -> None:
        """
        Export results to LaTeX table format.

        Args:
            output_path: Path to save the LaTeX file.
            metrics: Metrics to include in the table.
        """
        df = self.load_all_results()

        # Pivot for better table format
        pivot = df.pivot_table(
            values=metrics,
            index='model',
            columns='dataset',
            aggfunc='mean'
        ).round(4)

        latex_str = pivot.to_latex(
            caption='Comparison of Recommendation Models',
            label='tab:results',
            float_format='%.4f'
        )

        with open(output_path, 'w') as f:
            f.write(latex_str)

    def export_summary(self, output_path: str) -> None:
        """
        Export a summary of all results.

        Args:
            output_path: Path to save the summary.
        """
        df = self.load_all_results()

        summary = {
            'total_experiments': len(df),
            'models': df['model'].unique().tolist() if 'model' in df.columns else [],
            'datasets': df['dataset'].unique().tolist() if 'dataset' in df.columns else [],
            'embedding_sizes': df['embedding_size'].unique().tolist() if 'embedding_size' in df.columns else [],
            'sparsity_ratios': df['sparsity_ratio'].unique().tolist() if 'sparsity_ratio' in df.columns else [],
            'best_recall@10': df['recall@10'].max() if 'recall@10' in df.columns else None,
            'best_ndcg@10': df['ndcg@10'].max() if 'ndcg@10' in df.columns else None,
        }

        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
