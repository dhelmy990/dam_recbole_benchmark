"""
Data Sparsity Analysis Experiment

Evaluates model robustness by incrementally subsampling training data
at ratios: 0.8, 0.6, 0.4 (and full data at 1.0)
"""

import os
import sys
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import json
import numpy as np
import pandas as pd
from copy import deepcopy

from recbole.quick_start import run_recbole
from recbole.utils import init_seed
from recbole.data import create_dataset, data_preparation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.config_loader import load_config, merge_configs
from src.utils.logger import setup_logging
from src.utils.results_handler import save_results


class SparsityAnalysis:
    """
    Conduct data sparsity analysis by training models on
    different fractions of training data.
    """

    def __init__(
        self,
        models: List[str] = ['SASRec', 'LightGCN', 'SGL'],
        datasets: List[str] = ['ml-100k', 'amazon-beauty'],
        sparsity_ratios: List[float] = [1.0, 0.8, 0.6, 0.4],
        config_dir: str = 'configs',
        output_dir: str = 'results',
        seed: int = 42,
        n_runs: int = 1
    ):
        self.models = models
        self.datasets = datasets
        self.sparsity_ratios = sorted(sparsity_ratios, reverse=True)
        self.config_dir = config_dir
        self.output_dir = output_dir
        self.seed = seed
        self.n_runs = n_runs

        self.results: List[Dict[str, Any]] = []
        self.logger = setup_logging(
            log_file=os.path.join(output_dir, 'logs', 'sparsity_analysis.log')
        )

    def _subsample_dataset(self, dataset, ratio: float):
        """
        Subsample the training portion of the dataset.

        Args:
            dataset: RecBole dataset object.
            ratio: Fraction of training data to keep.

        Returns:
            Subsampled dataset interactions.
        """
        if ratio >= 1.0:
            return dataset

        # Get the number of interactions to sample
        n_interactions = len(dataset.inter_feat)
        n_sample = int(n_interactions * ratio)

        # Random sampling
        np.random.seed(self.seed)
        indices = np.random.choice(n_interactions, n_sample, replace=False)
        indices = np.sort(indices)

        # Update dataset
        for field in dataset.inter_feat:
            dataset.inter_feat[field] = dataset.inter_feat[field][indices]

        return dataset

    def run_single_experiment(
        self,
        model: str,
        dataset_name: str,
        sparsity_ratio: float,
        run_id: int = 0
    ) -> Dict[str, Any]:
        """
        Run a single experiment with specified parameters.

        Args:
            model: Model name.
            dataset_name: Dataset name.
            sparsity_ratio: Training data ratio.
            run_id: Run identifier for multiple runs.

        Returns:
            Results dictionary.
        """
        experiment_name = f"sparsity_{model}_{dataset_name}_r{sparsity_ratio}_run{run_id}"
        self.logger.info(f"Starting: {experiment_name}")

        # Load configs
        base_config = load_config(os.path.join(self.config_dir, 'base.yaml'))
        model_config = load_config(os.path.join(self.config_dir, 'models', f'{model.lower()}.yaml'))
        dataset_config = load_config(os.path.join(self.config_dir, 'datasets', f'{dataset_name}.yaml'))

        config_dict = merge_configs(base_config, model_config, dataset_config)
        config_dict['seed'] = self.seed + run_id
        config_dict['data_path'] = 'dataset/'

        # For subsampling, we modify the training ratio
        if sparsity_ratio < 1.0:
            # Adjust split ratios: keep same validation/test, reduce training
            train_ratio = 0.8 * sparsity_ratio
            config_dict['eval_args'] = {
                'split': {'RS': [train_ratio, 0.1, 0.1]},
                'group_by': 'user',
                'order': 'TO',
                'mode': 'full'
            }

        # Set seed
        init_seed(config_dict['seed'], reproducibility=True)

        try:
            result = run_recbole(
                model=model,
                dataset=dataset_name,
                config_dict=config_dict
            )

            metrics = {
                'experiment': 'sparsity_analysis',
                'model': model,
                'dataset': dataset_name,
                'sparsity_ratio': sparsity_ratio,
                'run_id': run_id,
                'seed': config_dict['seed'],
                'embedding_size': config_dict.get('embedding_size', 64),
                'recall@10': result['test_result'].get('recall@10', 0),
                'ndcg@10': result['test_result'].get('ndcg@10', 0),
                'best_valid_score': result.get('best_valid_score', 0),
                'status': 'success'
            }

            self.logger.info(
                f"Completed: {experiment_name} | "
                f"Recall@10: {metrics['recall@10']:.4f}, "
                f"NDCG@10: {metrics['ndcg@10']:.4f}"
            )

        except Exception as e:
            self.logger.error(f"Failed: {experiment_name} | Error: {str(e)}")
            metrics = {
                'experiment': 'sparsity_analysis',
                'model': model,
                'dataset': dataset_name,
                'sparsity_ratio': sparsity_ratio,
                'run_id': run_id,
                'status': 'failed',
                'error': str(e)
            }

        return metrics

    def run_all_experiments(self) -> pd.DataFrame:
        """
        Run all sparsity analysis experiments.

        Returns:
            DataFrame with all results.
        """
        self.logger.info("="*60)
        self.logger.info("STARTING DATA SPARSITY ANALYSIS")
        self.logger.info(f"Models: {self.models}")
        self.logger.info(f"Datasets: {self.datasets}")
        self.logger.info(f"Sparsity ratios: {self.sparsity_ratios}")
        self.logger.info(f"Number of runs: {self.n_runs}")
        self.logger.info("="*60)

        total_experiments = len(self.models) * len(self.datasets) * len(self.sparsity_ratios) * self.n_runs
        completed = 0

        for dataset in self.datasets:
            for model in self.models:
                for ratio in self.sparsity_ratios:
                    for run_id in range(self.n_runs):
                        completed += 1
                        self.logger.info(f"Progress: {completed}/{total_experiments}")

                        result = self.run_single_experiment(
                            model=model,
                            dataset_name=dataset,
                            sparsity_ratio=ratio,
                            run_id=run_id
                        )

                        self.results.append(result)

                        # Save intermediate results
                        save_results(
                            result,
                            os.path.join(
                                self.output_dir,
                                'metrics',
                                f"sparsity_{model}_{dataset}_r{ratio}_run{run_id}.json"
                            )
                        )

        self.logger.info("="*60)
        self.logger.info("DATA SPARSITY ANALYSIS COMPLETE")
        self.logger.info("="*60)

        return pd.DataFrame(self.results)

    def get_summary(self) -> pd.DataFrame:
        """
        Get summary statistics grouped by model, dataset, and sparsity ratio.

        Returns:
            Summary DataFrame.
        """
        df = pd.DataFrame(self.results)
        df = df[df['status'] == 'success']

        summary = df.groupby(['model', 'dataset', 'sparsity_ratio']).agg({
            'recall@10': ['mean', 'std'],
            'ndcg@10': ['mean', 'std']
        }).round(4)

        return summary

    def save_summary(self, filepath: Optional[str] = None) -> None:
        """Save summary to CSV file."""
        if filepath is None:
            filepath = os.path.join(self.output_dir, 'sparsity_analysis_summary.csv')

        summary = self.get_summary()
        summary.to_csv(filepath)
        self.logger.info(f"Summary saved to: {filepath}")


def main():
    """Run sparsity analysis from command line."""
    import argparse

    parser = argparse.ArgumentParser(description='Data Sparsity Analysis')
    parser.add_argument('--models', nargs='+', default=['SASRec', 'LightGCN', 'SGL'])
    parser.add_argument('--datasets', nargs='+', default=['ml-100k', 'amazon-beauty'])
    parser.add_argument('--ratios', nargs='+', type=float, default=[1.0, 0.8, 0.6, 0.4])
    parser.add_argument('--config_dir', type=str, default='configs')
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_runs', type=int, default=1)
    args = parser.parse_args()

    analysis = SparsityAnalysis(
        models=args.models,
        datasets=args.datasets,
        sparsity_ratios=args.ratios,
        config_dir=args.config_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        n_runs=args.n_runs
    )

    results = analysis.run_all_experiments()
    analysis.save_summary()

    print("\n" + "="*60)
    print("SPARSITY ANALYSIS SUMMARY")
    print("="*60)
    print(analysis.get_summary())


if __name__ == '__main__':
    main()
