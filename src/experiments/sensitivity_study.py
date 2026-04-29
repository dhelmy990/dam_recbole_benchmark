"""
Embedding Size Sensitivity Study

Evaluates how model performance varies with different embedding dimensions:
d in {32, 64, 128}
"""

import os
import sys
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import json
import pandas as pd

from recbole.quick_start import run_recbole
from recbole.utils import init_seed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.config_loader import load_config, merge_configs
from src.utils.logger import setup_logging
from src.utils.results_handler import save_results


class SensitivityStudy:
    """
    Conduct sensitivity analysis on embedding size parameter.
    """

    def __init__(
        self,
        models: List[str] = ['SASRec', 'LightGCN', 'SGL'],
        datasets: List[str] = ['ml-100k', 'amazon-beauty'],
        embedding_sizes: List[int] = [32, 64, 128],
        config_dir: str = 'configs',
        output_dir: str = 'results',
        seed: int = 42,
        n_runs: int = 1
    ):
        self.models = models
        self.datasets = datasets
        self.embedding_sizes = sorted(embedding_sizes)
        self.config_dir = config_dir
        self.output_dir = output_dir
        self.seed = seed
        self.n_runs = n_runs

        self.results: List[Dict[str, Any]] = []
        self.logger = setup_logging(
            log_file=os.path.join(output_dir, 'logs', 'sensitivity_study.log')
        )

    def run_single_experiment(
        self,
        model: str,
        dataset_name: str,
        embedding_size: int,
        run_id: int = 0
    ) -> Dict[str, Any]:
        """
        Run a single experiment with specified embedding size.

        Args:
            model: Model name.
            dataset_name: Dataset name.
            embedding_size: Embedding dimension.
            run_id: Run identifier.

        Returns:
            Results dictionary.
        """
        experiment_name = f"emb_{model}_{dataset_name}_d{embedding_size}_run{run_id}"
        self.logger.info(f"Starting: {experiment_name}")

        # Load configs
        base_config = load_config(os.path.join(self.config_dir, 'base.yaml'))
        model_config = load_config(os.path.join(self.config_dir, 'models', f'{model.lower()}.yaml'))
        dataset_config = load_config(os.path.join(self.config_dir, 'datasets', f'{dataset_name}.yaml'))

        config_dict = merge_configs(base_config, model_config, dataset_config)

        # Override embedding size
        config_dict['embedding_size'] = embedding_size

        # For SASRec, also update hidden_size to match embedding_size
        if model == 'SASRec':
            config_dict['hidden_size'] = embedding_size
            config_dict['inner_size'] = embedding_size * 4

        config_dict['seed'] = self.seed + run_id
        config_dict['data_path'] = 'dataset/'

        # Set seed
        init_seed(config_dict['seed'], reproducibility=True)

        try:
            result = run_recbole(
                model=model,
                dataset=dataset_name,
                config_dict=config_dict
            )

            metrics = {
                'experiment': 'sensitivity_study',
                'model': model,
                'dataset': dataset_name,
                'embedding_size': embedding_size,
                'sparsity_ratio': 1.0,
                'run_id': run_id,
                'seed': config_dict['seed'],
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
                'experiment': 'sensitivity_study',
                'model': model,
                'dataset': dataset_name,
                'embedding_size': embedding_size,
                'run_id': run_id,
                'status': 'failed',
                'error': str(e)
            }

        return metrics

    def run_all_experiments(self) -> pd.DataFrame:
        """
        Run all sensitivity study experiments.

        Returns:
            DataFrame with all results.
        """
        self.logger.info("="*60)
        self.logger.info("STARTING EMBEDDING SIZE SENSITIVITY STUDY")
        self.logger.info(f"Models: {self.models}")
        self.logger.info(f"Datasets: {self.datasets}")
        self.logger.info(f"Embedding sizes: {self.embedding_sizes}")
        self.logger.info(f"Number of runs: {self.n_runs}")
        self.logger.info("="*60)

        total_experiments = len(self.models) * len(self.datasets) * len(self.embedding_sizes) * self.n_runs
        completed = 0

        for dataset in self.datasets:
            for model in self.models:
                for emb_size in self.embedding_sizes:
                    for run_id in range(self.n_runs):
                        completed += 1
                        self.logger.info(f"Progress: {completed}/{total_experiments}")

                        result = self.run_single_experiment(
                            model=model,
                            dataset_name=dataset,
                            embedding_size=emb_size,
                            run_id=run_id
                        )

                        self.results.append(result)

                        # Save intermediate results
                        save_results(
                            result,
                            os.path.join(
                                self.output_dir,
                                'metrics',
                                f"emb_{model}_{dataset}_d{emb_size}_run{run_id}.json"
                            )
                        )

        self.logger.info("="*60)
        self.logger.info("SENSITIVITY STUDY COMPLETE")
        self.logger.info("="*60)

        return pd.DataFrame(self.results)

    def get_summary(self) -> pd.DataFrame:
        """
        Get summary statistics grouped by model, dataset, and embedding size.

        Returns:
            Summary DataFrame.
        """
        df = pd.DataFrame(self.results)
        df = df[df['status'] == 'success']

        summary = df.groupby(['model', 'dataset', 'embedding_size']).agg({
            'recall@10': ['mean', 'std'],
            'ndcg@10': ['mean', 'std']
        }).round(4)

        return summary

    def save_summary(self, filepath: Optional[str] = None) -> None:
        """Save summary to CSV file."""
        if filepath is None:
            filepath = os.path.join(self.output_dir, 'sensitivity_study_summary.csv')

        summary = self.get_summary()
        summary.to_csv(filepath)
        self.logger.info(f"Summary saved to: {filepath}")

    def find_optimal_embedding(self) -> pd.DataFrame:
        """
        Find optimal embedding size for each model-dataset combination.

        Returns:
            DataFrame with optimal configurations.
        """
        df = pd.DataFrame(self.results)
        df = df[df['status'] == 'success']

        optimal = df.loc[df.groupby(['model', 'dataset'])['recall@10'].idxmax()]
        return optimal[['model', 'dataset', 'embedding_size', 'recall@10', 'ndcg@10']]


def main():
    """Run sensitivity study from command line."""
    import argparse

    parser = argparse.ArgumentParser(description='Embedding Size Sensitivity Study')
    parser.add_argument('--models', nargs='+', default=['SASRec', 'LightGCN', 'SGL'])
    parser.add_argument('--datasets', nargs='+', default=['ml-100k', 'amazon-beauty', 'steam'])
    parser.add_argument('--embedding_sizes', nargs='+', type=int, default=[32, 64, 128])
    parser.add_argument('--config_dir', type=str, default='configs')
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_runs', type=int, default=1)
    args = parser.parse_args()

    study = SensitivityStudy(
        models=args.models,
        datasets=args.datasets,
        embedding_sizes=args.embedding_sizes,
        config_dir=args.config_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        n_runs=args.n_runs
    )

    results = study.run_all_experiments()
    study.save_summary()

    print("\n" + "="*60)
    print("SENSITIVITY STUDY SUMMARY")
    print("="*60)
    print(study.get_summary())

    print("\n" + "="*60)
    print("OPTIMAL EMBEDDING SIZES")
    print("="*60)
    print(study.find_optimal_embedding())


if __name__ == '__main__':
    main()
