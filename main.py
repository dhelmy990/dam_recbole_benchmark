"""
RecBole Benchmarking: Comparing SASRec, LightGCN, and SGL
Main entry point for training and evaluation.
"""

import argparse
import os
import logging
from datetime import datetime

from recbole.quick_start import run_recbole
from recbole.utils import init_seed

from src.utils.config_loader import load_config, merge_configs
from src.utils.logger import setup_logging, log_experiment_params
from src.utils.results_handler import save_results, ResultsCollector


def parse_args():
    parser = argparse.ArgumentParser(description='RecBole Benchmarking Framework')
    parser.add_argument('--model', type=str, required=True,
                        choices=['SASRec', 'LightGCN', 'SGL'],
                        help='Model to train: SASRec, LightGCN, or SGL')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['ml-100k', 'amazon-beauty', 'steam'],
                        help='Dataset to use: ml-100k, amazon-beauty, or steam')
    parser.add_argument('--config_dir', type=str, default='configs',
                        help='Directory containing configuration files')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--embedding_size', type=int, default=64,
                        help='Embedding dimension size')
    parser.add_argument('--sparsity_ratio', type=float, default=1.0,
                        help='Training data ratio (for sparsity analysis)')
    return parser.parse_args()


def run_experiment(args):
    """Run a single experiment with given configuration."""

    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = f"{args.model}_{args.dataset}_{timestamp}"
    log_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    logger = setup_logging(
        log_file=os.path.join(log_dir, f"{experiment_name}.log")
    )

    # Load configurations
    base_config = load_config(os.path.join(args.config_dir, 'base.yaml'))
    model_config = load_config(os.path.join(args.config_dir, 'models', f'{args.model.lower()}.yaml'))
    dataset_config = load_config(os.path.join(args.config_dir, 'datasets', f'{args.dataset}.yaml'))

    # Merge configurations
    config_dict = merge_configs(base_config, model_config, dataset_config)

    # Override with command line arguments
    config_dict['seed'] = args.seed
    config_dict['embedding_size'] = args.embedding_size

    # Handle sparsity ratio for data sampling
    if args.sparsity_ratio < 1.0:
        # Adjust split ratios: reduce training portion
        train_ratio = 0.8 * args.sparsity_ratio
        config_dict['eval_args'] = {
            'split': {'RS': [train_ratio, 0.1, 0.1]},
            'group_by': 'user',
            'order': 'TO',
            'mode': 'full'
        }
        # Store sparsity info for logging
        config_dict['sparsity_ratio'] = args.sparsity_ratio

    # Set dataset
    config_dict['data_path'] = 'dataset/'

    # Log experiment parameters
    log_experiment_params(logger, args, config_dict)

    logger.info(f"Starting experiment: {experiment_name}")
    logger.info(f"Model: {args.model}, Dataset: {args.dataset}")
    logger.info(f"Embedding size: {args.embedding_size}, Sparsity ratio: {args.sparsity_ratio}")

    # Set seed for reproducibility
    init_seed(args.seed, reproducibility=True)

    # Run RecBole
    result = run_recbole(
        model=args.model,
        dataset=args.dataset,
        config_dict=config_dict
    )

    # Extract metrics
    metrics = {
        'model': args.model,
        'dataset': args.dataset,
        'embedding_size': args.embedding_size,
        'sparsity_ratio': args.sparsity_ratio,
        'seed': args.seed,
        'recall@10': result['test_result'].get('recall@10', 0),
        'ndcg@10': result['test_result'].get('ndcg@10', 0),
        'best_valid_score': result.get('best_valid_score', 0),
        'best_valid_result': result.get('best_valid_result', {}),
        'test_result': result.get('test_result', {})
    }

    # Save results
    results_dir = os.path.join(args.output_dir, 'metrics')
    os.makedirs(results_dir, exist_ok=True)
    save_results(metrics, os.path.join(results_dir, f"{experiment_name}.json"))

    logger.info(f"Experiment completed: {experiment_name}")
    logger.info(f"Recall@10: {metrics['recall@10']:.4f}, NDCG@10: {metrics['ndcg@10']:.4f}")

    return metrics


def main():
    args = parse_args()

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)

    # Run experiment
    results = run_experiment(args)

    print("\n" + "="*50)
    print("EXPERIMENT RESULTS")
    print("="*50)
    print(f"Model: {results['model']}")
    print(f"Dataset: {results['dataset']}")
    print(f"Embedding Size: {results['embedding_size']}")
    print(f"Sparsity Ratio: {results['sparsity_ratio']}")
    print(f"Recall@10: {results['recall@10']:.4f}")
    print(f"NDCG@10: {results['ndcg@10']:.4f}")
    print("="*50)

    return results


if __name__ == '__main__':
    main()
