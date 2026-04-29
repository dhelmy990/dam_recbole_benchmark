#!/usr/bin/env python3
"""
Master script to run all experiments.

This script orchestrates:
1. Baseline model comparison (full data)
2. Data sparsity analysis
3. Embedding size sensitivity study
4. Results aggregation and visualization
"""

import os
import sys
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiments.sparsity_analysis import SparsityAnalysis
from src.experiments.sensitivity_study import SensitivityStudy
from src.utils.visualizer import ResultsVisualizer
from src.utils.results_handler import ResultsCollector
from src.utils.logger import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description='Run all benchmark experiments')
    parser.add_argument('--models', nargs='+', default=['SASRec', 'LightGCN', 'SGL'],
                        help='Models to benchmark')
    parser.add_argument('--datasets', nargs='+', default=['ml-100k', 'amazon-beauty', 'steam'],
                        help='Datasets to use')
    parser.add_argument('--config_dir', type=str, default='configs',
                        help='Configuration directory')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--n_runs', type=int, default=3,
                        help='Number of runs per experiment')
    parser.add_argument('--skip_sparsity', action='store_true',
                        help='Skip sparsity analysis')
    parser.add_argument('--skip_sensitivity', action='store_true',
                        help='Skip sensitivity study')
    parser.add_argument('--visualize_only', action='store_true',
                        help='Only generate visualizations from existing results')
    return parser.parse_args()


def main():
    args = parse_args()

    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'metrics'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'figures'), exist_ok=True)

    logger = setup_logging(
        log_file=os.path.join(args.output_dir, 'logs', 'master_experiment.log')
    )

    start_time = datetime.now()
    logger.info("="*70)
    logger.info("RECBOLE BENCHMARKING EXPERIMENT SUITE")
    logger.info(f"Started at: {start_time}")
    logger.info("="*70)
    logger.info(f"Models: {args.models}")
    logger.info(f"Datasets: {args.datasets}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Runs per experiment: {args.n_runs}")

    if args.visualize_only:
        logger.info("Visualization-only mode: Skipping experiments")
    else:
        # Run sparsity analysis
        if not args.skip_sparsity:
            logger.info("\n" + "="*70)
            logger.info("PHASE 1: DATA SPARSITY ANALYSIS")
            logger.info("="*70)

            sparsity = SparsityAnalysis(
                models=args.models,
                datasets=args.datasets,
                sparsity_ratios=[1.0, 0.8, 0.6, 0.4],
                config_dir=args.config_dir,
                output_dir=args.output_dir,
                seed=args.seed,
                n_runs=args.n_runs
            )
            sparsity.run_all_experiments()
            sparsity.save_summary()

        # Run sensitivity study
        if not args.skip_sensitivity:
            logger.info("\n" + "="*70)
            logger.info("PHASE 2: EMBEDDING SIZE SENSITIVITY STUDY")
            logger.info("="*70)

            sensitivity = SensitivityStudy(
                models=args.models,
                datasets=args.datasets,
                embedding_sizes=[32, 64, 128],
                config_dir=args.config_dir,
                output_dir=args.output_dir,
                seed=args.seed,
                n_runs=args.n_runs
            )
            sensitivity.run_all_experiments()
            sensitivity.save_summary()

    # Generate visualizations
    logger.info("\n" + "="*70)
    logger.info("PHASE 3: GENERATING VISUALIZATIONS")
    logger.info("="*70)

    visualizer = ResultsVisualizer(
        results_dir=os.path.join(args.output_dir, 'metrics'),
        output_dir=os.path.join(args.output_dir, 'figures')
    )
    visualizer.generate_all_plots()

    # Generate summary report
    logger.info("\n" + "="*70)
    logger.info("PHASE 4: GENERATING SUMMARY REPORT")
    logger.info("="*70)

    collector = ResultsCollector(
        results_dir=os.path.join(args.output_dir, 'metrics')
    )
    collector.export_summary(os.path.join(args.output_dir, 'experiment_summary.json'))

    # Export LaTeX table
    try:
        collector.export_to_latex(os.path.join(args.output_dir, 'results_table.tex'))
    except Exception as e:
        logger.warning(f"Could not generate LaTeX table: {e}")

    end_time = datetime.now()
    duration = end_time - start_time

    logger.info("\n" + "="*70)
    logger.info("EXPERIMENT SUITE COMPLETE")
    logger.info(f"Finished at: {end_time}")
    logger.info(f"Total duration: {duration}")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info("="*70)

    print(f"\nAll experiments completed in {duration}")
    print(f"Results saved to: {args.output_dir}")
    print(f"  - Metrics: {args.output_dir}/metrics/")
    print(f"  - Figures: {args.output_dir}/figures/")
    print(f"  - Logs: {args.output_dir}/logs/")


if __name__ == '__main__':
    main()
