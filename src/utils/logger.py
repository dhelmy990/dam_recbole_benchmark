"""Logging utilities for experiment tracking."""

import logging
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional


def setup_logging(
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging configuration.

    Args:
        log_file: Path to log file. If None, logs only to console.
        level: Logging level (default: INFO).
        format_string: Custom format string for log messages.

    Returns:
        Configured logger instance.
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    logger = logging.getLogger('recbole_benchmark')
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(format_string))
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(format_string))
        logger.addHandler(file_handler)

    return logger


def log_experiment_params(
    logger: logging.Logger,
    args: Any,
    config: Dict[str, Any]
) -> None:
    """
    Log experiment parameters for reproducibility.

    Args:
        logger: Logger instance.
        args: Command-line arguments.
        config: Configuration dictionary.
    """
    logger.info("="*60)
    logger.info("EXPERIMENT PARAMETERS")
    logger.info("="*60)

    # Log command-line arguments
    logger.info("Command-line Arguments:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")

    # Log key configuration parameters
    logger.info("\nKey Configuration:")
    key_params = [
        'embedding_size', 'hidden_size', 'num_layers', 'n_heads',
        'learning_rate', 'train_batch_size', 'epochs', 'seed',
        'topk', 'metrics', 'valid_metric', 'sparsity_ratio'
    ]

    for param in key_params:
        if param in config:
            logger.info(f"  {param}: {config[param]}")

    logger.info("="*60)


def log_results(
    logger: logging.Logger,
    results: Dict[str, Any],
    experiment_name: str
) -> None:
    """
    Log experiment results.

    Args:
        logger: Logger instance.
        results: Results dictionary.
        experiment_name: Name of the experiment.
    """
    logger.info("="*60)
    logger.info(f"RESULTS: {experiment_name}")
    logger.info("="*60)

    for key, value in results.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        elif isinstance(value, dict):
            logger.info(f"  {key}:")
            for k, v in value.items():
                if isinstance(v, float):
                    logger.info(f"    {k}: {v:.4f}")
                else:
                    logger.info(f"    {k}: {v}")
        else:
            logger.info(f"  {key}: {value}")

    logger.info("="*60)


class ExperimentLogger:
    """Context manager for experiment logging."""

    def __init__(self, experiment_name: str, log_dir: str = 'results/logs'):
        self.experiment_name = experiment_name
        self.log_dir = log_dir
        self.start_time = None
        self.logger = None

    def __enter__(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(self.log_dir, f"{self.experiment_name}_{timestamp}.log")
        self.logger = setup_logging(log_file=log_file)
        self.start_time = datetime.now()
        self.logger.info(f"Starting experiment: {self.experiment_name}")
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = datetime.now()
        duration = end_time - self.start_time
        self.logger.info(f"Experiment completed in {duration}")
        if exc_type:
            self.logger.error(f"Experiment failed: {exc_val}")
        return False
