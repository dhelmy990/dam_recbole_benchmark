"""Configuration loading and merging utilities."""

import os
import yaml
from typing import Dict, Any, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing the configuration.

    Raises:
        FileNotFoundError: If the configuration file doesn't exist.
        yaml.YAMLError: If the YAML is malformed.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config if config else {}


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries.
    Later configs override earlier ones.

    Args:
        *configs: Variable number of configuration dictionaries.

    Returns:
        Merged configuration dictionary.
    """
    result = {}
    for config in configs:
        if config:
            result = _deep_merge(result, config)
    return result


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.

    Args:
        base: Base dictionary.
        override: Dictionary with values to override.

    Returns:
        Merged dictionary.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Save a configuration dictionary to a YAML file.

    Args:
        config: Configuration dictionary to save.
        output_path: Path to save the YAML file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def get_config_for_experiment(
    config_dir: str,
    model: str,
    dataset: str,
    overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Load and merge all configs for an experiment.

    Args:
        config_dir: Base configuration directory.
        model: Model name (SASRec, LightGCN, SGL).
        dataset: Dataset name (ml-100k, amazon-beauty).
        overrides: Optional dictionary of config overrides.

    Returns:
        Complete configuration for the experiment.
    """
    base_config = load_config(os.path.join(config_dir, 'base.yaml'))
    model_config = load_config(os.path.join(config_dir, 'models', f'{model.lower()}.yaml'))
    dataset_config = load_config(os.path.join(config_dir, 'datasets', f'{dataset}.yaml'))

    config = merge_configs(base_config, model_config, dataset_config)

    if overrides:
        config = merge_configs(config, overrides)

    return config
