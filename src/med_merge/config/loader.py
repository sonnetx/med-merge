"""YAML configuration loading and Pydantic validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from omegaconf import OmegaConf

from med_merge.config.schema import ExperimentConfig


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return as dict."""
    cfg = OmegaConf.load(path)
    return OmegaConf.to_container(cfg, resolve=True)


def merge_configs(*configs: dict[str, Any]) -> dict[str, Any]:
    """Merge multiple config dicts (later overrides earlier)."""
    oc_configs = [OmegaConf.create(c) for c in configs]
    merged = OmegaConf.merge(*oc_configs)
    return OmegaConf.to_container(merged, resolve=True)


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply CLI overrides in 'key=value' or 'key.subkey=value' format."""
    oc = OmegaConf.create(config)
    for override in overrides:
        key, _, value = override.partition("=")
        OmegaConf.update(oc, key, value)
    return OmegaConf.to_container(oc, resolve=True)


def load_config(
    config_path: Optional[str | Path] = None,
    overrides: Optional[list[str]] = None,
    defaults_path: Optional[str | Path] = None,
) -> ExperimentConfig:
    """Load experiment configuration from YAML with optional overrides.

    The raw dict is validated through Pydantic's ``ExperimentConfig``.
    Unknown keys raise ``ValidationError`` instead of being silently dropped.

    Args:
        config_path: Path to experiment YAML config.
        overrides: CLI overrides in 'key=value' format.
        defaults_path: Path to defaults YAML config.

    Returns:
        Validated ExperimentConfig.
    """
    config: dict[str, Any] = {}

    if defaults_path is not None:
        config = load_yaml(defaults_path)

    if config_path is not None:
        experiment_config = load_yaml(config_path)
        config = merge_configs(config, experiment_config)

    if overrides:
        config = apply_overrides(config, overrides)

    return ExperimentConfig.model_validate(config)
