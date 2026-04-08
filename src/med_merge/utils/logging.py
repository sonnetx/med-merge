"""Experiment logging utilities."""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger("med_merge")


def setup_logging(level: str = "INFO") -> None:
    """Configure logging for med-merge."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def setup_wandb(
    project: str = "med-merge",
    entity: Optional[str] = None,
    mode: str = "online",
    config: Optional[dict[str, Any]] = None,
    name: Optional[str] = None,
) -> Any:
    """Initialize wandb run. Returns the run object or None if disabled."""
    try:
        import wandb

        run = wandb.init(
            project=project,
            entity=entity,
            mode=mode,
            config=config,
            name=name,
        )
        return run
    except ImportError:
        logger.warning("wandb not installed, skipping experiment tracking")
        return None


class ExperimentLogger:
    """Unified logging interface supporting wandb and stdout."""

    def __init__(self, wandb_run: Any = None):
        self.wandb_run = wandb_run

    def log_metrics(self, metrics: dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to wandb and stdout."""
        if self.wandb_run is not None:
            self.wandb_run.log(metrics, step=step)
        step_str = f" (step {step})" if step is not None else ""
        logger.info(f"Metrics{step_str}: {metrics}")

    def log_config(self, config: dict[str, Any]) -> None:
        """Log configuration."""
        if self.wandb_run is not None:
            self.wandb_run.config.update(config)
        logger.info(f"Config: {config}")

    def finish(self) -> None:
        """Finalize logging."""
        if self.wandb_run is not None:
            self.wandb_run.finish()
