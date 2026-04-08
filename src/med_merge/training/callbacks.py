"""Training callbacks for checkpointing and early stopping."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping based on validation metric."""

    def __init__(self, patience: int = 5, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.best_score: float | None = None
        self.counter = 0

    def __call__(self, score: float) -> bool:
        """Returns True if training should stop."""
        if self.best_score is None:
            self.best_score = score
            return False

        improved = (
            score > self.best_score if self.mode == "max"
            else score < self.best_score
        )

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            logger.info(
                f"Early stopping triggered after {self.counter} epochs without improvement"
            )
            return True
        return False


class CheckpointManager:
    """Manages model checkpoint saving."""

    def __init__(self, save_dir: Path, save_best_only: bool = True, mode: str = "max"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_best_only = save_best_only
        self.mode = mode
        self.best_score: float | None = None

    def save_if_best(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        score: float,
        metrics: dict[str, float],
    ) -> bool:
        """Save checkpoint if score is best. Returns True if saved."""
        is_best = False

        if self.best_score is None:
            is_best = True
        elif self.mode == "max" and score > self.best_score:
            is_best = True
        elif self.mode == "min" and score < self.best_score:
            is_best = True

        if is_best:
            self.best_score = score
            state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics,
                "best_score": score,
            }
            torch.save(state, self.save_dir / "best_model.pt")
            logger.info(f"Saved best model (epoch {epoch}, score={score:.4f})")

        if not self.save_best_only:
            state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "metrics": metrics,
            }
            torch.save(state, self.save_dir / f"checkpoint_epoch{epoch}.pt")

        return is_best
