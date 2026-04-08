"""Checkpoint and file I/O utilities."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch


def get_output_dir(base_dir: str, experiment_name: str) -> Path:
    """Create structured output path: base_dir/experiment_name/timestamp/."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(base_dir) / experiment_name / timestamp
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    metrics: dict[str, float],
    path: Path,
) -> None:
    """Save a training checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "metrics": metrics,
    }
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(state, path)


def load_checkpoint(
    path: Path,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> dict[str, Any]:
    """Load a training checkpoint."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    if model is not None:
        model.load_state_dict(state["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    return state


def save_state_dict(state_dict: dict[str, torch.Tensor], path: Path) -> None:
    """Save a state dict (encoder, head, or task vector)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, path)


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    """Load a state dict."""
    return torch.load(path, map_location="cpu", weights_only=True)


def save_json(data: Any, path: Path) -> None:
    """Save data as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: Path) -> Any:
    """Load data from JSON."""
    with open(path) as f:
        return json.load(f)
