"""Plotting utilities for merge analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def plot_layerwise_conflict(
    conflict_data: dict[int, float],
    output_path: Path,
    title: str = "Layer-wise Sign Agreement",
) -> None:
    """Bar chart of sign agreement per transformer layer."""
    import matplotlib.pyplot as plt

    layers = sorted(conflict_data.keys())
    values = [conflict_data[l] for l in layers]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(layers, values, color="steelblue")
    ax.set_xlabel("Layer Depth")
    ax.set_ylabel("Sign Agreement Fraction")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_attention_head_heatmap(
    importance: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    """Heatmap of per-head importance for each task."""
    import matplotlib.pyplot as plt

    n_tasks = len(importance)
    fig, axes = plt.subplots(1, n_tasks, figsize=(6 * n_tasks, 5))
    if n_tasks == 1:
        axes = [axes]

    for ax, (task_name, matrix) in zip(axes, importance.items()):
        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
        ax.set_xlabel("Head Index")
        ax.set_ylabel("Layer")
        ax.set_title(f"{task_name}")
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("Attention Head Importance (L2 norm of task vector)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_pairwise_interference_matrix(
    interference: dict[tuple[str, str], dict[str, float]],
    task_names: list[str],
    output_path: Path,
    metric: str = "cosine",
) -> None:
    """Symmetric heatmap of pairwise interference between tasks."""
    import matplotlib.pyplot as plt

    n = len(task_names)
    matrix = np.eye(n)
    name_to_idx = {name: i for i, name in enumerate(task_names)}

    for (a, b), metrics in interference.items():
        i, j = name_to_idx[a], name_to_idx[b]
        val = metrics.get(metric, 0.0)
        matrix[i, j] = val
        matrix[j, i] = val

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(task_names, rotation=45, ha="right")
    ax.set_yticklabels(task_names)
    ax.set_title(f"Pairwise {metric.replace('_', ' ').title()}")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_task_vector_norms(
    task_vectors: dict,
    output_path: Path,
) -> None:
    """Bar chart of task vector L2 norms."""
    import matplotlib.pyplot as plt
    import torch

    norms = {}
    for name, tv in task_vectors.items():
        flat = torch.cat([t.flatten() for t in tv.vector.values()])
        norms[name] = flat.norm().item()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(norms.keys(), norms.values(), color="coral")
    ax.set_ylabel("L2 Norm")
    ax.set_title("Task Vector Norms")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_sensitivity_landscape(
    sweep_2d: np.ndarray,
    param1_range: np.ndarray,
    param2_range: np.ndarray,
    output_path: Path,
    param1_name: str = "alpha",
    param2_name: str = "density",
    title: str = "Merge Quality Landscape",
) -> None:
    """Contour/heatmap plot of 2D hyperparameter sweep."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        sweep_2d, aspect="auto", origin="lower",
        extent=[param2_range[0], param2_range[-1], param1_range[0], param1_range[-1]],
        cmap="viridis",
    )
    ax.set_xlabel(param2_name)
    ax.set_ylabel(param1_name)
    ax.set_title(title)
    fig.colorbar(im, label="Aggregate Score")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
