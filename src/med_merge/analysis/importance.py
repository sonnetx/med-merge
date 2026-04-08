"""Parameter importance scoring and attention head analysis."""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
import torch

from med_merge.merging.task_vector import TaskVector


def compute_magnitude_importance(
    task_vectors: dict[str, TaskVector],
) -> dict[str, dict[str, float]]:
    """Per-task, per-key importance based on absolute magnitude.

    Returns ``{task_name: {key: mean_abs_magnitude}}``.
    """
    result = {}
    for name, tv in task_vectors.items():
        key_importance = {}
        for key, tensor in tv.vector.items():
            key_importance[key] = tensor.abs().mean().item()
        result[name] = key_importance
    return result


def compute_attention_head_importance(
    task_vectors: dict[str, TaskVector],
    n_heads: int = 12,
    hidden_dim: int = 768,
    layer_pattern: str = r"layers\.(\d+)\.",
    attn_keys: tuple[str, ...] = ("q_proj.weight", "k_proj.weight", "v_proj.weight", "out_proj.weight"),
) -> dict[str, np.ndarray]:
    """Score importance of each attention head per task.

    For ViT-B/16 with 12 heads of dim 64 each, reshapes attention weight
    task vectors into (n_heads, head_dim, ...) and computes per-head L2 norm.

    Returns ``{task_name: ndarray of shape (n_layers, n_heads)}``.
    """
    head_dim = hidden_dim // n_heads
    result = {}

    for task_name, tv in task_vectors.items():
        # Discover number of layers
        max_layer = -1
        for key in tv.vector:
            m = re.search(layer_pattern, key)
            if m:
                max_layer = max(max_layer, int(m.group(1)))

        if max_layer < 0:
            continue

        n_layers = max_layer + 1
        importance = np.zeros((n_layers, n_heads))

        for key, tensor in tv.vector.items():
            m = re.search(layer_pattern, key)
            if m is None:
                continue
            layer_idx = int(m.group(1))

            if not any(ak in key for ak in attn_keys):
                continue

            # Attention weight shape: (hidden_dim, hidden_dim) or (hidden_dim,)
            flat = tensor.float().flatten()
            if flat.numel() < n_heads * head_dim:
                continue

            # Reshape to (n_heads, -1) and compute per-head L2 norm
            try:
                reshaped = flat[:n_heads * (flat.numel() // n_heads)].reshape(n_heads, -1)
                head_norms = reshaped.norm(dim=1).numpy()
                importance[layer_idx] += head_norms
            except RuntimeError:
                continue

        result[task_name] = importance

    return result
