"""Fairness evaluation: subgroup metrics by Fitzpatrick skin type."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from med_merge.evaluation.metrics import compute_metrics

logger = logging.getLogger(__name__)

SKIN_TYPE_GROUPS = {
    "light": [1, 2],
    "medium": [3, 4],
    "dark": [5, 6],
}


def compute_fairness_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    skin_types: list[int],
    task_type: str,
    class_names: list[str] | None = None,
) -> dict[str, float]:
    """Compute metrics stratified by Fitzpatrick skin type group.

    Args:
        logits: Model predictions, shape (N, C).
        labels: Ground truth labels.
        skin_types: Fitzpatrick skin type (1-6) per sample.
        task_type: Task type string.
        class_names: Class names for per-class reporting.

    Returns:
        Dict with per-group and disparity metrics.
    """
    skin_types_arr = np.array(skin_types)
    metrics = {}

    group_scores = {}
    for group_name, types in SKIN_TYPE_GROUPS.items():
        mask = np.isin(skin_types_arr, types)
        if mask.sum() < 10:
            logger.warning(f"Skin type group '{group_name}' has <10 samples, skipping")
            continue

        group_logits = logits[mask]
        group_labels = labels[mask]

        group_metrics = compute_metrics(
            group_logits, group_labels, task_type, class_names
        )

        for k, v in group_metrics.items():
            metrics[f"{group_name}_{k}"] = v

        # Track primary metric for disparity
        primary = {
            "multiclass": "balanced_accuracy",
            "multilabel": "macro_auroc",
            "binary": "auroc",
            "ordinal": "qwk",
        }.get(task_type, "accuracy")
        if primary in group_metrics:
            group_scores[group_name] = group_metrics[primary]

    # Disparity metrics
    if len(group_scores) >= 2:
        scores = list(group_scores.values())
        metrics["fairness_max_min_gap"] = max(scores) - min(scores)
        metrics["fairness_worst_best_ratio"] = min(scores) / max(max(scores), 1e-8)
        metrics["fairness_worst_group"] = min(group_scores, key=group_scores.get)
        metrics["fairness_best_group"] = max(group_scores, key=group_scores.get)

    metrics["n_samples_by_group"] = {
        group: int(np.isin(skin_types_arr, types).sum())
        for group, types in SKIN_TYPE_GROUPS.items()
    }

    return metrics
