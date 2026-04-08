"""Metric computation for all task types."""

from __future__ import annotations

import logging

import numpy as np
from scipy.special import softmax as scipy_softmax
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def compute_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    task_type: str,
    class_names: list[str] | None = None,
) -> dict[str, float]:
    """Compute evaluation metrics based on task type.

    Args:
        logits: Raw model outputs, shape (N, C) or (N, 1).
        labels: Ground truth, shape (N,) for multiclass/ordinal,
                (N, C) for multilabel, (N,) or (N,1) for binary.
        task_type: One of 'multiclass', 'multilabel', 'binary', 'ordinal'.
        class_names: Names for per-class reporting.

    Returns:
        Dictionary of metric_name -> value.
    """
    if class_names is None:
        class_names = []

    metrics: dict[str, float] = {}

    try:
        if task_type == "multiclass":
            metrics = _multiclass_metrics(logits, labels, class_names)
        elif task_type == "multilabel":
            metrics = _multilabel_metrics(logits, labels, class_names)
        elif task_type == "binary":
            metrics = _binary_metrics(logits, labels)
        elif task_type == "ordinal":
            metrics = _ordinal_metrics(logits, labels, class_names)
        else:
            logger.warning(f"Unknown task type: {task_type}, computing basic metrics")
            preds = logits.argmax(axis=1)
            metrics["accuracy"] = accuracy_score(labels, preds)
    except Exception as e:
        logger.error(f"Error computing metrics: {e}")
        metrics["error"] = 1.0

    return metrics


def _multiclass_metrics(
    logits: np.ndarray, labels: np.ndarray, class_names: list[str]
) -> dict[str, float]:
    preds = logits.argmax(axis=1)
    probs = scipy_softmax(logits, axis=1)

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "balanced_accuracy": balanced_accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
    }

    # AUROC (one-vs-rest)
    try:
        n_classes = logits.shape[1]
        if n_classes > 2 and len(np.unique(labels)) > 1:
            metrics["macro_auroc"] = roc_auc_score(
                labels, probs, multi_class="ovr", average="macro"
            )
            # Per-class AUROC
            for i, name in enumerate(class_names):
                binary_labels = (labels == i).astype(int)
                if binary_labels.sum() > 0 and binary_labels.sum() < len(binary_labels):
                    metrics[f"auroc_{name}"] = roc_auc_score(binary_labels, probs[:, i])
    except Exception as e:
        logger.debug(f"AUROC computation failed: {e}")

    return metrics


def _multilabel_metrics(
    logits: np.ndarray, labels: np.ndarray, class_names: list[str]
) -> dict[str, float]:
    probs = _sigmoid(logits)
    preds = (probs > 0.5).astype(int)

    metrics = {}
    aurocs = []
    auprcs = []

    for i, name in enumerate(class_names):
        col_labels = labels[:, i]
        col_probs = probs[:, i]

        if col_labels.sum() > 0 and col_labels.sum() < len(col_labels):
            auc = roc_auc_score(col_labels, col_probs)
            metrics[f"auroc_{name}"] = auc
            aurocs.append(auc)

            ap = average_precision_score(col_labels, col_probs)
            metrics[f"auprc_{name}"] = ap
            auprcs.append(ap)

    if aurocs:
        metrics["macro_auroc"] = np.mean(aurocs)
    if auprcs:
        metrics["macro_auprc"] = np.mean(auprcs)

    return metrics


def _binary_metrics(logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    if logits.ndim == 2 and logits.shape[1] == 1:
        logits = logits[:, 0]
    if labels.ndim == 2 and labels.shape[1] == 1:
        labels = labels[:, 0]

    probs = _sigmoid(logits)
    preds = (probs > 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, zero_division=0),
    }

    try:
        if len(np.unique(labels)) > 1:
            metrics["auroc"] = roc_auc_score(labels, probs)
            metrics["auprc"] = average_precision_score(labels, probs)
    except Exception as e:
        logger.debug(f"Binary AUROC failed: {e}")

    return metrics


def _ordinal_metrics(
    logits: np.ndarray, labels: np.ndarray, class_names: list[str]
) -> dict[str, float]:
    preds = logits.argmax(axis=1)

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "balanced_accuracy": balanced_accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "qwk": cohen_kappa_score(labels, preds, weights="quadratic"),
    }

    # Also include multiclass AUROC
    try:
        probs = scipy_softmax(logits, axis=1)
        if len(np.unique(labels)) > 1:
            metrics["macro_auroc"] = roc_auc_score(
                labels, probs, multi_class="ovr", average="macro"
            )
    except Exception as e:
        logger.debug(f"Ordinal AUROC failed: {e}")

    return metrics
