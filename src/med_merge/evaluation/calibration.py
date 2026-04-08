"""Calibration metrics and reliability diagrams."""

from __future__ import annotations

import numpy as np
from scipy.special import softmax as scipy_softmax


def expected_calibration_error(
    logits: np.ndarray,
    labels: np.ndarray,
    task_type: str,
    n_bins: int = 15,
) -> float:
    """Compute Expected Calibration Error (ECE).

    For multiclass/ordinal: uses top-class confidence.
    For binary: uses predicted probability.
    For multilabel: mean ECE across labels.
    """
    if task_type == "multilabel":
        sigmoid = lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        probs = sigmoid(logits)
        eces = []
        for i in range(logits.shape[1]):
            ece = _binary_ece(probs[:, i], labels[:, i], n_bins)
            eces.append(ece)
        return float(np.mean(eces))

    elif task_type == "binary":
        if logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits[:, 0]
        if labels.ndim == 2 and labels.shape[1] == 1:
            labels = labels[:, 0]
        sigmoid = lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        probs = sigmoid(logits)
        return _binary_ece(probs, labels, n_bins)

    else:
        # Multiclass / ordinal: top-class confidence
        probs = scipy_softmax(logits, axis=1)
        confidences = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
        accuracies = (predictions == labels).astype(float)
        return _ece_from_confidence(confidences, accuracies, n_bins)


def _binary_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int) -> float:
    """ECE for binary predictions."""
    confidences = np.maximum(probs, 1 - probs)
    predictions = (probs >= 0.5).astype(float)
    accuracies = (predictions == labels).astype(float)
    return _ece_from_confidence(confidences, accuracies, n_bins)


def _ece_from_confidence(
    confidences: np.ndarray, accuracies: np.ndarray, n_bins: int
) -> float:
    """Core ECE computation from confidence and accuracy arrays."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(confidences)

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop = in_bin.sum() / max(total, 1)

        if in_bin.sum() > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].mean()
            ece += prop * abs(avg_accuracy - avg_confidence)

    return float(ece)


def reliability_diagram_data(
    logits: np.ndarray,
    labels: np.ndarray,
    task_type: str,
    n_bins: int = 15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (bin_centers, bin_accuracies, bin_counts) for reliability diagrams."""
    if task_type in ("multiclass", "ordinal"):
        probs = scipy_softmax(logits, axis=1)
        confidences = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
        accuracies = (predictions == labels).astype(float)
    elif task_type == "binary":
        if logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits[:, 0]
        if labels.ndim == 2 and labels.shape[1] == 1:
            labels = labels[:, 0]
        sigmoid = lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        probs_pos = sigmoid(logits)
        confidences = np.maximum(probs_pos, 1 - probs_pos)
        predictions = (probs_pos >= 0.5).astype(float)
        accuracies = (predictions == labels).astype(float)
    else:
        return np.array([]), np.array([]), np.array([])

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_accs = []
    bin_counts = []

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        count = in_bin.sum()
        bin_counts.append(count)

        if count > 0:
            bin_centers.append((bin_boundaries[i] + bin_boundaries[i + 1]) / 2)
            bin_accs.append(accuracies[in_bin].mean())
        else:
            bin_centers.append((bin_boundaries[i] + bin_boundaries[i + 1]) / 2)
            bin_accs.append(0.0)

    return np.array(bin_centers), np.array(bin_accs), np.array(bin_counts)
