"""Tests for evaluation metrics."""

import numpy as np
import pytest

from med_merge.evaluation.metrics import compute_metrics
from med_merge.evaluation.calibration import expected_calibration_error


class TestMulticlassMetrics:
    def test_perfect_predictions(self):
        logits = np.array([[10, 0, 0], [0, 10, 0], [0, 0, 10]])
        labels = np.array([0, 1, 2])
        metrics = compute_metrics(logits, labels, "multiclass", ["a", "b", "c"])
        assert metrics["accuracy"] == 1.0
        assert metrics["balanced_accuracy"] == 1.0

    def test_random_predictions(self):
        rng = np.random.RandomState(42)
        logits = rng.randn(100, 5)
        labels = rng.randint(0, 5, 100)
        metrics = compute_metrics(logits, labels, "multiclass")
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["balanced_accuracy"] <= 1


class TestBinaryMetrics:
    def test_perfect_binary(self):
        logits = np.array([10, -10, 10, -10])
        labels = np.array([1, 0, 1, 0])
        metrics = compute_metrics(logits.reshape(-1, 1), labels, "binary")
        assert metrics["accuracy"] == 1.0
        assert metrics["auroc"] == 1.0


class TestMultilabelMetrics:
    def test_perfect_multilabel(self):
        logits = np.array([[10, -10], [-10, 10], [10, 10]])
        labels = np.array([[1, 0], [0, 1], [1, 1]])
        metrics = compute_metrics(logits, labels, "multilabel", ["a", "b"])
        assert metrics["macro_auroc"] == 1.0


class TestOrdinalMetrics:
    def test_perfect_ordinal(self):
        logits = np.eye(5) * 10
        labels = np.array([0, 1, 2, 3, 4])
        metrics = compute_metrics(logits, labels, "ordinal")
        assert metrics["qwk"] == 1.0
        assert metrics["accuracy"] == 1.0


class TestCalibration:
    def test_ece_perfect(self):
        # Perfectly calibrated: confident and correct
        logits = np.eye(3) * 100
        labels = np.array([0, 1, 2])
        ece = expected_calibration_error(logits, labels, "multiclass")
        assert ece < 0.1  # should be near 0

    def test_ece_range(self):
        rng = np.random.RandomState(42)
        logits = rng.randn(200, 4)
        labels = rng.randint(0, 4, 200)
        ece = expected_calibration_error(logits, labels, "multiclass")
        assert 0 <= ece <= 1
