"""Tests for the analysis / interpretability framework."""

import torch
import pytest

from med_merge.merging.task_vector import TaskVector
from med_merge.analysis.conflict import (
    compute_sign_agreement,
    compute_layerwise_conflict,
    compute_pairwise_interference,
)
from med_merge.analysis.importance import (
    compute_magnitude_importance,
    compute_attention_head_importance,
)
from med_merge.analysis.diagnostics import (
    predict_merge_quality,
    compute_domain_gap,
)


def _make_tvs(n_tasks=3, dim=32):
    tvs = {}
    for i in range(n_tasks):
        vector = {
            f"encoder.layers.{j}.attn.q_proj.weight": torch.randn(dim, dim) * 0.01
            for j in range(4)
        }
        vector["encoder.layernorm.weight"] = torch.randn(dim) * 0.01
        tvs[f"task_{i}"] = TaskVector(vector=vector)
    return tvs


class TestConflict:
    def test_sign_agreement(self):
        tvs = _make_tvs()
        result = compute_sign_agreement(tvs)
        assert len(result) > 0
        for key, info in result.items():
            assert 0 <= info["agreement"] <= 1

    def test_layerwise_conflict(self):
        tvs = _make_tvs()
        result = compute_layerwise_conflict(tvs, layer_pattern=r"layers\.(\d+)\.")
        assert len(result) == 4  # 4 layers
        for depth, score in result.items():
            assert 0 <= score <= 1

    def test_pairwise_interference(self):
        tvs = _make_tvs()
        result = compute_pairwise_interference(tvs)
        assert len(result) == 3  # C(3,2) = 3 pairs
        for pair, metrics in result.items():
            assert "cosine" in metrics
            assert "sign_disagreement" in metrics
            assert -1 <= metrics["cosine"] <= 1
            assert 0 <= metrics["sign_disagreement"] <= 1


class TestImportance:
    def test_magnitude_importance(self):
        tvs = _make_tvs()
        result = compute_magnitude_importance(tvs)
        assert len(result) == 3
        for task, keys in result.items():
            for key, score in keys.items():
                assert score >= 0

    def test_attention_head_importance(self):
        tvs = _make_tvs(dim=64)
        result = compute_attention_head_importance(
            tvs, n_heads=4, hidden_dim=64,
            layer_pattern=r"layers\.(\d+)\.",
            attn_keys=("q_proj.weight",),
        )
        assert len(result) == 3
        for task, matrix in result.items():
            assert matrix.shape == (4, 4)  # 4 layers, 4 heads


class TestDiagnostics:
    def test_predict_merge_quality(self):
        tvs = _make_tvs()
        result = predict_merge_quality(tvs)
        assert "risk_score" in result
        assert 0 <= result["risk_score"] <= 1
        assert "task_vector_norms" in result
        assert len(result["task_vector_norms"]) == 3

    def test_compute_domain_gap(self):
        tvs = _make_tvs()
        result = compute_domain_gap(tvs)
        assert len(result) == 3  # 3 pairs
        for pair, metrics in result.items():
            assert "cosine_similarity" in metrics
            assert "l2_distance_normalized" in metrics
