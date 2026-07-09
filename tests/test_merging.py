"""Tests for merging methods using synthetic task vectors."""

import torch
import pytest

from med_merge.config.schema import MergingConfig
from med_merge.merging.task_vector import TaskVector
from med_merge.merging.simple_avg import SimpleAverageMerger
from med_merge.merging.task_arithmetic import TaskArithmeticMerger
from med_merge.merging.ties import TIESMerger
from med_merge.merging.dare import DAREMerger
from med_merge.merging.pcb_merging import PCBMerger
from med_merge.merging.lines import LiNeSMerger
from med_merge.merging.slerp import SLERPMerger
from med_merge.merging.fisher import FisherMerger
from med_merge.merging.iso_c import IsoCMerger
from med_merge.merging.tsv import TSVMerger
from med_merge.merging.gram_ls import GramLSMerger


def _make_synthetic_data(n_tasks: int = 3, dim: int = 64):
    """Create synthetic pretrained weights and task vectors."""
    pretrained = {
        "encoder.vision_model.encoder.layers.0.weight": torch.randn(dim, dim),
        "encoder.vision_model.encoder.layers.0.bias": torch.randn(dim),
        "encoder.vision_model.encoder.layers.5.weight": torch.randn(dim, dim),
        "encoder.vision_model.encoder.layers.11.weight": torch.randn(dim, dim),
        "encoder.vision_model.post_layernorm.weight": torch.randn(dim),
    }

    task_vectors = {}
    for i in range(n_tasks):
        vector = {k: torch.randn_like(v) * 0.1 for k, v in pretrained.items()}
        task_vectors[f"task_{i}"] = TaskVector(vector=vector)

    return pretrained, task_vectors


class TestTaskVector:
    def test_creation_from_diffs(self):
        pre = {"a": torch.ones(4), "b": torch.ones(4) * 2}
        fine = {"a": torch.ones(4) * 3, "b": torch.ones(4) * 5}
        tv = TaskVector(pretrained_state_dict=pre, finetuned_state_dict=fine)
        assert torch.allclose(tv.vector["a"], torch.ones(4) * 2)
        assert torch.allclose(tv.vector["b"], torch.ones(4) * 3)

    def test_addition(self):
        tv1 = TaskVector(vector={"a": torch.ones(4)})
        tv2 = TaskVector(vector={"a": torch.ones(4) * 2})
        result = tv1 + tv2
        assert torch.allclose(result.vector["a"], torch.ones(4) * 3)

    def test_scaling(self):
        tv = TaskVector(vector={"a": torch.ones(4) * 2})
        result = tv * 0.5
        assert torch.allclose(result.vector["a"], torch.ones(4))

    def test_apply_to(self):
        pre = {"a": torch.zeros(4)}
        tv = TaskVector(vector={"a": torch.ones(4)})
        result = tv.apply_to(pre, scaling_coef=0.5)
        assert torch.allclose(result["a"], torch.ones(4) * 0.5)

    def test_save_load(self, tmp_path):
        tv = TaskVector(vector={"a": torch.randn(4)})
        tv.save(tmp_path / "tv.pt")
        loaded = TaskVector.load(tmp_path / "tv.pt")
        assert torch.allclose(tv.vector["a"], loaded.vector["a"])


class TestMergers:
    def test_simple_avg(self):
        pretrained, task_vectors = _make_synthetic_data()
        config = MergingConfig(method="simple_avg")
        merger = SimpleAverageMerger(pretrained, config)
        result = merger.merge(task_vectors)
        assert set(result.keys()) == set(pretrained.keys())
        for k in pretrained:
            assert result[k].shape == pretrained[k].shape

    def test_task_arithmetic(self):
        pretrained, task_vectors = _make_synthetic_data()
        config = MergingConfig(method="task_arithmetic", alpha=0.5)
        merger = TaskArithmeticMerger(pretrained, config)
        result = merger.merge(task_vectors, alpha=0.5)
        assert set(result.keys()) == set(pretrained.keys())

    def test_ties(self):
        pretrained, task_vectors = _make_synthetic_data()
        config = MergingConfig(method="ties", alpha=0.3, trim_fraction=0.2)
        merger = TIESMerger(pretrained, config)
        result = merger.merge(task_vectors)
        assert set(result.keys()) == set(pretrained.keys())

    def test_dare(self):
        pretrained, task_vectors = _make_synthetic_data()
        config = MergingConfig(method="dare", alpha=0.3, drop_rate=0.5)
        merger = DAREMerger(pretrained, config)
        result = merger.merge(task_vectors)
        assert set(result.keys()) == set(pretrained.keys())

    def test_pcb_merging(self):
        pretrained, task_vectors = _make_synthetic_data()
        config = MergingConfig(method="pcb_merging")
        merger = PCBMerger(pretrained, config)
        result = merger.merge(task_vectors)
        assert set(result.keys()) == set(pretrained.keys())

    def test_lines(self):
        pretrained, task_vectors = _make_synthetic_data()
        config = MergingConfig(method="lines", alpha=0.3, lines_alpha=0.1, lines_beta=0.9)
        merger = LiNeSMerger(pretrained, config)
        result = merger.merge(task_vectors)
        assert set(result.keys()) == set(pretrained.keys())

    def test_slerp(self):
        pretrained, task_vectors = _make_synthetic_data(n_tasks=2)
        config = MergingConfig(method="slerp", slerp_t=0.5)
        merger = SLERPMerger(pretrained, config)
        result = merger.merge(task_vectors)
        assert set(result.keys()) == set(pretrained.keys())

    def test_slerp_requires_two_tasks(self):
        pretrained, task_vectors = _make_synthetic_data(n_tasks=3)
        config = MergingConfig(method="slerp", slerp_t=0.5)
        merger = SLERPMerger(pretrained, config)
        with pytest.raises(ValueError, match="exactly 2"):
            merger.merge(task_vectors)

    def test_fisher_fallback_uniform(self):
        """Fisher without validation data falls back to uniform importance."""
        pretrained, task_vectors = _make_synthetic_data()
        config = MergingConfig(method="fisher", alpha=0.3)
        merger = FisherMerger(pretrained, config)
        result = merger.merge(task_vectors)
        assert set(result.keys()) == set(pretrained.keys())

    def test_iso_c(self):
        pretrained, task_vectors = _make_synthetic_data()
        config = MergingConfig(method="iso_c", alpha=1.0)
        merger = IsoCMerger(pretrained, config)
        result = merger.merge(task_vectors)
        assert set(result.keys()) == set(pretrained.keys())
        for k in pretrained:
            assert result[k].shape == pretrained[k].shape

    def test_tsv_merge(self):
        pretrained, task_vectors = _make_synthetic_data()
        config = MergingConfig(method="tsv_merge", alpha=1.0)
        merger = TSVMerger(pretrained, config)
        result = merger.merge(task_vectors)
        assert set(result.keys()) == set(pretrained.keys())
        for k in pretrained:
            assert result[k].shape == pretrained[k].shape

    def test_gram_ls(self):
        pretrained, task_vectors = _make_synthetic_data()
        config = MergingConfig(method="gram_ls", alpha=1.0)
        merger = GramLSMerger(pretrained, config)
        result = merger.merge(task_vectors)
        assert set(result.keys()) == set(pretrained.keys())
        for k in pretrained:
            assert result[k].shape == pretrained[k].shape

    def test_gram_ls_orthogonal_recovers_sum(self):
        """For orthogonal task vectors the Gram system is diagonal, so the
        coefficients are ~1 and the merge reduces to a plain sum (each task's
        self-projection is preserved)."""
        dim = 128
        pretrained = {"encoder.layers.0.weight": torch.zeros(dim)}
        # Orthogonal one-hot directions with different norms.
        v0 = torch.zeros(dim); v0[0] = 3.0          # large-norm task
        v1 = torch.zeros(dim); v1[1] = 3.0          # large-norm task
        v2 = torch.zeros(dim); v2[2] = 0.3          # small-norm outlier task
        tvs = {
            "a": TaskVector(vector={"encoder.layers.0.weight": v0}),
            "b": TaskVector(vector={"encoder.layers.0.weight": v1}),
            "c": TaskVector(vector={"encoder.layers.0.weight": v2}),
        }
        config = MergingConfig(method="gram_ls", alpha=1.0, gram_lambda=0.0)
        merged = GramLSMerger(pretrained, config).merge(tvs)["encoder.layers.0.weight"]
        # Sum of the three orthogonal directions.
        expected = v0 + v1 + v2
        assert torch.allclose(merged, expected, atol=1e-4)

    def test_gram_ls_preserves_orthogonal_outlier_better_than_avg(self):
        """The core claim: a small-norm task vector near-orthogonal to the others
        keeps its signal under Gram-LS but is diluted by simple averaging."""
        dim = 256
        pretrained = {"encoder.layers.0.weight": torch.zeros(dim)}
        big_a = torch.zeros(dim); big_a[0] = 5.0
        big_b = torch.zeros(dim); big_b[1] = 5.0
        outlier = torch.zeros(dim); outlier[2] = 0.4
        tvs = {
            "a": TaskVector(vector={"encoder.layers.0.weight": big_a}),
            "b": TaskVector(vector={"encoder.layers.0.weight": big_b}),
            "outlier": TaskVector(vector={"encoder.layers.0.weight": outlier}),
        }

        avg = SimpleAverageMerger(pretrained, MergingConfig(method="simple_avg"))
        gram = GramLSMerger(pretrained, MergingConfig(method="gram_ls", alpha=1.0, gram_lambda=0.0))
        m_avg = avg.merge(tvs)["encoder.layers.0.weight"]
        m_gram = gram.merge(tvs)["encoder.layers.0.weight"]

        # Signal-retention ratio for the outlier task: <m, tau> / <tau, tau>.
        def retention(m, tau):
            return float(torch.dot(m, tau) / torch.dot(tau, tau))

        r_avg = retention(m_avg, outlier)
        r_gram = retention(m_gram, outlier)
        # Averaging retains ~1/K of the orthogonal outlier; Gram-LS retains ~1.
        assert r_avg < 0.5
        assert r_gram > 0.95
        assert r_gram > r_avg

    def test_dare_ties(self):
        pretrained, task_vectors = _make_synthetic_data()
        config = MergingConfig(
            method="dare", inner_method="ties",
            alpha=0.3, drop_rate=0.5, trim_fraction=0.2,
        )
        merger = DAREMerger(pretrained, config)
        result = merger.merge(task_vectors)
        assert set(result.keys()) == set(pretrained.keys())

    def test_all_methods_different_from_pretrained(self):
        """Merged results should differ from pretrained (merging did something)."""
        pretrained, task_vectors = _make_synthetic_data()

        methods = [
            ("simple_avg", SimpleAverageMerger, {}),
            ("task_arithmetic", TaskArithmeticMerger, {"alpha": 0.5}),
            ("ties", TIESMerger, {"alpha": 0.3, "trim_fraction": 0.2}),
            ("dare", DAREMerger, {"alpha": 0.3, "drop_rate": 0.5}),
            ("pcb_merging", PCBMerger, {}),
            ("lines", LiNeSMerger, {"alpha": 0.3, "lines_alpha": 0.1, "lines_beta": 0.9}),
            ("fisher", FisherMerger, {"alpha": 0.3}),
            ("iso_c", IsoCMerger, {"alpha": 1.0}),
            ("tsv_merge", TSVMerger, {"alpha": 1.0}),
            ("gram_ls", GramLSMerger, {"alpha": 1.0}),
        ]

        for name, cls, extra_config in methods:
            config = MergingConfig(method=name, **extra_config)
            merger = cls(pretrained, config)
            result = merger.merge(task_vectors)

            any_diff = False
            for k in pretrained:
                if not torch.allclose(result[k], pretrained[k], atol=1e-6):
                    any_diff = True
                    break
            assert any_diff, f"{name} merger produced identical weights to pretrained"

        # Also test SLERP with 2 tasks
        pretrained2, task_vectors2 = _make_synthetic_data(n_tasks=2)
        config = MergingConfig(method="slerp", slerp_t=0.5)
        merger = SLERPMerger(pretrained2, config)
        result = merger.merge(task_vectors2)
        any_diff = any(
            not torch.allclose(result[k], pretrained2[k], atol=1e-6)
            for k in pretrained2
        )
        assert any_diff, "slerp merger produced identical weights to pretrained"
