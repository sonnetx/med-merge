"""Tests for Pydantic configuration schema and YAML loading."""

import pytest
from pydantic import ValidationError

from med_merge.config.schema import (
    DatasetConfig,
    ExperimentConfig,
    MergingConfig,
    ModelConfig,
    TrainingConfig,
)
from med_merge.config.constants import ALL_DATASETS, ALL_METHODS, SEEDS, PRIMARY_METRICS


class TestModelConfig:
    def test_defaults(self):
        m = ModelConfig()
        assert m.backbone == "facebook/dinov3-vits16-pretrain-lvd1689m"
        assert m.hidden_size == 384
        assert m.num_layers == 12

    def test_custom(self):
        m = ModelConfig(backbone="facebook/dinov3-vits16-pretrain-lvd1689m", hidden_size=384)
        assert m.hidden_size == 384

    def test_rejects_unknown_field(self):
        with pytest.raises(ValidationError):
            ModelConfig(backbone="test", typo_field="oops")


class TestDatasetConfig:
    def test_defaults(self):
        d = DatasetConfig(name="test")
        assert d.task_type == "multiclass"
        assert d.augmentation == "standard"

    def test_rejects_bad_task_type(self):
        with pytest.raises(ValidationError):
            DatasetConfig(name="test", task_type="invalid")


class TestMergingConfig:
    def test_defaults(self):
        m = MergingConfig()
        assert m.method == "simple_avg"
        assert m.slerp_t == 0.5
        assert m.fisher_n_samples == 1000

    def test_model_dump_roundtrip(self):
        m = MergingConfig(method="ties", alpha=0.5, trim_fraction=0.1)
        d = m.model_dump()
        m2 = MergingConfig.model_validate(d)
        assert m2.method == "ties"
        assert m2.alpha == 0.5


class TestExperimentConfig:
    def test_has_model_field(self):
        e = ExperimentConfig()
        assert isinstance(e.model, ModelConfig)

    def test_mutation(self):
        e = ExperimentConfig()
        e.seed = 123
        e.dataset.name = "test"
        assert e.seed == 123
        assert e.dataset.name == "test"


class TestConstants:
    def test_datasets(self):
        from med_merge.config.constants import MULTICLASS_CORE, BINARY_CORE
        assert "isic2017" in ALL_DATASETS
        assert "chexpert" in ALL_DATASETS
        assert "pathmnist" in ALL_DATASETS
        assert MULTICLASS_CORE == ["isic2017", "chexpert", "pathmnist"]
        assert BINARY_CORE == ["isic_mel", "chexpert_pe", "patchcamelyon"]

    def test_methods(self):
        assert "slerp" in ALL_METHODS
        assert "fisher" in ALL_METHODS
        assert "dare_ties" in ALL_METHODS

    def test_seeds(self):
        assert SEEDS == [42, 123, 456]

    def test_primary_metrics(self):
        assert PRIMARY_METRICS["isic2017"] == "balanced_accuracy"
        assert PRIMARY_METRICS["chexpert"] == "macro_auroc"
        assert PRIMARY_METRICS["tcga"] == "auroc"


class TestYAMLLoading:
    def test_load_defaults(self):
        from med_merge.config.loader import load_config

        cfg = load_config(defaults_path="configs/defaults.yaml")
        assert cfg.seed == 42
        assert cfg.device == "cuda"

    def test_load_dataset_yaml(self):
        from med_merge.config.loader import load_yaml

        d = load_yaml("configs/datasets/isic2017.yaml")
        assert d["dataset"]["name"] == "isic2017"
        assert d["dataset"]["num_classes"] == 3

    def test_merge_configs(self):
        from med_merge.config.loader import load_yaml, merge_configs

        ds = load_yaml("configs/datasets/isic2017.yaml")
        tr = load_yaml("configs/training/isic2017.yaml")
        merged = merge_configs(ds, tr)
        assert "dataset" in merged
        assert "training" in merged
