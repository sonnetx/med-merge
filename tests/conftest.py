"""Shared test fixtures for med-merge.

Note: torch-dependent fixtures are guarded behind a try/except for
environments where the torch DLL may fail to load (Windows CI).
"""

from __future__ import annotations

import pytest

from med_merge.config.schema import (
    DatasetConfig,
    ExperimentConfig,
    MergingConfig,
    ModelConfig,
    TrainingConfig,
)


@pytest.fixture
def device():
    return "cpu"


@pytest.fixture
def model_config():
    return ModelConfig(backbone="openai/clip-vit-base-patch16")


@pytest.fixture
def dataset_config():
    return DatasetConfig(
        name="isic2017",
        source="local",
        data_dir="./data/isic2017",
        num_classes=3,
        task_type="multiclass",
        class_names=["nevus", "melanoma", "seborrheic_keratosis"],
    )


@pytest.fixture
def training_config():
    return TrainingConfig(epochs=2, batch_size=4, learning_rate=1e-4)


@pytest.fixture
def merging_config():
    return MergingConfig(method="task_arithmetic", alpha=0.3)
