"""Main evaluation orchestrator."""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader

from med_merge.config.schema import DatasetConfig, EvaluationConfig, ModelConfig
from med_merge.data.registry import build_dataset
from med_merge.data.transforms import get_eval_transform, norm_key_for_backbone
from med_merge.evaluation.calibration import (
    brier_score,
    expected_calibration_error,
    reliability_diagram_data,
)
from med_merge.evaluation.inference import InferenceResult, InferenceRunner
from med_merge.evaluation.metrics import compute_metrics
from med_merge.models.classifier import VisionClassifier
from med_merge.models.factory import create_model
from med_merge.models.heads import create_head

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluate a merged (or individual) encoder on benchmark datasets."""

    def __init__(
        self,
        eval_config: EvaluationConfig,
        model_config: Optional[ModelConfig] = None,
        device: str = "cuda",
    ):
        self.config = eval_config
        self.model_config = model_config or ModelConfig()
        self.device = device
        self.runner = InferenceRunner(device=device)

    def evaluate_single(
        self,
        encoder_state_dict: dict[str, torch.Tensor],
        dataset_config: DatasetConfig,
        head_state_dict: dict[str, torch.Tensor],
    ) -> dict[str, Any]:
        """Evaluate on a single dataset."""
        # Build model
        model = create_model(dataset_config, self.model_config)
        model.load_encoder_state_dict(encoder_state_dict)
        model.load_head_state_dict(head_state_dict)

        # Build test loader
        nk = norm_key_for_backbone(self.model_config.backbone)
        transform = get_eval_transform(dataset_config.image_size, norm_key=nk)
        extra_kw = {}
        if dataset_config.csv_path:
            extra_kw["csv_path"] = dataset_config.csv_path
        dataset = build_dataset(
            dataset_config.name,
            dataset_config.data_dir,
            split="test",
            transform=transform,
            **extra_kw,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

        # Inference
        result = self.runner.run(model, loader)

        # Metrics
        metrics = compute_metrics(
            result.logits, result.labels,
            task_type=dataset_config.task_type,
            class_names=dataset_config.class_names,
        )
        ece = expected_calibration_error(
            result.logits, result.labels,
            task_type=dataset_config.task_type,
            n_bins=self.config.num_calibration_bins,
        )
        metrics["ece"] = ece

        brier = brier_score(
            result.logits, result.labels,
            task_type=dataset_config.task_type,
        )
        metrics["brier"] = brier

        bin_centers, bin_accs, bin_counts = reliability_diagram_data(
            result.logits, result.labels,
            task_type=dataset_config.task_type,
            n_bins=self.config.num_calibration_bins,
        )

        return {
            "metrics": metrics,
            "calibration": {
                "ece": ece,
                "brier": brier,
                "bin_centers": bin_centers.tolist(),
                "bin_accuracies": bin_accs.tolist(),
                "bin_counts": bin_counts.tolist(),
            },
            "raw": {"logits": result.logits, "labels": result.labels},
        }

    def evaluate_all(
        self,
        encoder_state_dict: dict[str, torch.Tensor],
        dataset_configs: dict[str, DatasetConfig],
        heads: dict[str, dict[str, torch.Tensor]],
    ) -> dict[str, dict[str, Any]]:
        """Evaluate merged encoder on all datasets."""
        results = {}
        for ds_name, ds_config in dataset_configs.items():
            if ds_name not in heads:
                logger.warning(f"No head found for {ds_name}, skipping")
                continue
            logger.info(f"Evaluating on {ds_name}...")
            results[ds_name] = self.evaluate_single(
                encoder_state_dict, ds_config, heads[ds_name],
            )
            logger.info(f"  {ds_name}: {results[ds_name]['metrics']}")
        return results
