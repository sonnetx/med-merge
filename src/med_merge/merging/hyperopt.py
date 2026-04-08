"""Hyperparameter search for merging methods."""

from __future__ import annotations

import itertools
import logging
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader

from med_merge.config.schema import MergingConfig
from med_merge.merging.base import BaseMerger
from med_merge.merging.task_vector import TaskVector
from med_merge.models.clip_classifier import CLIPClassifier

logger = logging.getLogger(__name__)


def _evaluate_merged(
    merged_encoder: dict[str, torch.Tensor],
    val_loaders: dict[str, DataLoader],
    heads: dict[str, dict[str, torch.Tensor]],
    dataset_configs: dict[str, Any],
    device: str = "cuda",
) -> dict[str, float]:
    """Evaluate a merged encoder on all validation sets. Returns aggregate metric."""
    from med_merge.evaluation.metrics import compute_metrics

    all_metrics = {}

    for ds_name, loader in val_loaders.items():
        # Build model with merged encoder + dataset head
        config = dataset_configs[ds_name]
        model = CLIPClassifier(
            num_classes=config.num_classes,
            task_type=config.task_type,
        )
        model.load_encoder_state_dict(merged_encoder)
        model.load_head_state_dict(heads[ds_name])
        model = model.to(device).eval()

        all_logits = []
        all_labels = []

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device, non_blocking=True)
                logits = model(images)
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())

        logits = torch.cat(all_logits).numpy()
        labels = torch.cat(all_labels).numpy()

        metrics = compute_metrics(
            logits, labels,
            task_type=config.task_type,
            class_names=config.class_names,
        )
        # Use primary metric for this task type
        primary_key = {
            "multiclass": "balanced_accuracy",
            "multilabel": "macro_auroc",
            "binary": "auroc",
            "ordinal": "qwk",
        }.get(config.task_type, "accuracy")

        all_metrics[ds_name] = metrics.get(primary_key, 0.0)

    # Aggregate: mean across datasets
    aggregate = sum(all_metrics.values()) / max(len(all_metrics), 1)
    return {"aggregate": aggregate, **all_metrics}


class MergingHyperoptimizer:
    """Grid search over merging hyperparameters."""

    def __init__(
        self,
        merger_class: type[BaseMerger],
        pretrained_state_dict: dict[str, torch.Tensor],
        base_config: MergingConfig,
    ):
        self.merger_class = merger_class
        self.pretrained = pretrained_state_dict
        self.base_config = base_config

    def search(
        self,
        task_vectors: dict[str, TaskVector],
        val_loaders: dict[str, DataLoader],
        heads: dict[str, dict[str, torch.Tensor]],
        dataset_configs: dict[str, Any],
        device: str = "cuda",
    ) -> dict[str, Any]:
        """Run grid search and return best hyperparameters + results."""
        search_space = self._get_search_space()
        logger.info(
            f"Hyperopt for {self.merger_class.__name__}: "
            f"{len(search_space)} configurations"
        )

        best_score = -float("inf")
        best_params = {}
        best_metrics = {}
        all_results = []

        for params in search_space:
            config = self._apply_params(params)
            merger = self.merger_class(self.pretrained, config)
            merged = merger.merge(task_vectors, **params)

            metrics = _evaluate_merged(
                merged, val_loaders, heads, dataset_configs, device
            )

            score = metrics["aggregate"]
            all_results.append({"params": params, "metrics": metrics, "score": score})

            if score > best_score:
                best_score = score
                best_params = params
                best_metrics = metrics

            logger.info(f"  {params} -> aggregate={score:.4f}")

        logger.info(f"Best: {best_params} -> aggregate={best_score:.4f}")

        return {
            "best_params": best_params,
            "best_score": best_score,
            "best_metrics": best_metrics,
            "all_results": all_results,
        }

    def _get_search_space(self) -> list[dict]:
        """Build search space based on method."""
        method = self.base_config.method

        if method == "task_arithmetic":
            return [{"alpha": a} for a in self.base_config.alpha_search]

        elif method == "ties":
            return [
                {"alpha": a, "trim_fraction": t}
                for a, t in itertools.product(
                    self.base_config.alpha_search,
                    self.base_config.trim_fraction_search,
                )
            ]

        elif method == "dare":
            if self.base_config.inner_method == "ties":
                # DARE-TIES: 3D grid (alpha, drop_rate, trim_fraction)
                return [
                    {"alpha": a, "drop_rate": d, "trim_fraction": t}
                    for a, d, t in itertools.product(
                        self.base_config.alpha_search,
                        self.base_config.drop_rate_search,
                        self.base_config.trim_fraction_search,
                    )
                ]
            return [
                {"alpha": a, "drop_rate": d}
                for a, d in itertools.product(
                    self.base_config.alpha_search,
                    self.base_config.drop_rate_search,
                )
            ]

        elif method == "lines":
            return [
                {"alpha": a, "lines_alpha": la, "lines_beta": lb}
                for a, la, lb in itertools.product(
                    self.base_config.alpha_search,
                    self.base_config.lines_alpha_search,
                    self.base_config.lines_beta_search,
                )
            ]

        elif method == "slerp":
            return [{"t": t} for t in self.base_config.slerp_t_search]

        elif method == "fisher":
            return [{"alpha": a} for a in self.base_config.alpha_search]

        else:
            # No hyperparameters (simple_avg, pcb_merging)
            return [{}]

    def _apply_params(self, params: dict) -> MergingConfig:
        """Create a config with specific hyperparameters."""
        config_dict = self.base_config.model_dump()
        config_dict.update(params)
        return MergingConfig.model_validate(config_dict)
