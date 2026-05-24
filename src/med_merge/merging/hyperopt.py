"""
Hyperparameter search for merging methods.
"""

from __future__ import annotations

import itertools
import json
import logging
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader

from med_merge.config.schema import MergingConfig, ModelConfig
from med_merge.merging.base import BaseMerger
from med_merge.merging.task_vector import TaskVector

logger = logging.getLogger(__name__)


def _evaluate_merged(
    merged_encoder: dict[str, torch.Tensor],
    val_loaders: dict[str, DataLoader],
    heads: dict[str, dict[str, torch.Tensor]],
    dataset_configs: dict[str, Any],
    device: str = "cuda",
    model_config: Optional[ModelConfig] = None,
    model_cache: Optional[dict] = None,
) -> dict[str, float]:
    """Evaluate a merged encoder on all validation sets. Returns aggregate metric.

    ``model_cache`` is a dict {ds_name: model}; if provided, models are
    reused across trials instead of being rebuilt each time.
    """
    from med_merge.evaluation.metrics import compute_metrics
    from med_merge.models.factory import create_model

    if model_cache is None:
        model_cache = {}

    all_metrics = {}

    for ds_name, loader in val_loaders.items():
        config = dataset_configs[ds_name]

        if ds_name not in model_cache:
            model_cache[ds_name] = create_model(config, model_config=model_config).to(device)
        model = model_cache[ds_name]

        # Hot-swap state dicts on the cached model
        model.load_encoder_state_dict(merged_encoder)
        model.load_head_state_dict(heads[ds_name])
        model.eval()

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
        primary_key = {
            "multiclass": "balanced_accuracy",
            "multilabel": "macro_auroc",
            "binary": "auroc",
            "ordinal": "qwk",
        }.get(config.task_type, "accuracy")

        all_metrics[ds_name] = metrics.get(primary_key, 0.0)

    aggregate = sum(all_metrics.values()) / max(len(all_metrics), 1)
    return {"aggregate": aggregate, **all_metrics}


def _params_key(params: dict) -> str:
    """Stable string key for hashing a params dict."""
    return json.dumps(params, sort_keys=True)


class MergingHyperoptimizer:
    """Grid search over merging hyperparameters with checkpointing."""

    def __init__(
        self,
        merger_class: type[BaseMerger],
        pretrained_state_dict: dict[str, torch.Tensor],
        base_config: MergingConfig,
        model_config: Optional[ModelConfig] = None,
        state_file: Optional[str | Path] = None,
        fisher_cache_dir: Optional[str | Path] = None,
    ):
        self.merger_class = merger_class
        self.pretrained = pretrained_state_dict
        self.base_config = base_config
        self.model_config = model_config
        self.state_file = Path(state_file) if state_file else None
        self.fisher_cache_dir = str(fisher_cache_dir) if fisher_cache_dir else None

    # ----- checkpoint I/O -----

    def _load_state(self) -> list[dict]:
        if not self.state_file or not self.state_file.exists():
            return []
        try:
            data = json.loads(self.state_file.read_text())
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Could not load hyperopt state {self.state_file}: {e}")
            return []

    def _save_state(self, all_results: list[dict]) -> None:
        if not self.state_file:
            return
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps(all_results, indent=2))

    # ----- main search -----

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

        # Resume from saved state if available
        all_results = self._load_state()
        completed = {_params_key(t["params"]): t for t in all_results}
        if completed:
            logger.info(
                f"Resuming hyperopt for {self.merger_class.__name__}: "
                f"{len(completed)}/{len(search_space)} trials already complete"
            )
        else:
            logger.info(
                f"Hyperopt for {self.merger_class.__name__}: "
                f"{len(search_space)} configurations"
            )

        best_score = -float("inf")
        best_params: dict = {}
        best_metrics: dict = {}

        # Seed best from cached trials
        for trial in all_results:
            if trial["score"] > best_score:
                best_score = trial["score"]
                best_params = trial["params"]
                best_metrics = trial["metrics"]

        # Model cache shared across trials — avoids HF weight reload per trial
        model_cache: dict = {}

        for params in search_space:
            key = _params_key(params)
            if key in completed:
                logger.info(f"  [cached] {params} -> aggregate={completed[key]['score']:.4f}")
                continue

            config = self._apply_params(params)
            merger = self.merger_class(self.pretrained, config)

            # Fisher needs val data + heads + model_config + cache to compute
            # real Fisher matrices. Other mergers ignore these via **kwargs.
            extra_kwargs = {}
            if self.base_config.method == "fisher":
                extra_kwargs = {
                    "val_loaders": val_loaders,
                    "heads": heads,
                    "dataset_configs": dataset_configs,
                    "model_config": self.model_config,
                    "fisher_cache_dir": self.fisher_cache_dir,
                }
            merged = merger.merge(task_vectors, **params, **extra_kwargs)

            metrics = _evaluate_merged(
                merged, val_loaders, heads, dataset_configs, device,
                model_config=self.model_config,
                model_cache=model_cache,
            )

            score = metrics["aggregate"]
            trial = {"params": params, "metrics": metrics, "score": score}
            all_results.append(trial)
            completed[key] = trial

            # Persist after each trial so a crash doesn't lose progress
            self._save_state(all_results)

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
