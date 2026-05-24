"""Fisher-weighted merging: importance-aware parameter averaging."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from med_merge.config.schema import MergingConfig
from med_merge.merging.base import BaseMerger
from med_merge.merging.task_vector import TaskVector

logger = logging.getLogger(__name__)


class FisherMerger(BaseMerger):
    """Merge task vectors weighted by diagonal Fisher Information.

    Fisher-weighted averaging gives higher weight to parameters that are
    more important for each task (measured via the gradient of the
    log-likelihood).

    Requires validation data to compute Fisher matrices.  Pass pre-computed
    matrices via ``fisher_matrices`` kwarg, or provide ``val_loaders``,
    ``heads``, and ``dataset_configs`` to compute on-the-fly.
    """

    def merge(
        self,
        task_vectors: dict[str, TaskVector],
        alpha: Optional[float] = None,
        fisher_matrices: Optional[dict[str, dict[str, torch.Tensor]]] = None,
        val_loaders: Optional[dict[str, DataLoader]] = None,
        heads: Optional[dict[str, dict[str, torch.Tensor]]] = None,
        dataset_configs: Optional[dict[str, Any]] = None,
        fisher_cache_dir: Optional[str] = None,
        model_config: Optional[Any] = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        alpha = alpha if alpha is not None else (self.config.fisher_alpha or self.config.alpha or 0.3)
        n_samples = self.config.fisher_n_samples

        # Compute or load Fisher matrices
        if fisher_matrices is None:
            fisher_matrices = {}
            for ds_name, tv in task_vectors.items():
                cached = self._try_load_fisher(fisher_cache_dir, ds_name) if fisher_cache_dir else None
                if cached is not None:
                    fisher_matrices[ds_name] = cached
                elif val_loaders and heads and dataset_configs:
                    fm = self.compute_fisher_diagonal(
                        tv, val_loaders[ds_name], heads[ds_name],
                        dataset_configs[ds_name], n_samples=n_samples,
                        model_config=model_config,
                    )
                    fisher_matrices[ds_name] = fm
                    if fisher_cache_dir:
                        self._save_fisher(fisher_cache_dir, ds_name, fm)
                else:
                    # No val data available — log loudly and degenerate to simple_avg
                    logger.warning(
                        f"FisherMerger: no val data / head / config for {ds_name}; "
                        "falling back to uniform weights (equivalent to simple_avg)."
                    )
                    fisher_matrices[ds_name] = {
                        k: torch.ones_like(t) for k, t in tv.vector.items()
                    }

        # Fisher-weighted merge — keep everything on CPU
        merged_vector: dict[str, torch.Tensor] = {}
        keys = list(next(iter(task_vectors.values())).vector.keys())

        for key in keys:
            ref = task_vectors[list(task_vectors)[0]].vector[key].detach().cpu()
            numerator = torch.zeros_like(ref)
            denominator = torch.zeros_like(ref)
            for ds_name, tv in task_vectors.items():
                tv_t = tv.vector[key].detach().cpu()
                f = fisher_matrices[ds_name].get(key, torch.ones_like(tv_t))
                f = f.detach().cpu()
                numerator += f * tv_t
                denominator += f
            merged_vector[key] = numerator / denominator.clamp(min=1e-8)

        result = TaskVector(vector=merged_vector)
        return result.apply_to(self.pretrained, scaling_coef=alpha)

    def compute_fisher_diagonal(
        self,
        task_vector: TaskVector,
        val_loader: DataLoader,
        head_state_dict: dict[str, torch.Tensor],
        dataset_config: Any,
        n_samples: int = 1000,
        model_config: Optional[Any] = None,
    ) -> dict[str, torch.Tensor]:
        """Compute diagonal Fisher Information for encoder parameters."""
        from med_merge.models.factory import create_model

        # Build model for the correct backbone (CLIP, ViT, DINOv3, RAD-DINO)
        model = create_model(dataset_config, model_config=model_config)
        encoder_sd = task_vector.apply_to(self.pretrained, scaling_coef=1.0)
        model.load_encoder_state_dict(encoder_sd)
        model.load_head_state_dict(head_state_dict)
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()

        # Compute loss function
        loss_fn = nn.CrossEntropyLoss() if dataset_config.task_type != "multilabel" else nn.BCEWithLogitsLoss()

        fisher: dict[str, torch.Tensor] = {}
        count = 0

        for images, labels in val_loader:
            if count >= n_samples:
                break
            images = images.to(next(model.parameters()).device)
            labels = labels.to(images.device)

            model.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()

            for name, param in model.named_parameters():
                if name.startswith("encoder.") and param.grad is not None:
                    grad_sq = param.grad.detach() ** 2
                    if name not in fisher:
                        fisher[name] = grad_sq
                    else:
                        fisher[name] += grad_sq

            count += images.size(0)

        # Average and move to CPU for portable caching + cross-device merge math
        for k in fisher:
            fisher[k] = (fisher[k] / max(count, 1)).cpu()

        logger.info(f"Computed Fisher diagonal ({count} samples, {len(fisher)} params)")
        return fisher

    @staticmethod
    def _try_load_fisher(cache_dir: str, ds_name: str) -> Optional[dict[str, torch.Tensor]]:
        path = Path(cache_dir) / ds_name / "fisher.pt"
        if path.exists():
            logger.info(f"Loading cached Fisher from {path}")
            return torch.load(path, map_location="cpu", weights_only=True)
        return None

    @staticmethod
    def _save_fisher(cache_dir: str, ds_name: str, fisher: dict[str, torch.Tensor]) -> None:
        path = Path(cache_dir) / ds_name
        path.mkdir(parents=True, exist_ok=True)
        torch.save(fisher, path / "fisher.pt")
        logger.info(f"Cached Fisher to {path / 'fisher.pt'}")

    @property
    def name(self) -> str:
        return "fisher"

    @property
    def hyperparameters(self) -> dict:
        return {
            "fisher_alpha": self.config.fisher_alpha or self.config.alpha,
            "fisher_n_samples": self.config.fisher_n_samples,
        }
