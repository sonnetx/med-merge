"""TIES-Merging: Trim, Elect Sign, Disjoint Merge."""

from __future__ import annotations

from typing import Optional

import torch

from med_merge.config.schema import MergingConfig
from med_merge.merging.base import BaseMerger
from med_merge.merging.task_vector import TaskVector


class TIESMerger(BaseMerger):
    """TIES-Merging with sign conflict resolution.

    Steps:
    1. TRIM: Zero out parameters below threshold percentile in each task vector.
    2. ELECT SIGN: For each parameter, vote on dominant sign across task vectors.
    3. DISJOINT MERGE: Average only parameters agreeing with elected sign.
    4. Apply: pretrained + alpha * merged_vector.
    """

    def merge(
        self,
        task_vectors: dict[str, TaskVector],
        alpha: Optional[float] = None,
        trim_fraction: Optional[float] = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        alpha = alpha if alpha is not None else self.config.alpha
        if alpha is None:
            alpha = 0.3
        trim_fraction = trim_fraction if trim_fraction is not None else self.config.trim_fraction

        keys = list(next(iter(task_vectors.values())).vector.keys())
        merged_vector = {}

        for key in keys:
            tensors = [tv.vector[key].float() for tv in task_vectors.values() if key in tv.vector]
            if not tensors:
                continue

            stacked = torch.stack(tensors)  # (N, *shape)

            # Step 1: TRIM — zero out small-magnitude parameters
            trimmed = self._trim(stacked, trim_fraction)

            # Step 2: ELECT SIGN — majority vote on sign
            elected_sign = self._elect_sign(trimmed)

            # Step 3: DISJOINT MERGE — average only agreeing parameters
            merged_vector[key] = self._disjoint_merge(trimmed, elected_sign)

        tv = TaskVector(vector=merged_vector)
        return tv.apply_to(self.pretrained, scaling_coef=alpha)

    def _trim(self, stacked: torch.Tensor, fraction: float) -> torch.Tensor:
        """Zero out parameters below the top-k percentile per task vector."""
        trimmed = stacked.clone()
        for i in range(stacked.shape[0]):
            flat = stacked[i].abs().flatten()
            if flat.numel() == 0:
                continue
            threshold = torch.quantile(flat.float(), fraction)
            mask = stacked[i].abs() < threshold
            trimmed[i][mask] = 0.0
        return trimmed

    def _elect_sign(self, trimmed: torch.Tensor) -> torch.Tensor:
        """Elect sign by summing signed magnitudes across task vectors."""
        sign_sum = trimmed.sum(dim=0)
        elected = torch.sign(sign_sum)
        # Where sum is zero, default to positive
        elected[elected == 0] = 1.0
        return elected

    def _disjoint_merge(
        self, trimmed: torch.Tensor, elected_sign: torch.Tensor
    ) -> torch.Tensor:
        """Average only parameters whose sign matches the elected sign."""
        # Mask: keep only parameters with matching sign
        matching = (torch.sign(trimmed) == elected_sign.unsqueeze(0)) | (trimmed == 0)
        # For zero entries, exclude from average
        valid = (trimmed != 0) & (torch.sign(trimmed) == elected_sign.unsqueeze(0))
        count = valid.float().sum(dim=0).clamp(min=1.0)

        merged = (trimmed * valid.float()).sum(dim=0) / count
        return merged

    @property
    def name(self) -> str:
        return "ties"

    @property
    def hyperparameters(self) -> dict:
        return {"alpha": self.config.alpha, "trim_fraction": self.config.trim_fraction}
