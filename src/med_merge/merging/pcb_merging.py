"""PCB-Merging: Parameter Competition Balancing (NeurIPS 2024)."""

from __future__ import annotations

import torch

from med_merge.config.schema import MergingConfig
from med_merge.merging.base import BaseMerger
from med_merge.merging.task_vector import TaskVector


class PCBMerger(BaseMerger):
    """Parameter Competition Balancing for model merging.

    Training-free method with intra-task and inter-task balancing:
    1. INTRA-BALANCING: Score each parameter within a task by importance
       (magnitude relative to layer statistics).
    2. INTER-BALANCING: Score parameter similarity across tasks
       (cosine similarity of parameter vectors at each position).
    3. Combined score: intra_score * inter_score.
    4. Apply scores as per-parameter scaling during merge.
    """

    def merge(
        self,
        task_vectors: dict[str, TaskVector],
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        keys = list(next(iter(task_vectors.values())).vector.keys())
        tv_names = list(task_vectors.keys())
        n_tasks = len(tv_names)

        merged_vector = {}

        for key in keys:
            tensors = []
            for name in tv_names:
                if key in task_vectors[name].vector:
                    tensors.append(task_vectors[name].vector[key].float())
            if not tensors:
                continue

            stacked = torch.stack(tensors)  # (N, *shape)
            flat_shape = (n_tasks, -1)
            flat = stacked.reshape(flat_shape)

            # Intra-task balancing: normalize by layer statistics
            intra_scores = self._intra_balance(flat)

            # Inter-task balancing: measure parameter agreement
            inter_scores = self._inter_balance(flat)

            # Combined scores
            combined = intra_scores * inter_scores  # (N, D)

            # Normalize scores to sum to 1 across tasks for each parameter
            score_sum = combined.sum(dim=0, keepdim=True).clamp(min=1e-8)
            weights = combined / score_sum  # (N, D)

            # Weighted merge
            merged_flat = (flat * weights).sum(dim=0)
            merged_vector[key] = merged_flat.reshape(stacked.shape[1:])

        tv = TaskVector(vector=merged_vector)
        return tv.apply_to(self.pretrained)

    def _intra_balance(self, flat: torch.Tensor) -> torch.Tensor:
        """Score parameters by importance within each task.

        Uses magnitude relative to mean magnitude in the layer.
        """
        # (N, D)
        magnitudes = flat.abs()
        mean_mag = magnitudes.mean(dim=1, keepdim=True).clamp(min=1e-8)
        return magnitudes / mean_mag

    def _inter_balance(self, flat: torch.Tensor) -> torch.Tensor:
        """Score parameter agreement across tasks.

        High score = parameters agree in direction across tasks.
        Low score = conflicting directions.
        """
        n_tasks = flat.shape[0]

        if n_tasks <= 1:
            return torch.ones_like(flat)

        # Sign agreement: fraction of tasks with same sign
        signs = torch.sign(flat)  # (N, D)
        # For each parameter, count how many tasks agree with the majority sign
        sign_sum = signs.sum(dim=0)  # (D,)
        majority_sign = torch.sign(sign_sum)
        majority_sign[majority_sign == 0] = 1.0

        agreement = (signs == majority_sign.unsqueeze(0)).float()  # (N, D)
        agreement_fraction = agreement  # already per-task, per-param

        return agreement_fraction

    @property
    def name(self) -> str:
        return "pcb_merging"

    @property
    def hyperparameters(self) -> dict:
        return {}  # training-free, no hyperparameters
