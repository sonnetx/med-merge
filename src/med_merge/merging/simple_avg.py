"""Simple weight averaging merger (baseline)."""

from __future__ import annotations

import torch

from med_merge.config.schema import MergingConfig
from med_merge.merging.base import BaseMerger
from med_merge.merging.task_vector import TaskVector


class SimpleAverageMerger(BaseMerger):
    """Naive averaging: pretrained + (1/N) * sum(task_vectors)."""

    def merge(self, task_vectors: dict[str, TaskVector], **kwargs) -> dict[str, torch.Tensor]:
        n = len(task_vectors)
        combined = None
        for tv in task_vectors.values():
            if combined is None:
                combined = tv
            else:
                combined = combined + tv
        combined = combined * (1.0 / n)
        return combined.apply_to(self.pretrained)

    @property
    def name(self) -> str:
        return "simple_avg"
