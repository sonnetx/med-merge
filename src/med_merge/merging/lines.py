"""LiNeS: Layer-Increasing Network Scaling."""

from __future__ import annotations

import re
from typing import Optional

import torch

from med_merge.config.schema import MergingConfig
from med_merge.merging.base import BaseMerger
from med_merge.merging.task_arithmetic import TaskArithmeticMerger
from med_merge.merging.task_vector import TaskVector


class LiNeSMerger(BaseMerger):
    """Layer-Increasing Network Scaling.

    Post-processing applied to merged task vectors:
    - Scale each layer's parameters by a factor that increases linearly with depth.
    - Layer i gets: alpha + (beta * i / (L-1))
    - Shallow layers (general features) → scaled DOWN toward pretrained.
    - Deep layers (task-specific) → scaled UP to preserve task knowledge.

    Wraps Task Arithmetic as the inner merger by default.
    """

    def __init__(
        self,
        pretrained_state_dict: dict[str, torch.Tensor],
        config: MergingConfig,
        num_layers: int = 12,
        layer_key_pattern: str = r"layers\.(\d+)\.",
    ):
        super().__init__(pretrained_state_dict, config)
        self.inner_merger = TaskArithmeticMerger(pretrained_state_dict, config)
        self._num_layers = num_layers
        self._layer_key_pattern = layer_key_pattern

    def merge(
        self,
        task_vectors: dict[str, TaskVector],
        lines_alpha: Optional[float] = None,
        lines_beta: Optional[float] = None,
        alpha: Optional[float] = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        lines_alpha = lines_alpha if lines_alpha is not None else self.config.lines_alpha
        lines_beta = lines_beta if lines_beta is not None else self.config.lines_beta

        # First merge with inner method to get a combined task vector
        combined = None
        for tv in task_vectors.values():
            if combined is None:
                combined = tv
            else:
                combined = combined + tv

        # Apply layer-increasing scaling to the combined task vector
        scaled_vector = self._apply_layer_scaling(combined, lines_alpha, lines_beta)

        # Apply with inner alpha
        inner_alpha = alpha if alpha is not None else self.config.alpha
        if inner_alpha is None:
            inner_alpha = 0.3

        return scaled_vector.apply_to(self.pretrained, scaling_coef=inner_alpha)

    def _apply_layer_scaling(
        self,
        task_vector: TaskVector,
        base_alpha: float,
        beta: float,
    ) -> TaskVector:
        """Apply linearly increasing scale factors per layer depth."""
        new_vector = {}
        max_depth = self._num_layers - 1

        for key, tensor in task_vector.vector.items():
            depth = self._get_layer_depth(key)
            if depth is not None and max_depth > 0:
                scale = base_alpha + (beta * depth / max_depth)
            else:
                # Non-layer params (embeddings, final LN) → use base_alpha (conservative)
                scale = base_alpha
            new_vector[key] = tensor * scale

        return TaskVector(vector=new_vector)

    def _get_layer_depth(self, key: str) -> Optional[int]:
        """Extract transformer layer depth from parameter key."""
        match = re.search(self._layer_key_pattern, key)
        if match:
            return int(match.group(1))
        return None

    @property
    def name(self) -> str:
        return "lines"

    @property
    def hyperparameters(self) -> dict:
        return {
            "lines_alpha": self.config.lines_alpha,
            "lines_beta": self.config.lines_beta,
            "inner_alpha": self.config.alpha,
        }
