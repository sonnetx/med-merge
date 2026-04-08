"""Merging method registry."""

from __future__ import annotations

import torch

from med_merge.config.schema import MergingConfig
from med_merge.merging.base import BaseMerger
from med_merge.merging.dare import DAREMerger
from med_merge.merging.fisher import FisherMerger
from med_merge.merging.lines import LiNeSMerger
from med_merge.merging.pcb_merging import PCBMerger
from med_merge.merging.simple_avg import SimpleAverageMerger
from med_merge.merging.slerp import SLERPMerger
from med_merge.merging.task_arithmetic import TaskArithmeticMerger
from med_merge.merging.ties import TIESMerger

MERGER_REGISTRY: dict[str, type[BaseMerger]] = {
    "simple_avg": SimpleAverageMerger,
    "task_arithmetic": TaskArithmeticMerger,
    "ties": TIESMerger,
    "dare": DAREMerger,
    "dare_ties": DAREMerger,  # convenience alias — uses inner_method="ties"
    "pcb_merging": PCBMerger,
    "lines": LiNeSMerger,
    "slerp": SLERPMerger,
    "fisher": FisherMerger,
}


def build_merger(
    pretrained_state_dict: dict[str, torch.Tensor],
    config: MergingConfig,
) -> BaseMerger:
    """Build a merger by method name."""
    method = config.method
    if method not in MERGER_REGISTRY:
        raise ValueError(
            f"Unknown merging method: {method}. "
            f"Available: {list(MERGER_REGISTRY.keys())}"
        )

    # For dare_ties alias, set inner_method automatically
    if method == "dare_ties":
        config.method = "dare"
        config.inner_method = "ties"

    return MERGER_REGISTRY[method](pretrained_state_dict, config)
