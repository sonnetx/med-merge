"""Merging method registry."""

from __future__ import annotations

import torch

from med_merge.config.schema import MergingConfig
from med_merge.merging.base import BaseMerger
from med_merge.merging.dare import DAREMerger
from med_merge.merging.fisher import FisherMerger
from med_merge.merging.gram_ls import GramLSMerger
from med_merge.merging.iso_c import IsoCMerger
from med_merge.merging.iso_cts import IsoCTSMerger
from med_merge.merging.lines import LiNeSMerger
from med_merge.merging.pcb_merging import PCBMerger
from med_merge.merging.simple_avg import SimpleAverageMerger
from med_merge.merging.slerp import SLERPMerger
from med_merge.merging.task_arithmetic import TaskArithmeticMerger
from med_merge.merging.ties import TIESMerger
from med_merge.merging.tsv import TSVMerger

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
    "iso_c": IsoCMerger,
    "iso_cts": IsoCTSMerger,
    "tsv_merge": TSVMerger,
    "gram_ls": GramLSMerger,
    "gram_ls_na": GramLSMerger,  # norm-aware variant (config.gram_norm_aware set below)
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

    # Norm-aware Gram-LS: same merger, ridge scaled by max task norm.
    if method == "gram_ls_na":
        config.gram_norm_aware = True

    return MERGER_REGISTRY[method](pretrained_state_dict, config)
