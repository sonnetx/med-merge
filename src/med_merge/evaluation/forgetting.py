"""Forgetting metric: drop in primary metric from oracle (individual fine-tuned) to merged."""

from __future__ import annotations

from typing import Mapping

from med_merge.config.constants import PRIMARY_METRICS


def compute_forgetting(
    merged_results: Mapping[str, Mapping[str, float]],
    oracle_results: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    """Per-task drop in primary metric from oracle to merged model.

    Args:
        merged_results: ``{dataset_name: {metric: value}}`` from the merged model.
        oracle_results: ``{dataset_name: {metric: value}}`` from individually
            fine-tuned models (one model per dataset).

    Returns:
        ``{dataset_name: oracle - merged}`` for each dataset present in both,
        plus ``"mean_forgetting"``. Positive values mean merging hurt the task.
    """
    out: dict[str, float] = {}
    deltas: list[float] = []

    for ds_name, oracle_metrics in oracle_results.items():
        if ds_name not in merged_results:
            continue
        metric_key = PRIMARY_METRICS.get(ds_name)
        if metric_key is None:
            continue
        merged_metrics = merged_results[ds_name]
        if metric_key not in oracle_metrics or metric_key not in merged_metrics:
            continue
        delta = float(oracle_metrics[metric_key]) - float(merged_metrics[metric_key])
        out[ds_name] = delta
        deltas.append(delta)

    if deltas:
        out["mean_forgetting"] = float(sum(deltas) / len(deltas))
    return out
