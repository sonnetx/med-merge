"""Task vector conflict analysis: sign agreement, pairwise interference."""

from __future__ import annotations

import re
from itertools import combinations
from typing import Optional

import numpy as np
import torch

from med_merge.merging.task_vector import TaskVector


def compute_sign_agreement(
    task_vectors: dict[str, TaskVector],
) -> dict[str, dict[str, float]]:
    """For each parameter key, compute fraction of task vectors that agree in sign.

    Returns ``{key: {"agreement": float, "n_positive": int, "n_negative": int}}``.
    """
    names = list(task_vectors.keys())
    any_tv = task_vectors[names[0]]
    result = {}

    for key in any_tv.vector:
        signs = []
        for name in names:
            t = task_vectors[name].vector.get(key)
            if t is not None:
                signs.append(torch.sign(t).flatten())

        if not signs:
            continue

        stacked = torch.stack(signs)  # (N, params)
        pos = (stacked > 0).sum(dim=0)
        neg = (stacked < 0).sum(dim=0)
        majority = torch.maximum(pos, neg)
        agreement = (majority.float() / len(signs)).mean().item()

        result[key] = {
            "agreement": agreement,
            "n_positive": int(pos.sum()),
            "n_negative": int(neg.sum()),
        }

    return result


def compute_layerwise_conflict(
    task_vectors: dict[str, TaskVector],
    layer_pattern: str = r"layers\.(\d+)\.",
) -> dict[int, float]:
    """Aggregate sign agreement per transformer layer depth.

    Returns ``{layer_idx: mean_agreement_fraction}``.
    """
    per_key = compute_sign_agreement(task_vectors)
    layer_scores: dict[int, list[float]] = {}

    for key, info in per_key.items():
        m = re.search(layer_pattern, key)
        if m:
            depth = int(m.group(1))
            layer_scores.setdefault(depth, []).append(info["agreement"])

    return {d: float(np.mean(scores)) for d, scores in sorted(layer_scores.items())}


def compute_pairwise_interference(
    task_vectors: dict[str, TaskVector],
) -> dict[tuple[str, str], dict[str, float]]:
    """For each pair of tasks, compute cosine similarity, sign disagreement, etc.

    Returns ``{(task_a, task_b): {"cosine": float, "sign_disagreement": float}}``.
    """
    result = {}
    names = list(task_vectors.keys())

    for a, b in combinations(names, 2):
        tv_a = task_vectors[a]
        tv_b = task_vectors[b]

        # Flatten all keys into single vectors
        flat_a = torch.cat([tv_a.vector[k].flatten() for k in tv_a.vector])
        flat_b = torch.cat([tv_b.vector[k].flatten() for k in tv_b.vector])

        # Cosine similarity
        cos = torch.nn.functional.cosine_similarity(
            flat_a.unsqueeze(0), flat_b.unsqueeze(0)
        ).item()

        # Sign disagreement
        signs_a = torch.sign(flat_a)
        signs_b = torch.sign(flat_b)
        disagree = ((signs_a != signs_b) & (signs_a != 0) & (signs_b != 0)).float().mean().item()

        # Magnitude correlation
        mag_a = flat_a.abs().numpy()
        mag_b = flat_b.abs().numpy()
        corr = float(np.corrcoef(mag_a, mag_b)[0, 1]) if len(mag_a) > 1 else 0.0

        result[(a, b)] = {
            "cosine": cos,
            "sign_disagreement": disagree,
            "magnitude_correlation": corr,
        }

    return result
