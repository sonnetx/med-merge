"""Pre-merge diagnostics: quality prediction, domain gap quantification."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import torch

from med_merge.merging.task_vector import TaskVector


def predict_merge_quality(
    task_vectors: dict[str, TaskVector],
) -> dict[str, Any]:
    """Compute diagnostic metrics that predict merge success.

    Returns dict with norm statistics, pairwise cosine, and a heuristic
    risk score (0 = low risk, 1 = high risk).
    """
    names = list(task_vectors.keys())

    # Task vector norms
    norms = {}
    for name, tv in task_vectors.items():
        flat = torch.cat([t.flatten() for t in tv.vector.values()])
        norms[name] = flat.norm().item()

    norm_values = list(norms.values())
    norm_ratio = max(norm_values) / (min(norm_values) + 1e-8)

    # Pairwise cosine similarities
    cosines = []
    for a, b in combinations(names, 2):
        flat_a = torch.cat([task_vectors[a].vector[k].flatten() for k in task_vectors[a].vector])
        flat_b = torch.cat([task_vectors[b].vector[k].flatten() for k in task_vectors[b].vector])
        cos = torch.nn.functional.cosine_similarity(
            flat_a.unsqueeze(0), flat_b.unsqueeze(0)
        ).item()
        cosines.append(cos)

    mean_cosine = float(np.mean(cosines)) if cosines else 0.0

    # Risk heuristic: high norm ratio + low cosine = high risk
    norm_risk = min(norm_ratio / 10.0, 1.0)  # normalize
    cosine_risk = max(0.0, 0.5 - mean_cosine)  # negative cosine = risk
    risk_score = 0.5 * norm_risk + 0.5 * cosine_risk

    return {
        "task_vector_norms": norms,
        "mean_norm": float(np.mean(norm_values)),
        "norm_ratio_max_min": norm_ratio,
        "mean_pairwise_cosine": mean_cosine,
        "pairwise_cosines": cosines,
        "risk_score": min(risk_score, 1.0),
    }


def compute_domain_gap(
    task_vectors: dict[str, TaskVector],
) -> dict[tuple[str, str], dict[str, float]]:
    """Quantify domain gap between task vectors.

    For each pair, computes L2 distance and cosine similarity of
    normalized task vectors.  Directly addresses RQ1 (radiology vs
    pathology vs dermoscopy distance).
    """
    names = list(task_vectors.keys())
    result = {}

    for a, b in combinations(names, 2):
        flat_a = torch.cat([task_vectors[a].vector[k].flatten() for k in task_vectors[a].vector])
        flat_b = torch.cat([task_vectors[b].vector[k].flatten() for k in task_vectors[b].vector])

        # Normalize
        na = flat_a / (flat_a.norm() + 1e-8)
        nb = flat_b / (flat_b.norm() + 1e-8)

        l2_dist = (na - nb).norm().item()
        cosine = torch.dot(na, nb).item()

        result[(a, b)] = {
            "l2_distance_normalized": l2_dist,
            "cosine_similarity": cosine,
            "l2_distance_raw": (flat_a - flat_b).norm().item(),
        }

    return result
