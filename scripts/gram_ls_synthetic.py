"""Synthetic validation of Gram-LS Merge.

Demonstrates the core mechanism without any GPU or trained checkpoints: when one task
vector is small in norm and near-orthogonal to a cluster of larger task vectors (the
cross-domain medical regime), simple averaging dilutes the outlier's signal while
Gram-LS preserves it. We sweep the outlier's cosine similarity to the cluster and
measure per-task *signal retention*  r_i = <m, tau_i> / <tau_i, tau_i>  (1.0 = the
merged vector fully reproduces task i's own signal under projection).

Run:  python scripts/gram_ls_synthetic.py
Writes: CS_229_Project_Milestone/gram_ls_synthetic.csv
"""

from __future__ import annotations

import csv
from pathlib import Path

import torch

from med_merge.config.schema import MergingConfig
from med_merge.merging.gram_ls import GramLSMerger
from med_merge.merging.iso_c import IsoCMerger
from med_merge.merging.simple_avg import SimpleAverageMerger
from med_merge.merging.task_arithmetic import TaskArithmeticMerger
from med_merge.merging.task_vector import TaskVector

KEY = "encoder.layers.0.weight"


def _retention(merged: torch.Tensor, tau: torch.Tensor) -> float:
    return float(torch.dot(merged, tau) / torch.dot(tau, tau).clamp(min=1e-12))


def _mean_cosine_to_others(target: torch.Tensor, others: list[torch.Tensor]) -> float:
    cos = [float(torch.nn.functional.cosine_similarity(target, o, dim=0)) for o in others]
    return sum(cos) / len(cos)


def build_task_vectors(
    dim: int,
    n_cluster: int,
    cluster_norm: float,
    outlier_norm: float,
    outlier_cos: float,
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    """A cluster of correlated large-norm tasks plus one small-norm outlier whose
    mean alignment with the cluster is approximately ``outlier_cos``."""
    shared = torch.randn(dim, generator=generator)
    shared = shared / shared.norm()

    tasks: dict[str, torch.Tensor] = {}
    for i in range(n_cluster):
        private = torch.randn(dim, generator=generator)
        private = private - torch.dot(private, shared) * shared
        private = private / private.norm()
        # 80% shared direction, 20% private -> the cluster is mutually correlated.
        v = 0.8 * shared + 0.2 * private
        v = v / v.norm() * cluster_norm
        tasks[f"cluster_{i}"] = v

    # Outlier: interpolate between the shared direction (cos=1) and an orthogonal
    # direction (cos=0) to hit the requested alignment.
    orth = torch.randn(dim, generator=generator)
    orth = orth - torch.dot(orth, shared) * shared
    orth = orth / orth.norm()
    out = outlier_cos * shared + (1.0 - outlier_cos**2) ** 0.5 * orth
    out = out / out.norm() * outlier_norm
    tasks["outlier"] = out
    return tasks


def evaluate(tasks: dict[str, torch.Tensor]) -> dict[str, dict[str, float]]:
    pretrained = {KEY: torch.zeros_like(tasks["outlier"])}
    tvs = {name: TaskVector(vector={KEY: v}) for name, v in tasks.items()}

    mergers = {
        "simple_avg": SimpleAverageMerger(pretrained, MergingConfig(method="simple_avg")),
        "task_arith": TaskArithmeticMerger(pretrained, MergingConfig(method="task_arithmetic", alpha=1.0)),
        "iso_c": IsoCMerger(pretrained, MergingConfig(method="iso_c", alpha=1.0)),
        "gram_ls": GramLSMerger(pretrained, MergingConfig(method="gram_ls", alpha=1.0, gram_lambda=0.0)),
    }

    out: dict[str, dict[str, float]] = {}
    for mname, merger in mergers.items():
        merged = merger.merge(tvs)[KEY]
        retentions = {name: _retention(merged, v) for name, v in tasks.items()}
        cluster_ret = [r for n, r in retentions.items() if n.startswith("cluster")]
        out[mname] = {
            "outlier_retention": retentions["outlier"],
            "mean_cluster_retention": sum(cluster_ret) / len(cluster_ret),
        }
    return out


def main() -> None:
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)

    dim = 1024
    n_cluster = 2          # two correlated large tasks (radiology-like)
    cluster_norm = 5.0
    outlier_norm = 0.5     # small-norm outlier (histopathology-like)

    rows = []
    print(f"\n{'cos(outlier, cluster)':>22} | {'method':>11} | {'outlier r_i':>12} | {'cluster r_i':>12}")
    print("-" * 70)
    for outlier_cos in [0.0, 0.1, 0.2, 0.4, 0.6, 0.8]:
        tasks = build_task_vectors(dim, n_cluster, cluster_norm, outlier_norm, outlier_cos, gen)
        others = [v for n, v in tasks.items() if n != "outlier"]
        measured_cos = _mean_cosine_to_others(tasks["outlier"], others)
        res = evaluate(tasks)
        for mname, metrics in res.items():
            rows.append({
                "requested_cos": outlier_cos,
                "measured_cos": round(measured_cos, 4),
                "method": mname,
                "outlier_retention": round(metrics["outlier_retention"], 4),
                "mean_cluster_retention": round(metrics["mean_cluster_retention"], 4),
            })
            print(f"{measured_cos:>22.3f} | {mname:>11} | "
                  f"{metrics['outlier_retention']:>12.3f} | {metrics['mean_cluster_retention']:>12.3f}")
        print("-" * 70)

    out_path = Path(__file__).parent.parent / "CS_229_Project_Milestone" / "gram_ls_synthetic.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {out_path}")
    print("\nReading: r_i = 1.0 means the merged vector fully preserves that task's own")
    print("signal. Near cos=0 (disjoint domains), simple_avg dilutes the outlier to ~1/K")
    print("while gram_ls holds it at ~1.0. The gap is the method's headline advantage.")


if __name__ == "__main__":
    main()
