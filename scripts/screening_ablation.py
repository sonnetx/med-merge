"""Cosine-screening ablation: sweep a cosine threshold, skip below-threshold
(backbone, dataset) cells and fall back to MTL or oracle, take best-of-8 merge
on the kept cells. Reports compute saved vs aggregate primary metric. Writes
screening_ablation.csv and screening_ablation.png under outputs/_figures/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from med_merge.config.constants import ALL_DATASETS, ALL_METHODS, PRIMARY_METRICS

BACKBONES = ["clip", "vit", "dinov3", "rad_dino", "dinov2", "mae", "beit"]
OUT_DIR = Path("outputs/_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_oracle(bb: str, seed: int, dataset: str) -> float | None:
    p = Path(f"outputs/{bb}/seed_{seed}/checkpoints/{dataset}/best_metrics.json")
    if not p.exists():
        return None
    return json.loads(p.read_text()).get(PRIMARY_METRICS[dataset])


def load_merged(bb: str, seed: int, method: str, dataset: str) -> float | None:
    p = Path(f"outputs/{bb}/seed_{seed}/results/{method}/results.json")
    if not p.exists():
        return None
    return json.loads(p.read_text()).get(dataset, {}).get(PRIMARY_METRICS[dataset])


def load_mtl(bb: str, seed: int, dataset: str) -> float | None:
    p = Path(f"outputs/{bb}/seed_{seed}/mtl/results.json")
    if not p.exists():
        return None
    return json.loads(p.read_text()).get(dataset, {}).get(PRIMARY_METRICS[dataset])


def load_cos(bb: str, seed: int, dataset: str) -> float | None:
    p = Path(f"outputs/{bb}/seed_{seed}/analysis/geometry_features.csv")
    if not p.exists():
        return None
    df = pd.read_csv(p)
    row = df[df["dataset"] == dataset]
    if row.empty:
        return None
    return float(row["mean_cos"].iloc[0])


def assemble(seed: int) -> pd.DataFrame:
    rows = []
    for bb in BACKBONES:
        for ds in ALL_DATASETS:
            cos = load_cos(bb, seed, ds)
            oracle = load_oracle(bb, seed, ds)
            mtl = load_mtl(bb, seed, ds)
            best_merge = None
            best_method = None
            for m in ALL_METHODS:
                v = load_merged(bb, seed, m, ds)
                if v is None:
                    continue
                if best_merge is None or v > best_merge:
                    best_merge = v
                    best_method = m
            if cos is None or oracle is None or best_merge is None:
                continue
            rows.append({
                "backbone": bb, "dataset": ds,
                "mean_cos": cos,
                "oracle": oracle,
                "mtl": mtl,
                "best_merge": best_merge,
                "best_method": best_method,
                "merge_forgetting": oracle - best_merge,
            })
    return pd.DataFrame(rows)


def evaluate_strategy(df: pd.DataFrame, threshold: float, fallback: str) -> dict:
    """Apply screening: low-cos cells fall back to MTL (or oracle/specialist).

    Returns aggregate performance and the number of cells we still need
    to merge-hyperopt.
    """
    if fallback == "mtl":
        fb = df["mtl"].fillna(df["oracle"])  # fall back to oracle if MTL absent
    elif fallback == "oracle":
        fb = df["oracle"]
    else:
        raise ValueError(fallback)
    kept = df["mean_cos"] >= threshold
    achieved = np.where(kept, df["best_merge"], fb)
    n_kept = int(kept.sum())
    n_total = len(df)
    return {
        "threshold": threshold,
        "fallback": fallback,
        "n_cells_merged": n_kept,
        "n_cells_total": n_total,
        "compute_saved_pct": 100.0 * (1 - n_kept / max(n_total, 1)),
        "mean_achieved_metric": float(np.mean(achieved)),
        "mean_merge_only": float(df["best_merge"].mean()),
        "mean_fallback_only": float(fb.mean()),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    df = assemble(args.seed)
    if df.empty:
        print("No data. Run compute_geometry.py and ensure mtl/results.json exists.")
        return

    print(f"Assembled {len(df)} cells")
    print(df.round(3).to_string(index=False))

    # Sweep the threshold over observed cosine values
    thresholds = sorted(df["mean_cos"].unique().tolist())
    rows = []
    for t in [0.0] + thresholds + [df["mean_cos"].max() + 0.01]:
        for fb in ["mtl", "oracle"]:
            rows.append(evaluate_strategy(df, t, fb))
    sweep = pd.DataFrame(rows)
    sweep.to_csv(OUT_DIR / "screening_ablation.csv", index=False)
    print(f"\nWrote {OUT_DIR / 'screening_ablation.csv'}")

    # Headline takeaways at the median threshold
    med = float(df["mean_cos"].median())
    for fb in ["mtl", "oracle"]:
        r = evaluate_strategy(df, med, fb)
        print(
            f"\nThreshold = median cos ({med:.3f}), fallback = {fb}:"
            f"\n  cells merged: {r['n_cells_merged']}/{r['n_cells_total']} "
            f"({r['compute_saved_pct']:.0f}% compute saved)"
            f"\n  achieved metric: {r['mean_achieved_metric']:.3f} "
            f"(vs full merge sweep: {r['mean_merge_only']:.3f}, "
            f"vs fallback only: {r['mean_fallback_only']:.3f})"
        )

    # Plot: achieved metric vs compute saved, for both fallbacks
    fig, ax = plt.subplots(figsize=(8, 5))
    for fb, color in [("mtl", "tab:blue"), ("oracle", "tab:orange")]:
        sub = sweep[sweep["fallback"] == fb].sort_values("compute_saved_pct")
        ax.plot(sub["compute_saved_pct"], sub["mean_achieved_metric"],
                marker="o", label=f"fallback = {fb}", color=color)
    ax.axhline(df["best_merge"].mean(), color="gray", linestyle=":",
               label=f"full merge sweep ({df['best_merge'].mean():.3f})")
    ax.set_xlabel("compute saved (% of cells skipped)")
    ax.set_ylabel("achieved mean primary metric across grid")
    ax.set_title("Cosine-screened practitioner workflow:\n"
                 "How much performance you sacrifice for how much compute saved")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "screening_ablation.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {OUT_DIR / 'screening_ablation.png'}")


if __name__ == "__main__":
    main()
