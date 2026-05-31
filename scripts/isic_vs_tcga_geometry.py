"""(norm, 1-cos) forensic scatter across (backbone, dataset) cells, used as
Figure 1 in the paper. Each cell gets one point: x = task-vector norm,
y = mean 1-cos to other tasks' task vectors, color = dataset, marker size
proportional to mean forgetting. Writes norm_vs_cos_scatter.png and
norm_vs_cos_table.csv under outputs/_figures/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from med_merge.config.constants import ALL_DATASETS, ALL_METHODS, PRIMARY_METRICS

BACKBONES = ["clip", "vit", "dinov3", "rad_dino", "dinov2", "mae", "beit"]
DATASET_COLORS = {
    "isic2017": "#1f77b4",
    "chexpert": "#2ca02c",
    "tcga": "#d62728",
    "nih_cxr": "#9467bd",
}
DATASET_LABELS = {"isic2017": "ISIC", "chexpert": "CheXpert",
                  "tcga": "TCGA", "nih_cxr": "NIH"}

OUT_DIR = Path("outputs/_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_features(bb: str, seed: int) -> pd.DataFrame | None:
    p = Path(f"outputs/{bb}/seed_{seed}/analysis/geometry_features.csv")
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["backbone"] = bb
    return df


def mean_forgetting(bb: str, seed: int, dataset: str) -> float | None:
    """Mean (oracle - merged) across all merging methods, in PRIMARY_METRIC."""
    oracle_p = Path(f"outputs/{bb}/seed_{seed}/checkpoints/{dataset}/best_metrics.json")
    if not oracle_p.exists():
        return None
    oracle = json.loads(oracle_p.read_text()).get(PRIMARY_METRICS[dataset])
    if oracle is None:
        return None
    drops = []
    for m in ALL_METHODS:
        p = Path(f"outputs/{bb}/seed_{seed}/results/{m}/results.json")
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        v = r.get(dataset, {}).get(PRIMARY_METRICS[dataset])
        if v is not None:
            drops.append(oracle - v)
    if not drops:
        return None
    return float(sum(drops) / len(drops))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rows = []
    for bb in BACKBONES:
        feats = load_features(bb, args.seed)
        if feats is None:
            continue
        for _, r in feats.iterrows():
            ds = r["dataset"]
            mf = mean_forgetting(bb, args.seed, ds)
            rows.append({
                "backbone": bb,
                "dataset": ds,
                "norm": float(r["norm"]),
                "mean_cos": float(r["mean_cos"]),
                "one_minus_cos": 1.0 - float(r["mean_cos"]),
                "interference": float(r["interference"]),
                "norm_times_one_minus_cos": float(r["norm"]) * (1.0 - float(r["mean_cos"])),
                "mean_forgetting": mf if mf is not None else float("nan"),
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "norm_vs_cos_table.csv", index=False)
    print(df.round(3).to_string(index=False))

    # Scatter
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    # Point size proportional to mean_forgetting (rescaled), with a floor
    valid = df["mean_forgetting"].notna()
    sizes = (df.loc[valid, "mean_forgetting"].clip(lower=0.0) * 1200 + 40).values
    for ds in DATASET_LABELS:
        mask = (df["dataset"] == ds) & valid
        if mask.sum() == 0:
            continue
        sub = df[mask]
        ax.scatter(sub["norm"], sub["one_minus_cos"],
                   s=(sub["mean_forgetting"].clip(lower=0.0) * 1200 + 40).values,
                   c=DATASET_COLORS[ds], alpha=0.7,
                   edgecolor="black", linewidth=1.0,
                   label=DATASET_LABELS[ds])
        for _, r in sub.iterrows():
            ax.annotate(r["backbone"], (r["norm"], r["one_minus_cos"]),
                        fontsize=8, alpha=0.7,
                        xytext=(5, 5), textcoords="offset points")

    ax.set_xlabel(r"task-vector norm $\|\tau_d\|$")
    ax.set_ylabel(r"mean misalignment to others $1 - \cos(\tau_d, \tau_{\neq d})$")
    ax.set_title(
        "Geometry of merge-failure: TCGA is big AND misaligned; "
        "ISIC is small even when misaligned\n"
        "(point size = mean forgetting across 8 merging methods)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    out = OUT_DIR / "norm_vs_cos_scatter.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")

    # Quick correlation summary: how well does norm * (1-cos) predict
    # mean forgetting?
    sub = df[valid].copy()
    rho = sub[["norm_times_one_minus_cos", "mean_forgetting"]].corr(method="spearman").iloc[0, 1]
    rho_cos = sub[["one_minus_cos", "mean_forgetting"]].corr(method="spearman").iloc[0, 1]
    rho_norm = sub[["norm", "mean_forgetting"]].corr(method="spearman").iloc[0, 1]
    print(f"\nSpearman(norm * (1-cos), forgetting) = {rho:.3f}")
    print(f"Spearman((1-cos),         forgetting) = {rho_cos:.3f}")
    print(f"Spearman(norm,            forgetting) = {rho_norm:.3f}")


if __name__ == "__main__":
    main()
