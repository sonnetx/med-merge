"""Plot task-vector cosine similarity matrices + correlate with merged-model F1.

Run from repo root:
    python scripts/plot_cosine_analysis.py

Outputs:
    outputs/_figures/cosine_similarity_4backbones.png
    outputs/_figures/cosine_vs_tcga_f1.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BACKBONES = ["clip", "vit", "dinov3", "rad_dino"]
BACKBONE_LABELS = {
    "clip": "CLIP ViT-B/16",
    "vit": "ViT-B/16",
    "dinov3": "DINOv3 ViT-S/16",
    "rad_dino": "RAD-DINO",
}
DATASETS = ["isic2017", "chexpert", "tcga", "nih_cxr"]
DATASET_LABELS = {
    "isic2017": "ISIC",
    "chexpert": "CheXpert",
    "tcga": "TCGA",
    "nih_cxr": "NIH",
}
METHODS = ["simple_avg", "task_arithmetic", "ties", "dare",
           "dare_ties", "pcb_merging", "lines", "fisher"]
SEED = 42

OUT_DIR = Path("outputs/_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_cosine_matrix(backbone: str) -> pd.DataFrame | None:
    path = Path(f"outputs/{backbone}/seed_{SEED}/analysis/task_vector_cosine.csv")
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0)
    # Reorder rows/cols consistently
    available = [d for d in DATASETS if d in df.index]
    return df.loc[available, available]


def load_specialist_metric(backbone: str, dataset: str, key: str) -> float | None:
    path = Path(f"outputs/{backbone}/seed_{SEED}/checkpoints/{dataset}/best_metrics.json")
    if not path.exists():
        return None
    m = json.loads(path.read_text())
    return m.get(key)


def load_merged_metric(backbone: str, method: str, dataset: str, key: str) -> float | None:
    path = Path(f"outputs/{backbone}/seed_{SEED}/results/{method}/results.json")
    if not path.exists():
        return None
    r = json.loads(path.read_text())
    return r.get(dataset, {}).get(key)


def plot_cosine_heatmaps():
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.flatten()

    for ax, bb in zip(axes, BACKBONES):
        mat = load_cosine_matrix(bb)
        if mat is None:
            ax.set_title(f"{BACKBONE_LABELS[bb]} (no data)")
            ax.axis("off")
            continue

        labels = [DATASET_LABELS.get(c, c) for c in mat.columns]
        sns.heatmap(
            mat.values,
            xticklabels=labels,
            yticklabels=labels,
            annot=True,
            fmt=".2f",
            cmap="viridis",
            vmin=0,
            vmax=1,
            cbar=True,
            ax=ax,
            square=True,
            annot_kws={"size": 10},
        )
        ax.set_title(BACKBONE_LABELS[bb], fontsize=12, pad=8)
        ax.set_xlabel("")
        ax.set_ylabel("")

    fig.suptitle(
        "Pairwise Task-Vector Cosine Similarity (seed 42)",
        fontsize=14, y=1.00,
    )
    fig.tight_layout()
    out = OUT_DIR / "cosine_similarity_4backbones.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")
    plt.close(fig)


def plot_cosine_vs_tcga_f1():
    rows = []
    for bb in BACKBONES:
        mat = load_cosine_matrix(bb)
        if mat is None or "tcga" not in mat.index:
            continue
        others = [d for d in mat.index if d != "tcga"]
        tcga_mean_cos = float(mat.loc["tcga", others].mean())

        specialist_f1 = load_specialist_metric(bb, "tcga", "f1")
        for method in METHODS:
            f1 = load_merged_metric(bb, method, "tcga", "f1")
            if f1 is None:
                continue
            retention = f1 / specialist_f1 if specialist_f1 else 0.0
            rows.append({
                "backbone": bb,
                "method": method,
                "tcga_mean_cosine": tcga_mean_cos,
                "merged_f1": f1,
                "specialist_f1": specialist_f1 or float("nan"),
                "f1_retention": retention,
            })

    if not rows:
        print("No data for TCGA scatter plot")
        return

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "tcga_cosine_vs_f1.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for bb in BACKBONES:
        sub = df[df["backbone"] == bb]
        if len(sub) == 0:
            continue
        ax.scatter(
            sub["tcga_mean_cosine"], sub["merged_f1"],
            label=BACKBONE_LABELS[bb], s=80, alpha=0.75, edgecolor="black",
        )
    ax.set_xlabel("Mean cosine similarity (TCGA ↔ {ISIC, CheXpert, NIH})", fontsize=11)
    ax.set_ylabel("Merged-model TCGA F1", fontsize=11)
    ax.set_title("TCGA task-vector isolation predicts F1 collapse", fontsize=12)
    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for bb in BACKBONES:
        sub = df[df["backbone"] == bb]
        if len(sub) == 0:
            continue
        ax.scatter(
            sub["tcga_mean_cosine"], sub["f1_retention"],
            label=BACKBONE_LABELS[bb], s=80, alpha=0.75, edgecolor="black",
        )
    ax.axhline(1.0, color="green", linestyle="--", alpha=0.5, label="No degradation")
    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Mean cosine similarity (TCGA ↔ others)", fontsize=11)
    ax.set_ylabel("F1 retention (merged / specialist)", fontsize=11)
    ax.set_title("Higher cosine sim → better F1 retention", fontsize=12)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = OUT_DIR / "cosine_vs_tcga_f1.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")
    plt.close(fig)

    try:
        from scipy.stats import spearmanr
        rho, p = spearmanr(df["tcga_mean_cosine"], df["merged_f1"])
        rho_ret, p_ret = spearmanr(df["tcga_mean_cosine"], df["f1_retention"])
        print(f"Spearman(cos, merged_f1)    = {rho:.3f}  (p={p:.3g})")
        print(f"Spearman(cos, f1_retention) = {rho_ret:.3f}  (p={p_ret:.3g})")
    except ImportError:
        pass


if __name__ == "__main__":
    print("=== Cosine similarity heatmaps ===")
    plot_cosine_heatmaps()
    print()
    print("=== TCGA cosine vs F1 scatter ===")
    plot_cosine_vs_tcga_f1()
