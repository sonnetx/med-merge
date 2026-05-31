"""Regenerate the two paper figures with seeds pooled (no seed-42 callout).

Reads `CS_229_Project_Milestone/lobo_grid.csv` and writes both PNGs alongside it:
    lobo_predictions.png       (single-panel LOBO scatter, all 3 seeds pooled)
    norm_vs_cos_scatter.png    (42 (backbone, dataset) cells, all-seed average)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = REPO_ROOT / "CS_229_Project_Milestone"
GRID_CSV = PAPER_DIR / "lobo_grid.csv"

BACKBONES = ["clip", "vit", "dinov3", "rad_dino", "dinov2", "mae", "beit"]
BACKBONE_COLORS = {
    "clip":     "#1f77b4",
    "vit":      "#ff7f0e",
    "dinov3":   "#2ca02c",
    "rad_dino": "#d62728",
    "dinov2":   "#9467bd",
    "mae":      "#8c564b",
    "beit":     "#e377c2",
}

DATASET_COLORS = {
    "isic2017":     "#1f77b4",
    "chexpert":     "#ff7f0e",
    "tcga":         "#2ca02c",
    "nih_cxr":      "#d62728",
    "pathmnist":    "#9467bd",
    "retinamnist":  "#8c564b",
}

DATASET_LABEL = {
    "isic2017":    "ISIC",
    "chexpert":    "CheXpert",
    "tcga":        "TCGA",
    "nih_cxr":     "NIH",
    "pathmnist":   "PathMNIST",
    "retinamnist": "RetinaMNIST",
}


def spearman(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ra = pd.Series(a).rank().values
    rb = pd.Series(b).rank().values
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = float(np.sqrt((ra ** 2).sum()) * np.sqrt((rb ** 2).sum()))
    return float((ra * rb).sum() / denom) if denom else float("nan")


def lobo_predictions(df, feature_cols=("mean_cos",), include_dummies=True):
    """Leave-one-backbone-out OLS predictions of `forgetting`.

    Returns an array of predicted forgetting aligned to df.index, with NaN for
    rows whose held-out backbone had no training data left.
    """
    df = df.reset_index(drop=True).copy()
    backbones_present = df["backbone"].unique().tolist()
    feat_mat = df[list(feature_cols)].astype(float).values
    if include_dummies:
        ds_dum = pd.get_dummies(df["dataset"], drop_first=True, dtype=float).values
        mt_dum = pd.get_dummies(df["method"], drop_first=True, dtype=float).values
        X = np.hstack([feat_mat, ds_dum, mt_dum])
    else:
        X = feat_mat
    X = np.hstack([np.ones((len(X), 1)), X])
    y = df["forgetting"].astype(float).values
    preds = np.full(len(df), np.nan)
    for held in backbones_present:
        train_mask = (df["backbone"].values != held)
        test_mask  = ~train_mask
        if train_mask.sum() < X.shape[1] + 2:
            continue
        Xt = X[train_mask]; yt = y[train_mask]
        try:
            beta, *_ = np.linalg.lstsq(Xt, yt, rcond=None)
        except np.linalg.LinAlgError:
            continue
        preds[test_mask] = X[test_mask] @ beta
    return preds


def fig_lobo(df, out_path):
    """Single-panel pooled-across-seeds LOBO scatter."""
    df = df.dropna(subset=["forgetting", "mean_cos"]).copy()
    preds = lobo_predictions(df)
    valid = ~np.isnan(preds)
    y = df["forgetting"].values
    n = int(valid.sum())
    r2 = 1.0 - ((y[valid] - preds[valid]) ** 2).sum() / ((y[valid] - y[valid].mean()) ** 2).sum()
    rho = spearman(y[valid], preds[valid])
    mae = float(np.abs(y[valid] - preds[valid]).mean())

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for bb in BACKBONES:
        m = (df["backbone"].values == bb) & valid
        if not m.any():
            continue
        ax.scatter(y[m], preds[m], s=24, alpha=0.55,
                   color=BACKBONE_COLORS.get(bb, "gray"),
                   edgecolor="black", linewidth=0.3, label=bb)
    lo = float(min(y[valid].min(), preds[valid].min())) - 0.05
    hi = float(max(y[valid].max(), preds[valid].max())) + 0.05
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4)
    ax.axhline(float(y.mean()), color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("actual forgetting (oracle $-$ merged)")
    ax.set_ylabel("predicted forgetting (LOBO)")
    ax.set_title(f"$R^2$={r2:.3f}, Spearman $\\rho$={rho:.3f}, MAE={mae:.3f}, $N$={n}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=7, ncol=4, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}  (R2={r2:.3f}, rho={rho:.3f}, MAE={mae:.3f}, N={n})")


def fig_norm_vs_cos(df, out_path):
    """Seed-pooled (backbone, dataset) scatter in the (||tau||, 1-cos) plane.

    Marker size encodes mean forgetting across methods x seeds.
    """
    agg = (
        df.groupby(["backbone", "dataset"])
          .agg(norm=("norm", "mean"),
               mean_cos=("mean_cos", "mean"),
               forgetting=("forgetting", "mean"))
          .reset_index()
    )
    agg["one_minus_cos"] = 1.0 - agg["mean_cos"]

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for ds, sub in agg.groupby("dataset"):
        ax.scatter(sub["norm"], sub["one_minus_cos"],
                   s=80 + sub["forgetting"] * 600,
                   color=DATASET_COLORS.get(ds, "gray"),
                   alpha=0.55, edgecolor="black", linewidth=0.4,
                   label=DATASET_LABEL.get(ds, ds))
        for _, r in sub.iterrows():
            ax.annotate(r["backbone"], (r["norm"], r["one_minus_cos"]),
                        fontsize=6, xytext=(4, 3), textcoords="offset points",
                        color="black", alpha=0.7)
    ax.set_xlabel(r"task-vector norm $\|\tau\|$")
    ax.set_ylabel(r"mean misalignment to other tasks $1-\cos(\tau_i, \tau_j)$")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=7, ncol=2, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}  (N={len(agg)} cells, seed-pooled)")


def main(datasets=None):
    df = pd.read_csv(GRID_CSV)
    print(f"Loaded {GRID_CSV} ({len(df)} rows, seeds={sorted(df['seed'].unique().tolist())})")
    if datasets is not None:
        df = df[df["dataset"].isin(datasets)].copy()
        print(f"Filtered to datasets {datasets}: {len(df)} rows")
    fig_lobo(df, PAPER_DIR / "lobo_predictions.png")
    fig_norm_vs_cos(df, PAPER_DIR / "norm_vs_cos_scatter.png")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=None,
                   help="Subset of datasets to include (default: all in lobo_grid.csv)")
    args = p.parse_args()
    main(args.datasets)
