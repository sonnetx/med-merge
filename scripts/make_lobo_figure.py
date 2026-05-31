"""Regenerate lobo_predictions.png with one panel per seed. Reads
outputs/_figures/lobo_grid.csv. Each panel pools all held-out-backbone LOBO
predictions for that seed, colored by held-out backbone, with y=x diagonal
and train-mean line.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from predict_forgetting_lobo import BACKBONES, lobo_score

GRID_CSV = Path("outputs/_figures/lobo_grid.csv")
OUT_PNG = Path("outputs/_figures/lobo_predictions.png")

# Distinct colors for the 7 backbones — colorblind-friendly tab palette
BACKBONE_COLORS = {
    "clip":     "#1f77b4",
    "vit":      "#ff7f0e",
    "dinov3":   "#2ca02c",
    "rad_dino": "#d62728",
    "dinov2":   "#9467bd",
    "mae":      "#8c564b",
    "beit":     "#e377c2",
}


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = pd.Series(a).rank().values
    rb = pd.Series(b).rank().values
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = float((np.sqrt((ra ** 2).sum()) * np.sqrt((rb ** 2).sum())))
    return float((ra * rb).sum() / denom) if denom else float("nan")


def main():
    if not GRID_CSV.exists():
        print(f"Missing {GRID_CSV}. Run predict_forgetting_lobo.py first.")
        return
    df = pd.read_csv(GRID_CSV)
    seeds = sorted(df["seed"].unique().tolist())

    fig, axes = plt.subplots(len(seeds), 1, figsize=(9, 4 * len(seeds)),
                              sharex=True, sharey=True)
    if len(seeds) == 1:
        axes = [axes]

    for ax, s in zip(axes, seeds):
        df_s = df[df["seed"] == s].copy()
        # Run LOBO on this seed's grid; reuse the canonical implementation
        r2, mae, rho, preds = lobo_score(df_s, ["mean_cos"], include_dummies=True)
        y = df_s["forgetting"].values

        for bb in BACKBONES:
            mask = (df_s["backbone"].values == bb)
            if mask.sum() == 0:
                continue
            valid = ~np.isnan(preds[mask])
            if not valid.any():
                continue
            ax.scatter(
                y[mask][valid], preds[mask][valid],
                s=40, alpha=0.7,
                color=BACKBONE_COLORS.get(bb, "gray"),
                edgecolor="black", linewidth=0.4,
                label=bb,
            )

        # Reference lines
        all_x = y[~np.isnan(preds)]
        all_y = preds[~np.isnan(preds)]
        if len(all_x) == 0:
            continue
        lo = float(min(all_x.min(), all_y.min())) - 0.05
        hi = float(max(all_x.max(), all_y.max())) + 0.05
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, label="y = x")
        train_mean = float(df_s["forgetting"].mean())
        ax.axhline(train_mean, color="gray", linestyle=":", alpha=0.5,
                   label=f"global mean={train_mean:.2f}")

        ax.set_title(f"seed {s}: aggregate $R^2$={r2:.3f}, "
                     f"$\\rho$={rho:.3f}, MAE={mae:.3f} "
                     f"(N={(~np.isnan(preds)).sum()} cells)",
                     fontsize=11)
        ax.set_ylabel("predicted forgetting (LOBO)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=7, ncol=3, framealpha=0.85)

    axes[-1].set_xlabel("actual forgetting (oracle $-$ merged)")
    fig.suptitle(
        "LOBO predictions of merge forgetting from cosine + dataset/method dummies\n"
        "(faceted by seed; same regression specification across panels)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PNG}")


def make_hook_figure():
    """Two-panel conceptual hook figure:
    Left: within-family cosine vs forgetting (single backbone, scatter shows ρ~0.7-0.8).
    Right: cross-backbone LOBO predictions vs truth (all 7 backbones, seed 42).
    Caption tells the paper's whole story in one image.
    """
    if not GRID_CSV.exists():
        print("Skipping hook figure (no grid CSV)")
        return
    df = pd.read_csv(GRID_CSV)
    df42 = df[df["seed"] == 42].copy()
    if df42.empty:
        return

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.5))

    # LEFT: within-family scatter, pick the backbone with the most cells available
    counts = df42.groupby("backbone").size()
    pick_bb = counts.idxmax()
    sub = df42[df42["backbone"] == pick_bb]
    # Aggregate per dataset: mean forgetting across methods, mean cosine
    agg = sub.groupby("dataset").agg(
        forgetting=("forgetting", "mean"),
        mean_cos=("mean_cos", "first"),
    ).reset_index()
    axL.scatter(agg["mean_cos"], agg["forgetting"],
                s=120, c=axL._get_lines.get_next_color(),
                edgecolor="black", linewidth=0.8)
    for _, r in agg.iterrows():
        axL.annotate(r["dataset"], (r["mean_cos"], r["forgetting"]),
                     fontsize=8, xytext=(6, 4), textcoords="offset points")
    rho = _spearman_local(agg["mean_cos"].values, agg["forgetting"].values)
    axL.set_xlabel("mean cosine to other tasks")
    axL.set_ylabel("mean forgetting across methods")
    axL.set_title(f"Within a single backbone ({pick_bb}, seed 42):\n"
                  f"cosine tracks forgetting (Spearman $\\rho$={rho:.2f}, $n$={len(agg)} cells)",
                  fontsize=10)
    axL.grid(True, alpha=0.3)

    # RIGHT: cross-backbone LOBO scatter at seed 42 (all 7 backbones overlaid)
    r2, mae, rho2, preds = lobo_score(df42, ["mean_cos"], include_dummies=True)
    y = df42["forgetting"].values
    for bb in BACKBONES:
        mask = (df42["backbone"].values == bb)
        if mask.sum() == 0:
            continue
        valid = ~np.isnan(preds[mask])
        axR.scatter(y[mask][valid], preds[mask][valid], s=30, alpha=0.65,
                    color=BACKBONE_COLORS.get(bb, "gray"),
                    edgecolor="black", linewidth=0.3, label=bb)
    valid_all = ~np.isnan(preds)
    if valid_all.any():
        lo = float(min(y[valid_all].min(), preds[valid_all].min())) - 0.05
        hi = float(max(y[valid_all].max(), preds[valid_all].max())) + 0.05
        axR.plot([lo, hi], [lo, hi], "k--", alpha=0.4, label="y = x")
    axR.set_xlabel("actual forgetting (oracle $-$ merged)")
    axR.set_ylabel("predicted forgetting (LOBO)")
    axR.set_title(f"Across backbones (LOBO, seed 42):\n"
                  f"the signal does not generalize ($R^2$={r2:.2f}, $\\rho$={rho2:.2f})",
                  fontsize=10)
    axR.grid(True, alpha=0.3)
    axR.legend(loc="best", fontsize=6, ncol=2, framealpha=0.85)

    fig.suptitle("Within-family cosine signal does not generalize across backbones",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = Path("outputs/_figures/hook_figure.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def _spearman_local(a, b):
    ra = pd.Series(a).rank().values
    rb = pd.Series(b).rank().values
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = float((np.sqrt((ra ** 2).sum()) * np.sqrt((rb ** 2).sum())))
    return float((ra * rb).sum() / denom) if denom else float("nan")


if __name__ == "__main__":
    main()
    make_hook_figure()
