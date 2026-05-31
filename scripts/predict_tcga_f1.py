"""Leave-one-out linear regression of merged-model TCGA F1 on TCGA task-vector
cosine similarity + method indicators, across (backbone, method) cells.
Writes tcga_loo_predictions.csv and tcga_loo_predictions.png under
outputs/_figures/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

BACKBONES = ["clip", "vit", "dinov3", "rad_dino"]
DATASETS = ["isic2017", "chexpert", "tcga", "nih_cxr"]
METHODS = ["simple_avg", "task_arithmetic", "ties", "dare", "dare_ties",
           "pcb_merging", "lines", "fisher"]
SEED = 42

OUT_DIR = Path("outputs/_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_cosine_matrix(backbone: str) -> pd.DataFrame | None:
    path = Path(f"outputs/{backbone}/seed_{SEED}/analysis/task_vector_cosine.csv")
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0)


def main():
    # Build the data table
    rows = []
    for bb in BACKBONES:
        mat = load_cosine_matrix(bb)
        if mat is None or "tcga" not in mat.index:
            continue
        others = [d for d in mat.index if d != "tcga"]
        tcga_mean_cos = float(mat.loc["tcga", others].mean())
        tcga_min_cos = float(mat.loc["tcga", others].min())

        for method in METHODS:
            f = Path(f"outputs/{bb}/seed_{SEED}/results/{method}/results.json")
            if not f.exists():
                continue
            r = json.loads(f.read_text())
            if "tcga" not in r:
                continue
            tcga_metrics = r["tcga"]
            f1 = tcga_metrics.get("f1")
            auroc = tcga_metrics.get("auroc")
            if f1 is None:
                continue
            rows.append({
                "backbone": bb,
                "method": method,
                "tcga_mean_cos": tcga_mean_cos,
                "tcga_min_cos": tcga_min_cos,
                "f1": f1,
                "auroc": auroc,
            })

    df = pd.DataFrame(rows)
    if len(df) < 5:
        print(f"Only {len(df)} samples — need more data for prediction")
        return

    print(f"Data: {len(df)} (backbone, method) points on TCGA")
    print(df.describe()[["tcga_mean_cos", "f1", "auroc"]].round(3))

    # Features: cosine sim + one-hot method indicator
    X_base = df[["tcga_mean_cos"]].values  # (N, 1)
    method_dummies = pd.get_dummies(df["method"], prefix="m", drop_first=True).values
    X = np.concatenate([X_base, method_dummies], axis=1)
    y = df["f1"].values

    # Leave-one-out predictions
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import LeaveOneOut
    loo = LeaveOneOut()
    preds = np.zeros_like(y)
    for train_idx, test_idx in loo.split(X):
        reg = LinearRegression()
        reg.fit(X[train_idx], y[train_idx])
        preds[test_idx] = reg.predict(X[test_idx])

    residuals = y - preds
    # R^2 of LOO predictions
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2_loo = 1.0 - ss_res / ss_tot
    mae = float(np.mean(np.abs(residuals)))

    # Compare to cosine-only baseline
    reg_simple = LinearRegression().fit(X_base, y)
    preds_simple = np.zeros_like(y)
    for train_idx, test_idx in loo.split(X_base):
        r2 = LinearRegression().fit(X_base[train_idx], y[train_idx])
        preds_simple[test_idx] = r2.predict(X_base[test_idx])
    ss_res_s = float(np.sum((y - preds_simple) ** 2))
    r2_simple = 1.0 - ss_res_s / ss_tot
    mae_simple = float(np.mean(np.abs(y - preds_simple)))

    print(f"\nLeave-one-out prediction performance:")
    print(f"  cosine-only model:   R^2 = {r2_simple:.3f}, MAE = {mae_simple:.3f}")
    print(f"  cosine + method-id:  R^2 = {r2_loo:.3f}, MAE = {mae:.3f}")

    # Scatter plot: predicted vs actual
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, p, lab, r2v, maev in [
        (axes[0], preds_simple, "cosine only", r2_simple, mae_simple),
        (axes[1], preds, "cosine + method id", r2_loo, mae),
    ]:
        for bb in BACKBONES:
            mask = df["backbone"].values == bb
            ax.scatter(y[mask], p[mask], label=bb, s=70, alpha=0.75,
                       edgecolor="black")
        lo = min(y.min(), p.min()) - 0.05
        hi = max(y.max(), p.max()) + 0.05
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="y = x")
        ax.set_xlabel("Actual TCGA F1")
        ax.set_ylabel("Predicted TCGA F1 (LOO)")
        ax.set_title(f"{lab}: R²={r2v:.3f}, MAE={maev:.3f}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    fig.suptitle("Leave-one-out prediction of merged TCGA F1 from cosine sim",
                 fontsize=13)
    fig.tight_layout()
    out = OUT_DIR / "tcga_loo_predictions.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")
    plt.close(fig)

    df["pred_cos_only"] = preds_simple
    df["pred_cos_method"] = preds
    df["residual_cos_method"] = residuals
    df.to_csv(OUT_DIR / "tcga_loo_predictions.csv", index=False)
    print(f"Wrote {OUT_DIR / 'tcga_loo_predictions.csv'}")


if __name__ == "__main__":
    main()
