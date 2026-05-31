"""3-domain robustness check: collapse CheXpert + NIH into one 'radiology'
cluster (since they share label set and modality), re-run LOBO predictor at
3-domain granularity, and report within- vs cross-cluster cosines. Writes
three_domain_table.csv and within_vs_cross_cluster_cosines.csv under
outputs/_figures/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from med_merge.config.constants import ALL_METHODS, PRIMARY_METRICS, SEEDS

BACKBONES = ["clip", "vit", "dinov3", "rad_dino", "dinov2", "mae", "beit"]
DOMAIN_MAP = {
    "isic2017": "dermoscopy",
    "chexpert": "radiology",
    "nih_cxr": "radiology",
    "tcga": "histopathology",
    "pathmnist": "histopathology",
    "retinamnist": "fundus",
}
DOMAINS = ["dermoscopy", "radiology", "histopathology", "fundus"]
OUT_DIR = Path("outputs/_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_cos_matrix(bb: str, seed: int) -> pd.DataFrame | None:
    p = Path(f"outputs/{bb}/seed_{seed}/analysis/task_vector_cosine.csv")
    if not p.exists():
        return None
    return pd.read_csv(p, index_col=0)


def load_oracle(bb: str, seed: int, ds: str) -> float | None:
    p = Path(f"outputs/{bb}/seed_{seed}/checkpoints/{ds}/best_metrics.json")
    if not p.exists():
        return None
    return json.loads(p.read_text()).get(PRIMARY_METRICS[ds])


def load_merged(bb: str, seed: int, method: str, ds: str) -> float | None:
    p = Path(f"outputs/{bb}/seed_{seed}/results/{method}/results.json")
    if not p.exists():
        return None
    return json.loads(p.read_text()).get(ds, {}).get(PRIMARY_METRICS[ds])


def cluster_cosines(cos: pd.DataFrame) -> tuple[float, float]:
    """Return (within-radiology cos, mean cross-cluster cos)."""
    if "chexpert" not in cos.index or "nih_cxr" not in cos.index:
        return float("nan"), float("nan")
    within = float(cos.loc["chexpert", "nih_cxr"])
    cross = []
    for rad in ["chexpert", "nih_cxr"]:
        for other in ["isic2017", "tcga"]:
            if other in cos.index:
                cross.append(float(cos.loc[rad, other]))
    return within, float(np.mean(cross)) if cross else float("nan")


def assemble_3domain(seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    cluster_rows = []
    domain_rows = []
    for bb in BACKBONES:
        cos = load_cos_matrix(bb, seed)
        if cos is None:
            continue
        within, cross = cluster_cosines(cos)
        cluster_rows.append({"backbone": bb, "within_radiology_cos": within,
                             "cross_cluster_cos": cross})

        # Per-domain forgetting averaged across methods
        for domain in DOMAINS:
            ds_in_domain = [d for d, dom in DOMAIN_MAP.items() if dom == domain]
            for method in ALL_METHODS:
                drops = []
                for ds in ds_in_domain:
                    oracle = load_oracle(bb, seed, ds)
                    merged = load_merged(bb, seed, method, ds)
                    if oracle is not None and merged is not None:
                        drops.append(oracle - merged)
                if not drops:
                    continue
                # Per-domain cosine: mean cosine from any task vector in the
                # domain to any task vector outside it
                domain_cos = []
                for ds in ds_in_domain:
                    if ds not in cos.index:
                        continue
                    for other in cos.index:
                        if other != ds and DOMAIN_MAP.get(other) != domain:
                            domain_cos.append(float(cos.loc[ds, other]))
                if not domain_cos:
                    continue
                domain_rows.append({
                    "backbone": bb,
                    "domain": domain,
                    "method": method,
                    "mean_cos_to_other_domains": float(np.mean(domain_cos)),
                    "forgetting": float(np.mean(drops)),
                })
    return pd.DataFrame(cluster_rows), pd.DataFrame(domain_rows)


def lobo_score(df: pd.DataFrame, feature_cols: list[str], include_dummies: bool):
    from sklearn.linear_model import LinearRegression
    y = df["forgetting"].values
    if not feature_cols and not include_dummies:
        preds = np.full_like(y, y.mean(), dtype=float)
    else:
        parts = []
        if feature_cols:
            parts.append(df[feature_cols].astype(float))
        if include_dummies:
            parts.append(pd.get_dummies(df["method"], prefix="m", drop_first=True))
            parts.append(pd.get_dummies(df["domain"], prefix="d", drop_first=True))
        X = pd.concat(parts, axis=1).values
        preds = np.full_like(y, fill_value=np.nan, dtype=float)
        for held in df["backbone"].unique():
            train_mask = (df["backbone"] != held).values
            test_mask = ~train_mask
            if test_mask.sum() == 0 or train_mask.sum() == 0:
                continue
            reg = LinearRegression().fit(X[train_mask], y[train_mask])
            preds[test_mask] = reg.predict(X[test_mask])
    valid = ~np.isnan(preds)
    y_v, p_v = y[valid], preds[valid]
    ss_res = float(((y_v - p_v) ** 2).sum())
    ss_tot = float(((y_v - y_v.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    mae = float(np.abs(y_v - p_v).mean())
    rho = float(pd.Series(y_v).corr(pd.Series(p_v), method="spearman"))
    return r2, mae, rho


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    args = p.parse_args()

    all_cluster = []
    summary_rows = []
    for seed in args.seeds:
        cluster, domain = assemble_3domain(seed)
        if cluster.empty:
            continue
        all_cluster.append(cluster.assign(seed=seed))

        # LOBO with 3 domains x 4 backbones x 8 methods = 96 rows (full grid)
        for name, feats, dummies in [
            ("mean_of_train baseline", [], False),
            ("dummies only", [], True),
            ("cosine + dummies", ["mean_cos_to_other_domains"], True),
        ]:
            r2, mae, rho = lobo_score(domain, feats, dummies)
            summary_rows.append({
                "seed": seed,
                "predictor": name,
                "R2": r2,
                "MAE": mae,
                "Spearman": rho,
            })

    cluster_df = pd.concat(all_cluster, ignore_index=True) if all_cluster else pd.DataFrame()
    if not cluster_df.empty:
        print("\n=== Within-radiology vs cross-cluster task-vector cosines ===")
        print(cluster_df.round(3).to_string(index=False))
        cluster_df.to_csv(OUT_DIR / "within_vs_cross_cluster_cosines.csv", index=False)
        print(f"Wrote {OUT_DIR / 'within_vs_cross_cluster_cosines.csv'}")
        print(f"\nMean within-radiology cosine: {cluster_df['within_radiology_cos'].mean():.3f}")
        print(f"Mean cross-cluster cosine:    {cluster_df['cross_cluster_cos'].mean():.3f}")
        print("(If within >> cross, the 2-dataset radiology framing was indeed redundant.)")

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        print("\n=== 3-domain LOBO (per-seed) ===")
        print(summary.round(3).to_string(index=False))
        agg = summary.groupby("predictor").agg(
            R2_median=("R2", "median"),
            R2_min=("R2", "min"),
            R2_max=("R2", "max"),
            Spearman_median=("Spearman", "median"),
        ).reset_index()
        print("\n=== 3-domain LOBO (median across seeds) ===")
        print(agg.round(3).to_string(index=False))
        summary.to_csv(OUT_DIR / "three_domain_table.csv", index=False)
        agg.to_csv(OUT_DIR / "three_domain_summary.csv", index=False)
        print(f"Wrote {OUT_DIR / 'three_domain_table.csv'}")
        print(f"Wrote {OUT_DIR / 'three_domain_summary.csv'}")


if __name__ == "__main__":
    main()
