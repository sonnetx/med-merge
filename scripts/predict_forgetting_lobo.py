"""Leave-one-backbone-out (LOBO) prediction of merge forgetting from
task-vector geometry. Plain leave-one-point-out leaks via the dataset and
method dummies (which are identical across backbones), so LOBO holds out the
whole backbone instead.

Writes feature_ablation.csv, lobo_results_table.csv, lobo_predictions.png,
lobo_results_multiseed.csv, lobo_jackknife.csv, and lobo_permutation.csv
under outputs/_figures/.
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

from med_merge.config.constants import ALL_DATASETS, ALL_METHODS, PRIMARY_METRICS, SEEDS

BACKBONES = ["clip", "vit", "dinov3", "rad_dino", "dinov2", "mae", "beit"]

OUT_DIR = Path("outputs/_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_geometry(bb: str, seed: int) -> pd.DataFrame | None:
    p = Path(f"outputs/{bb}/seed_{seed}/analysis/geometry_features.csv")
    if not p.exists():
        return None
    return pd.read_csv(p)


def load_oracle(bb: str, seed: int, dataset: str) -> float | None:
    p = Path(f"outputs/{bb}/seed_{seed}/checkpoints/{dataset}/best_metrics.json")
    if not p.exists():
        return None
    m = json.loads(p.read_text())
    return m.get(PRIMARY_METRICS[dataset])


def load_merged(bb: str, seed: int, method: str, dataset: str) -> float | None:
    p = Path(f"outputs/{bb}/seed_{seed}/results/{method}/results.json")
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    return r.get(dataset, {}).get(PRIMARY_METRICS[dataset])


def assemble(seeds: list[int]) -> pd.DataFrame:
    rows = []
    for bb in BACKBONES:
        for s in seeds:
            geom = load_geometry(bb, s)
            if geom is None:
                continue
            geom_by_ds = geom.set_index("dataset")
            for ds in ALL_DATASETS:
                if ds not in geom_by_ds.index:
                    continue
                oracle = load_oracle(bb, s, ds)
                if oracle is None:
                    continue
                gfeat = geom_by_ds.loc[ds]
                for method in ALL_METHODS:
                    merged = load_merged(bb, s, method, ds)
                    if merged is None:
                        continue
                    rows.append({
                        "backbone": bb,
                        "seed": s,
                        "dataset": ds,
                        "method": method,
                        "oracle": oracle,
                        "merged": merged,
                        "forgetting": oracle - merged,
                        "mean_cos": float(gfeat["mean_cos"]),
                        "min_cos": float(gfeat["min_cos"]),
                        "max_cos": float(gfeat["max_cos"]),
                        "norm": float(gfeat["norm"]),
                        "mean_l2": float(gfeat["mean_l2"]),
                        "max_l2": float(gfeat["max_l2"]),
                        "mean_norm_others": float(gfeat["mean_norm_others"]),
                        "interference": float(gfeat["interference"]),
                        "norm_times_one_minus_cos": float(gfeat["norm"]) * (1.0 - float(gfeat["mean_cos"])),
                    })
    return pd.DataFrame(rows)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = pd.Series(a).rank().values
    rb = pd.Series(b).rank().values
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = float((np.sqrt((ra ** 2).sum()) * np.sqrt((rb ** 2).sum())))
    if denom == 0.0:
        return float("nan")
    return float((ra * rb).sum() / denom)


def _fit_predict(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray) -> np.ndarray:
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression().fit(X_tr, y_tr)
    return reg.predict(X_te)


def lobo_score(df: pd.DataFrame, feature_cols: list[str], include_dummies: bool):
    """Return aggregated (R^2, MAE, Spearman) under leave-one-backbone-out."""
    y_all = df["forgetting"].values
    preds = np.full_like(y_all, fill_value=np.nan, dtype=float)

    method_dummies = pd.get_dummies(df["method"], prefix="m", drop_first=True)
    dataset_dummies = pd.get_dummies(df["dataset"], prefix="d", drop_first=True)

    parts = [df[feature_cols].astype(float)]
    if include_dummies:
        parts.extend([method_dummies, dataset_dummies])
    X = pd.concat(parts, axis=1).values

    for held in BACKBONES:
        train_mask = (df["backbone"] != held).values
        test_mask = ~train_mask
        if test_mask.sum() == 0 or train_mask.sum() == 0:
            continue
        preds[test_mask] = _fit_predict(X[train_mask], y_all[train_mask], X[test_mask])

    valid = ~np.isnan(preds)
    y_v = y_all[valid]
    p_v = preds[valid]
    ss_res = float(((y_v - p_v) ** 2).sum())
    ss_tot = float(((y_v - y_v.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    mae = float(np.abs(y_v - p_v).mean())
    rho = _spearman(y_v, p_v)
    return r2, mae, rho, preds


def jackknife_lobo_delta_r2(
    df: pd.DataFrame,
    feature_cols: list[str] = ("mean_cos",),
    include_dummies: bool = True,
):
    """Jackknife-over-backbones CI for the cosine cross-backbone lift.

    Leave one backbone out of the analysis entirely (NOT just the LOBO fold);
    on the remaining backbones run a full LOBO; record dummies-only R^2,
    cosine+dummies R^2, and their difference. Repeat for each backbone.

    The std of the N "delta R^2" estimates is the jackknife SE of the lift.

    Returns dict with per-fold estimates plus the jackknife mean / SE / 95% CI.
    """
    backbones = sorted(df["backbone"].unique().tolist())
    rows = []
    for held in backbones:
        sub = df[df["backbone"] != held].copy()
        if sub["backbone"].nunique() < 2:
            continue
        r2_d, _, rho_d, _ = lobo_score(sub, [], include_dummies=True)
        r2_c, _, rho_c, _ = lobo_score(sub, list(feature_cols),
                                        include_dummies=include_dummies)
        rows.append({
            "excluded_backbone": held,
            "n_train_backbones": sub["backbone"].nunique(),
            "dummies_r2": r2_d,
            "cosine_r2": r2_c,
            "delta_r2": r2_c - r2_d,
            "delta_rho": rho_c - rho_d,
        })
    jk = pd.DataFrame(rows)
    if jk.empty:
        return jk, {}
    n = len(jk)
    # True jackknife SE: sqrt((n-1)/n * sum((theta_i - theta_bar)^2))
    delta_r2 = jk["delta_r2"].values
    delta_rho = jk["delta_rho"].values
    mean_dr2 = float(delta_r2.mean())
    se_dr2 = float(np.sqrt((n - 1) / n * ((delta_r2 - mean_dr2) ** 2).sum()))
    mean_drho = float(delta_rho.mean())
    se_drho = float(np.sqrt((n - 1) / n * ((delta_rho - mean_drho) ** 2).sum()))
    summary = {
        "n_folds": n,
        "delta_r2_mean": mean_dr2,
        "delta_r2_jackknife_se": se_dr2,
        "delta_r2_ci_lo_95": mean_dr2 - 1.96 * se_dr2,
        "delta_r2_ci_hi_95": mean_dr2 + 1.96 * se_dr2,
        "delta_rho_mean": mean_drho,
        "delta_rho_jackknife_se": se_drho,
        "delta_rho_ci_lo_95": mean_drho - 1.96 * se_drho,
        "delta_rho_ci_hi_95": mean_drho + 1.96 * se_drho,
    }
    return jk, summary


def permutation_test_cosine_lift(
    df: pd.DataFrame,
    feature_cols: list[str] = ("mean_cos",),
    n_permutations: int = 1000,
    rng_seed: int = 0,
):
    """Permutation test: is the observed cosine LOBO lift distinguishable
    from random shuffles of the cosine feature?

    For each permutation:
      - Shuffle the cosine column across (backbone, dataset) cells (preserving
        the dummy structure).
      - Refit LOBO with permuted cosine + dummies; compute R^2.
      - Null delta R^2 = (permuted cosine+dummies R^2) - (dummies-only R^2).

    Empirical p-value (two-sided): fraction of |null delta R^2| >= |observed|.
    """
    rng = np.random.default_rng(rng_seed)

    # Observed values
    r2_d_obs, _, rho_d_obs, _ = lobo_score(df, [], include_dummies=True)
    r2_c_obs, _, rho_c_obs, _ = lobo_score(df, list(feature_cols), include_dummies=True)
    obs_dr2 = r2_c_obs - r2_d_obs
    obs_drho = rho_c_obs - rho_d_obs

    null_dr2 = np.zeros(n_permutations)
    null_drho = np.zeros(n_permutations)
    df_perm = df.copy()
    for i in range(n_permutations):
        for col in feature_cols:
            df_perm[col] = rng.permutation(df[col].values)
        r2_c_perm, _, rho_c_perm, _ = lobo_score(df_perm, list(feature_cols),
                                                  include_dummies=True)
        null_dr2[i] = r2_c_perm - r2_d_obs
        null_drho[i] = rho_c_perm - rho_d_obs

    p_r2 = float((np.abs(null_dr2) >= abs(obs_dr2)).mean())
    p_rho = float((np.abs(null_drho) >= abs(obs_drho)).mean())
    summary = {
        "n_permutations": n_permutations,
        "observed_delta_r2": obs_dr2,
        "observed_delta_rho": obs_drho,
        "null_delta_r2_mean": float(null_dr2.mean()),
        "null_delta_r2_std": float(null_dr2.std(ddof=1)),
        "null_delta_r2_p05": float(np.percentile(null_dr2, 5)),
        "null_delta_r2_p95": float(np.percentile(null_dr2, 95)),
        "null_delta_rho_mean": float(null_drho.mean()),
        "null_delta_rho_std": float(null_drho.std(ddof=1)),
        "p_value_two_sided_r2": p_r2,
        "p_value_two_sided_rho": p_rho,
    }
    return summary, null_dr2, null_drho


def loo_score(df: pd.DataFrame, feature_cols: list[str], include_dummies: bool):
    """Plain leave-one-point-out, for the 'leaks' baseline row."""
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import LeaveOneOut

    y_all = df["forgetting"].values

    method_dummies = pd.get_dummies(df["method"], prefix="m", drop_first=True)
    dataset_dummies = pd.get_dummies(df["dataset"], prefix="d", drop_first=True)
    parts = [df[feature_cols].astype(float)]
    if include_dummies:
        parts.extend([method_dummies, dataset_dummies])
    X = pd.concat(parts, axis=1).values

    preds = np.zeros_like(y_all, dtype=float)
    for tr, te in LeaveOneOut().split(X):
        reg = LinearRegression().fit(X[tr], y_all[tr])
        preds[te] = reg.predict(X[te])

    ss_res = float(((y_all - preds) ** 2).sum())
    ss_tot = float(((y_all - y_all.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    mae = float(np.abs(y_all - preds).mean())
    rho = _spearman(y_all, preds)
    return r2, mae, rho, preds


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    args = p.parse_args()

    df = assemble(args.seeds)
    if df.empty:
        print("No data. Did you run scripts/compute_geometry.py?")
        return
    print(f"Assembled {len(df)} (backbone, seed, dataset, method) rows "
          f"across seeds={sorted(df['seed'].unique().tolist())}")
    df_head = df[df["seed"] == 42].copy() if 42 in df["seed"].values else df.copy()
    print(f"Headline grid (seed 42): {len(df_head)} rows")

    feature_sets = [
        ("mean_of_train (baseline)", []),
        ("cosine only", ["mean_cos"]),
        ("norm only", ["norm"]),
        ("interference (theory term)", ["interference"]),
        ("norm * (1 - cos)", ["norm_times_one_minus_cos"]),
        ("cosine + dummies", ["mean_cos"]),
        ("interference + dummies", ["interference"]),
        ("dummies only (no geom)", []),
        ("cosine + norm + dummies", ["mean_cos", "norm"]),
        ("all geom + dummies", ["mean_cos", "min_cos", "norm", "mean_l2", "interference"]),
    ]
    include_dummies_for = {
        "cosine + dummies", "interference + dummies", "dummies only (no geom)",
        "cosine + norm + dummies", "all geom + dummies",
    }

    rows = []
    for name, feats in feature_sets:
        if not feats and name != "dummies only (no geom)" and name != "mean_of_train (baseline)":
            continue
        if name == "mean_of_train (baseline)":
            y = df_head["forgetting"].values
            preds = np.full_like(y, y.mean())
            ss_res = float(((y - preds) ** 2).sum())
            ss_tot = float(((y - y.mean()) ** 2).sum())
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            mae = float(np.abs(y - preds).mean())
            rho = float("nan")
        else:
            r2, mae, rho, _ = lobo_score(df_head, feats, include_dummies=(name in include_dummies_for))
        rows.append({"features": name, "lobo_r2": r2, "lobo_mae": mae, "lobo_spearman": rho})
    table = pd.DataFrame(rows)
    print("\n=== LOBO feature ablation (seed 42) ===")
    print(table.round(3).to_string(index=False))
    table.to_csv(OUT_DIR / "feature_ablation.csv", index=False)
    print(f"Wrote {OUT_DIR / 'feature_ablation.csv'}")

    leak_r2, leak_mae, leak_rho, _ = loo_score(df_head, ["mean_cos"], include_dummies=True)
    lobo_r2, lobo_mae, lobo_rho, lobo_preds = lobo_score(df_head, ["mean_cos"], include_dummies=True)
    best_row = table.iloc[table["lobo_r2"].idxmax()]
    print("\n=== Headline comparison ===")
    head = pd.DataFrame([
        {"predictor": "naive LOO (leaks via dummies)", "R2": leak_r2, "MAE": leak_mae, "Spearman": leak_rho},
        {"predictor": "LOBO: cosine + dummies",         "R2": lobo_r2, "MAE": lobo_mae, "Spearman": lobo_rho},
        {"predictor": f"LOBO: {best_row['features']} (best)",
         "R2": best_row["lobo_r2"], "MAE": best_row["lobo_mae"], "Spearman": best_row["lobo_spearman"]},
    ])
    print(head.round(3).to_string(index=False))
    head.to_csv(OUT_DIR / "lobo_results_table.csv", index=False)
    print(f"Wrote {OUT_DIR / 'lobo_results_table.csv'}")

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharex=True, sharey=True)
    y_head = df_head["forgetting"].values
    for ax, held in zip(axes, BACKBONES):
        mask = (df_head["backbone"] == held).values
        ax.scatter(y_head[mask], lobo_preds[mask], s=60, alpha=0.75,
                   edgecolor="black", label=held)
        all_x = y_head
        lo = float(min(all_x.min(), lobo_preds.min())) - 0.05
        hi = float(max(all_x.max(), lobo_preds.max())) + 0.05
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, label="y = x")
        train_mean = df_head.loc[~mask, "forgetting"].mean()
        ax.axhline(train_mean, color="gray", linestyle=":", alpha=0.5,
                   label=f"train mean={train_mean:.2f}")
        ax.set_title(f"held-out: {held}\n(R^2 below is on this fold only)")
        ax.set_xlabel("actual forgetting (oracle - merged)")
        ax.set_ylabel("predicted forgetting (LOBO)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle("LOBO predictions of merge forgetting from task-vector geometry "
                 f"(cosine + dummies; aggregate R^2={lobo_r2:.3f}, "
                 f"rho={lobo_rho:.3f})", fontsize=12)
    fig.tight_layout()
    fig_path = OUT_DIR / "lobo_predictions.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {fig_path}")

    df.to_csv(OUT_DIR / "lobo_grid.csv", index=False)
    print(f"Wrote {OUT_DIR / 'lobo_grid.csv'}")

    multi_seed_rows = []
    for s in sorted(df["seed"].unique().tolist()):
        df_s = df[df["seed"] == s].copy()
        if len(df_s) == 0:
            continue
        r2_d, mae_d, rho_d, _ = lobo_score(df_s, [], include_dummies=True)
        r2_c, mae_c, rho_c, _ = lobo_score(df_s, ["mean_cos"], include_dummies=True)
        multi_seed_rows.extend([
            {"seed": s, "predictor": "dummies only",         "R2": r2_d, "MAE": mae_d, "Spearman": rho_d, "n_rows": len(df_s)},
            {"seed": s, "predictor": "cosine + dummies",     "R2": r2_c, "MAE": mae_c, "Spearman": rho_c, "n_rows": len(df_s)},
            {"seed": s, "predictor": "cosine ΔR2 over dummies",
             "R2": r2_c - r2_d, "MAE": mae_d - mae_c, "Spearman": rho_c - rho_d, "n_rows": len(df_s)},
        ])
    ms = pd.DataFrame(multi_seed_rows)
    print("\n=== Multi-seed LOBO (per-seed) ===")
    print(ms.round(3).to_string(index=False))
    ms.to_csv(OUT_DIR / "lobo_results_multiseed.csv", index=False)
    print(f"Wrote {OUT_DIR / 'lobo_results_multiseed.csv'}")

    if not ms.empty:
        agg = ms.groupby("predictor").agg(
            R2_median=("R2", "median"),
            R2_min=("R2", "min"),
            R2_max=("R2", "max"),
            MAE_median=("MAE", "median"),
            Spearman_median=("Spearman", "median"),
            Spearman_min=("Spearman", "min"),
            Spearman_max=("Spearman", "max"),
            n_seeds=("seed", "nunique"),
        ).reset_index()
        print("\n=== LOBO summary across seeds (median, min, max) ===")
        print(agg.round(3).to_string(index=False))
        agg.to_csv(OUT_DIR / "lobo_results_seed_summary.csv", index=False)
        print(f"Wrote {OUT_DIR / 'lobo_results_seed_summary.csv'}")

    print("\n=== Jackknife-over-backbones CI on cosine LOBO lift ===")
    jk_rows = []
    jk_summary_rows = []
    for s in sorted(df["seed"].unique().tolist()):
        df_s = df[df["seed"] == s].copy()
        if df_s["backbone"].nunique() < 3:
            continue
        jk, summary = jackknife_lobo_delta_r2(df_s)
        jk["seed"] = s
        jk_rows.append(jk)
        if summary:
            summary_with_seed = {"seed": s, **summary}
            jk_summary_rows.append(summary_with_seed)
            print(f"  seed {s}: ΔR² mean={summary['delta_r2_mean']:+.3f} "
                  f"jackknife SE={summary['delta_r2_jackknife_se']:.3f} "
                  f"[95% CI {summary['delta_r2_ci_lo_95']:+.3f}, "
                  f"{summary['delta_r2_ci_hi_95']:+.3f}]; "
                  f"Δρ mean={summary['delta_rho_mean']:+.3f} "
                  f"SE={summary['delta_rho_jackknife_se']:.3f}")
    if jk_rows:
        jk_all = pd.concat(jk_rows, ignore_index=True)
        jk_all.to_csv(OUT_DIR / "lobo_jackknife.csv", index=False)
        pd.DataFrame(jk_summary_rows).to_csv(OUT_DIR / "lobo_jackknife_summary.csv", index=False)
        print(f"Wrote {OUT_DIR / 'lobo_jackknife.csv'}")
        print(f"Wrote {OUT_DIR / 'lobo_jackknife_summary.csv'}")

    n_perm = int(__import__("os").environ.get("N_PERM", "1000"))
    print(f"\n=== Permutation test (n={n_perm}) on cosine LOBO lift ===")
    perm_rows = []
    for s in sorted(df["seed"].unique().tolist()):
        df_s = df[df["seed"] == s].copy()
        if df_s["backbone"].nunique() < 3:
            continue
        summary, _, _ = permutation_test_cosine_lift(
            df_s, n_permutations=n_perm, rng_seed=s,
        )
        summary_with_seed = {"seed": s, **summary}
        perm_rows.append(summary_with_seed)
        print(
            f"  seed {s}: observed ΔR²={summary['observed_delta_r2']:+.3f}; "
            f"null mean={summary['null_delta_r2_mean']:+.3f}, "
            f"std={summary['null_delta_r2_std']:.3f}; "
            f"two-sided p={summary['p_value_two_sided_r2']:.3f} "
            f"(Δρ p={summary['p_value_two_sided_rho']:.3f})"
        )
    if perm_rows:
        pd.DataFrame(perm_rows).to_csv(OUT_DIR / "lobo_permutation.csv", index=False)
        print(f"Wrote {OUT_DIR / 'lobo_permutation.csv'}")


if __name__ == "__main__":
    main()
