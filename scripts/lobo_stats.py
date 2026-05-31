"""Jackknife-over-backbones CI + permutation test on the cosine LOBO lift.
Standalone companion to predict_forgetting_lobo.py.

Writes lobo_jackknife.csv, lobo_jackknife_summary.csv, and
lobo_permutation.csv under outputs/_figures/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from med_merge.config.constants import SEEDS

from predict_forgetting_lobo import (
    BACKBONES,
    assemble,
    lobo_score,
)

OUT_DIR = Path("outputs/_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def jackknife_lobo_delta_r2(
    df: pd.DataFrame,
    feature_cols: tuple[str, ...] = ("mean_cos",),
) -> tuple[pd.DataFrame, dict]:
    """Leave-one-backbone-out of the analysis entirely; recompute LOBO ΔR²
    on the remaining backbones. Returns per-fold table + jackknife summary.
    """
    rows = []
    for held in sorted(df["backbone"].unique().tolist()):
        sub = df[df["backbone"] != held].copy()
        if sub["backbone"].nunique() < 2:
            continue
        r2_d, _, rho_d, _ = lobo_score(sub, [], include_dummies=True)
        r2_c, _, rho_c, _ = lobo_score(sub, list(feature_cols), include_dummies=True)
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
    dr2 = jk["delta_r2"].values
    drho = jk["delta_rho"].values
    mean_dr2 = float(dr2.mean())
    se_dr2 = float(np.sqrt((n - 1) / n * ((dr2 - mean_dr2) ** 2).sum()))
    mean_drho = float(drho.mean())
    se_drho = float(np.sqrt((n - 1) / n * ((drho - mean_drho) ** 2).sum()))
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


def permutation_test(
    df: pd.DataFrame,
    feature_cols: tuple[str, ...] = ("mean_cos",),
    n_permutations: int = 1000,
    rng_seed: int = 0,
) -> dict:
    """Shuffle the cosine column, refit LOBO, build null distribution of ΔR²."""
    rng = np.random.default_rng(rng_seed)
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
        r2_c_p, _, rho_c_p, _ = lobo_score(df_perm, list(feature_cols), include_dummies=True)
        null_dr2[i] = r2_c_p - r2_d_obs
        null_drho[i] = rho_c_p - rho_d_obs

    return {
        "n_permutations": n_permutations,
        "observed_delta_r2": obs_dr2,
        "observed_delta_rho": obs_drho,
        "null_delta_r2_mean": float(null_dr2.mean()),
        "null_delta_r2_std": float(null_dr2.std(ddof=1)),
        "null_delta_r2_p05": float(np.percentile(null_dr2, 5)),
        "null_delta_r2_p95": float(np.percentile(null_dr2, 95)),
        "null_delta_rho_mean": float(null_drho.mean()),
        "null_delta_rho_std": float(null_drho.std(ddof=1)),
        "p_value_two_sided_r2": float((np.abs(null_dr2) >= abs(obs_dr2)).mean()),
        "p_value_two_sided_rho": float((np.abs(null_drho) >= abs(obs_drho)).mean()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    p.add_argument("--n-perm", type=int, default=1000)
    args = p.parse_args()

    df = assemble(args.seeds)
    if df.empty:
        print("No data. Run compute_geometry.py and predict_forgetting_lobo.py first.")
        return
    print(f"Assembled {len(df)} rows across seeds={sorted(df['seed'].unique().tolist())}, "
          f"backbones={sorted(df['backbone'].unique().tolist())}")

    print("\n=== Jackknife-over-backbones (cosine ΔR²) ===")
    jk_rows, jk_summary_rows = [], []
    for s in sorted(df["seed"].unique().tolist()):
        df_s = df[df["seed"] == s].copy()
        if df_s["backbone"].nunique() < 3:
            print(f"  seed {s}: too few backbones ({df_s['backbone'].nunique()}), skipping")
            continue
        jk, summary = jackknife_lobo_delta_r2(df_s)
        jk["seed"] = s
        jk_rows.append(jk)
        if summary:
            jk_summary_rows.append({"seed": s, **summary})
            print(
                f"  seed {s} (n_folds={summary['n_folds']}): "
                f"ΔR² mean={summary['delta_r2_mean']:+.3f} "
                f"SE={summary['delta_r2_jackknife_se']:.3f} "
                f"[95% CI {summary['delta_r2_ci_lo_95']:+.3f}, "
                f"{summary['delta_r2_ci_hi_95']:+.3f}]"
            )
            print(
                f"            Δρ mean={summary['delta_rho_mean']:+.3f} "
                f"SE={summary['delta_rho_jackknife_se']:.3f} "
                f"[95% CI {summary['delta_rho_ci_lo_95']:+.3f}, "
                f"{summary['delta_rho_ci_hi_95']:+.3f}]"
            )
    if jk_rows:
        pd.concat(jk_rows, ignore_index=True).to_csv(OUT_DIR / "lobo_jackknife.csv", index=False)
        pd.DataFrame(jk_summary_rows).to_csv(OUT_DIR / "lobo_jackknife_summary.csv", index=False)
        print(f"\nWrote {OUT_DIR / 'lobo_jackknife.csv'}")
        print(f"Wrote {OUT_DIR / 'lobo_jackknife_summary.csv'}")

    print(f"\n=== Permutation test (n={args.n_perm}) ===")
    perm_rows = []
    for s in sorted(df["seed"].unique().tolist()):
        df_s = df[df["seed"] == s].copy()
        if df_s["backbone"].nunique() < 3:
            continue
        summary = permutation_test(df_s, n_permutations=args.n_perm, rng_seed=s)
        perm_rows.append({"seed": s, **summary})
        print(
            f"  seed {s}: observed ΔR²={summary['observed_delta_r2']:+.3f}, "
            f"null mean={summary['null_delta_r2_mean']:+.3f} std={summary['null_delta_r2_std']:.3f}; "
            f"two-sided p_R²={summary['p_value_two_sided_r2']:.3f}, "
            f"p_ρ={summary['p_value_two_sided_rho']:.3f}"
        )
    if perm_rows:
        pd.DataFrame(perm_rows).to_csv(OUT_DIR / "lobo_permutation.csv", index=False)
        print(f"\nWrote {OUT_DIR / 'lobo_permutation.csv'}")


if __name__ == "__main__":
    main()
