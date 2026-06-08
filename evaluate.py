"""
Bootstrap confidence-interval evaluation for all models.

Expects a prediction CSV containing at minimum these columns:
    pred          – model's scalar prediction (same scale as scaled_rating)
    scaled_rating – ground-truth standardized rating
    instance_id   – snippet / conversation identifier for snippet-level aggregation
    seen_flag     – boolean; True if the annotator appeared in training data

All training scripts (train_moe.py, train_uwua.py, train_jury.py,
train_mbert.py) produce prediction CSVs in this format.

Usage:
    python evaluate.py --pred_csv harm_moe_test_preds.csv
    python evaluate.py --pred_csv paact_uwua_test_preds.csv --n_bootstrap 1000

    # With demographic subgroup analysis:
    python evaluate.py --pred_csv paact_moe_test_preds.csv \
        --demo_cols "hcp_freq,edu_level,age_group,gender,race,occupation,doc_trust_category,ethnic_trust_category"

Metrics reported (seen / unseen annotator splits):
    indiv_corr  – Pearson r between per-rating predictions and labels
    indiv_mae   – mean absolute error on individual ratings
    snip_corr   – Pearson r on per-snippet mean predictions vs. mean labels
    snip_mae    – MAE on per-snippet means
    emd         – mean Earth Mover's Distance per snippet

Demographic subgroup analysis (--demo_cols):
    For each demographic attribute, independently bootstraps each subgroup
    (e.g. Man / Woman / Non-binary for gender), pools all bootstrap snip_agg_mae
    values across subgroups, and reports a t-interval 95% CI on the pool.
    Output format per attribute:
        gender: mean=0.887, ci=(0.871, 0.902), error=0.016
    The demographic columns must be present in the prediction CSV (all training
    scripts preserve the original data columns alongside pred).

Output:
    A formatted table with point estimates and 95% CI for each metric × split,
    followed by the demographic subgroup table if --demo_cols is provided.
"""

import argparse

import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.stats import wasserstein_distance
from sklearn.metrics import mean_absolute_error


# ── Core metric computation ───────────────────────────────────────────────────

def slice_metrics(df: pd.DataFrame):
    """Compute all five metrics for a dataframe slice.

    Returns (indiv_corr, indiv_mae, snip_corr, snip_mae, emd).
    Returns NaN for any metric that cannot be computed (e.g. only 1 sample).
    """
    preds   = df["pred"].to_numpy(dtype=float)
    targets = df["scaled_rating"].to_numpy(dtype=float)

    if len(preds) < 2:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")

    indiv_corr = np.corrcoef(preds, targets)[0, 1]
    indiv_mae  = mean_absolute_error(targets, preds)

    grp_pred = df.groupby("instance_id")["pred"].mean()
    grp_true = df.groupby("instance_id")["scaled_rating"].mean()

    if len(grp_pred) < 2:
        snip_corr = float("nan")
    else:
        snip_corr = np.corrcoef(grp_pred.values, grp_true.values)[0, 1]
    snip_mae = mean_absolute_error(grp_true.values, grp_pred.values)

    emd_per_snip = df.groupby("instance_id").apply(
        lambda g: wasserstein_distance(g["pred"].tolist(), g["scaled_rating"].tolist())
    )
    emd = emd_per_snip.mean()

    return indiv_corr, indiv_mae, snip_corr, snip_mae, emd


METRIC_NAMES = ["indiv_corr", "indiv_mae", "snip_corr", "snip_mae", "emd"]


# ── Bootstrap ─────────────────────────────────────────────────────────────────

def bootstrap_ci(df: pd.DataFrame, n_bootstrap: int, seed: int = 42):
    """Run bootstrap on df, return dict metric → (mean, lower_95, upper_95)."""
    rng     = np.random.default_rng(seed)
    records = {m: [] for m in METRIC_NAMES}

    for _ in range(n_bootstrap):
        sample = df.sample(n=len(df), replace=True, random_state=rng.integers(1 << 31))
        vals   = slice_metrics(sample)
        for name, val in zip(METRIC_NAMES, vals):
            records[name].append(val)

    result = {}
    for name, vals in records.items():
        arr = np.array(vals, dtype=float)
        result[name] = (
            float(np.nanmean(arr)),
            float(np.nanpercentile(arr, 2.5)),
            float(np.nanpercentile(arr, 97.5)),
        )
    return result


# ── Printing ──────────────────────────────────────────────────────────────────

def print_table(split_name: str, point_vals, ci_dict: dict):
    """Print one section of the results table."""
    print(f"\n{'─' * 60}")
    print(f"  {split_name.upper()} ANNOTATORS")
    print(f"{'─' * 60}")
    header = f"  {'Metric':<14} {'Point':>8}  {'95% CI':>22}"
    print(header)
    print(f"  {'─' * 50}")
    for name, pval in zip(METRIC_NAMES, point_vals):
        mean, lo, hi = ci_dict[name]
        print(f"  {name:<14} {pval:>8.4f}  [{lo:.4f}, {hi:.4f}]")


# ── Demographic subgroup analysis ─────────────────────────────────────────────

def demo_subgroup_analysis(df: pd.DataFrame, demo_col: str, n_bootstrap: int, seed: int = 42):
    """
    For each subgroup within demo_col, independently bootstrap snip_agg_mae.
    Pool all bootstrap values across subgroups and compute a t-interval CI.

    Matches the notebook analysis (demo_analysis_paact.ipynb):
        - per-subgroup bootstrapping
        - values pooled across subgroups before CI computation
        - scipy.stats.t.interval CI (not percentile)

    Returns (mean, ci_lower, ci_upper, error) where error = (ci_upper - ci_lower) / 2.
    """
    rng       = np.random.default_rng(seed)
    subgroups = df[demo_col].dropna().unique()
    all_vals  = []

    for grp in subgroups:
        subset = df[df[demo_col] == grp]
        if len(subset) == 0:
            continue
        for _ in range(n_bootstrap):
            sample   = subset.sample(n=len(subset), replace=True,
                                     random_state=int(rng.integers(1 << 31)))
            grp_pred = sample.groupby("instance_id")["pred"].mean()
            grp_true = sample.groupby("instance_id")["scaled_rating"].mean()
            all_vals.append(mean_absolute_error(grp_true.values, grp_pred.values))

    if len(all_vals) < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")

    mean_val              = float(np.mean(all_vals))
    ci_lower, ci_upper    = st.t.interval(
        0.95, len(all_vals) - 1, loc=mean_val, scale=st.sem(all_vals)
    )
    error = (ci_upper - ci_lower) / 2
    return mean_val, float(ci_lower), float(ci_upper), float(error)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Bootstrap CI evaluation")
    parser.add_argument("--pred_csv",    required=True,
                        help="Path to prediction CSV produced by a training script")
    parser.add_argument("--n_bootstrap", type=int, default=500,
                        help="Number of bootstrap resamples (default: 500)")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--demo_cols",   default=None,
                        help="Comma-separated demographic column names for subgroup "
                             "analysis, e.g. 'gender,race,age'")
    args = parser.parse_args()

    df = pd.read_csv(args.pred_csv)

    required = {"pred", "scaled_rating", "instance_id", "seen_flag"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Prediction CSV is missing columns: {missing}")

    df["seen_flag"] = df["seen_flag"].astype(bool)
    df_seen   = df[df["seen_flag"]].copy()
    df_unseen = df[~df["seen_flag"]].copy()

    print(f"\nLoaded {args.pred_csv}")
    print(f"  Total rows : {len(df)}")
    print(f"  Seen       : {len(df_seen)}")
    print(f"  Unseen     : {len(df_unseen)}")

    print(f"\nRunning {args.n_bootstrap} bootstrap resamples …")

    for split_name, split_df in [("seen", df_seen), ("unseen", df_unseen)]:
        if len(split_df) == 0:
            print(f"\n  (no {split_name} annotator rows — skipping)")
            continue
        point_vals = slice_metrics(split_df)
        ci         = bootstrap_ci(split_df, args.n_bootstrap, seed=args.seed)
        print_table(split_name, point_vals, ci)

    if args.demo_cols:
        demo_cols   = [c.strip() for c in args.demo_cols.split(",") if c.strip()]
        missing     = [c for c in demo_cols if c not in df.columns]
        if missing:
            print(f"\nWarning: columns not found in CSV and will be skipped: {missing}")
        demo_cols = [c for c in demo_cols if c in df.columns]

        if demo_cols:
            print(f"\n{'─' * 60}")
            print("  DEMOGRAPHIC SUBGROUP ANALYSIS  (snip_agg_mae)")
            print(f"{'─' * 60}")
            print(f"  {'Category':<26} {'mean':>7}  {'95% CI':>18}  {'error':>7}")
            print(f"  {'─' * 62}")
            for col in demo_cols:
                mean_val, ci_lo, ci_hi, err = demo_subgroup_analysis(
                    df, col, args.n_bootstrap, seed=args.seed
                )
                if np.isnan(mean_val):
                    print(f"  {col:<26}  (insufficient data)")
                else:
                    print(f"  {col:<26} {mean_val:>7.3f}  "
                          f"({ci_lo:.3f}, {ci_hi:.3f})  {err:>7.3f}")

    print()


if __name__ == "__main__":
    main()
