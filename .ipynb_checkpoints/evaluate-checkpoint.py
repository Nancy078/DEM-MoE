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

Metrics reported (seen / unseen annotator splits):
    indiv_corr  – Pearson r between per-rating predictions and labels
    indiv_mae   – mean absolute error on individual ratings
    snip_corr   – Pearson r on per-snippet mean predictions vs. mean labels
    snip_mae    – MAE on per-snippet means
    emd         – mean Earth Mover's Distance per snippet

Output:
    A formatted table with point estimates and 95% CI for each metric × split.
"""

import argparse

import numpy as np
import pandas as pd
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Bootstrap CI evaluation")
    parser.add_argument("--pred_csv",    required=True,
                        help="Path to prediction CSV produced by a training script")
    parser.add_argument("--n_bootstrap", type=int, default=500,
                        help="Number of bootstrap resamples (default: 500)")
    parser.add_argument("--seed",        type=int, default=42)
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

    print()


if __name__ == "__main__":
    main()
