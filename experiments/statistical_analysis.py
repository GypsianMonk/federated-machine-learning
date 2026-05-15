"""
Statistical Analysis for FL Compression Benchmarks
====================================================
Computes confidence intervals, runs paired t-tests / Wilcoxon
signed-rank tests, and generates a publication-ready stats table.

Usage:
    python experiments/statistical_analysis.py
    python experiments/statistical_analysis.py --input results/real_mitbih_full_3seeds.csv
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

# Fix Windows console encoding
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def parse_args():
    parser = argparse.ArgumentParser(description="Statistical analysis of FL benchmarks.")
    parser.add_argument(
        "--input",
        type=Path,
        default=RESULTS_DIR / "real_mitbih_full_3seeds.csv",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def confidence_interval(data, confidence=0.95):
    """Compute mean and CI for a 1D array."""
    n = len(data)
    mean = np.mean(data)
    if n < 2:
        return mean, mean, mean
    se = scipy_stats.sem(data)
    h = se * scipy_stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean, mean - h, mean + h


def main():
    args = parse_args()

    if not args.input.exists():
        print(f"ERROR: {args.input} not found.")
        sys.exit(1)

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows from {args.input.name}")

    has_seeds = "seed" in df.columns and df["seed"].nunique() > 1
    n_seeds = df["seed"].nunique() if has_seeds else 1
    print(f"Seeds: {n_seeds}")

    # Get final-round results per experiment per seed
    group_cols = ["experiment"]
    if has_seeds:
        group_cols.append("seed")

    final = df.sort_values("round").groupby(group_cols, as_index=False).tail(1)
    experiments = sorted(final["experiment"].unique())
    baseline_name = "fedavg_fp32"

    # --- Table 1: Summary Statistics with CIs ---
    print("\n" + "=" * 90)
    print("  TABLE 1: Final-Round Performance with 95% Confidence Intervals")
    print("=" * 90)

    header = f"  {'Method':<20} {'Accuracy':>16} {'Macro F1':>16} {'Log Loss':>16} {'Compress':>9}"
    print(header)
    print("  " + "-" * 86)

    summary_rows = []
    for exp in experiments:
        exp_data = final[final["experiment"] == exp]

        acc_vals = exp_data["accuracy"].values
        f1_vals = exp_data["macro_f1"].values
        ll_vals = exp_data["log_loss"].values
        cr = exp_data["compression_ratio"].values[0]

        acc_mean, acc_lo, acc_hi = confidence_interval(acc_vals)
        f1_mean, f1_lo, f1_hi = confidence_interval(f1_vals)
        ll_mean, ll_lo, ll_hi = confidence_interval(ll_vals)

        if n_seeds > 1:
            acc_str = f"{acc_mean:.4f} [{acc_lo:.4f}, {acc_hi:.4f}]"
            f1_str = f"{f1_mean:.4f} [{f1_lo:.4f}, {f1_hi:.4f}]"
            ll_str = f"{ll_mean:.4f} [{ll_lo:.4f}, {ll_hi:.4f}]"
        else:
            acc_str = f"{acc_mean:.4f}"
            f1_str = f"{f1_mean:.4f}"
            ll_str = f"{ll_mean:.4f}"

        print(f"  {exp:<20} {acc_str:>16} {f1_str:>16} {ll_str:>16} {cr:>8.2f}x")

        summary_rows.append({
            "experiment": exp,
            "accuracy_mean": acc_mean,
            "accuracy_ci_lo": acc_lo,
            "accuracy_ci_hi": acc_hi,
            "f1_mean": f1_mean,
            "f1_ci_lo": f1_lo,
            "f1_ci_hi": f1_hi,
            "logloss_mean": ll_mean,
            "logloss_ci_lo": ll_lo,
            "logloss_ci_hi": ll_hi,
            "compression_ratio": cr,
        })

    # --- Table 2: Statistical Tests vs Baseline ---
    if has_seeds and n_seeds >= 3:
        print("\n" + "=" * 90)
        print("  TABLE 2: Statistical Tests vs FP32 Baseline (alpha={:.2f})".format(args.alpha))
        print("=" * 90)

        baseline_data = final[final["experiment"] == baseline_name]
        if baseline_data.empty:
            print("  WARNING: No FP32 baseline found, skipping statistical tests.")
        else:
            baseline_acc = baseline_data.set_index("seed")["accuracy"]
            baseline_f1 = baseline_data.set_index("seed")["macro_f1"]

            print(f"  {'Method':<20} {'Acc Diff':>9} {'t-stat':>8} {'p-value':>9} {'Signif?':>8} {'Wilcoxon p':>11}")
            print("  " + "-" * 70)

            for exp in experiments:
                if exp == baseline_name:
                    continue

                exp_data = final[final["experiment"] == exp]
                exp_acc = exp_data.set_index("seed")["accuracy"]

                # Align by seed
                common_seeds = baseline_acc.index.intersection(exp_acc.index)
                if len(common_seeds) < 2:
                    print(f"  {exp:<20} Insufficient paired data (n={len(common_seeds)})")
                    continue

                b = baseline_acc.loc[common_seeds].values
                e = exp_acc.loc[common_seeds].values
                diff = e - b

                # Paired t-test
                t_stat, p_ttest = scipy_stats.ttest_rel(e, b)

                # Wilcoxon signed-rank (non-parametric)
                try:
                    _, p_wilcox = scipy_stats.wilcoxon(diff, alternative="two-sided")
                except ValueError:
                    p_wilcox = 1.0  # All differences are zero

                significant = "YES" if p_ttest < args.alpha else "no"
                mean_diff = np.mean(diff)

                print(
                    f"  {exp:<20} {mean_diff:>+8.4f} {t_stat:>8.3f} {p_ttest:>9.4f} "
                    f"{significant:>8} {p_wilcox:>11.4f}"
                )

        # --- Table 3: Accuracy Retention ---
        print("\n" + "=" * 90)
        print("  TABLE 3: Accuracy Retention vs FP32 Baseline")
        print("=" * 90)

        baseline_mean = baseline_data["accuracy"].mean()
        print(f"  FP32 baseline mean accuracy: {baseline_mean:.4f}\n")

        print(f"  {'Method':<20} {'Retention':>10} {'Bits/Val':>9} {'Verdict':>30}")
        print("  " + "-" * 73)

        for row in summary_rows:
            if row["experiment"] == baseline_name:
                continue
            retention = row["accuracy_mean"] / max(baseline_mean, 1e-12)
            bits = row["compression_ratio"]

            if retention >= 0.99:
                verdict = "Excellent - near lossless"
            elif retention >= 0.95:
                verdict = "Good - acceptable for most uses"
            elif retention >= 0.90:
                verdict = "Fair - some accuracy loss"
            else:
                verdict = "Poor - significant degradation"

            print(f"  {row['experiment']:<20} {retention:>9.1%} {bits:>8.2f}x {verdict:>30}")

    # Save summary CSV
    output_path = args.output or (RESULTS_DIR / f"statistical_summary_{args.input.stem}.csv")
    pd.DataFrame(summary_rows).to_csv(output_path, index=False)
    print(f"\nSummary saved to: {output_path}")


if __name__ == "__main__":
    main()
