"""
Compression Method Recommender for Federated Learning
=====================================================
Analyzes all available benchmark results and recommends the best
compression method based on your priorities.

Usage:
    python experiments/recommend.py
"""

import sys
import os
from pathlib import Path

# Fix Windows console encoding for emoji output
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd


RESULTS_DIR = Path("results")

BENCHMARK_FILES = [
    ("Synthetic ECG (delta+EF)", "compression_benchmark_delta_ef.csv"),
    ("Real MIT-BIH (3 seeds)", "real_mitbih_full_3seeds.csv"),
    ("Real MIT-BIH (smoke)", "real_mitbih_smoke.csv"),
]


def load_all_benchmarks():
    """Load all available benchmark CSVs into a single tagged DataFrame."""
    frames = []
    for label, filename in BENCHMARK_FILES:
        path = RESULTS_DIR / filename
        if path.exists():
            df = pd.read_csv(path)
            df["benchmark"] = label
            frames.append(df)
    if not frames:
        print("ERROR: No benchmark CSV files found in", RESULTS_DIR)
        sys.exit(1)
    return pd.concat(frames, ignore_index=True)


def extract_final_rounds(df):
    """Get the final round for each (benchmark, experiment, seed) combo."""
    group_cols = ["benchmark", "experiment"]
    sort_cols = ["benchmark", "experiment", "round"]
    if "seed" in df.columns:
        group_cols.append("seed")
        sort_cols.append("seed")

    final = df.sort_values(sort_cols).groupby(group_cols, as_index=False).tail(1)

    # Average across seeds if present
    agg_cols = {
        "accuracy": "mean",
        "macro_f1": "mean",
        "log_loss": "mean",
        "compression_ratio": "first",
        "bits_per_value": "first",
        "avg_client_latency_sec": "mean",
    }
    summary = (
        final.groupby(["benchmark", "experiment"], as_index=False)
        .agg(**{k: (k, v) for k, v in agg_cols.items()})
    )
    return summary


def compute_scores(summary):
    """
    Score each method on three dimensions:
      - accuracy_retention: how close to FP32 baseline (higher = better)
      - bandwidth_savings: compression_ratio normalized (higher = better)
      - speed: inverse of latency normalized (higher = better)
    """
    rows = []

    for benchmark in summary["benchmark"].unique():
        bench_df = summary[summary["benchmark"] == benchmark].copy()
        baseline = bench_df[bench_df["experiment"] == "fedavg_fp32"]

        if baseline.empty:
            continue

        baseline_acc = baseline["accuracy"].values[0]
        baseline_latency = baseline["avg_client_latency_sec"].values[0]

        for _, row in bench_df.iterrows():
            if row["experiment"] == "fedavg_fp32":
                continue

            acc_retention = row["accuracy"] / max(baseline_acc, 1e-12)
            bandwidth_savings = row["compression_ratio"]
            # Speed relative to baseline (>1 = faster than FP32)
            speed_ratio = baseline_latency / max(row["avg_client_latency_sec"], 1e-12)

            rows.append({
                "benchmark": benchmark,
                "experiment": row["experiment"],
                "accuracy": row["accuracy"],
                "accuracy_retention": acc_retention,
                "compression_ratio": bandwidth_savings,
                "bits_per_value": row["bits_per_value"],
                "avg_latency_sec": row["avg_client_latency_sec"],
                "speed_vs_fp32": speed_ratio,
                "macro_f1": row["macro_f1"],
                "log_loss": row["log_loss"],
            })

    return pd.DataFrame(rows)


def print_separator(char="=", width=80):
    print(char * width)


def print_header(title):
    print()
    print_separator()
    print(f"  {title}")
    print_separator()
    print()


def recommend(scores):
    """Print a comprehensive recommendation report."""

    print_header("FEDERATED LEARNING COMPRESSION RECOMMENDER")

    # --- Per-benchmark breakdown ---
    for benchmark in scores["benchmark"].unique():
        bench = scores[scores["benchmark"] == benchmark].copy()
        print(f"📊 Benchmark: {benchmark}")
        print(f"   Methods tested: {len(bench)}")
        print()

        # Sort by accuracy retention, then compression
        bench = bench.sort_values(
            ["accuracy_retention", "compression_ratio"],
            ascending=[False, False],
        )

        print(f"   {'Method':<20} {'Acc%':>6} {'Retain':>7} {'Compress':>9} {'Bits':>5} {'Latency':>8} {'F1':>6}")
        print(f"   {'-'*20} {'-'*6} {'-'*7} {'-'*9} {'-'*5} {'-'*8} {'-'*6}")

        for _, row in bench.iterrows():
            flag = " ⭐" if row["accuracy_retention"] >= 0.99 and row["compression_ratio"] >= 7 else ""
            print(
                f"   {row['experiment']:<20} "
                f"{row['accuracy']:>5.1%} "
                f"{row['accuracy_retention']:>6.1%} "
                f"{row['compression_ratio']:>8.2f}x "
                f"{row['bits_per_value']:>4.1f}b "
                f"{row['avg_latency_sec']:>7.3f}s "
                f"{row['macro_f1']:>5.3f}"
                f"{flag}"
            )
        print()

    # --- Overall recommendation ---
    print_header("RECOMMENDATION")

    # Method that appears best across all benchmarks
    avg_scores = (
        scores.groupby("experiment", as_index=False)
        .agg(
            mean_retention=("accuracy_retention", "mean"),
            mean_compression=("compression_ratio", "mean"),
            mean_latency=("avg_latency_sec", "mean"),
            mean_f1=("macro_f1", "mean"),
            count=("benchmark", "count"),
        )
    )

    # Composite score: prioritize accuracy retention, then compression
    avg_scores["composite"] = (
        avg_scores["mean_retention"] * 0.5
        + (avg_scores["mean_compression"] / avg_scores["mean_compression"].max()) * 0.3
        + (1.0 / (1.0 + avg_scores["mean_latency"])) * 0.2
    )

    avg_scores = avg_scores.sort_values("composite", ascending=False)

    print("  Across ALL benchmarks (averaged):\n")
    print(f"  {'Rank':<5} {'Method':<20} {'Avg Retain':>10} {'Avg Compress':>13} {'Avg Latency':>12} {'Score':>6}")
    print(f"  {'-'*5} {'-'*20} {'-'*10} {'-'*13} {'-'*12} {'-'*6}")

    for rank, (_, row) in enumerate(avg_scores.iterrows(), 1):
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, "  ")
        print(
            f"  {medal} {rank:<2} {row['experiment']:<20} "
            f"{row['mean_retention']:>9.1%} "
            f"{row['mean_compression']:>12.2f}x "
            f"{row['mean_latency']:>11.3f}s "
            f"{row['composite']:>5.3f}"
        )

    best = avg_scores.iloc[0]
    print()
    print_separator("-")
    print()

    # Decision guide
    print("  🎯 WHICH ONE IS BEST FOR YOU?\n")

    print("  ┌─────────────────────────────────────────────────────────────────┐")
    print("  │  YOUR PRIORITY              →  USE THIS METHOD                 │")
    print("  ├─────────────────────────────────────────────────────────────────┤")
    print("  │  Maximum accuracy           →  fedavg_fp32 (no compression)    │")
    print("  │  Best accuracy/bandwidth    →  uniform_4bit  ⭐ recommended    │")
    print("  │  Most bandwidth savings     →  turboquant_2bit (15.3x)         │")
    print("  │  Fastest experiment time    →  uniform_4bit or uniform_8bit    │")
    print("  │  Research novelty           →  turboquant_4bit (rotation+quant)│")
    print("  └─────────────────────────────────────────────────────────────────┘")

    print()
    print(f"  ✅ OVERALL WINNER: {best['experiment']}")
    print(f"     Retains {best['mean_retention']:.1%} of FP32 accuracy")
    print(f"     at {best['mean_compression']:.1f}x compression")
    print(f"     with {best['mean_latency']:.3f}s avg client latency")
    print()

    # When NOT to use compression
    print("  ⚠️  WHEN TO SKIP COMPRESSION:")
    print("     - If your network is fast and bandwidth is not a bottleneck")
    print("     - If you need bit-exact reproducibility across runs")
    print("     - If your model is very small (<10K params) — overhead isn't worth it")
    print()
    print_separator()


def main():
    print("Loading benchmark data...")
    all_data = load_all_benchmarks()
    print(f"  Loaded {len(all_data)} rows from {all_data['benchmark'].nunique()} benchmarks")

    summary = extract_final_rounds(all_data)
    scores = compute_scores(summary)

    if scores.empty:
        print("ERROR: No compressed experiments found to compare.")
        sys.exit(1)

    recommend(scores)

    # Save scores to CSV for reference
    output_path = RESULTS_DIR / "recommendation_scores.csv"
    scores.to_csv(output_path, index=False)
    print(f"\nDetailed scores saved to: {output_path}")


if __name__ == "__main__":
    main()
