"""
Convergence Plot Generator
===========================
Generates publication-ready convergence plots (accuracy vs round)
from benchmark CSV files using matplotlib.

Usage:
    python experiments/plot_convergence.py
    python experiments/plot_convergence.py --input results/real_mitbih_full_3seeds.csv
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate convergence plots.")
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        default=[
            RESULTS_DIR / "compression_benchmark_delta_ef.csv",
            RESULTS_DIR / "real_mitbih_full_3seeds.csv",
        ],
    )
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "plots")
    return parser.parse_args()


STYLE_MAP = {
    "fedavg_fp32":      {"color": "#2c3e50", "linestyle": "-",  "marker": "o", "label": "FedAvg FP32 (baseline)"},
    "uniform_8bit":     {"color": "#2980b9", "linestyle": "--", "marker": "s", "label": "Uniform 8-bit"},
    "uniform_4bit":     {"color": "#27ae60", "linestyle": "--", "marker": "^", "label": "Uniform 4-bit"},
    "turboquant_8bit":  {"color": "#e67e22", "linestyle": "-.", "marker": "D", "label": "TurboQuant 8-bit"},
    "turboquant_4bit":  {"color": "#e74c3c", "linestyle": "-.", "marker": "v", "label": "TurboQuant 4-bit"},
    "turboquant_2bit":  {"color": "#9b59b6", "linestyle": ":",  "marker": "X", "label": "TurboQuant 2-bit"},
    "topk_4bit":        {"color": "#1abc9c", "linestyle": ":",  "marker": "P", "label": "Top-K (12.5%)"},
    "topk_8bit":        {"color": "#16a085", "linestyle": ":",  "marker": "H", "label": "Top-K (25%)"},
}


def get_style(experiment):
    return STYLE_MAP.get(experiment, {
        "color": "#7f8c8d",
        "linestyle": "-",
        "marker": ".",
        "label": experiment,
    })


def plot_metric(ax, df, metric, ylabel, title):
    """Plot a single metric across rounds for all experiments."""
    experiments = df["experiment"].unique()

    for experiment in sorted(experiments):
        style = get_style(experiment)
        subset = df[df["experiment"] == experiment]

        if "seed" in subset.columns and subset["seed"].nunique() > 1:
            # Multiple seeds: plot mean with shaded CI
            grouped = subset.groupby("round")[metric]
            mean = grouped.mean()
            std = grouped.std()
            rounds = mean.index

            ax.plot(rounds, mean, label=style["label"],
                    color=style["color"], linestyle=style["linestyle"],
                    marker=style["marker"], markersize=5, linewidth=1.8)
            ax.fill_between(rounds, mean - std, mean + std,
                            alpha=0.15, color=style["color"])
        else:
            # Single seed: simple line
            subset_sorted = subset.sort_values("round")
            ax.plot(subset_sorted["round"], subset_sorted[metric],
                    label=style["label"], color=style["color"],
                    linestyle=style["linestyle"], marker=style["marker"],
                    markersize=5, linewidth=1.8)

    ax.set_xlabel("Communication Round", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=9)


def generate_plots(csv_path, output_dir):
    """Generate a full set of convergence plots for one CSV file."""
    df = pd.read_csv(csv_path)
    stem = csv_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Figure 1: Accuracy + F1 convergence ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_metric(axes[0], df, "accuracy", "Test Accuracy",
                "Accuracy vs Communication Round")
    plot_metric(axes[1], df, "macro_f1", "Macro F1",
                "Macro-F1 vs Communication Round")
    fig.suptitle(f"Convergence Analysis: {stem}", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    path1 = output_dir / f"{stem}_convergence.png"
    fig.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path1}")

    # --- Figure 2: Log-loss convergence ---
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    plot_metric(ax, df, "log_loss", "Log Loss",
                "Log-Loss vs Communication Round")
    fig.tight_layout()
    path2 = output_dir / f"{stem}_logloss.png"
    fig.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path2}")

    # --- Figure 3: Accuracy vs Bits tradeoff (final round) ---
    group_cols = ["experiment"]
    if "seed" in df.columns:
        group_cols.append("seed")

    final = df.sort_values("round").groupby(group_cols, as_index=False).tail(1)
    final_avg = final.groupby("experiment", as_index=False).agg(
        accuracy=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        bits_per_value=("bits_per_value", "first"),
        compression_ratio=("compression_ratio", "first"),
    )
    final_avg["accuracy_std"] = final_avg["accuracy_std"].fillna(0)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for _, row in final_avg.iterrows():
        style = get_style(row["experiment"])
        ax.errorbar(
            row["bits_per_value"], row["accuracy"],
            yerr=row["accuracy_std"],
            fmt=style["marker"], color=style["color"],
            markersize=10, capsize=4, linewidth=1.5,
            label=style["label"],
        )

    ax.set_xlabel("Bits per Value", fontsize=11)
    ax.set_ylabel("Final Accuracy", fontsize=11)
    ax.set_title("Accuracy vs Communication Cost", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()  # Lower bits = better compression = right side
    fig.tight_layout()
    path3 = output_dir / f"{stem}_acc_vs_bits.png"
    fig.savefig(path3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path3}")


def main():
    args = parse_args()
    print("Generating convergence plots...")

    for csv_path in args.input:
        if not csv_path.exists():
            print(f"  Skipping {csv_path} (not found)")
            continue
        print(f"\n  Processing: {csv_path.name}")
        generate_plots(csv_path, args.output_dir)

    print("\nDone! All plots saved to:", args.output_dir)


if __name__ == "__main__":
    main()
