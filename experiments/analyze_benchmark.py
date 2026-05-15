import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze which benchmark configurations performed best."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/compression_benchmark_delta_ef.csv"),
    )
    return parser.parse_args()


def explain_result(row, baseline_accuracy, best_compressed_experiment):
    experiment = row["experiment"]
    retained = row["accuracy"] / max(baseline_accuracy, 1e-12)

    if experiment == "fedavg_fp32":
        return (
            "Best overall because client updates are sent without quantization, "
            "so FedAvg aggregates exact floating-point weights."
        )

    if experiment == "turboquant_8bit":
        lead_in = "Current best compressed result" if experiment == best_compressed_experiment else "Strong compressed result"
        return (
            f"{lead_in} because 8-bit quantization keeps enough codebook "
            "resolution to preserve most of the update signal after rotation."
        )

    if experiment == "uniform_8bit":
        lead_in = "Current best compressed result" if experiment == best_compressed_experiment else "Strong baseline"
        return (
            f"{lead_in} because simple per-tensor min-max quantization keeps most "
            "signal at 8 bits, but it lacks TurboQuant's rotation step for harder distributions."
        )

    if experiment == "uniform_4bit":
        if retained >= 0.95:
            lead_in = (
                "Current best compressed result"
                if experiment == best_compressed_experiment
                else "Strong low-bit result"
            )
            return (
                f"{lead_in} because delta compression plus error feedback keeps "
                "4-bit uniform quantization stable while nearly matching FP32 accuracy."
            )
        return (
            "Useful mid-compression baseline, but 4-bit uniform bins are too coarse here "
            "and performance falls well below the 8-bit variants."
        )

    if experiment == "turboquant_4bit":
        if retained >= 0.95:
            lead_in = (
                "Current best compressed result"
                if experiment == best_compressed_experiment
                else "Strong low-bit result"
            )
            return (
                f"{lead_in} because delta compression plus error feedback "
                "lets TurboQuant hold FP32-level accuracy at roughly 7.9x compression."
            )
        return (
            "Communication improves strongly, but 4-bit quantization is too coarse "
            "for this current implementation and the model collapses toward chance accuracy."
        )

    if experiment == "turboquant_2bit":
        if retained >= 0.95:
            return (
                "Very strong extreme-compression result because the corrected delta path "
                "keeps even 2-bit TurboQuant close to FP32 performance."
            )
        return (
            "Maximum compression, but only four quantization levels survive, "
            "so the update distortion is too high for useful learning here."
        )

    return f"Retains {retained:.2%} of baseline accuracy."


def main():
    args = parse_args()
    results = pd.read_csv(args.input)

    group_columns = ["experiment"]
    sort_columns = ["experiment", "round"]
    if "seed" in results.columns:
        group_columns = ["seed", "experiment"]
        sort_columns = ["seed", "experiment", "round"]

    final_round = (
        results.sort_values(sort_columns)
        .groupby(group_columns, as_index=False)
        .tail(1)
    )

    if "seed" in results.columns:
        final_round = (
            final_round.groupby("experiment", as_index=False)
            .agg(
                accuracy=("accuracy", "mean"),
                macro_f1=("macro_f1", "mean"),
                log_loss=("log_loss", "mean"),
                compression_ratio=("compression_ratio", "mean"),
                bits_per_value=("bits_per_value", "mean"),
                avg_client_latency_sec=("avg_client_latency_sec", "mean"),
            )
        )

    final_round = (
        final_round.sort_values(
            ["accuracy", "compression_ratio", "log_loss"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )

    baseline_row = final_round.loc[final_round["experiment"] == "fedavg_fp32"].iloc[0]
    best_compressed = final_round.loc[final_round["compression_ratio"] > 1].iloc[0]

    print("Benchmark analysis for", args.input)
    print(
        "Best overall:",
        f"{baseline_row['experiment']} with accuracy={baseline_row['accuracy']:.4f},",
        f"macro_f1={baseline_row['macro_f1']:.4f}",
    )
    print(
        "Best compressed:",
        f"{best_compressed['experiment']} with accuracy={best_compressed['accuracy']:.4f},",
        f"compression={best_compressed['compression_ratio']:.2f}x,",
        f"accuracy_retention={best_compressed['accuracy'] / baseline_row['accuracy']:.2%}",
    )
    print("Per-configuration analysis:")

    for _, row in final_round.iterrows():
        print(
            f"- {row['experiment']}: "
            f"accuracy={row['accuracy']:.4f}, "
            f"macro_f1={row['macro_f1']:.4f}, "
            f"log_loss={row['log_loss']:.4f}, "
            f"compression={row['compression_ratio']:.2f}x. "
            f"{explain_result(row, baseline_row['accuracy'], best_compressed['experiment'])}"
        )


if __name__ == "__main__":
    main()
