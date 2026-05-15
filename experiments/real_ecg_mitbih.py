import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.mitbih_beats import (
    compute_class_weights,
    load_mitbih_interpatient_dataset,
    make_non_iid_record_partitions,
)
from federated.experiment import make_client_partitions, run_federated_experiment
from model.cnn_lstm import CNNLSTM


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a real MIT-BIH inter-patient federated compression experiment."
    )
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--clients", type=int, default=4)
    parser.add_argument("--window-size", type=int, default=180)
    parser.add_argument("--per-record-limit", type=int, default=None)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--iid", action="store_true")
    parser.add_argument("--train-records", nargs="*", default=None)
    parser.add_argument("--test-records", nargs="*", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/real_mitbih_compression.csv"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = load_mitbih_interpatient_dataset(
        window_size=args.window_size,
        per_record_limit=args.per_record_limit,
        force_rebuild=args.force_rebuild,
        download=True,
        train_records=args.train_records,
        test_records=args.test_records,
    )

    X_train = dataset["X_train"]
    y_train = dataset["y_train"]
    train_record_ids = dataset["train_record_ids"]
    X_test = dataset["X_test"]
    y_test = dataset["y_test"]

    if args.iid:
        clients_data, clients_labels = make_client_partitions(
            X_train,
            y_train,
            num_clients=args.clients,
            non_iid=False,
        )
        partition_mode = "iid"
    else:
        clients_data, clients_labels = make_non_iid_record_partitions(
            X_train,
            y_train,
            train_record_ids,
            num_clients=args.clients,
        )
        partition_mode = "record_non_iid"

    class_weights = compute_class_weights(y_train)

    configs = [
        {"name": "fedavg_fp32", "compression_mode": "none", "compression_bits": 32},
        {"name": "uniform_4bit", "compression_mode": "uniform", "compression_bits": 4},
        {"name": "turboquant_4bit", "compression_mode": "turboquant", "compression_bits": 4},
        {"name": "turboquant_2bit", "compression_mode": "turboquant", "compression_bits": 2},
    ]

    records = []
    for seed in args.seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)

        for config in configs:
            torch.manual_seed(seed)
            _, history = run_federated_experiment(
                model_factory=CNNLSTM,
                clients_data=clients_data,
                clients_labels=clients_labels,
                X_test=X_test,
                y_test=y_test,
                rounds=args.rounds,
                local_epochs=args.local_epochs,
                lr=args.lr,
                compression_mode=config["compression_mode"],
                compression_bits=config["compression_bits"],
                batch_size=args.batch_size,
                class_weights=class_weights,
            )

            for row in history:
                row = dict(row)
                row["experiment"] = config["name"]
                row["seed"] = seed
                row["partition_mode"] = partition_mode
                records.append(row)

    results = pd.DataFrame.from_records(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)

    final_round = (
        results.sort_values(["seed", "experiment", "round"])
        .groupby(["seed", "experiment"], as_index=False)
        .tail(1)
    )

    summary = (
        final_round.groupby("experiment", as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            log_loss_mean=("log_loss", "mean"),
            compression_ratio_mean=("compression_ratio", "mean"),
            bits_per_value_mean=("bits_per_value", "mean"),
            avg_client_latency_sec_mean=("avg_client_latency_sec", "mean"),
        )
        .sort_values(
            ["accuracy_mean", "compression_ratio_mean", "log_loss_mean"],
            ascending=[False, False, True],
        )
    )

    print("Real MIT-BIH federated benchmark completed")
    print(
        "Dataset:",
        f"train_beats={len(X_train)}, test_beats={len(X_test)}, "
        f"partition={partition_mode}, clients={args.clients}, seeds={args.seeds}",
    )
    print("Class weights:", class_weights)
    print("Client sizes:", [len(labels) for labels in clients_labels])
    print("Saved detailed results to", args.output)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
