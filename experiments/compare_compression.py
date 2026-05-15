import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.synthetic_ecg import generate_ecg_data
from federated.experiment import make_client_partitions, run_federated_experiment
from model.cnn_lstm import CNNLSTM


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare uncompressed FedAvg against TurboQuant-compressed FL."
    )
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--timesteps", type=int, default=200)
    parser.add_argument("--clients", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--non-iid", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/compression_benchmark_delta_ef.csv"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    X, y = generate_ecg_data(
        samples=args.samples,
        timesteps=args.timesteps,
        seed=args.seed,
    )
    y = y.reshape(-1)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=args.seed,
        stratify=y,
    )

    clients_data, clients_labels = make_client_partitions(
        X_train,
        y_train,
        num_clients=args.clients,
        non_iid=args.non_iid,
    )

    configs = [
        {"name": "fedavg_fp32", "compression_mode": "none", "compression_bits": 32},
        {"name": "uniform_8bit", "compression_mode": "uniform", "compression_bits": 8},
        {"name": "uniform_4bit", "compression_mode": "uniform", "compression_bits": 4},
        {"name": "topk_8bit", "compression_mode": "topk", "compression_bits": 8},
        {"name": "topk_4bit", "compression_mode": "topk", "compression_bits": 4},
        {"name": "turboquant_8bit", "compression_mode": "turboquant", "compression_bits": 8},
        {"name": "turboquant_4bit", "compression_mode": "turboquant", "compression_bits": 4},
        {"name": "turboquant_2bit", "compression_mode": "turboquant", "compression_bits": 2},
    ]

    records = []
    for config in configs:
        torch.manual_seed(args.seed)
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
        )

        for row in history:
            row = dict(row)
            row["experiment"] = config["name"]
            records.append(row)

    results = pd.DataFrame.from_records(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)

    final_round = (
        results.sort_values(["experiment", "round"])
        .groupby("experiment", as_index=False)
        .tail(1)
        .sort_values(
            ["accuracy", "compression_ratio", "log_loss"],
            ascending=[False, False, True],
        )
    )

    print("Saved detailed results to", args.output)
    print(
        final_round[
            [
                "experiment",
                "round",
                "accuracy",
                "macro_f1",
                "log_loss",
                "compression_ratio",
                "bits_per_value",
                "avg_client_latency_sec",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
