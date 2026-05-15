import numpy as np
import time
import torch
from sklearn.model_selection import train_test_split

from data.synthetic_ecg import generate_ecg_data
from explainability.shap_explainer import explain
from federated.experiment import make_client_partitions, run_federated_experiment
from model.cnn_lstm import CNNLSTM


SEED = 42
SAMPLES = 600
TIMESTEPS = 200
NUM_CLIENTS = 3
ROUNDS = 8
LOCAL_EPOCHS = 2
LEARNING_RATE = 1e-3
NON_IID = False


def main():
    total_start = time.perf_counter()

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    X, y = generate_ecg_data(samples=SAMPLES, timesteps=TIMESTEPS, seed=SEED)
    y = y.reshape(-1)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=SEED,
        stratify=y,
    )

    clients_data, clients_labels = make_client_partitions(
        X_train,
        y_train,
        num_clients=NUM_CLIENTS,
        non_iid=NON_IID,
    )

    configs = [
        {
            "name": "fedavg_fp32",
            "compression_mode": "none",
            "compression_bits": 32,
        },
        {
            "name": "uniform_8bit",
            "compression_mode": "uniform",
            "compression_bits": 8,
        },
        {
            "name": "turboquant_8bit",
            "compression_mode": "turboquant",
            "compression_bits": 8,
        },
        {
            "name": "uniform_4bit",
            "compression_mode": "uniform",
            "compression_bits": 4,
        },
        {
            "name": "turboquant_4bit",
            "compression_mode": "turboquant",
            "compression_bits": 4,
        },
    ]

    final_results = []
    trained_models = {}

    for config in configs:
        torch.manual_seed(SEED)
        exp_start = time.perf_counter()
        model, history = run_federated_experiment(
            model_factory=CNNLSTM,
            clients_data=clients_data,
            clients_labels=clients_labels,
            X_test=X_test,
            y_test=y_test,
            rounds=ROUNDS,
            local_epochs=LOCAL_EPOCHS,
            lr=LEARNING_RATE,
            compression_mode=config["compression_mode"],
            compression_bits=config["compression_bits"],
        )
        exp_elapsed = time.perf_counter() - exp_start
        trained_models[config["name"]] = model

        final_round = dict(history[-1])
        final_round["experiment"] = config["name"]
        final_round["total_experiment_sec"] = exp_elapsed
        final_results.append(final_round)

    final_results.sort(
        key=lambda row: (row["accuracy"], row["compression_ratio"], -row["log_loss"]),
        reverse=True,
    )
    baseline_result = next(row for row in final_results if row["experiment"] == "fedavg_fp32")
    best_compressed = max(
        (row for row in final_results if row["experiment"] != "fedavg_fp32"),
        key=lambda row: (row["accuracy"], row["compression_ratio"], -row["log_loss"]),
    )

    accuracy_retention = best_compressed["accuracy"] / max(baseline_result["accuracy"], 1e-12)
    sample = X_test[:20]
    explain(trained_models[best_compressed["experiment"]], sample)

    total_elapsed = time.perf_counter() - total_start

    print("Proper federated evaluation completed")
    print(
        "Setup:",
        f"samples={SAMPLES}, rounds={ROUNDS}, local_epochs={LOCAL_EPOCHS},",
        f"clients={NUM_CLIENTS}, non_iid={NON_IID}",
    )
    print("Final round metrics:")
    for row in final_results:
        print(
            f"  {row['experiment']}: "
            f"accuracy={row['accuracy']:.4f}, "
            f"macro_f1={row['macro_f1']:.4f}, "
            f"log_loss={row['log_loss']:.4f}, "
            f"compression={row['compression_ratio']:.2f}x, "
            f"bits/value={row['bits_per_value']:.2f}, "
            f"avg_latency={row['avg_client_latency_sec']:.3f}s, "
            f"exp_time={row['total_experiment_sec']:.2f}s"
        )

    print(
        "Best compressed result:",
        f"{best_compressed['experiment']} preserves {accuracy_retention:.2%} "
        "of FP32 accuracy while reducing communication cost.",
    )
    print("SHAP explanation generated for the best compressed model")
    print(f"Total pipeline time: {total_elapsed:.2f}s")


if __name__ == "__main__":
    main()
