import concurrent.futures
import copy
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, log_loss

from federated.client import FLClient
from federated.server import FLServer


def make_client_partitions(
    X: np.ndarray,
    y: np.ndarray,
    num_clients: int,
    non_iid: bool = True,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    if non_iid:
        order = np.argsort(y.reshape(-1))
        X = X[order]
        y = y[order]

    return list(np.array_split(X, num_clients)), list(np.array_split(y, num_clients))


def evaluate_model(
    model: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    device: str = "cpu",
) -> Dict[str, float]:
    model = model.to(device)
    model.eval()

    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y, dtype=torch.long, device=device).reshape(-1)

    with torch.inference_mode():
        logits = model(X_tensor)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        predictions = np.argmax(probabilities, axis=1)

    targets = y_tensor.cpu().numpy()
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "log_loss": float(log_loss(targets, probabilities, labels=[0, 1])),
    }


def run_federated_experiment(
    model_factory: Callable[[], torch.nn.Module],
    clients_data: Sequence[np.ndarray],
    clients_labels: Sequence[np.ndarray],
    X_test: np.ndarray,
    y_test: np.ndarray,
    rounds: int = 5,
    local_epochs: int = 1,
    lr: float = 0.001,
    compression_mode: str = "turboquant",
    compression_bits: int = 4,
    device: str = "cpu",
    batch_size: int | None = None,
    class_weights: Sequence[float] | None = None,
) -> Tuple[torch.nn.Module, List[Dict[str, Any]]]:
    base_model = model_factory().to(device)
    clients = [
        FLClient(
            model_factory(),
            data,
            labels,
            device=device,
            compression_mode=compression_mode,
            compression_bits=compression_bits,
            batch_size=batch_size,
            class_weights=class_weights,
        )
        for data, labels in zip(clients_data, clients_labels)
    ]
    server = FLServer(base_model)
    history: List[Dict[str, Any]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(clients)) as executor:
        for round_index in range(1, rounds + 1):
            global_state = copy.deepcopy(server.global_model.state_dict())
            client_updates = []
            latencies = []

            futures = [
                executor.submit(
                    client.train,
                    global_state=global_state,
                    epochs=local_epochs,
                    lr=lr,
                )
                for client in clients
            ]

            for future in concurrent.futures.as_completed(futures):
                update, latency = future.result()
                client_updates.append(update)
                latencies.append(latency)

            server.aggregate(client_updates)
            metrics = evaluate_model(server.global_model, X_test, y_test, device=device)
            compression_stats = server.last_compression_stats or {}

            history.append(
                {
                    "round": round_index,
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "log_loss": metrics["log_loss"],
                    "avg_client_latency_sec": float(np.mean(latencies)),
                    "compression_ratio": float(compression_stats.get("compression_ratio", 1.0)),
                    "bits_per_value": float(compression_stats.get("bits_per_value", 32.0)),
                }
            )

    return server.global_model, history
