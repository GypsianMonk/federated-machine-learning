import copy
from typing import Sequence

import torch


def fed_avg(
    global_model,
    client_updates,
    client_weights: Sequence[float] | None = None,
    update_type: str = "weights",
):
    if not client_updates:
        return global_model

    if client_weights is None:
        client_weights = [1.0] * len(client_updates)

    total_weight = float(sum(client_weights))
    if total_weight <= 0:
        raise ValueError("Client weights must sum to a positive value.")

    aggregated_state = copy.deepcopy(global_model.state_dict())

    for key in aggregated_state:
        reference = aggregated_state[key]

        if not reference.is_floating_point():
            if update_type == "weights":
                aggregated_state[key] = client_updates[0][key].clone()
            continue

        weighted_sum = sum(
            update[key].to(dtype=reference.dtype) * (weight / total_weight)
            for update, weight in zip(client_updates, client_weights)
        )

        if update_type == "weights":
            aggregated_state[key] = weighted_sum
        elif update_type == "delta":
            aggregated_state[key] = reference + weighted_sum
        else:
            raise ValueError(f"Unsupported update_type: {update_type}")

    global_model.load_state_dict(aggregated_state)
    return global_model
