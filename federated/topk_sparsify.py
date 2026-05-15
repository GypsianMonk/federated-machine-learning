"""
Top-K Sparsification for Federated Learning
============================================
Keeps only the top-K largest (by magnitude) values in each tensor,
zeroing the rest. This is a standard FL communication baseline.
"""

import math
from typing import Any, Dict, Mapping

import torch


def is_topk_payload(payload: Any) -> bool:
    return isinstance(payload, dict) and payload.get("format") == "topk_v1"


def quantize_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    num_bits: int = 4,
) -> Dict[str, Any]:
    """
    Top-K sparsification. The `num_bits` parameter controls the sparsity:
      - 8 bits -> keep ~25% of values (mild sparsification)
      - 4 bits -> keep ~12.5% of values (moderate)
      - 2 bits -> keep ~6.25% of values  (aggressive)
    """
    keep_ratio = num_bits / 32.0

    payload: Dict[str, Any] = {
        "format": "topk_v1",
        "keep_ratio": keep_ratio,
        "tensors": {},
    }
    original_bits = 0
    quantized_bits = 0
    total_values = 0

    for name, tensor in state_dict.items():
        tensor = tensor.detach().cpu()
        original_bits += tensor.numel() * tensor.element_size() * 8
        total_values += tensor.numel()

        if not tensor.is_floating_point():
            payload["tensors"][name] = {
                "kind": "raw",
                "tensor": tensor.clone(),
            }
            quantized_bits += tensor.numel() * tensor.element_size() * 8
            continue

        compressed = _sparsify_tensor(tensor.float(), keep_ratio)
        compressed["dtype"] = str(tensor.dtype).split(".")[-1]
        payload["tensors"][name] = compressed
        quantized_bits += compressed["estimated_bits"]

    payload["stats"] = {
        "original_bits": original_bits,
        "quantized_bits": quantized_bits,
        "total_values": total_values,
        "compression_ratio": original_bits / max(quantized_bits, 1),
        "bits_per_value": quantized_bits / max(total_values, 1),
    }
    return payload


def dequantize_state_dict(payload: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
    if not is_topk_payload(payload):
        raise ValueError("Expected a Top-K payload.")

    state_dict: Dict[str, torch.Tensor] = {}
    for name, entry in payload["tensors"].items():
        if entry.get("kind") == "raw":
            state_dict[name] = entry["tensor"].clone()
            continue

        tensor = _reconstruct_tensor(entry)
        dtype = getattr(torch, entry["dtype"])
        state_dict[name] = tensor.to(dtype=dtype)

    return state_dict


def _sparsify_tensor(tensor: torch.Tensor, keep_ratio: float) -> Dict[str, Any]:
    shape = list(tensor.shape)
    flat = tensor.reshape(-1)
    numel = flat.numel()
    k = max(1, int(math.ceil(numel * keep_ratio)))

    if numel == 0:
        return {
            "kind": "topk",
            "shape": shape,
            "numel": numel,
            "indices": torch.empty(0, dtype=torch.long),
            "values": torch.empty(0, dtype=torch.float32),
            "estimated_bits": 64,
        }

    _, top_indices = torch.topk(flat.abs(), k, sorted=False)
    top_values = flat[top_indices]

    # Cost: indices (64 bits each) + values (32 bits each) + metadata
    estimated_bits = k * (64 + 32) + 128

    return {
        "kind": "topk",
        "shape": shape,
        "numel": numel,
        "indices": top_indices,
        "values": top_values,
        "estimated_bits": estimated_bits,
    }


def _reconstruct_tensor(entry: Mapping[str, Any]) -> torch.Tensor:
    numel = entry["numel"]
    if numel == 0:
        return torch.zeros(entry["shape"], dtype=torch.float32)

    flat = torch.zeros(numel, dtype=torch.float32)
    flat[entry["indices"]] = entry["values"]
    return flat.reshape(entry["shape"])
