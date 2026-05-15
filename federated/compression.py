from typing import Any, Dict, Mapping

import torch

from federated.turboquant import (
    dequantize_state_dict as dequantize_turboquant_state_dict,
    is_turboquant_payload,
    quantize_state_dict as quantize_turboquant_state_dict,
)
from federated.uniform_quant import (
    dequantize_state_dict as dequantize_uniform_state_dict,
    is_uniform_quant_payload,
    quantize_state_dict as quantize_uniform_state_dict,
)
from federated.topk_sparsify import (
    dequantize_state_dict as dequantize_topk_state_dict,
    is_topk_payload,
    quantize_state_dict as quantize_topk_state_dict,
)


def clone_state_dict(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        name: tensor.detach().clone() if tensor.device.type == "cpu"
        else tensor.detach().cpu().clone()
        for name, tensor in state_dict.items()
    }


def subtract_state_dicts(
    minuend: Mapping[str, torch.Tensor],
    subtrahend: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    result: Dict[str, torch.Tensor] = {}

    for name, tensor in minuend.items():
        reference = subtrahend[name]
        if tensor.is_floating_point():
            result[name] = tensor.detach().cpu() - reference.detach().cpu()
        else:
            result[name] = tensor.detach().cpu().clone()

    return result


def add_state_dicts(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    result: Dict[str, torch.Tensor] = {}

    for name, tensor in left.items():
        other = right[name]
        if tensor.is_floating_point():
            result[name] = tensor.detach().cpu() + other.detach().cpu()
        else:
            result[name] = tensor.detach().cpu().clone()

    return result


def zero_state_dict_like(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    zeros: Dict[str, torch.Tensor] = {}

    for name, tensor in state_dict.items():
        base = tensor.detach().cpu()
        if base.is_floating_point():
            zeros[name] = torch.zeros_like(base)
        else:
            zeros[name] = base.clone()

    return zeros


def serialize_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    compression_mode: str = "turboquant",
    compression_bits: int = 8,
) -> Dict[str, Any] | Dict[str, torch.Tensor]:
    cloned_state_dict = clone_state_dict(state_dict)

    if compression_mode == "none":
        return cloned_state_dict

    if compression_mode == "turboquant":
        return quantize_turboquant_state_dict(
            cloned_state_dict,
            num_bits=compression_bits,
        )

    if compression_mode == "uniform":
        return quantize_uniform_state_dict(
            cloned_state_dict,
            num_bits=compression_bits,
        )

    if compression_mode == "topk":
        return quantize_topk_state_dict(
            cloned_state_dict,
            num_bits=compression_bits,
        )

    raise ValueError(f"Unsupported compression mode: {compression_mode}")


def is_serialized_payload(payload: Any) -> bool:
    return (
        is_turboquant_payload(payload)
        or is_uniform_quant_payload(payload)
        or is_topk_payload(payload)
    )


def deserialize_state_dict(payload: Any) -> Dict[str, torch.Tensor]:
    if is_turboquant_payload(payload):
        return dequantize_turboquant_state_dict(payload)

    if is_uniform_quant_payload(payload):
        return dequantize_uniform_state_dict(payload)

    if is_topk_payload(payload):
        return dequantize_topk_state_dict(payload)

    raise ValueError("Unsupported compressed payload.")


def summarize_payloads(payloads: list[Any]) -> Dict[str, float] | None:
    original_bits = 0
    quantized_bits = 0
    total_values = 0

    for payload in payloads:
        if is_serialized_payload(payload):
            stats = payload["stats"]
            original_bits += stats["original_bits"]
            quantized_bits += stats["quantized_bits"]
            total_values += stats["total_values"]

    if original_bits == 0:
        return None

    return {
        "original_bits": original_bits,
        "quantized_bits": quantized_bits,
        "total_values": total_values,
        "compression_ratio": original_bits / max(quantized_bits, 1),
        "bits_per_value": quantized_bits / max(total_values, 1),
    }
