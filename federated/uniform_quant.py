import math
from typing import Any, Dict, Mapping

import torch


def is_uniform_quant_payload(payload: Any) -> bool:
    return isinstance(payload, dict) and payload.get("format") == "uniform_quant_v1"


def quantize_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    num_bits: int = 8,
) -> Dict[str, Any]:
    if num_bits < 2 or num_bits > 8:
        raise ValueError("Uniform quantization bit-width must be between 2 and 8.")

    payload: Dict[str, Any] = {
        "format": "uniform_quant_v1",
        "num_bits": num_bits,
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

        compressed = _quantize_tensor(tensor.float(), num_bits=num_bits)
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
    if not is_uniform_quant_payload(payload):
        raise ValueError("Expected a uniform quantization payload.")

    state_dict: Dict[str, torch.Tensor] = {}
    for name, tensor_payload in payload["tensors"].items():
        if tensor_payload.get("kind") == "raw":
            state_dict[name] = tensor_payload["tensor"].clone()
            continue

        tensor = _dequantize_tensor(tensor_payload)
        dtype = getattr(torch, tensor_payload["dtype"])
        state_dict[name] = tensor.to(dtype=dtype)

    return state_dict


def _quantize_tensor(tensor: torch.Tensor, num_bits: int) -> Dict[str, Any]:
    shape = list(tensor.shape)
    flat = tensor.reshape(-1)
    min_val = flat.min().item()
    max_val = flat.max().item()

    if min_val == max_val:
        return {
            "kind": "quantized",
            "shape": shape,
            "numel": flat.numel(),
            "num_bits": num_bits,
            "min_val": min_val,
            "max_val": max_val,
            "packed_codes": torch.empty(0, dtype=torch.uint8),
            "estimated_bits": 128,
        }

    levels = (2**num_bits) - 1
    scale = (max_val - min_val) / levels
    normalized = torch.clamp(torch.round((flat - min_val) / scale), 0, levels)
    codes = normalized.to(torch.uint8)
    packed_codes = _pack_codes(codes, num_bits)

    return {
        "kind": "quantized",
        "shape": shape,
        "numel": flat.numel(),
        "num_bits": num_bits,
        "min_val": min_val,
        "max_val": max_val,
        "packed_codes": packed_codes,
        "estimated_bits": (packed_codes.numel() * 8) + 128,
    }


def _dequantize_tensor(payload: Mapping[str, Any]) -> torch.Tensor:
    if payload["packed_codes"].numel() == 0:
        return torch.full(payload["shape"], payload["min_val"], dtype=torch.float32)

    num_bits = payload["num_bits"]
    levels = (2**num_bits) - 1
    scale = (payload["max_val"] - payload["min_val"]) / levels
    codes = _unpack_codes(payload["packed_codes"], payload["numel"], num_bits)
    values = codes.float() * scale + payload["min_val"]
    return values.reshape(payload["shape"])


def _pack_codes(codes: torch.Tensor, num_bits: int) -> torch.Tensor:
    if num_bits == 8:
        return codes.view(-1).to(torch.uint8).clone()

    if num_bits == 4:
        codes = codes.view(-1).long()
        pad = (2 - (codes.numel() % 2)) % 2
        if pad > 0:
            codes = torch.cat([codes, torch.zeros(pad, dtype=codes.dtype, device=codes.device)])
        packed = (codes[0::2] | (codes[1::2] << 4)).to(torch.uint8)
        return packed

    packed_length = math.ceil((codes.numel() * num_bits) / 8)
    packed = torch.zeros(packed_length, dtype=torch.uint8)
    bit_offset = 0

    for code in codes.tolist():
        byte_index = bit_offset // 8
        shift = bit_offset % 8
        packed[byte_index] |= (code << shift) & 0xFF

        if shift + num_bits > 8 and byte_index + 1 < packed_length:
            packed[byte_index + 1] |= code >> (8 - shift)

        bit_offset += num_bits

    return packed


def _unpack_codes(packed_codes: torch.Tensor, count: int, num_bits: int) -> torch.Tensor:
    if num_bits == 8:
        return packed_codes[:count].to(torch.uint8).clone()

    if num_bits == 4:
        codes = torch.empty(count, dtype=torch.uint8, device=packed_codes.device)
        codes_even = packed_codes & 0x0F
        codes_odd = (packed_codes >> 4) & 0x0F
        if count % 2 == 0:
            codes[0::2] = codes_even
            codes[1::2] = codes_odd
        else:
            codes[0::2] = codes_even[: count // 2 + 1]
            codes[1::2] = codes_odd[: count // 2]
        return codes

    codes = torch.zeros(count, dtype=torch.uint8)
    mask = (1 << num_bits) - 1
    bit_offset = 0
    source = packed_codes.tolist()

    for index in range(count):
        byte_index = bit_offset // 8
        shift = bit_offset % 8
        value = source[byte_index] >> shift

        if shift + num_bits > 8 and byte_index + 1 < len(source):
            value |= source[byte_index + 1] << (8 - shift)

        codes[index] = value & mask
        bit_offset += num_bits

    return codes
