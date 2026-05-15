import hashlib
import math
from functools import lru_cache
from typing import Any, Dict, Mapping, Tuple

import torch


# Practical TurboQuant adaptation for FL updates:
# random structured rotation + scalar quantization on approximately Gaussian coordinates.


def is_turboquant_payload(payload: Any) -> bool:
    return isinstance(payload, dict) and payload.get("format") == "turboquant_v1"


def quantize_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    num_bits: int = 4,
) -> Dict[str, Any]:
    if num_bits < 2 or num_bits > 8:
        raise ValueError("TurboQuant bit-width must be between 2 and 8.")

    payload: Dict[str, Any] = {
        "format": "turboquant_v1",
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

        seed = _seed_from_name(name)
        compressed = _quantize_tensor(tensor.float(), num_bits=num_bits, seed=seed)
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
    if not is_turboquant_payload(payload):
        raise ValueError("Expected a TurboQuant payload.")

    state_dict: Dict[str, torch.Tensor] = {}
    for name, tensor_payload in payload["tensors"].items():
        kind = tensor_payload.get("kind")
        if kind == "raw":
            state_dict[name] = tensor_payload["tensor"].clone()
            continue

        tensor = _dequantize_tensor(tensor_payload)
        dtype = getattr(torch, tensor_payload["dtype"])
        state_dict[name] = tensor.to(dtype=dtype)

    return state_dict


def summarize_payloads(payloads: list[Any]) -> Dict[str, float] | None:
    original_bits = 0
    quantized_bits = 0
    total_values = 0

    for payload in payloads:
        if is_turboquant_payload(payload):
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


def _quantize_tensor(tensor: torch.Tensor, num_bits: int, seed: int) -> Dict[str, Any]:
    shape = list(tensor.shape)
    flat = tensor.reshape(-1)
    numel = flat.numel()
    padded_length = _next_power_of_two(max(numel, 1))
    norm = flat.norm(p=2).item()

    if norm == 0.0:
        return {
            "kind": "quantized",
            "shape": shape,
            "numel": numel,
            "padded_length": padded_length,
            "num_bits": num_bits,
            "norm": 0.0,
            "seed": seed,
            "packed_codes": torch.empty(0, dtype=torch.uint8),
            "estimated_bits": 64,
        }

    padded = torch.zeros(padded_length, dtype=torch.float32)
    padded[:numel] = flat / norm

    rotated = _apply_rotation(padded, seed)
    scaled = rotated * math.sqrt(padded_length)

    codebook = _normal_codebook(num_bits)
    boundaries = _codebook_boundaries(codebook)
    codes = torch.bucketize(scaled, boundaries).to(torch.uint8)
    packed_codes = _pack_codes(codes, num_bits)

    # Estimated transport cost: quantized values plus norm and seed metadata.
    estimated_bits = (packed_codes.numel() * 8) + 64

    return {
        "kind": "quantized",
        "shape": shape,
        "numel": numel,
        "padded_length": padded_length,
        "num_bits": num_bits,
        "norm": norm,
        "seed": seed,
        "packed_codes": packed_codes,
        "estimated_bits": estimated_bits,
    }


def _dequantize_tensor(payload: Mapping[str, Any]) -> torch.Tensor:
    if payload["norm"] == 0.0:
        return torch.zeros(payload["shape"], dtype=torch.float32)

    padded_length = payload["padded_length"]
    codebook = _normal_codebook(payload["num_bits"])
    codes = _unpack_codes(payload["packed_codes"], padded_length, payload["num_bits"])
    scaled = codebook[codes.long()]
    rotated = scaled / math.sqrt(padded_length)
    recovered = _invert_rotation(rotated, payload["seed"])
    tensor = recovered[: payload["numel"]] * payload["norm"]
    return tensor.reshape(payload["shape"])


def _seed_from_name(name: str) -> int:
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _next_power_of_two(value: int) -> int:
    return 1 if value <= 1 else 1 << (value - 1).bit_length()


def _apply_rotation(vector: torch.Tensor, seed: int) -> torch.Tensor:
    perm, signs = _rotation_params(vector.numel(), seed)
    prepared = vector[perm] * signs
    return _fwht(prepared)


def _invert_rotation(vector: torch.Tensor, seed: int) -> torch.Tensor:
    perm, signs = _rotation_params(vector.numel(), seed)
    prepared = _fwht(vector)
    recovered = torch.empty_like(prepared)
    recovered[perm] = prepared * signs
    return recovered


@lru_cache(maxsize=32)
def _rotation_params(length: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    perm = torch.randperm(length, generator=generator)
    signs = torch.randint(0, 2, (length,), generator=generator, dtype=torch.int8)
    signs = (signs.float() * 2.0) - 1.0
    return perm, signs


def _fwht(vector: torch.Tensor) -> torch.Tensor:
    output = vector.clone()
    width = 1
    total = output.numel()
    buf = torch.empty(total, dtype=output.dtype, device=output.device)

    while width < total:
        output_view = output.view(-1, width * 2)
        left = output_view[:, :width]
        right = output_view[:, width:]
        buf_view = buf.view(-1, width * 2)
        buf_view[:, :width] = left + right
        buf_view[:, width:] = left - right
        output, buf = buf, output
        width *= 2

    return output.reshape(total) / math.sqrt(total)


def _codebook_boundaries(codebook: torch.Tensor) -> torch.Tensor:
    return (codebook[:-1] + codebook[1:]) / 2.0


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


@lru_cache(maxsize=None)
def _normal_codebook(num_bits: int) -> torch.Tensor:
    levels = 2 ** num_bits
    generator = torch.Generator()
    generator.manual_seed(2026 + num_bits)
    samples = torch.randn(50000, generator=generator)

    quantiles = torch.linspace(0.5 / levels, 1.0 - (0.5 / levels), levels)
    normal = torch.distributions.Normal(0.0, 1.0)
    centroids = normal.icdf(quantiles)

    for _ in range(20):
        boundaries = _codebook_boundaries(centroids)
        assignments = torch.bucketize(samples, boundaries)
        updated = centroids.clone()

        for index in range(levels):
            mask = assignments == index
            if mask.any():
                updated[index] = samples[mask].mean()

        if torch.max(torch.abs(updated - centroids)).item() < 1e-4:
            centroids = updated
            break
        centroids = updated

    return centroids.to(torch.float32)
