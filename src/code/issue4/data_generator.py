#!/usr/bin/env python3
# Copyright (c) 2025, TENCENT CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information.

"""Deterministic rank-distinct tensor generation for ISSUE4.

The generator intentionally does not use PRNG state.  It maps
``(seed, rank, call_index, element_index, dtype)`` to finite IEEE-like bit
patterns through SplitMix64-style integer mixing.  The same tuple therefore
recreates exactly the same input bytes across independent worker processes.
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch


SUPPORTED_DTYPES = ("float32", "float16", "bfloat16")


@dataclass(frozen=True)
class DTypeInfo:
    name: str
    itemsize: int
    torch_dtype: torch.dtype


DTYPE_INFO = {
    "float32": DTypeInfo("float32", 4, torch.float32),
    "float16": DTypeInfo("float16", 2, torch.float16),
    "bfloat16": DTypeInfo("bfloat16", 2, torch.bfloat16),
}


def dtype_info(dtype: str) -> DTypeInfo:
    try:
        return DTYPE_INFO[dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype {dtype!r}; expected one of {SUPPORTED_DTYPES}") from exc


def numel_from_size(size_bytes: int, dtype: str) -> int:
    info = dtype_info(dtype)
    if size_bytes <= 0:
        raise ValueError("size_bytes must be positive")
    if size_bytes % info.itemsize != 0:
        raise ValueError(f"size_bytes={size_bytes} is not divisible by itemsize={info.itemsize} for {dtype}")
    return size_bytes // info.itemsize


def parse_size_bytes(value: str) -> int:
    """Parse sizes such as 1024, 1KiB, 64K, 16MiB, 128M."""

    text = value.strip()
    if not text:
        raise argparse.ArgumentTypeError("size must not be empty")
    units = {
        "b": 1,
        "k": 1000,
        "kb": 1000,
        "m": 1000**2,
        "mb": 1000**2,
        "g": 1000**3,
        "gb": 1000**3,
        "kib": 1024,
        "mib": 1024**2,
        "gib": 1024**3,
    }
    lower = text.lower()
    for suffix in sorted(units, key=len, reverse=True):
        if lower.endswith(suffix):
            number = lower[: -len(suffix)].strip()
            try:
                return int(float(number) * units[suffix])
            except ValueError as exc:
                raise argparse.ArgumentTypeError(f"invalid size {value!r}") from exc
    try:
        return int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid size {value!r}") from exc


def _splitmix64(values: np.ndarray) -> np.ndarray:
    """Vectorized SplitMix64 finalizer over uint64 values."""

    x = values.astype(np.uint64, copy=False)
    x = x + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def _hash_stream(numel: int, seed: int, rank: int, call_index: int, stream_id: int = 0) -> np.ndarray:
    idx = np.arange(numel, dtype=np.uint64)
    key = idx
    mask = 0xFFFFFFFFFFFFFFFF
    key ^= np.uint64(seed & mask)
    key ^= np.uint64(((rank + 1) * 0xD1B54A32D192ED03) & mask)
    key ^= np.uint64(((call_index + 1) * 0xABC98388FB8FAC03) & mask)
    key ^= np.uint64(((stream_id + 1) * 0x8CB92BA72F3D8DD7) & mask)
    return _splitmix64(key)


def deterministic_numpy(dtype: str, numel: int, seed: int, rank: int, call_index: int) -> np.ndarray:
    """Return deterministic finite values as a NumPy array.

    Different ranks receive different values for the same element.  The values
    deliberately cover a modest exponent range to make floating-point reduction
    order observable when algorithms differ, while avoiding NaN/Inf inputs.
    """

    if numel < 0:
        raise ValueError("numel must be non-negative")
    h = _hash_stream(numel, seed=seed, rank=rank, call_index=call_index)

    if dtype == "float32":
        sign = ((h >> np.uint64(63)).astype(np.uint32) & np.uint32(1)) << np.uint32(31)
        exponent = (np.uint32(104) + ((h >> np.uint64(24)).astype(np.uint32) % np.uint32(47))) << np.uint32(23)
        mantissa = h.astype(np.uint32) & np.uint32(0x007FFFFF)
        return (sign | exponent | mantissa).view(np.float32)

    if dtype == "float16":
        sign = ((h >> np.uint64(63)).astype(np.uint16) & np.uint16(1)) << np.uint16(15)
        exponent = (np.uint16(5) + ((h >> np.uint64(20)).astype(np.uint16) % np.uint16(20))) << np.uint16(10)
        mantissa = h.astype(np.uint16) & np.uint16(0x03FF)
        return (sign | exponent | mantissa).view(np.float16)

    if dtype == "bfloat16":
        sign = ((h >> np.uint64(63)).astype(np.uint32) & np.uint32(1)) << np.uint32(31)
        exponent = (np.uint32(104) + ((h >> np.uint64(24)).astype(np.uint32) % np.uint32(47))) << np.uint32(23)
        mantissa_top = (h.astype(np.uint32) & np.uint32(0x0000007F)) << np.uint32(16)
        bits32 = sign | exponent | mantissa_top
        return bits32.view(np.float32)

    raise ValueError(f"unsupported dtype {dtype!r}; expected one of {SUPPORTED_DTYPES}")


def deterministic_tensor(dtype: str, numel: int, seed: int, rank: int, call_index: int, device: torch.device | str) -> torch.Tensor:
    np_values = deterministic_numpy(dtype, numel, seed, rank, call_index)
    tensor = torch.from_numpy(np_values.copy())
    return tensor.to(device=device, dtype=dtype_info(dtype).torch_dtype, non_blocking=False)


def tensor_raw_bytes(tensor: torch.Tensor) -> bytes:
    """Return exact tensor storage bytes in logical contiguous order."""

    cpu = tensor.detach().contiguous().cpu()
    byte_view = cpu.view(torch.uint8)
    return byte_view.numpy().tobytes()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    return sha256_bytes(tensor_raw_bytes(tensor))


def write_tensor_raw(tensor: torch.Tensor, path: str) -> str:
    data = tensor_raw_bytes(tensor)
    with open(path, "wb") as f:
        f.write(data)
    return sha256_bytes(data)


def iter_relevant_nccl_env(env: dict[str, str]) -> Iterable[tuple[str, str]]:
    prefixes = ("NCCL", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "CUBLAS", "TORCH")
    for key in sorted(env):
        if key.startswith(prefixes):
            yield key, env[key]


def _main() -> int:
    parser = argparse.ArgumentParser(description="Generate and hash deterministic ISSUE4 tensors")
    parser.add_argument("--dtype", choices=SUPPORTED_DTYPES, default="float32")
    parser.add_argument("--size-bytes", type=parse_size_bytes, default=parse_size_bytes("1KiB"))
    parser.add_argument("--seed", type=int, default=202604)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--call", type=int, default=0)
    args = parser.parse_args()

    numel = numel_from_size(args.size_bytes, args.dtype)
    tensor = deterministic_tensor(args.dtype, numel, args.seed, args.rank, args.call, device="cpu")
    print({
        "dtype": args.dtype,
        "size_bytes": args.size_bytes,
        "numel": numel,
        "rank": args.rank,
        "call": args.call,
        "sha256": tensor_sha256(tensor),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
