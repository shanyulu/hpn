#!/usr/bin/env python3
# Copyright (c) 2025, TENCENT CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information.

"""Raw-byte and floating-point forensic comparator for ISSUE4."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np


FLOAT_DTYPES = ("float32", "float16", "bfloat16")


@dataclass(frozen=True)
class CompareResult:
    equal: bool
    dtype: str
    shape: list[int]
    size_bytes: int
    numel: int
    changed_elements: int
    changed_bytes: int
    changed_bits: int
    first_differing_element: int | None
    first_differing_byte: int | None
    first_differing_bit: int | None
    first_value_a: Any = None
    first_value_b: Any = None
    first_raw_a_hex: str | None = None
    first_raw_b_hex: str | None = None
    max_abs_error: float | None = None
    max_rel_error: float | None = None
    max_unsigned_ulp: int | None = None
    max_ordered_ulp: int | None = None
    first_mismatch_elements: list[int] = field(default_factory=list)
    byte_mismatch_histogram_16: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def dtype_itemsize(dtype: str) -> int:
    if dtype == "float32":
        return 4
    if dtype in ("float16", "bfloat16"):
        return 2
    raise ValueError(f"unsupported dtype {dtype!r}")


def _dtype_raw_uint(dtype: str) -> np.dtype:
    if dtype == "float32":
        return np.dtype("<u4")
    if dtype in ("float16", "bfloat16"):
        return np.dtype("<u2")
    raise ValueError(f"unsupported dtype {dtype!r}")


def _decode_values(raw: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "float32":
        return raw.view("<f4").astype(np.float64)
    if dtype == "float16":
        return raw.view("<f2").astype(np.float32)
    if dtype == "bfloat16":
        raw32 = raw.astype(np.uint32) << np.uint32(16)
        return raw32.view("<f4").astype(np.float32)
    raise ValueError(f"unsupported dtype {dtype!r}")


def _python_value(values: np.ndarray, index: int) -> Any:
    if index is None or index < 0:
        return None
    value = values[index]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
    return value


def _bit_count_uint8(values: np.ndarray) -> int:
    table = np.array([int(i).bit_count() for i in range(256)], dtype=np.uint8)
    return int(table[values].sum(dtype=np.uint64))


def _first_set_bit_lsb(byte_value: int) -> int:
    for bit in range(8):
        if byte_value & (1 << bit):
            return bit
    raise ValueError("byte_value has no set bit")


def _ordered_float_int(raw: np.ndarray, bitwidth: int) -> np.ndarray:
    """Map IEEE sign-magnitude bits to monotonically ordered unsigned ints."""

    raw64 = raw.astype(np.uint64)
    sign = np.uint64(1 << (bitwidth - 1))
    mask = np.uint64((1 << bitwidth) - 1)
    return np.where((raw64 & sign) != 0, (~raw64) & mask, raw64 | sign)


def _ulp_metrics(raw_a: np.ndarray, raw_b: np.ndarray, dtype: str) -> tuple[int, int]:
    if dtype not in FLOAT_DTYPES:
        return 0, 0
    bitwidth = 32 if dtype == "float32" else 16
    a64 = raw_a.astype(np.int64)
    b64 = raw_b.astype(np.int64)
    unsigned = np.abs(a64 - b64)
    ordered_a = _ordered_float_int(raw_a, bitwidth).astype(np.int64)
    ordered_b = _ordered_float_int(raw_b, bitwidth).astype(np.int64)
    ordered = np.abs(ordered_a - ordered_b)
    return int(unsigned.max(initial=0)), int(ordered.max(initial=0))


def _error_metrics(values_a: np.ndarray, values_b: np.ndarray, changed_mask: np.ndarray) -> tuple[float, float]:
    if not changed_mask.any():
        return 0.0, 0.0
    a = values_a[changed_mask].astype(np.float64)
    b = values_b[changed_mask].astype(np.float64)
    finite = np.isfinite(a) & np.isfinite(b)
    if not finite.any():
        return float("nan"), float("nan")
    diff = np.abs(a[finite] - b[finite])
    max_abs = float(diff.max(initial=0.0))
    denom = np.maximum(np.abs(a[finite]), np.finfo(np.float64).tiny)
    max_rel = float((diff / denom).max(initial=0.0))
    return max_abs, max_rel


def _histogram_16(byte_diff_mask: np.ndarray) -> list[int]:
    if byte_diff_mask.size == 0:
        return [0] * 16
    indices = np.nonzero(byte_diff_mask)[0]
    if indices.size == 0:
        return [0] * 16
    bins = np.linspace(0, byte_diff_mask.size, 17, dtype=np.int64)
    hist, _ = np.histogram(indices, bins=bins)
    return [int(x) for x in hist.tolist()]


def compare_raw_bytes(data_a: bytes | bytearray | memoryview, data_b: bytes | bytearray | memoryview, dtype: str, shape: Sequence[int]) -> CompareResult:
    itemsize = dtype_itemsize(dtype)
    shape_list = [int(x) for x in shape]
    numel = int(np.prod(shape_list, dtype=np.int64)) if shape_list else 1
    size_bytes = numel * itemsize
    if len(data_a) != len(data_b):
        raise ValueError(f"byte length mismatch: {len(data_a)} vs {len(data_b)}")
    if len(data_a) != size_bytes:
        raise ValueError(f"expected {size_bytes} bytes for dtype={dtype}, shape={shape_list}; got {len(data_a)}")

    bytes_a = np.frombuffer(data_a, dtype=np.uint8)
    bytes_b = np.frombuffer(data_b, dtype=np.uint8)
    xor = np.bitwise_xor(bytes_a, bytes_b)
    byte_diff = xor != 0
    changed_bytes = int(byte_diff.sum(dtype=np.int64))

    if changed_bytes == 0:
        return CompareResult(
            equal=True,
            dtype=dtype,
            shape=shape_list,
            size_bytes=size_bytes,
            numel=numel,
            changed_elements=0,
            changed_bytes=0,
            changed_bits=0,
            first_differing_element=None,
            first_differing_byte=None,
            first_differing_bit=None,
            max_abs_error=0.0,
            max_rel_error=0.0,
            max_unsigned_ulp=0,
            max_ordered_ulp=0,
            byte_mismatch_histogram_16=[0] * 16,
        )

    first_byte = int(np.nonzero(byte_diff)[0][0])
    first_bit_in_byte = _first_set_bit_lsb(int(xor[first_byte]))
    first_bit = first_byte * 8 + first_bit_in_byte
    elem_diff = byte_diff.reshape(numel, itemsize).any(axis=1)
    changed_elements = int(elem_diff.sum(dtype=np.int64))
    first_elem = int(np.nonzero(elem_diff)[0][0])

    raw_dtype = _dtype_raw_uint(dtype)
    raw_a = np.frombuffer(data_a, dtype=raw_dtype, count=numel)
    raw_b = np.frombuffer(data_b, dtype=raw_dtype, count=numel)
    values_a = _decode_values(raw_a, dtype)
    values_b = _decode_values(raw_b, dtype)
    max_abs, max_rel = _error_metrics(values_a, values_b, elem_diff)
    max_unsigned_ulp, max_ordered_ulp = _ulp_metrics(raw_a[elem_diff], raw_b[elem_diff], dtype)

    first_raw_a = bytes(bytes_a[first_elem * itemsize : (first_elem + 1) * itemsize].tolist())
    first_raw_b = bytes(bytes_b[first_elem * itemsize : (first_elem + 1) * itemsize].tolist())
    first_elements = np.nonzero(elem_diff)[0][:10]

    return CompareResult(
        equal=False,
        dtype=dtype,
        shape=shape_list,
        size_bytes=size_bytes,
        numel=numel,
        changed_elements=changed_elements,
        changed_bytes=changed_bytes,
        changed_bits=_bit_count_uint8(xor[byte_diff]),
        first_differing_element=first_elem,
        first_differing_byte=first_byte,
        first_differing_bit=first_bit,
        first_value_a=_python_value(values_a, first_elem),
        first_value_b=_python_value(values_b, first_elem),
        first_raw_a_hex=first_raw_a.hex(),
        first_raw_b_hex=first_raw_b.hex(),
        max_abs_error=max_abs,
        max_rel_error=max_rel,
        max_unsigned_ulp=max_unsigned_ulp,
        max_ordered_ulp=max_ordered_ulp,
        first_mismatch_elements=[int(x) for x in first_elements.tolist()],
        byte_mismatch_histogram_16=_histogram_16(byte_diff),
    )


def compare_files(path_a: str | Path, path_b: str | Path, dtype: str, shape: Sequence[int]) -> CompareResult:
    data_a = Path(path_a).read_bytes()
    data_b = Path(path_b).read_bytes()
    return compare_raw_bytes(data_a, data_b, dtype=dtype, shape=shape)


def _main() -> int:
    parser = argparse.ArgumentParser(description="Compare two raw tensor capture files bitwise")
    parser.add_argument("path_a")
    parser.add_argument("path_b")
    parser.add_argument("--dtype", choices=FLOAT_DTYPES, required=True)
    parser.add_argument("--shape", required=True, help="Comma-separated shape, e.g. 1024 or 4,1024")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    shape = [int(x) for x in args.shape.split(",") if x]
    result = compare_files(args.path_a, args.path_b, dtype=args.dtype, shape=shape)
    print(json.dumps(result.to_dict(), indent=2 if args.pretty else None, sort_keys=True, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
