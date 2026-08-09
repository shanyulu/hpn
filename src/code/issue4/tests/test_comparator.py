# Copyright (c) 2025, TENCENT CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information.

from __future__ import annotations

import struct

import numpy as np

from issue4.comparator import compare_raw_bytes


def f32_bytes(values: list[float]) -> bytes:
    return b"".join(struct.pack("<f", x) for x in values)


def test_exact_bytes_equal_float32() -> None:
    data = f32_bytes([1.0, -2.0, 3.5])
    result = compare_raw_bytes(data, data, "float32", [3])
    assert result.equal
    assert result.changed_elements == 0
    assert result.changed_bytes == 0
    assert result.changed_bits == 0


def test_single_bit_flip_localization_float32() -> None:
    a = bytearray(f32_bytes([1.0, 2.0, 3.0]))
    b = bytearray(a)
    b[4] ^= 0b0000_0100
    result = compare_raw_bytes(a, b, "float32", [3])
    assert not result.equal
    assert result.changed_elements == 1
    assert result.changed_bytes == 1
    assert result.changed_bits == 1
    assert result.first_differing_element == 1
    assert result.first_differing_byte == 4
    assert result.first_differing_bit == 34


def test_one_ulp_float32() -> None:
    a = bytearray(struct.pack("<I", 0x3F800000))
    b = bytearray(struct.pack("<I", 0x3F800001))
    result = compare_raw_bytes(a, b, "float32", [1])
    assert not result.equal
    assert result.max_unsigned_ulp == 1
    assert result.max_ordered_ulp == 1
    assert result.max_abs_error == np.float32(float.fromhex("0x1p-23")).item()


def test_multiple_element_changes() -> None:
    a = bytearray(f32_bytes([1.0, 2.0, 3.0, 4.0]))
    b = bytearray(a)
    b[0] ^= 1
    b[8] ^= 1
    b[15] ^= 0x80
    result = compare_raw_bytes(a, b, "float32", [4])
    assert not result.equal
    assert result.changed_elements == 3
    assert result.first_differing_element == 0
    assert result.first_mismatch_elements == [0, 2, 3]


def test_nan_inf_same_bits_are_equal() -> None:
    values = [float("nan"), float("inf"), float("-inf")]
    data = f32_bytes(values)
    result = compare_raw_bytes(data, data, "float32", [3])
    assert result.equal


def test_positive_zero_negative_zero_are_bitwise_different() -> None:
    a = struct.pack("<f", 0.0)
    b = struct.pack("<I", 0x80000000)
    result = compare_raw_bytes(a, b, "float32", [1])
    assert not result.equal
    assert result.first_value_a == 0.0
    assert result.first_value_b == -0.0
    assert result.max_ordered_ulp == 1


def test_float16_one_ulp() -> None:
    a = struct.pack("<H", 0x3C00)
    b = struct.pack("<H", 0x3C01)
    result = compare_raw_bytes(a, b, "float16", [1])
    assert not result.equal
    assert result.max_unsigned_ulp == 1
    assert result.max_ordered_ulp == 1
    assert result.first_differing_element == 0


def test_bfloat16_one_ulp() -> None:
    a = struct.pack("<H", 0x3F80)
    b = struct.pack("<H", 0x3F81)
    result = compare_raw_bytes(a, b, "bfloat16", [1])
    assert not result.equal
    assert result.max_unsigned_ulp == 1
    assert result.max_ordered_ulp == 1
    assert result.first_value_a == 1.0
    assert result.first_value_b > 1.0
