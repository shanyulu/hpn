# Copyright (c) 2025, TENCENT CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information.

from __future__ import annotations

from issue4.data_generator import deterministic_tensor, numel_from_size, tensor_raw_bytes, tensor_sha256


def test_numel_from_size() -> None:
    assert numel_from_size(1024, "float32") == 256
    assert numel_from_size(1024, "float16") == 512
    assert numel_from_size(1024, "bfloat16") == 512


def test_same_tuple_same_bytes() -> None:
    a = deterministic_tensor("float32", 1024, seed=7, rank=2, call_index=3, device="cpu")
    b = deterministic_tensor("float32", 1024, seed=7, rank=2, call_index=3, device="cpu")
    assert tensor_raw_bytes(a) == tensor_raw_bytes(b)
    assert tensor_sha256(a) == tensor_sha256(b)


def test_different_rank_and_call_different_bytes() -> None:
    base = deterministic_tensor("float32", 1024, seed=7, rank=0, call_index=0, device="cpu")
    other_rank = deterministic_tensor("float32", 1024, seed=7, rank=1, call_index=0, device="cpu")
    other_call = deterministic_tensor("float32", 1024, seed=7, rank=0, call_index=1, device="cpu")
    assert tensor_sha256(base) != tensor_sha256(other_rank)
    assert tensor_sha256(base) != tensor_sha256(other_call)


def test_bfloat16_raw_bytes_available() -> None:
    tensor = deterministic_tensor("bfloat16", 128, seed=11, rank=0, call_index=0, device="cpu")
    assert len(tensor_raw_bytes(tensor)) == 256
