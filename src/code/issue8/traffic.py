"""Traffic accounting with explicit evidence levels."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import torch

from routing import destination_ranks


ModeName = Literal["A", "B", "C"]


@dataclass(frozen=True)
class TrafficAccounting:
    mode: ModeName
    evidence_level: str
    logical_payload_tokens: int
    payload_bytes: int
    metadata_bytes: int
    total_bytes: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def dtype_bytes(dtype: str) -> int:
    sizes = {"bf16": 2, "fp16": 2, "fp32": 4}
    try:
        return sizes[dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype: {dtype}") from exc


def logical_return_tokens(
    topk_idx: torch.Tensor, *, experts_per_rank: int, mode: ModeName
) -> int:
    """Payload units returned by the source-level layouts.

    A/B return one unit per distinct destination rank. C returns every valid
    top-k replica. The value excludes transport protocol overhead and does not
    claim a hardware counter measurement.
    """
    if mode == "C":
        return int(topk_idx.numel())
    ranks = destination_ranks(topk_idx, experts_per_rank)
    return int(sum(torch.unique(row).numel() for row in ranks))


def analytical_traffic(
    topk_idx: torch.Tensor,
    *,
    hidden: int,
    experts_per_rank: int,
    dtype: str,
    mode: ModeName,
    metadata_bytes_per_payload: int = 0,
) -> TrafficAccounting:
    tokens = logical_return_tokens(topk_idx, experts_per_rank=experts_per_rank, mode=mode)
    payload_bytes = tokens * hidden * dtype_bytes(dtype)
    metadata_bytes = tokens * metadata_bytes_per_payload
    return TrafficAccounting(
        mode=mode,
        evidence_level="L1_analytical_logical_payload",
        logical_payload_tokens=tokens,
        payload_bytes=payload_bytes,
        metadata_bytes=metadata_bytes,
        total_bytes=payload_bytes + metadata_bytes,
    )


def software_instrumented_traffic(
    *,
    mode: ModeName,
    payload_slots_from_metadata: int,
    hidden: int,
    dtype: str,
    metadata_bytes: int = 0,
) -> TrafficAccounting:
    """Record a Level-2 metadata-derived payload count from a real run."""
    payload_bytes = payload_slots_from_metadata * hidden * dtype_bytes(dtype)
    return TrafficAccounting(
        mode=mode,
        evidence_level="L2_software_instrumented_metadata",
        logical_payload_tokens=payload_slots_from_metadata,
        payload_bytes=payload_bytes,
        metadata_bytes=metadata_bytes,
        total_bytes=payload_bytes + metadata_bytes,
    )
