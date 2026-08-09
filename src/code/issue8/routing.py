"""Deterministic routing cases shared by the three Issue8 modes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import torch


DuplicateBucket = Literal["zero", "low", "medium", "high", "max"]


@dataclass(frozen=True)
class RoutingCase:
    case_id: str
    seed: int
    num_tokens: int
    hidden: int
    topk: int
    num_experts: int
    world_size: int
    duplicate_bucket: str
    rank_duplicate_ratio: float
    expert_duplicate_ratio: float

    def metadata(self) -> dict[str, object]:
        return asdict(self)


_COLLIDING_SLOTS: dict[str, int] = {
    "zero": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "max": 99,
}


def _validate(num_experts: int, world_size: int, topk: int) -> int:
    if num_experts <= 0 or world_size <= 0 or topk <= 0:
        raise ValueError("num_experts, world_size, and topk must be positive")
    if num_experts % world_size:
        raise ValueError("num_experts must be divisible by world_size")
    if topk > num_experts:
        raise ValueError("topk cannot exceed num_experts")
    return num_experts // world_size


def destination_ranks(topk_idx: torch.Tensor, experts_per_rank: int) -> torch.Tensor:
    """Map expert ids to destination ranks, preserving the top-k axis."""
    if topk_idx.ndim != 2:
        raise ValueError("topk_idx must have shape [tokens, topk]")
    return topk_idx.to(torch.int64) // experts_per_rank


def rank_duplicate_ratio(topk_idx: torch.Tensor, experts_per_rank: int) -> float:
    """Mean fraction of top-k slots that collide at their destination rank.

    It is 1 - mean_t(|unique(destination_rank[t])| / K).  This is the
    duplicate quantity that Mode B can merge locally; repeated expert ids by
    themselves are not sufficient.
    """
    ranks = destination_ranks(topk_idx, experts_per_rank)
    return float(_row_rank_duplicates(ranks))


def _row_rank_duplicates(ranks: torch.Tensor) -> torch.Tensor:
    unique_count = torch.tensor(
        [torch.unique(row).numel() for row in ranks], dtype=torch.float64
    )
    return 1.0 - unique_count.mean() / ranks.shape[1]


def _expert_duplicate_ratio(topk_idx: torch.Tensor) -> float:
    unique_count = torch.tensor(
        [torch.unique(row).numel() for row in topk_idx], dtype=torch.float64
    )
    return float(1.0 - unique_count.mean() / topk_idx.shape[1])


def build_case(
    *,
    seed: int,
    num_tokens: int,
    hidden: int,
    topk: int,
    num_experts: int,
    world_size: int,
    duplicate_bucket: DuplicateBucket,
) -> tuple[RoutingCase, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build one reproducible activation/routing/weight tuple.

    The generator deliberately controls destination-rank collision. Expert
    indices remain distinct within a row whenever possible, separating rank
    duplicate from accidental duplicate expert selection.
    """
    experts_per_rank = _validate(num_experts, world_size, topk)
    if duplicate_bucket not in _COLLIDING_SLOTS:
        raise ValueError(f"unknown duplicate bucket: {duplicate_bucket}")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn((num_tokens, hidden), generator=generator, dtype=torch.float32, device="cpu")
    weights = torch.rand((num_tokens, topk), generator=generator, dtype=torch.float32, device="cpu")
    weights = weights / weights.sum(dim=1, keepdim=True)

    collision_slots = min(_COLLIDING_SLOTS[duplicate_bucket], topk - 1)
    rows: list[list[int]] = []
    for token in range(num_tokens):
        base_rank = token % world_size
        ranks = [base_rank] * (collision_slots + 1)
        for offset in range(1, topk - collision_slots):
            ranks.append((base_rank + offset) % world_size)
        row = [rank * experts_per_rank + ((token + slot) % experts_per_rank) for slot, rank in enumerate(ranks)]
        rows.append(row)
    topk_idx = torch.tensor(rows, dtype=torch.int64, device="cpu")
    ranks = destination_ranks(topk_idx, experts_per_rank)
    rank_ratio = float(_row_rank_duplicates(ranks))
    expert_ratio = _expert_duplicate_ratio(topk_idx)
    case_id = (
        f"s{seed}-t{num_tokens}-h{hidden}-k{topk}-e{num_experts}-"
        f"w{world_size}-dup{duplicate_bucket}"
    )
    case = RoutingCase(
        case_id=case_id,
        seed=seed,
        num_tokens=num_tokens,
        hidden=hidden,
        topk=topk,
        num_experts=num_experts,
        world_size=world_size,
        duplicate_bucket=duplicate_bucket,
        rank_duplicate_ratio=rank_ratio,
        expert_duplicate_ratio=expert_ratio,
    )
    return case, x, topk_idx, weights
