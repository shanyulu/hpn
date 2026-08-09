"""Numerical reference and error metrics for Issue8."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class PrecisionMetrics:
    max_abs_error: float
    mean_abs_error: float
    rmse: float
    relative_l2_error: float
    max_relative_error: float
    mismatch_count: int
    nan_count: int
    inf_count: int

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def fp32_full_return_reference(values: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    """Deterministic FP32 full-return reduction over the replica axis."""
    values = values.to(torch.float32)
    if weights is not None:
        values = values * weights.to(torch.float32).unsqueeze(-1)
    return values.sum(dim=1, dtype=torch.float32)


def error_metrics(
    observed: torch.Tensor,
    reference: torch.Tensor,
    *,
    relative_epsilon: float = 1e-8,
    mismatch_atol: float = 0.0,
) -> PrecisionMetrics:
    observed32 = observed.to(torch.float32)
    reference32 = reference.to(torch.float32)
    diff = (observed32 - reference32).abs()
    finite = torch.isfinite(observed32) & torch.isfinite(reference32)
    finite_diff = torch.where(finite, diff, torch.zeros_like(diff))
    denom = reference32.abs().clamp_min(relative_epsilon)
    rel = finite_diff / denom
    ref_norm = torch.linalg.vector_norm(reference32)
    diff_norm = torch.linalg.vector_norm(finite_diff)
    return PrecisionMetrics(
        max_abs_error=float(finite_diff.max().item()),
        mean_abs_error=float(finite_diff.mean().item()),
        rmse=float(torch.sqrt(torch.mean(finite_diff.square())).item()),
        relative_l2_error=float((diff_norm / ref_norm.clamp_min(relative_epsilon)).item()),
        max_relative_error=float(rel.max().item()),
        mismatch_count=int((finite_diff > mismatch_atol).sum().item()),
        nan_count=int((torch.isnan(observed32) | torch.isnan(reference32)).sum().item()),
        inf_count=int((torch.isinf(observed32) | torch.isinf(reference32)).sum().item()),
    )
