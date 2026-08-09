from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision import error_metrics, fp32_full_return_reference
from merge_measurements import merge
from report import decision_rows
from routing import build_case, rank_duplicate_ratio
from traffic import analytical_traffic


def test_rank_duplicate_buckets_are_controlled() -> None:
    ratios = []
    for bucket in ("zero", "low", "medium", "high", "max"):
        case, _x, routing, _weights = build_case(
            seed=7, num_tokens=8, hidden=16, topk=4, num_experts=16,
            world_size=4, duplicate_bucket=bucket,
        )
        ratios.append(case.rank_duplicate_ratio)
        assert rank_duplicate_ratio(routing, 4) == case.rank_duplicate_ratio
    assert ratios == sorted(ratios)
    assert ratios[0] == 0.0
    assert ratios[-1] == 0.75


def test_mode_c_has_one_payload_per_replica() -> None:
    _case, _x, routing, _weights = build_case(
        seed=7, num_tokens=8, hidden=16, topk=4, num_experts=16,
        world_size=4, duplicate_bucket="max",
    )
    a = analytical_traffic(routing, hidden=16, experts_per_rank=4, dtype="bf16", mode="A")
    b = analytical_traffic(routing, hidden=16, experts_per_rank=4, dtype="bf16", mode="B")
    c = analytical_traffic(routing, hidden=16, experts_per_rank=4, dtype="bf16", mode="C")
    assert a.logical_payload_tokens == b.logical_payload_tokens == 8
    assert c.logical_payload_tokens == 32
    assert c.payload_bytes == 4 * b.payload_bytes


def test_precision_metrics_guard_near_zero() -> None:
    replicas = torch.tensor([[[1.0, 1.0e-9], [-1.0, 1.0e-9]]])
    reference = fp32_full_return_reference(replicas)
    metrics = error_metrics(torch.zeros_like(reference), reference)
    assert metrics.nan_count == 0
    assert metrics.inf_count == 0
    assert 0.0 < metrics.relative_l2_error <= 1.0
    assert metrics.max_relative_error > 0.0


def test_decision_requires_measured_critical_path_latency() -> None:
    rows = [
        {"duplicate_bucket": "high", "message_bucket": "medium", "mode": "A", "world_size": "1", "completion_time_evidence": "GPU_event_max_rank_completion", "measured_combine_completion_median_ms": "0.1"},
        {"duplicate_bucket": "high", "message_bucket": "medium", "mode": "C", "world_size": "4", "completion_time_evidence": "GPU_event_max_rank_completion", "measured_combine_completion_median_ms": "1.2"},
        {"duplicate_bucket": "high", "message_bucket": "medium", "mode": "B", "world_size": "4", "completion_time_evidence": "GPU_event_max_rank_completion", "measured_combine_completion_median_ms": "0.9"},
    ]
    decisions = decision_rows(rows)
    assert len(decisions) == 1
    assert decisions[0]["recommended_mode"] == "B"


def test_merge_rejects_one_rank_cuda_event_records() -> None:
    record = {
        "case_id": "one-rank",
        "mode": "A",
        "status": "measured",
        "execution_scope": "one_rank",
        "completion_time_evidence": "one_rank_cuda_event_not_multi_rank",
        "world_size": 1,
        "critical_path_samples_ms": [0.1],
    }
    try:
        merge([record])
    except ValueError as exc:
        assert "only multi-rank" in str(exc)
    else:
        raise AssertionError("one-rank event record was accepted for aggregation")


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
            print(name)
