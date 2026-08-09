#!/usr/bin/env python3
"""Run one real DeepEP V2 combine mode under ``torchrun``.

The program deliberately has no synthetic latency fallback. A result receives
``GPU_event_max_rank_completion`` only after the DeepEP combine call completes
on every rank. Unsupported NCCL Gin or topology configurations write a blocked
JSON record instead of a number.
"""

from __future__ import annotations

import argparse
import json
import statistics
import traceback
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from precision import error_metrics, fp32_full_return_reference
from routing import build_case
from traffic import analytical_traffic


@dataclass(frozen=True)
class Mode:
    name: str
    do_expand: bool
    allow_multiple_reduction: bool


MODES = {
    "A": Mode("A", do_expand=False, allow_multiple_reduction=True),
    "B": Mode("B", do_expand=True, allow_multiple_reduction=True),
    "C": Mode("C", do_expand=True, allow_multiple_reduction=False),
}


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("no timing samples")
    index = (len(ordered) - 1) * percentile
    low, high = int(index), min(int(index) + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def _write(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_combine_input(
    *,
    handle: object,
    recv_topk_idx: torch.Tensor,
    num_recv_tokens: int,
    num_tokens: int,
    topk: int,
    hidden: int,
    expanded: bool,
    deep_ep: object,
) -> torch.Tensor:
    """Mirror the layout preparation in DeepEP's upstream elastic test."""
    from deep_ep.utils.refs import generate_pre_combine_data, ordered_accumulate

    src_metadata = handle.recv_src_metadata[:num_recv_tokens]
    local_y = generate_pre_combine_data(
        src_metadata[:, 0], num_tokens, topk, hidden
    )
    if not expanded:
        local_y[recv_topk_idx[:num_recv_tokens] == -1] = 0
        return ordered_accumulate(local_y)

    # Metadata columns 2.. are the expanded per-top-k return slots.  C must
    # receive one independently populated slot per valid replica.
    slots = src_metadata[:, 2:].flatten()
    valid = slots >= 0
    result = torch.zeros((int(slots.max().item()) + 1, hidden), dtype=torch.bfloat16, device="cuda")
    result[slots[valid]] = local_y.view(-1, hidden)[valid]
    return result


def _run(args: argparse.Namespace, rank: int, world_size: int, group: object) -> dict[str, object]:
    import deep_ep
    from deep_ep.utils.refs import generate_pre_combine_data

    mode = MODES[args.mode]
    if args.num_experts % world_size:
        raise ValueError("num-experts must be divisible by runtime world size")
    case, x_fp32, topk_cpu, _weights = build_case(
        seed=args.seed + rank,
        num_tokens=args.num_tokens,
        hidden=args.hidden,
        topk=args.topk,
        num_experts=args.num_experts,
        world_size=world_size,
        duplicate_bucket=args.duplicate_bucket,
    )
    x = x_fp32.to(device="cuda", dtype=torch.bfloat16)
    topk_idx = topk_cpu.to(device="cuda", dtype=deep_ep.topk_idx_t)
    buffer = deep_ep.ElasticBuffer(
        group,
        num_max_tokens_per_rank=args.num_tokens,
        hidden=args.hidden,
        deterministic=True,
        allow_hybrid_mode=args.allow_hybrid_mode,
        allow_multiple_reduction=mode.allow_multiple_reduction,
        prefer_overlap_with_compute=False,
        explicitly_destroy=True,
        num_gpu_timeout_secs=args.timeout_seconds,
        num_cpu_timeout_secs=args.timeout_seconds,
    )
    try:
        sm_count = torch.cuda.get_device_properties("cuda").multi_processor_count
        num_sms = min(args.num_sms, sm_count)
        num_qps = buffer.get_theoretical_num_qps(num_sms)
        recv_x, recv_topk_idx, _recv_weights, handle, _ = buffer.dispatch(
            x,
            topk_idx=topk_idx,
            topk_weights=None,
            num_experts=args.num_experts,
            num_max_tokens_per_rank=args.num_tokens,
            expert_alignment=1,
            num_sms=num_sms,
            num_qps=num_qps,
            do_expand=mode.do_expand,
            do_cpu_sync=True,
            async_with_compute_stream=False,
        )
        num_recv_tokens = int(handle.psum_num_recv_tokens_per_scaleup_rank[-1].item())
        combine_x = _build_combine_input(
            handle=handle,
            recv_topk_idx=recv_topk_idx,
            num_recv_tokens=num_recv_tokens,
            num_tokens=args.num_tokens,
            topk=args.topk,
            hidden=args.hidden,
            expanded=mode.do_expand,
            deep_ep=deep_ep,
        )
        combined_x, _, _ = buffer.combine(
            combine_x, handle=handle, num_sms=num_sms, num_qps=num_qps,
            async_with_compute_stream=False,
        )
        torch.cuda.synchronize()

        # The official DeepEP helper produces the identical BF16 source data.
        # Casting it to FP32 before the complete K-way reduction is the formal
        # baseline here: it isolates combine rounding from input quantization.
        token_ids = rank * args.num_tokens + torch.arange(args.num_tokens, device="cuda")
        full_return_fp32 = fp32_full_return_reference(
            generate_pre_combine_data(token_ids, args.num_tokens, args.topk, args.hidden)
        )
        precision = error_metrics(combined_x, full_return_fp32)

        for _ in range(args.warmup):
            buffer.combine(combine_x, handle=handle, num_sms=num_sms, num_qps=num_qps, async_with_compute_stream=False)
        torch.cuda.synchronize()
        critical_path_ms: list[float] = []
        for _ in range(args.iterations):
            dist.barrier(group=group)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            buffer.combine(combine_x, handle=handle, num_sms=num_sms, num_qps=num_qps, async_with_compute_stream=False)
            end.record()
            end.synchronize()
            local_ms = torch.tensor([start.elapsed_time(end)], device="cuda", dtype=torch.float64)
            all_rank_ms = [torch.zeros_like(local_ms) for _ in range(world_size)]
            dist.all_gather(all_rank_ms, local_ms, group=group)
            if rank == 0:
                critical_path_ms.append(max(float(item.item()) for item in all_rank_ms))

        traffic = analytical_traffic(
            topk_cpu,
            hidden=args.hidden,
            experts_per_rank=args.num_experts // world_size,
            dtype="bf16",
            mode=args.mode,
        )
        if rank != 0:
            return {}
        return {
            "status": "measured",
            **case.metadata(),
            "mode": args.mode,
            "dtype": "bf16",
            "deep_ep_version": getattr(deep_ep, "__version__", "unknown"),
            **traffic.as_dict(),
            "latency_evidence": "GPU_event_max_rank_completion",
            "timing_definition": "per-iteration max of all rank CUDA Event completion durations",
            "warmup_iterations": args.warmup,
            "measurement_iterations": args.iterations,
            "median_ms": statistics.median(critical_path_ms),
            "p95_ms": _percentile(critical_path_ms, 0.95),
            "min_ms": min(critical_path_ms),
            "max_ms": max(critical_path_ms),
            "stddev_ms": statistics.pstdev(critical_path_ms),
            "critical_path_samples_ms": critical_path_ms,
            "precision_evidence": "FP32_full_return_deterministic_reduction_after_BF16_input",
            **precision.as_dict(),
            "hardware_traffic_evidence": "not_hardware_counter_validated",
        }
    finally:
        buffer.destroy()


def _worker(local_rank: int, args: argparse.Namespace) -> None:
    """Use DeepEP's native multiprocessing contract, not torchrun's contract."""
    group = None
    global_rank = local_rank
    try:
        from deep_ep.utils.envs import init_dist

        global_rank, _, group = init_dist(local_rank, args.num_processes, seed=args.seed)
        record = _run(args, global_rank, args.num_processes, group)
        if global_rank == 0:
            _write(args.output, record)
    except Exception as exc:
        # A rank-zero blocker record makes unsupported topology reproducible,
        # rather than converting an initialization failure into silent missing data.
        if global_rank == 0:
            _write(args.output, {
                "status": "blocked",
                "mode": args.mode,
                "world_size": args.num_processes,
                "reason": str(exc),
                "exception_type": type(exc).__name__,
                "traceback_tail": traceback.format_exc().splitlines()[-12:],
            })
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=tuple(MODES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--num-tokens", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--num-experts", type=int, default=32)
    parser.add_argument("--duplicate-bucket", choices=("zero", "low", "medium", "high", "max"), default="medium")
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--num-sms", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--allow-hybrid-mode", action="store_true")
    args = parser.parse_args()
    if args.num_processes < 1:
        parser.error("--num-processes must be positive")
    if args.num_processes == 1:
        _worker(0, args)
    else:
        mp.spawn(_worker, args=(args,), nprocs=args.num_processes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
