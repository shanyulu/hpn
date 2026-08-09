#!/usr/bin/env python3
"""Issue6 MoE AGScatter benchmark harness.

Run with torchrun, for example:
  torchrun --standalone --nproc_per_node=4 benchmark.py --variant optimized --component fused
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flux-root", default="/root/tencent-hpn-issue6/src/flux")
    parser.add_argument("--output-dir", default="/root/tencent-hpn-issue6/artifacts/issue6_benchmark")
    parser.add_argument("--variant", choices=["original", "optimized"], default="optimized")
    parser.add_argument("--component", choices=["fused", "compute_only", "comm_only"], default="fused")
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--S", type=int, default=256)
    parser.add_argument("--H", type=int, default=7168)
    parser.add_argument("--ffn-hidden-size", type=int, default=16384)
    parser.add_argument("--G", type=int, default=4)
    parser.add_argument("--E", type=int, default=1)
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--dist", choices=["uniform", "random_uniform", "random"], default="uniform")
    parser.add_argument("--warmup-iters", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--sm-margin", type=int, default=0)
    parser.add_argument(
        "--tune-top-index",
        type=int,
        default=0,
        help="For --variant optimized, load this zero-based tuning rank after profiling.",
    )
    parser.add_argument("--ring-mode", choices=["auto", "all2all", "ring1d", "ring2d"], default="ring1d")
    parser.add_argument("--use-cuda-core-ag", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--stable-index", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--correctness", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--nvtx-loop-label", default="ISSUE6_MEASURED_LOOP")
    parser.add_argument("--seed", type=int, default=12345)
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    idx = min(len(values) - 1, max(0, math.ceil(q * len(values)) - 1))
    return sorted(values)[idx]


def stats(values: list[float]) -> dict[str, float]:
    return {
        "count": float(len(values)),
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "min_ms": min(values),
        "stddev_ms": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def gather_object(torch_mod: Any, obj: Any, group: Any, world_size: int) -> list[Any]:
    gathered = [None for _ in range(world_size)]
    torch_mod.distributed.all_gather_object(gathered, obj, group=group)
    return gathered


def set_bench_env(component: str, variant: str) -> dict[str, str | None]:
    keys = [
        "FLUX_MOE_AG_BENCH_COMPUTE_ONLY",
        "FLUX_MOE_AG_BENCH_COMM_ONLY",
        "FLUX_MOE_AG_FUSED_TOPK1_SORT",
    ]
    old = {k: os.environ.get(k) for k in keys}
    for k in keys:
        os.environ.pop(k, None)
    os.environ["FLUX_MOE_AG_FUSED_TOPK1_SORT"] = "1" if variant == "optimized" else "0"
    if component == "compute_only":
        os.environ["FLUX_MOE_AG_BENCH_COMPUTE_ONLY"] = "1"
    elif component == "comm_only":
        os.environ["FLUX_MOE_AG_BENCH_COMM_ONLY"] = "1"
    return old


def restore_env(old: dict[str, str | None]) -> None:
    for key, value in old.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def main() -> int:
    args = parse_args()
    flux_root = Path(args.flux_root).resolve()
    sys.path.insert(0, str(flux_root / "python"))

    import torch

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))

    import flux
    import flux.testing
    from flux.testing import DTYPE_MAP, RING_MODE_MAP, MoeAgScatterWithTorch, MoeMlp1Ctx

    dist_env = flux.get_dist_env()
    tp_group = dist_env.get_world()
    torch.cuda.set_device(dist_env.LOCAL_RANK)

    def init_ep_group(ep_size: int) -> Any:
        assert dist_env.WORLD_SIZE % ep_size == 0
        ffn_tp_size = tp_group.size() // ep_size
        ep_group = None
        temp_groups = []
        for i in range(ffn_tp_size):
            temp_groups.append(list(range(i, dist_env.WORLD_SIZE, ffn_tp_size)))
        for group_ranks in temp_groups:
            for start in range(0, len(group_ranks), ep_size):
                ranks = group_ranks[start : start + ep_size]
                group = dist_env.new_group(ranks)
                if dist_env.RANK in ranks:
                    ep_group = group
        assert ep_group is not None
        return ep_group

    ep_group = init_ep_group(args.E)
    rank = tp_group.rank()
    world_size = tp_group.size()
    out_dir = Path(args.output_dir).resolve()
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    flux.init_flux_shm(tp_group)
    torch.cuda.synchronize()

    input_dtype = DTYPE_MAP[args.dtype]
    output_dtype = torch.bfloat16 if flux.util.is_fp8_dtype(input_dtype) else input_dtype
    generator = torch.Generator(device="cuda")
    generator.manual_seed(args.seed)
    ctx = MoeMlp1Ctx(
        tp_group,
        ep_group,
        b=args.B,
        s=args.S,
        h=args.H,
        ffn_size=args.ffn_hidden_size,
        nexperts=args.G,
        topk=args.topk,
        input_dtype=input_dtype,
        output_dtype=output_dtype,
        dist=args.dist,
        fast_accum=False,
        weight_groups=1,
        drop_token=False,
        debug=False,
        generator=generator,
        stable=args.stable_index,
    )

    tp_env = flux.DistEnvTPWithEP(tp_group=tp_group, nnodes=dist_env.NNODES, ep_group=ep_group)
    moe_args = flux.MoeArguments(
        max_ntokens=args.B * args.S,
        hidden=args.H,
        ffn_hidden=args.ffn_hidden_size,
        nexperts=args.G,
        topk=args.topk,
        input_dtype=ctx.inputs_shard.dtype,
        output_dtype=ctx.outputs[0].dtype,
    )
    if flux.util.get_arch() >= 90:
        raise RuntimeError("Issue6 benchmark expects the SM120-compatible V2 path")
    op = flux.GemmGroupedV2AGScatterOp(tp_env=tp_env, moe_args=moe_args)

    ag_option = flux.AllGatherOption()
    ag_option.mode = RING_MODE_MAP[args.ring_mode]
    ag_option.use_cuda_core_ag = args.use_cuda_core_ag
    ag_option.use_cuda_core_local = False
    os.environ["FLUX_MOE_AG_FUSED_TOPK1_SORT"] = "1" if args.variant == "optimized" else "0"

    tuning_info: dict[str, Any] = {"enabled": args.variant == "optimized"}
    if args.variant == "optimized":
        if args.tune_top_index < 0:
            raise ValueError("--tune-top-index must be non-negative")
        prof_ctx = flux.ProfilingContext("issue6_ag_scatter_sm80")
        op.profiling(
            inputs_shard=ctx.inputs_shard,
            weights=ctx.weights,
            splits_gpu=ctx.splits_gpu,
            scatter_index=ctx.scatter_index,
            output_scale=ctx.output_scale,
            outputs_buf=ctx.outputs,
            allgather_output=None,
            fast_accum=False,
            sm_margin=args.sm_margin,
            ag_option=ag_option,
            prof_ctx=prof_ctx,
        )
        torch.cuda.synchronize()
        if not hasattr(prof_ctx, "get_latest_record_at"):
            raise RuntimeError("Flux patch missing ProfilingContext.get_latest_record_at")
        flux.load_tuning_record(prof_ctx.get_latest_record_at(args.tune_top_index))
        if rank == 0:
            tuning_info = {
                "enabled": True,
                "tune_top_index": args.tune_top_index,
                "results": prof_ctx.get_all_prof_results(),
                "generated_config": prof_ctx.get_code(),
            }

    correctness: dict[str, Any] = {"checked": False}
    if args.correctness and args.component == "fused":
        gemm_only_op = flux.GemmOnly(ctx.inputs.dtype, ctx.outputs[0].dtype)
        ctx.clear_outputs()
        MoeAgScatterWithTorch.comm_impl(ctx, tp_group)
        MoeAgScatterWithTorch.scatter_impl(ctx)
        MoeAgScatterWithTorch.gemm_impl(ctx, gemm_only_op)
        ref_outputs = ctx.get_outputs_clone()
        ctx.clear_outputs()
        op.forward(
            inputs_shard=ctx.inputs_shard,
            weights=ctx.weights[0],
            splits_gpu=ctx.splits_gpu,
            scatter_index=ctx.scatter_index,
            output_scale=ctx.output_scale[0],
            outputs_buf=ctx.outputs[0],
            allgather_output=None,
            fast_accum=False,
            sm_margin=args.sm_margin,
            ag_option=ag_option,
        )
        torch.cuda.synchronize()
        max_abs = 0.0
        max_rel = 0.0
        bitwise = True
        allclose = True
        atol, rtol = (1e-2, 1.5e-2) if input_dtype == torch.bfloat16 else (1e-2, 1e-3)
        for got, ref in zip(ctx.get_outputs_clone(), ref_outputs):
            diff = (got - ref).abs().float()
            denom = ref.abs().float().clamp_min(1e-12)
            max_abs = max(max_abs, float(diff.max().item()))
            max_rel = max(max_rel, float((diff / denom).max().item()))
            bitwise = bitwise and bool(torch.equal(got, ref))
            allclose = allclose and bool(torch.allclose(got, ref, atol=atol, rtol=rtol))
        correctness = {
            "checked": True,
            "bitwise": bitwise,
            "allclose": allclose,
            "atol": atol,
            "rtol": rtol,
            "max_abs_error": max_abs,
            "max_relative_error": max_rel,
            "shape": list(ctx.outputs[0].shape),
            "dtype": str(ctx.outputs[0].dtype),
            "nan_count": int(torch.isnan(ctx.outputs[0].float()).sum().item()),
            "inf_count": int(torch.isinf(ctx.outputs[0].float()).sum().item()),
        }

    if args.component == "compute_only":
        MoeAgScatterWithTorch.comm_impl(ctx, tp_group)
        torch.cuda.synchronize()
        tp_group.barrier()

    torch.cuda.reset_peak_memory_stats()
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(args.warmup_iters + args.iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(args.warmup_iters + args.iters)]
    local_times: list[float] = []

    tp_group.barrier()
    torch.cuda.synchronize()
    old_env = set_bench_env(args.component, args.variant)
    nvtx_loop_pushed = False
    try:
        if args.nvtx_loop_label:
            torch.cuda.nvtx.range_push(args.nvtx_loop_label)
            nvtx_loop_pushed = True
        for i in range(args.warmup_iters + args.iters):
            ctx.clear_outputs()
            op.clear_buffers()
            start_events[i].record()
            op.forward(
                inputs_shard=ctx.inputs_shard,
                weights=ctx.weights[0],
                splits_gpu=ctx.splits_gpu,
                scatter_index=ctx.scatter_index,
                output_scale=ctx.output_scale[0],
                outputs_buf=ctx.outputs[0],
                allgather_output=ctx.inputs if args.component == "compute_only" else None,
                fast_accum=False,
                sm_margin=args.sm_margin,
                ag_option=ag_option,
            )
            end_events[i].record()
        for i in range(args.warmup_iters + args.iters):
            end_events[i].synchronize()
            if i >= args.warmup_iters:
                local_times.append(start_events[i].elapsed_time(end_events[i]))
        if args.nvtx_loop_label:
            torch.cuda.nvtx.range_pop()
            nvtx_loop_pushed = False
    finally:
        if nvtx_loop_pushed:
            torch.cuda.nvtx.range_pop()
        restore_env(old_env)

    peak_alloc = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    gathered_times = gather_object(torch, local_times, tp_group, world_size)
    gathered_mem = gather_object(
        torch,
        {"rank": rank, "peak_allocated_bytes": peak_alloc, "peak_reserved_bytes": peak_reserved},
        tp_group,
        world_size,
    )
    gathered_correctness = gather_object(torch, correctness, tp_group, world_size)

    if rank == 0:
        method = f"{args.variant}_{args.component}"
        iter_csv = out_dir / f"{method}_iterations.csv"
        with iter_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["method", "rank", "iteration", "latency_ms"])
            writer.writeheader()
            for r, times in enumerate(gathered_times):
                for i, value in enumerate(times):
                    writer.writerow({"method": method, "rank": r, "iteration": i, "latency_ms": value})
        distributed_times = [max(float(gathered_times[r][i]) for r in range(world_size)) for i in range(args.iters)]
        n = args.ffn_hidden_size // world_size
        flops_per_rank = 2.0 * (args.B * args.S) * n * args.H
        summary = {
            "method": method,
            "variant": args.variant,
            "component": args.component,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "shape": {"Group": args.G, "M": args.B * args.S, "N": n, "K": args.H},
            "world_size": world_size,
            "warmup_iters": args.warmup_iters,
            "iters": args.iters,
            "dtype": args.dtype,
            "ring_mode": args.ring_mode,
            "use_cuda_core_ag": args.use_cuda_core_ag,
            "stats_by_rank": {str(r): stats([float(x) for x in gathered_times[r]]) for r in range(world_size)},
            "distributed": stats(distributed_times),
            "throughput": {
                "tokens_per_second": (args.B * args.S) / (statistics.median(distributed_times) / 1000.0),
                "tflops_per_rank": flops_per_rank / (statistics.median(distributed_times) / 1000.0) / 1e12,
            },
            "memory": gathered_mem,
            "correctness_by_rank": gathered_correctness,
            "tuning": tuning_info,
            "artifacts": {"iterations_csv": str(iter_csv)},
        }
        summary_path = out_dir / f"{method}_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)

    tp_group.barrier()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
