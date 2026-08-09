#!/usr/bin/env python3
# Copyright (c) 2025, TENCENT CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information.

"""Single independent distributed NCCL run for ISSUE4 diagnostics.

This script is launched by ``diagnose.py`` through ``torch.distributed.run``.
Each launch creates a fresh process group / NCCL communicator, executes a fixed
collective sequence, writes raw output bytes per rank/call, and exits.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from issue4.data_generator import (  # type: ignore
        SUPPORTED_DTYPES,
        deterministic_tensor,
        dtype_info,
        iter_relevant_nccl_env,
        numel_from_size,
        parse_size_bytes,
        sha256_bytes,
        tensor_raw_bytes,
        tensor_sha256,
    )
else:
    from .data_generator import (
        SUPPORTED_DTYPES,
        deterministic_tensor,
        dtype_info,
        iter_relevant_nccl_env,
        numel_from_size,
        parse_size_bytes,
        sha256_bytes,
        tensor_raw_bytes,
        tensor_sha256,
    )


COLLECTIVES = ("all_reduce", "reduce_scatter")


def _device_metadata(local_rank: int) -> dict[str, Any]:
    props = torch.cuda.get_device_properties(local_rank)
    return {
        "device_index": local_rank,
        "name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "total_memory": int(props.total_memory),
        "multi_processor_count": int(props.multi_processor_count),
        "pci_bus_id_decimal": getattr(props, "pci_bus_id", None),
    }


def _runtime_versions() -> dict[str, Any]:
    return {
        "python": sys.version.replace("\n", " "),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_nccl": list(torch.cuda.nccl.version()) if hasattr(torch.cuda, "nccl") else None,
        "cudnn": torch.backends.cudnn.version(),
    }


def _mutate_raw_for_injection(raw: bytes, dtype: str, element: int, ulp_delta: int, bit: int | None) -> tuple[bytes, dict[str, Any]]:
    itemsize = dtype_info(dtype).itemsize
    if element < 0 or element * itemsize >= len(raw):
        raise ValueError(f"injection element {element} outside tensor with {len(raw) // itemsize} elements")
    data = bytearray(raw)
    element_offset = element * itemsize

    info: dict[str, Any] = {
        "element": element,
        "dtype": dtype,
        "before_hex": bytes(data[element_offset : element_offset + itemsize]).hex(),
        "ulp_delta": ulp_delta,
        "bit": bit,
    }

    if ulp_delta != 0:
        if dtype == "float32":
            raw_arr = np.frombuffer(data, dtype="<u4")
            raw_arr[element] = np.uint32((int(raw_arr[element]) + ulp_delta) & 0xFFFFFFFF)
        elif dtype in ("float16", "bfloat16"):
            raw_arr = np.frombuffer(data, dtype="<u2")
            raw_arr[element] = np.uint16((int(raw_arr[element]) + ulp_delta) & 0xFFFF)
        else:
            raise ValueError(f"ULP injection unsupported for dtype {dtype}")

    if bit is not None:
        if bit < 0 or bit >= itemsize * 8:
            raise ValueError(f"bit must be in [0, {itemsize * 8}) for dtype {dtype}")
        byte_index = element_offset + bit // 8
        data[byte_index] ^= 1 << (bit % 8)

    info["after_hex"] = bytes(data[element_offset : element_offset + itemsize]).hex()
    return bytes(data), info


def _save_raw(path: Path, raw: bytes) -> str:
    path.write_bytes(raw)
    return sha256_bytes(raw)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ISSUE4 distributed worker; launch with torchrun")
    parser.add_argument("--out-dir", required=True, help="Directory for this independent run")
    parser.add_argument("--run-index", type=int, required=True)
    parser.add_argument("--collective", choices=COLLECTIVES, required=True)
    parser.add_argument("--dtype", choices=SUPPORTED_DTYPES, default="float32")
    parser.add_argument("--size-bytes", type=parse_size_bytes, required=True, help="Output tensor bytes per rank/call")
    parser.add_argument("--calls", type=int, default=20)
    parser.add_argument("--seed", type=int, default=202604)
    parser.add_argument("--timeout-sec", type=int, default=180)
    parser.add_argument("--inject-run", type=int, default=None)
    parser.add_argument("--inject-call", type=int, default=None)
    parser.add_argument("--inject-rank", type=int, default=None)
    parser.add_argument("--inject-element", type=int, default=0)
    parser.add_argument("--inject-ulp", type=int, default=0)
    parser.add_argument("--inject-bit", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.calls <= 0:
        raise ValueError("--calls must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("NCCL worker requires CUDA availability")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=args.timeout_sec))

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    output_numel = numel_from_size(args.size_bytes, args.dtype)
    input_numel = output_numel if args.collective == "all_reduce" else output_numel * world_size
    run_dir = Path(args.out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, Any] = {
        "schema_version": 1,
        "run_index": args.run_index,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "collective": args.collective,
        "dtype": args.dtype,
        "output_numel": output_numel,
        "output_shape": [output_numel],
        "input_numel": input_numel,
        "input_shape": [input_numel],
        "size_bytes": args.size_bytes,
        "calls": args.calls,
        "seed": args.seed,
        "device": _device_metadata(local_rank),
        "versions": _runtime_versions(),
        "env": dict(iter_relevant_nccl_env(os.environ)),
        "records": [],
        "started_unix": time.time(),
    }

    try:
        for call_index in range(args.calls):
            inp = deterministic_tensor(args.dtype, input_numel, args.seed, rank, call_index, device=device)
            input_hash = tensor_sha256(inp)

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(device)
            start.record()
            if args.collective == "all_reduce":
                out = inp
                dist.all_reduce(out, op=dist.ReduceOp.SUM)
            else:
                out = torch.empty(output_numel, device=device, dtype=dtype_info(args.dtype).torch_dtype)
                dist.reduce_scatter_tensor(out, inp, op=dist.ReduceOp.SUM)
            end.record()
            torch.cuda.synchronize(device)
            elapsed_ms = float(start.elapsed_time(end))

            raw = tensor_raw_bytes(out)
            injection_info = None
            if (
                args.inject_run == args.run_index
                and args.inject_call == call_index
                and args.inject_rank == rank
                and (args.inject_ulp != 0 or args.inject_bit is not None)
            ):
                raw, injection_info = _mutate_raw_for_injection(
                    raw,
                    dtype=args.dtype,
                    element=args.inject_element,
                    ulp_delta=args.inject_ulp,
                    bit=args.inject_bit,
                )

            filename = f"rank{rank:03d}_call{call_index:06d}.bin"
            output_hash = _save_raw(run_dir / filename, raw)
            metadata["records"].append(
                {
                    "call_index": call_index,
                    "input_sha256": input_hash,
                    "output_sha256": output_hash,
                    "output_file": filename,
                    "elapsed_ms_cuda_event": elapsed_ms,
                    "injection": injection_info,
                }
            )

        metadata["finished_unix"] = time.time()
        metadata["status"] = "ok"
    except Exception as exc:
        metadata["finished_unix"] = time.time()
        metadata["status"] = "error"
        metadata["error"] = repr(exc)
        raise
    finally:
        (run_dir / f"rank{rank:03d}_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
        dist.destroy_process_group()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
