#!/usr/bin/env python3
"""Capture an Nsight Systems timeline for the Issue6 benchmark loop."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flux-root", type=Path, default=Path("/root/tencent-hpn-issue6/src/flux"))
    parser.add_argument("--output", type=Path, default=Path("/root/tencent-hpn-issue6/artifacts/profiling/raw/issue6_profile"))
    parser.add_argument("--variant", choices=["original", "optimized"], default="optimized")
    parser.add_argument("--sm-margin", type=int, default=32)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup-iters", type=int, default=5)
    args = parser.parse_args()

    nsys = Path("/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys")
    bench = Path(__file__).resolve().parent / "benchmark.py"
    env = os.environ.copy()
    env.setdefault("CUDA_HOME", "/usr/local/cuda-12.8")
    nvshmem = "/root/miniconda3/lib/python3.12/site-packages/nvidia/nvshmem"
    env["NVSHMEM_HOME"] = nvshmem
    env["LD_LIBRARY_PATH"] = (
        f"{env['CUDA_HOME']}/lib64:{nvshmem}/lib:{args.flux_root}/python/flux/lib:"
        + env.get("LD_LIBRARY_PATH", "")
    )
    env["PYTHONPATH"] = f"{args.flux_root}/python:" + env.get("PYTHONPATH", "")
    env.setdefault("FLUX_SM120_LOGICAL_ARCH", "80")
    env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    cmd = [
        str(nsys),
        "profile",
        "--force-overwrite=true",
        "--trace=cuda,nvtx,osrt",
        "--sample=none",
        "--cpuctxsw=none",
        f"--output={args.output}",
        str(args.flux_root / "launch.sh"),
        str(bench),
        "--variant",
        args.variant,
        "--component",
        "fused",
        "--iters",
        str(args.iters),
        "--warmup-iters",
        str(args.warmup_iters),
        "--sm-margin",
        str(args.sm_margin),
        "--tune-top-index",
        "0",
        "--output-dir",
        str(args.output.parent / args.output.name),
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, env=env, cwd=args.flux_root, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
