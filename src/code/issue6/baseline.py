#!/usr/bin/env python3
"""Run the Issue6 official-shape benchmark through Flux launch.sh."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flux-root", type=Path, default=Path("/root/tencent-hpn-issue6/src/flux"))
    parser.add_argument("--output-dir", type=Path, default=Path("/root/tencent-hpn-issue6/artifacts/issue6_repro"))
    parser.add_argument("--variant", choices=["original", "optimized"], default="optimized")
    parser.add_argument("--sm-margin", type=int, default=32)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup-iters", type=int, default=20)
    args = parser.parse_args()

    repo_file = Path(__file__).resolve()
    bench = repo_file.parent / "benchmark.py"
    env = os.environ.copy()
    env.setdefault("CUDA_HOME", "/usr/local/cuda-12.8")
    env["PATH"] = f"{env['CUDA_HOME']}/bin:/root/miniconda3/bin:" + env.get("PATH", "")
    nvshmem = "/root/miniconda3/lib/python3.12/site-packages/nvidia/nvshmem"
    env["NVSHMEM_HOME"] = nvshmem
    env["LD_LIBRARY_PATH"] = (
        f"{env['CUDA_HOME']}/lib64:{nvshmem}/lib:{args.flux_root}/python/flux/lib:"
        + env.get("LD_LIBRARY_PATH", "")
    )
    env["PYTHONPATH"] = f"{args.flux_root}/python:" + env.get("PYTHONPATH", "")
    env.setdefault("FLUX_SM120_LOGICAL_ARCH", "80")
    env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    env["FLUX_EXTRA_TORCHRUN_ARGS"] = "--standalone"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for component in ["fused", "compute_only", "comm_only"]:
        cmd = [
            str(args.flux_root / "launch.sh"),
            str(bench),
            "--variant",
            args.variant,
            "--component",
            component,
            "--iters",
            str(args.iters),
            "--warmup-iters",
            str(args.warmup_iters),
            "--sm-margin",
            str(args.sm_margin),
            "--tune-top-index",
            "0",
            "--output-dir",
            str(args.output_dir),
        ]
        log = args.output_dir / f"{args.variant}_{component}.log"
        with log.open("w") as f:
            subprocess.run(cmd, env=env, cwd=args.flux_root, stdout=f, stderr=subprocess.STDOUT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
