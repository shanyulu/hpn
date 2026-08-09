#!/usr/bin/env python3
# Copyright (c) 2025, TENCENT CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information.

"""Run ISSUE4 diagnostic experiment matrices."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from issue4.data_generator import SUPPORTED_DTYPES, parse_size_bytes  # type: ignore
else:
    from .data_generator import SUPPORTED_DTYPES, parse_size_bytes


def _split_csv(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def _size_label(size_bytes: int) -> str:
    for suffix, factor in (("GiB", 1024**3), ("MiB", 1024**2), ("KiB", 1024)):
        if size_bytes % factor == 0 and size_bytes >= factor:
            return f"{size_bytes // factor}{suffix}"
    return f"{size_bytes}B"


def _run_one(args: argparse.Namespace, output_root: Path, collective: str, algo: str, proto: str, size_bytes: int) -> dict[str, Any]:
    tag = f"matrix_{collective}_{args.dtype}_{_size_label(size_bytes)}_{algo}_{proto}"
    cmd = [
        sys.executable,
        "-m",
        "issue4.diagnose",
        "--collective",
        collective,
        "--dtype",
        args.dtype,
        "--size-bytes",
        str(size_bytes),
        "--runs",
        str(args.runs),
        "--calls",
        str(args.calls),
        "--nproc-per-node",
        str(args.nproc_per_node),
        "--algo",
        algo,
        "--proto",
        proto,
        "--output-root",
        str(output_root),
        "--tag",
        tag,
        "--launch-timeout-sec",
        str(args.launch_timeout_sec),
        "--timeout-sec",
        str(args.timeout_sec),
        "--allow-failure",
    ]
    if args.nccl_debug == "none":
        cmd.extend(["--nccl-debug", "none"])
    else:
        cmd.extend(["--nccl-debug", args.nccl_debug, "--nccl-debug-subsys", args.nccl_debug_subsys])

    started = time.time()
    proc = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parents[1]), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    exp_dir = None
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and "experiment_dir" in line:
            try:
                exp_dir = json.loads(line)["experiment_dir"]
                break
            except Exception:
                pass
    summary: dict[str, Any] | None = None
    if exp_dir and Path(exp_dir, "summary.json").exists():
        summary = json.loads(Path(exp_dir, "summary.json").read_text(encoding="utf-8"))
    return {
        "collective": collective,
        "dtype": args.dtype,
        "size_bytes": size_bytes,
        "size_label": _size_label(size_bytes),
        "algo": algo,
        "proto": proto,
        "returncode": proc.returncode,
        "started_unix": started,
        "finished_unix": time.time(),
        "experiment_dir": exp_dir,
        "status": summary.get("status") if summary else "missing_summary",
        "input_hashes_consistent": summary.get("input_hashes_consistent") if summary else None,
        "comparison_count": summary.get("comparison_count") if summary else None,
        "divergent_comparison_count": summary.get("divergent_comparison_count") if summary else None,
        "max_abs_error": summary.get("max_abs_error") if summary else None,
        "max_rel_error": summary.get("max_rel_error") if summary else None,
        "max_ordered_ulp": summary.get("max_ordered_ulp") if summary else None,
        "stdout_tail": "\n".join(proc.stdout.splitlines()[-20:]),
        "stderr_tail": "\n".join(proc.stderr.splitlines()[-20:]),
        "cmd": cmd,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ISSUE4 diagnostic matrix")
    parser.add_argument("--collectives", default="all_reduce,reduce_scatter")
    parser.add_argument("--dtype", choices=SUPPORTED_DTYPES, default="float32")
    parser.add_argument("--sizes", default="1KiB,64KiB,1MiB,16MiB,128MiB")
    parser.add_argument("--algos", default="default", help="Comma-separated NCCL_ALGO values: default,Ring,Tree")
    parser.add_argument("--protos", default="default", help="Comma-separated NCCL_PROTO values: default,Simple,LL,LL128")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--calls", type=int, default=20)
    parser.add_argument("--nproc-per-node", type=int, default=4)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--tag", default="matrix")
    parser.add_argument("--timeout-sec", type=int, default=180)
    parser.add_argument("--launch-timeout-sec", type=int, default=900)
    parser.add_argument("--nccl-debug", choices=("none", "WARN", "INFO", "TRACE"), default="INFO")
    parser.add_argument("--nccl-debug-subsys", default="INIT,GRAPH,COLL")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    issue4_dir = Path(__file__).resolve().parent
    output_root = Path(args.output_root) if args.output_root else issue4_dir / "results" / f"{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}_{args.tag}"
    output_root.mkdir(parents=True, exist_ok=False)
    collectives = _split_csv(args.collectives)
    algos = _split_csv(args.algos)
    protos = _split_csv(args.protos)
    sizes = [parse_size_bytes(x) for x in _split_csv(args.sizes)]
    rows: list[dict[str, Any]] = []
    matrix_manifest = {
        "argv": sys.argv,
        "parsed_args": vars(args),
        "output_root": str(output_root.resolve()),
        "collectives": collectives,
        "algos": algos,
        "protos": protos,
        "sizes": sizes,
    }
    (output_root / "matrix_manifest.json").write_text(json.dumps(matrix_manifest, indent=2, sort_keys=True), encoding="utf-8")
    for collective in collectives:
        for size_bytes in sizes:
            for algo in algos:
                for proto in protos:
                    print(f"[issue4] matrix collective={collective} size={_size_label(size_bytes)} algo={algo} proto={proto}", flush=True)
                    row = _run_one(args, output_root, collective, algo, proto, size_bytes)
                    rows.append(row)
                    (output_root / "matrix_summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True, allow_nan=True), encoding="utf-8")
                    with (output_root / "matrix_summary.csv").open("w", newline="", encoding="utf-8") as f:
                        fieldnames = [
                            "collective",
                            "dtype",
                            "size_label",
                            "size_bytes",
                            "algo",
                            "proto",
                            "status",
                            "returncode",
                            "input_hashes_consistent",
                            "comparison_count",
                            "divergent_comparison_count",
                            "max_abs_error",
                            "max_rel_error",
                            "max_ordered_ulp",
                            "experiment_dir",
                        ]
                        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                        writer.writeheader()
                        writer.writerows(rows)
    print(json.dumps({"output_root": str(output_root.resolve()), "experiments": len(rows)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
