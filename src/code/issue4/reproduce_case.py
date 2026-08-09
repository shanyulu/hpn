#!/usr/bin/env python3
# Copyright (c) 2025, TENCENT CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information.

"""Reproducible ISSUE4 diagnostic cases.

These cases are intentionally labeled: injection is a detector self-test; the
Ring-vs-Tree case is a cross-configuration numerical sensitivity test, not a
same-configuration nondeterminism claim.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from issue4.data_generator import SUPPORTED_DTYPES, parse_size_bytes  # type: ignore
else:
    from .data_generator import SUPPORTED_DTYPES, parse_size_bytes


def _code_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _last_experiment_dir(stdout: str) -> str:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and "experiment_dir" in line:
            return json.loads(line)["experiment_dir"]
    raise RuntimeError("diagnose output did not include experiment_dir")


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    print(" ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(_code_root()), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, end="")
    if proc.returncode != 0:
        raise RuntimeError(f"command failed with returncode {proc.returncode}: {' '.join(cmd)}")
    return proc


def run_injection(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root) if args.output_root else Path(__file__).resolve().parent / "results" / f"{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}_reproduce_injection"
    output_root.mkdir(parents=True, exist_ok=False)
    cmd = [
        sys.executable,
        "-m",
        "issue4.diagnose",
        "--collective",
        args.collective,
        "--dtype",
        args.dtype,
        "--size-bytes",
        str(args.size_bytes),
        "--runs",
        "2",
        "--calls",
        str(args.calls),
        "--nproc-per-node",
        str(args.nproc_per_node),
        "--output-root",
        str(output_root),
        "--tag",
        "controlled_1ulp_injection",
        "--inject-run",
        "1",
        "--inject-call",
        str(args.inject_call),
        "--inject-rank",
        str(args.inject_rank),
        "--inject-element",
        str(args.inject_element),
        "--inject-ulp",
        str(args.inject_ulp),
    ]
    proc = _run(cmd)
    print(json.dumps({"case": "controlled_injection", "experiment_dir": _last_experiment_dir(proc.stdout)}, sort_keys=True))
    return 0


def run_cross_algo(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root) if args.output_root else Path(__file__).resolve().parent / "results" / f"{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}_reproduce_cross_algo"
    output_root.mkdir(parents=True, exist_ok=False)
    common = [
        sys.executable,
        "-m",
        "issue4.diagnose",
        "--collective",
        args.collective,
        "--dtype",
        args.dtype,
        "--size-bytes",
        str(args.size_bytes),
        "--runs",
        str(args.runs),
        "--calls",
        str(args.calls),
        "--nproc-per-node",
        str(args.nproc_per_node),
        "--output-root",
        str(output_root),
    ]
    ring = _run(common + ["--algo", "Ring", "--tag", "cross_algo_Ring"])
    tree = _run(common + ["--algo", "Tree", "--tag", "cross_algo_Tree"])
    ring_dir = _last_experiment_dir(ring.stdout)
    tree_dir = _last_experiment_dir(tree.stdout)
    compare_dir = output_root / "ring_vs_tree_compare"
    compare_cmd = [
        sys.executable,
        "-m",
        "issue4.compare_experiments",
        "--exp-a",
        ring_dir,
        "--exp-b",
        tree_dir,
        "--output-dir",
        str(compare_dir),
    ]
    _run(compare_cmd)
    print(json.dumps({"case": "cross_algorithm", "ring_dir": ring_dir, "tree_dir": tree_dir, "compare_dir": str(compare_dir.resolve())}, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run labeled ISSUE4 reproduction cases")
    sub = parser.add_subparsers(dest="case", required=True)

    inj = sub.add_parser("injection", help="Controlled 1-ULP detector self-test")
    inj.add_argument("--collective", choices=("all_reduce", "reduce_scatter"), default="all_reduce")
    inj.add_argument("--dtype", choices=SUPPORTED_DTYPES, default="float32")
    inj.add_argument("--size-bytes", type=parse_size_bytes, default=parse_size_bytes("1KiB"))
    inj.add_argument("--calls", type=int, default=3)
    inj.add_argument("--nproc-per-node", type=int, default=1)
    inj.add_argument("--inject-call", type=int, default=1)
    inj.add_argument("--inject-rank", type=int, default=0)
    inj.add_argument("--inject-element", type=int, default=5)
    inj.add_argument("--inject-ulp", type=int, default=1)
    inj.add_argument("--output-root", default=None)
    inj.set_defaults(func=run_injection)

    cross = sub.add_parser("cross-algo", help="Ring vs Tree cross-configuration numerical sensitivity")
    cross.add_argument("--collective", choices=("all_reduce", "reduce_scatter"), default="all_reduce")
    cross.add_argument("--dtype", choices=SUPPORTED_DTYPES, default="float32")
    cross.add_argument("--size-bytes", type=parse_size_bytes, default=parse_size_bytes("1MiB"))
    cross.add_argument("--runs", type=int, default=2)
    cross.add_argument("--calls", type=int, default=5)
    cross.add_argument("--nproc-per-node", type=int, default=4)
    cross.add_argument("--output-root", default=None)
    cross.set_defaults(func=run_cross_algo)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
