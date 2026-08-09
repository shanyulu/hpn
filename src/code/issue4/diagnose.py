#!/usr/bin/env python3
# Copyright (c) 2025, TENCENT CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information.

"""Orchestrate independent NCCL runs and bitwise forensic comparison."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from issue4.comparator import compare_files  # type: ignore
    from issue4.data_generator import SUPPORTED_DTYPES, parse_size_bytes  # type: ignore
    from issue4.env_audit import collect_environment  # type: ignore
    from issue4.reporter import write_csv, write_json, write_markdown_summary  # type: ignore
else:
    from .comparator import compare_files
    from .data_generator import SUPPORTED_DTYPES, parse_size_bytes
    from .env_audit import collect_environment
    from .reporter import write_csv, write_json, write_markdown_summary


COLLECTIVES = ("all_reduce", "reduce_scatter")
ALGORITHMS = ("default", "Ring", "Tree")
PROTOCOLS = ("default", "Simple", "LL", "LL128")
SANITIZED_NCCL_KEYS = (
    "NCCL_ALGO",
    "NCCL_PROTO",
    "NCCL_P2P_DISABLE",
    "NCCL_NVLS_ENABLE",
)


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


def _repo_paths() -> tuple[Path, Path, Path]:
    issue4_dir = Path(__file__).resolve().parent
    code_root = issue4_dir.parent
    hpn_root = issue4_dir.parents[2]
    workspace_root = issue4_dir.parents[3]
    return code_root, hpn_root, workspace_root


def _auto_world_size() -> int:
    count = torch.cuda.device_count()
    if count <= 0:
        raise RuntimeError("no CUDA devices visible")
    return count


def _experiment_dir(args: argparse.Namespace) -> Path:
    _, hpn_root, _ = _repo_paths()
    root = Path(args.output_root) if args.output_root else hpn_root / "src" / "code" / "issue4" / "results"
    tag = args.tag or f"{args.collective}_{args.dtype}_{args.size_bytes}B_{args.algo}_{args.proto}"
    base = root / f"{_timestamp()}_{tag}"
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = Path(f"{base}_{suffix}")
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate.resolve()


def _build_env(args: argparse.Namespace, exp_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if args.sanitize_nccl_env:
        for key in SANITIZED_NCCL_KEYS:
            env.pop(key, None)
    if args.algo != "default":
        env["NCCL_ALGO"] = args.algo
    if args.proto != "default":
        env["NCCL_PROTO"] = args.proto
    if args.nccl_debug != "none":
        env["NCCL_DEBUG"] = args.nccl_debug
        if args.nccl_debug_subsys:
            env["NCCL_DEBUG_SUBSYS"] = args.nccl_debug_subsys
    env["ISSUE4_EXPERIMENT_DIR"] = str(exp_dir)
    return env


def _worker_command(args: argparse.Namespace, run_dir: Path, run_index: int) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={args.nproc_per_node}",
        "-m",
        "issue4.worker",
        "--out-dir",
        str(run_dir),
        "--run-index",
        str(run_index),
        "--collective",
        args.collective,
        "--dtype",
        args.dtype,
        "--size-bytes",
        str(args.size_bytes),
        "--calls",
        str(args.calls),
        "--seed",
        str(args.seed),
        "--timeout-sec",
        str(args.timeout_sec),
    ]
    if args.inject_run is not None:
        cmd.extend(["--inject-run", str(args.inject_run)])
    if args.inject_call is not None:
        cmd.extend(["--inject-call", str(args.inject_call)])
    if args.inject_rank is not None:
        cmd.extend(["--inject-rank", str(args.inject_rank)])
    if args.inject_element is not None:
        cmd.extend(["--inject-element", str(args.inject_element)])
    if args.inject_ulp:
        cmd.extend(["--inject-ulp", str(args.inject_ulp)])
    if args.inject_bit is not None:
        cmd.extend(["--inject-bit", str(args.inject_bit)])
    return cmd


def _run_independent_launch(args: argparse.Namespace, exp_dir: Path, env: dict[str, str], run_index: int) -> dict[str, Any]:
    code_root, _, _ = _repo_paths()
    run_dir = exp_dir / f"run_{run_index:03d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    cmd = _worker_command(args, run_dir, run_index)
    started = time.time()
    with (run_dir / "stdout.log").open("w", encoding="utf-8") as stdout, (run_dir / "stderr.log").open("w", encoding="utf-8") as stderr:
        proc = subprocess.run(cmd, cwd=str(code_root), env=env, stdout=stdout, stderr=stderr, text=True, timeout=args.launch_timeout_sec, check=False)
    return {
        "run_index": run_index,
        "run_dir": str(run_dir),
        "cmd": cmd,
        "returncode": proc.returncode,
        "started_unix": started,
        "finished_unix": time.time(),
        "stdout_log": str(run_dir / "stdout.log"),
        "stderr_log": str(run_dir / "stderr.log"),
    }


def _load_run_metadata(run_dir: Path) -> dict[int, dict[str, Any]]:
    metas: dict[int, dict[str, Any]] = {}
    for path in sorted(run_dir.glob("rank*_metadata.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        metas[int(payload["rank"])] = payload
    return metas


def _all_metadata(exp_dir: Path, runs: int) -> dict[int, dict[int, dict[str, Any]]]:
    return {run: _load_run_metadata(exp_dir / f"run_{run:03d}") for run in range(runs)}


def _validate_input_hashes(all_meta: dict[int, dict[int, dict[str, Any]]], calls: int) -> tuple[bool, list[dict[str, Any]]]:
    mismatches: list[dict[str, Any]] = []
    base = all_meta.get(0, {})
    for run, ranks in sorted(all_meta.items()):
        if run == 0:
            continue
        for rank, meta in sorted(ranks.items()):
            base_meta = base.get(rank)
            if not base_meta:
                mismatches.append({"run": run, "rank": rank, "reason": "missing_rank_in_run0"})
                continue
            for call in range(calls):
                got = meta["records"][call]["input_sha256"]
                expected = base_meta["records"][call]["input_sha256"]
                if got != expected:
                    mismatches.append({"run": run, "rank": rank, "call_index": call, "run0": expected, "run": got})
    return len(mismatches) == 0, mismatches


def _compare_outputs(exp_dir: Path, all_meta: dict[int, dict[int, dict[str, Any]]], runs: int, calls: int) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    rows: list[dict[str, Any]] = []
    first_divergence: dict[str, Any] | None = None
    base = all_meta[0]
    ranks = sorted(base)
    for run_b in range(1, runs):
        for call_index in range(calls):
            for rank in ranks:
                meta_a = base[rank]
                meta_b = all_meta[run_b][rank]
                rec_a = meta_a["records"][call_index]
                rec_b = meta_b["records"][call_index]
                path_a = exp_dir / "run_000" / rec_a["output_file"]
                path_b = exp_dir / f"run_{run_b:03d}" / rec_b["output_file"]
                result = compare_files(path_a, path_b, dtype=meta_a["dtype"], shape=meta_a["output_shape"])
                result_dict = result.to_dict()
                row = {
                    "run_a": 0,
                    "run_b": run_b,
                    "call_index": call_index,
                    "rank": rank,
                    **result_dict,
                    "path_a": str(path_a),
                    "path_b": str(path_b),
                }
                rows.append(row)
                if not result.equal and first_divergence is None:
                    first_divergence = {
                        "run_a": 0,
                        "run_b": run_b,
                        "call_index": call_index,
                        "rank": rank,
                        "path_a": str(path_a),
                        "path_b": str(path_b),
                        "comparison": result_dict,
                    }
    return rows, first_divergence


def _summarize(args: argparse.Namespace, exp_dir: Path, launches: list[dict[str, Any]]) -> dict[str, Any]:
    failed = [x for x in launches if x["returncode"] != 0]
    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": "ok" if not failed else "launch_failed",
        "experiment_dir": str(exp_dir),
        "config": {
            "collective": args.collective,
            "dtype": args.dtype,
            "size_bytes": args.size_bytes,
            "runs": args.runs,
            "calls": args.calls,
            "nproc_per_node": args.nproc_per_node,
            "seed": args.seed,
            "algo": args.algo,
            "proto": args.proto,
            "sanitize_nccl_env": args.sanitize_nccl_env,
            "nccl_debug": args.nccl_debug,
            "nccl_debug_subsys": args.nccl_debug_subsys,
            "injection": {
                "run": args.inject_run,
                "call": args.inject_call,
                "rank": args.inject_rank,
                "element": args.inject_element,
                "ulp": args.inject_ulp,
                "bit": args.inject_bit,
            },
        },
        "launches": launches,
    }
    if failed:
        unsupported_markers = (
            "no algorithm/protocol available",
            "invalid value for NCCL_ALGO",
            "invalid value for NCCL_PROTO",
            "unsupported",
        )
        failure_evidence = []
        unsupported = False
        for launch in failed:
            snippets = []
            for key in ("stdout_log", "stderr_log"):
                path = Path(launch[key])
                text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
                for line in text.splitlines():
                    lower = line.lower()
                    if any(marker in lower for marker in unsupported_markers):
                        snippets.append(line)
                        unsupported = True
            failure_evidence.append({"run_index": launch["run_index"], "evidence": snippets[:20]})
        summary["status"] = "unsupported_configuration" if unsupported else "launch_failed"
        summary["failed_launches"] = failed
        summary["failure_evidence"] = failure_evidence
        return summary

    all_meta = _all_metadata(exp_dir, args.runs)
    missing = {run: sorted(set(range(args.nproc_per_node)) - set(ranks)) for run, ranks in all_meta.items()}
    summary["missing_rank_metadata"] = {str(k): v for k, v in missing.items() if v}
    if any(missing.values()):
        summary["status"] = "metadata_incomplete"
        return summary

    inputs_ok, input_mismatches = _validate_input_hashes(all_meta, args.calls)
    rows, first = _compare_outputs(exp_dir, all_meta, args.runs, args.calls)
    divergent = [row for row in rows if not row["equal"]]
    summary.update(
        {
            "status": "ok",
            "input_hashes_consistent": inputs_ok,
            "input_hash_mismatches": input_mismatches,
            "comparison_count": len(rows),
            "divergent_comparison_count": len(divergent),
            "first_divergence": first,
            "max_changed_elements": max((int(row["changed_elements"]) for row in divergent), default=0),
            "max_changed_bytes": max((int(row["changed_bytes"]) for row in divergent), default=0),
            "max_changed_bits": max((int(row["changed_bits"]) for row in divergent), default=0),
            "max_abs_error": max((float(row["max_abs_error"]) for row in divergent if row["max_abs_error"] is not None), default=0.0),
            "max_rel_error": max((float(row["max_rel_error"]) for row in divergent if row["max_rel_error"] is not None), default=0.0),
            "max_ordered_ulp": max((int(row["max_ordered_ulp"]) for row in divergent if row["max_ordered_ulp"] is not None), default=0),
        }
    )
    write_csv(exp_dir / "summary.csv", rows)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ISSUE4 independent-run NCCL bitwise diagnostic orchestrator")
    parser.add_argument("--collective", choices=COLLECTIVES, required=True)
    parser.add_argument("--dtype", choices=SUPPORTED_DTYPES, default="float32")
    parser.add_argument("--size-bytes", type=parse_size_bytes, required=True, help="Output tensor bytes per rank/call")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--calls", type=int, default=20)
    parser.add_argument("--nproc-per-node", type=int, default=0, help="0 means all visible CUDA devices")
    parser.add_argument("--seed", type=int, default=202604)
    parser.add_argument("--algo", choices=ALGORITHMS, default="default")
    parser.add_argument("--proto", choices=PROTOCOLS, default="default")
    parser.add_argument("--nccl-debug", choices=("none", "WARN", "INFO", "TRACE"), default="INFO")
    parser.add_argument("--nccl-debug-subsys", default="INIT,GRAPH,COLL")
    parser.add_argument("--no-sanitize-nccl-env", dest="sanitize_nccl_env", action="store_false")
    parser.set_defaults(sanitize_nccl_env=True)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--timeout-sec", type=int, default=180)
    parser.add_argument("--launch-timeout-sec", type=int, default=600)
    parser.add_argument("--allow-failure", action="store_true", help="Exit zero even when launch fails; useful for unsupported matrix entries")
    parser.add_argument("--inject-run", type=int, default=None)
    parser.add_argument("--inject-call", type=int, default=None)
    parser.add_argument("--inject-rank", type=int, default=None)
    parser.add_argument("--inject-element", type=int, default=0)
    parser.add_argument("--inject-ulp", type=int, default=0)
    parser.add_argument("--inject-bit", type=int, default=None)
    args = parser.parse_args()
    if args.runs <= 0:
        raise ValueError("--runs must be positive")
    if args.calls <= 0:
        raise ValueError("--calls must be positive")
    if args.nproc_per_node == 0:
        args.nproc_per_node = _auto_world_size()
    if args.nproc_per_node <= 0:
        raise ValueError("--nproc-per-node must be positive")
    return args


def main() -> int:
    args = parse_args()
    exp_dir = _experiment_dir(args)
    _, _, workspace_root = _repo_paths()
    write_json(exp_dir / "command.json", {"argv": sys.argv, "parsed_args": vars(args), "cwd": os.getcwd()})
    write_json(exp_dir / "environment.json", collect_environment(workspace_root))
    env = _build_env(args, exp_dir)
    write_json(exp_dir / "launcher_env_relevant.json", {k: v for k, v in sorted(env.items()) if k.startswith(("NCCL", "CUDA", "CUBLAS", "TORCH", "LD_LIBRARY_PATH", "PYTHON"))})

    launches = []
    for run_index in range(args.runs):
        print(f"[issue4] launch run {run_index}/{args.runs - 1}: {exp_dir / f'run_{run_index:03d}'}", flush=True)
        launches.append(_run_independent_launch(args, exp_dir, env, run_index))
        if launches[-1]["returncode"] != 0 and not args.allow_failure:
            print(f"[issue4] run {run_index} failed with returncode {launches[-1]['returncode']}", file=sys.stderr, flush=True)
            break

    summary = _summarize(args, exp_dir, launches)
    write_json(exp_dir / "summary.json", summary)
    write_markdown_summary(exp_dir / "SUMMARY.md", summary)
    print(json.dumps({"experiment_dir": str(exp_dir), "status": summary.get("status"), "divergent_comparison_count": summary.get("divergent_comparison_count")}, sort_keys=True), flush=True)
    if summary.get("status") != "ok" and not args.allow_failure:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
