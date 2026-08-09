#!/usr/bin/env python3
"""Create the deterministic analytical Issue8 matrix."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from routing import build_case
from traffic import analytical_traffic


def message_bucket(tokens: int, hidden: int, dtype: str) -> str:
    payload = tokens * hidden * {"bf16": 2, "fp16": 2, "fp32": 4}[dtype]
    if payload < 1 << 20:
        return "small_<1MiB"
    if payload < 16 << 20:
        return "medium_1-16MiB"
    return "large_>=16MiB"


def make_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for bucket in args.duplicate_buckets:
        for tokens, hidden in args.shapes:
            case, _x, topk_idx, _weights = build_case(
                seed=args.seed,
                num_tokens=tokens,
                hidden=hidden,
                topk=args.topk,
                num_experts=args.num_experts,
                world_size=args.world_size,
                duplicate_bucket=bucket,
            )
            experts_per_rank = args.num_experts // args.world_size
            for mode in ("A", "B", "C"):
                traffic = analytical_traffic(
                    topk_idx,
                    hidden=hidden,
                    experts_per_rank=experts_per_rank,
                    dtype=args.dtype,
                    mode=mode,
                )
                rows.append(
                    {
                        **case.metadata(),
                        "mode": mode,
                        "dtype": args.dtype,
                        "message_bucket": message_bucket(tokens, hidden, args.dtype),
                        **traffic.as_dict(),
                        "evidence_classification": "analytical",
                        "completion_time_evidence": "unavailable_on_tested_host",
                        "measured_combine_completion_median_ms": None,
                        "measured_combine_completion_p95_ms": None,
                        "precision_evidence": "unavailable_in_analytical_matrix",
                        "max_abs_error": None,
                        "relative_l2_error": None,
                        "status": "analytical_only",
                    }
                )
    return rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "N/A" if value is None else value for key, value in row.items()})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--num-experts", type=int, default=32)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--analytical-only", action="store_true")
    parser.add_argument("--duplicate-buckets", nargs="+", default=["zero", "low", "medium", "high", "max"])
    parser.add_argument(
        "--shapes",
        nargs="+",
        default=["256x1024", "1024x4096", "4096x7168"],
        help="tokensxhidden entries grounded in DeepEP test scale, e.g. 1024x4096",
    )
    args = parser.parse_args()
    args.shapes = [tuple(map(int, shape.split("x", 1))) for shape in args.shapes]
    rows = make_rows(args)
    csv_path = args.output_dir / "summary.csv"
    write_csv(rows, csv_path)
    (args.output_dir / "summary.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not args.analytical_only:
        print(
            "Matrix contains no measured completion time. Use benchmark.py only on a "
            "host where the real multi-rank DeepEP path completes.",
            file=sys.stderr,
        )
    from report import write_report

    write_report(csv_path, args.output_dir)
    print(json.dumps({"rows": len(rows), "summary_csv": str(csv_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
