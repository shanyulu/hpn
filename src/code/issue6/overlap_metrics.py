#!/usr/bin/env python3
"""Compute Issue6 derived overlap metrics from benchmark summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_summary(path: Path) -> dict:
    return json.loads(path.read_text())


def derived_overlap(compute_ms: float, comm_ms: float, fused_ms: float) -> dict[str, float]:
    serial = compute_ms + comm_ms
    ideal = max(compute_ms, comm_ms)
    denom = serial - ideal
    efficiency = (serial - fused_ms) / denom if denom > 0 else 0.0
    return {
        "compute_ms": compute_ms,
        "comm_ms": comm_ms,
        "fused_ms": fused_ms,
        "serial_ms": serial,
        "ideal_ms": ideal,
        "derived_overlap_pct": efficiency * 100.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    fused = read_summary(args.input_dir / f"{args.variant}_fused_summary.json")
    compute = read_summary(args.input_dir / f"{args.variant}_compute_only_summary.json")
    comm = read_summary(args.input_dir / f"{args.variant}_comm_only_summary.json")
    row = {"variant": args.variant}
    row.update(
        derived_overlap(
            compute["distributed"]["median_ms"],
            comm["distributed"]["median_ms"],
            fused["distributed"]["median_ms"],
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    print(json.dumps(row, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
