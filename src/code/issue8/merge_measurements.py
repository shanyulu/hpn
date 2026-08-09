#!/usr/bin/env python3
"""Merge independent DeepEP benchmark launches into one measured summary CSV."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def percentile(values: list[float], value: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * value
    low, high = int(index), min(int(index) + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def message_bucket(row: dict[str, object]) -> str:
    size = int(row["num_tokens"]) * int(row["hidden"]) * 2
    if size < 1 << 20:
        return "small_<1MiB"
    if size < 16 << 20:
        return "medium_1-16MiB"
    return "large_>=16MiB"


def is_multi_rank_completion(record: dict[str, object]) -> bool:
    """Return whether a record is eligible for a completion-time aggregate."""
    try:
        return (
            record.get("status") == "measured"
            and record.get("execution_scope") == "multi_rank"
            and record.get("completion_time_evidence") == "GPU_event_max_rank_completion"
            and int(record.get("world_size", 0)) > 1
        )
    except (TypeError, ValueError):
        return False


def merge(records: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, object], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[(record.get("case_id"), record.get("mode"))].append(record)
    merged: list[dict[str, object]] = []
    for (_case_id, _mode), group in sorted(grouped.items(), key=str):
        blocked = [item for item in group if item.get("status") != "measured"]
        if blocked:
            merged.append({
                "status": "blocked",
                "case_id": _case_id,
                "mode": _mode,
                "independent_launches": len(group),
                "reason": "; ".join(str(item.get("reason", "unknown")) for item in blocked),
            })
            continue
        invalid = [item for item in group if not is_multi_rank_completion(item)]
        if invalid:
            raise ValueError(
                f"{_case_id}/{_mode}: only multi-rank GPU_event_max_rank_completion "
                "records can be merged"
            )
        samples = [float(v) for item in group for v in item.get("critical_path_samples_ms", [])]
        if not samples:
            raise ValueError(f"{_case_id}/{_mode}: no raw critical-path samples")
        row = dict(group[0])
        row.update({
            "independent_launches": len(group),
            "message_bucket": message_bucket(row),
            "measured_combine_completion_median_ms": statistics.median(samples),
            "measured_combine_completion_p95_ms": percentile(samples, 0.95),
            "measured_combine_completion_min_ms": min(samples),
            "measured_combine_completion_max_ms": max(samples),
            "measured_combine_completion_stddev_ms": statistics.pstdev(samples),
            "total_measurement_samples": len(samples),
        })
        row.pop("critical_path_samples_ms", None)
        merged.append(row)
    return merged


def write_csv(rows: list[dict[str, object]], output: Path) -> None:
    keys = sorted({key for row in rows for key in row})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=Path("results/measured_summary.csv"))
    args = parser.parse_args()
    records = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    rows = merge(records)
    write_csv(rows, args.output)
    from report import write_report

    write_report(args.output, args.output.parent)
    print(json.dumps({"rows": len(rows), "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
