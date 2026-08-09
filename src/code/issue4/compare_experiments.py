#!/usr/bin/env python3
# Copyright (c) 2025, TENCENT CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information.

"""Compare captures from two completed ISSUE4 experiments."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from issue4.comparator import compare_files  # type: ignore
else:
    from .comparator import compare_files


def _load_run_metadata(exp_dir: Path, run_index: int) -> dict[int, dict[str, Any]]:
    run_dir = exp_dir / f"run_{run_index:03d}"
    out = {}
    for path in sorted(run_dir.glob("rank*_metadata.json")):
        meta = json.loads(path.read_text(encoding="utf-8"))
        out[int(meta["rank"])] = meta
    if not out:
        raise FileNotFoundError(f"no rank metadata in {run_dir}")
    return out


def compare_experiments(exp_a: Path, exp_b: Path, run_a: int, run_b: int) -> dict[str, Any]:
    meta_a = _load_run_metadata(exp_a, run_a)
    meta_b = _load_run_metadata(exp_b, run_b)
    ranks = sorted(set(meta_a) & set(meta_b))
    rows: list[dict[str, Any]] = []
    first = None
    input_mismatches = []
    for rank in ranks:
        ma = meta_a[rank]
        mb = meta_b[rank]
        if ma["dtype"] != mb["dtype"] or ma["output_shape"] != mb["output_shape"] or ma["calls"] != mb["calls"]:
            raise ValueError(f"incompatible rank {rank} metadata")
        for call_index in range(ma["calls"]):
            ra = ma["records"][call_index]
            rb = mb["records"][call_index]
            if ra["input_sha256"] != rb["input_sha256"]:
                input_mismatches.append({"rank": rank, "call_index": call_index, "exp_a": ra["input_sha256"], "exp_b": rb["input_sha256"]})
            path_a = exp_a / f"run_{run_a:03d}" / ra["output_file"]
            path_b = exp_b / f"run_{run_b:03d}" / rb["output_file"]
            result = compare_files(path_a, path_b, ma["dtype"], ma["output_shape"])
            result_dict = result.to_dict()
            row = {
                "rank": rank,
                "call_index": call_index,
                **result_dict,
                "path_a": str(path_a),
                "path_b": str(path_b),
            }
            rows.append(row)
            if not result.equal and first is None:
                first = {"rank": rank, "call_index": call_index, "comparison": result_dict, "path_a": str(path_a), "path_b": str(path_b)}
    divergent = [row for row in rows if not row["equal"]]
    return {
        "schema_version": 1,
        "exp_a": str(exp_a),
        "exp_b": str(exp_b),
        "run_a": run_a,
        "run_b": run_b,
        "input_hashes_consistent": len(input_mismatches) == 0,
        "input_hash_mismatches": input_mismatches,
        "comparison_count": len(rows),
        "divergent_comparison_count": len(divergent),
        "first_difference": first,
        "max_changed_elements": max((int(row["changed_elements"]) for row in divergent), default=0),
        "max_changed_bytes": max((int(row["changed_bytes"]) for row in divergent), default=0),
        "max_changed_bits": max((int(row["changed_bits"]) for row in divergent), default=0),
        "max_abs_error": max((float(row["max_abs_error"]) for row in divergent if row["max_abs_error"] is not None), default=0.0),
        "max_rel_error": max((float(row["max_rel_error"]) for row in divergent if row["max_rel_error"] is not None), default=0.0),
        "max_ordered_ulp": max((int(row["max_ordered_ulp"]) for row in divergent if row["max_ordered_ulp"] is not None), default=0),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two ISSUE4 experiment captures")
    parser.add_argument("--exp-a", required=True)
    parser.add_argument("--exp-b", required=True)
    parser.add_argument("--run-a", type=int, default=0)
    parser.add_argument("--run-b", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = compare_experiments(Path(args.exp_a), Path(args.exp_b), args.run_a, args.run_b)
    rows = summary.pop("rows")
    (output_dir / "cross_compare_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=True), encoding="utf-8")
    with (output_dir / "cross_compare_rows.csv").open("w", newline="", encoding="utf-8") as f:
        fields = [
            "rank",
            "call_index",
            "equal",
            "changed_elements",
            "changed_bytes",
            "changed_bits",
            "first_differing_element",
            "first_differing_byte",
            "first_differing_bit",
            "max_abs_error",
            "max_rel_error",
            "max_ordered_ulp",
            "path_a",
            "path_b",
        ]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"output_dir": str(output_dir.resolve()), "divergent_comparison_count": summary["divergent_comparison_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
