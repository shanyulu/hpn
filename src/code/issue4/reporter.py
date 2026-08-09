#!/usr/bin/env python3
# Copyright (c) 2025, TENCENT CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information.

"""Report writers for ISSUE4 diagnostic experiments."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


CSV_FIELDS = [
    "run_a",
    "run_b",
    "call_index",
    "rank",
    "equal",
    "changed_elements",
    "changed_bytes",
    "changed_bits",
    "first_differing_element",
    "first_differing_byte",
    "first_differing_bit",
    "max_abs_error",
    "max_rel_error",
    "max_unsigned_ulp",
    "max_ordered_ulp",
]


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True), encoding="utf-8")


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown_summary(path: str | Path, summary: dict[str, Any]) -> None:
    first = summary.get("first_divergence")
    lines = [
        "# ISSUE4 Diagnostic Summary",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Collective: `{summary.get('config', {}).get('collective')}`",
        f"- DType: `{summary.get('config', {}).get('dtype')}`",
        f"- Size bytes per output tensor: `{summary.get('config', {}).get('size_bytes')}`",
        f"- Independent runs: `{summary.get('config', {}).get('runs')}`",
        f"- Calls per run: `{summary.get('config', {}).get('calls')}`",
        f"- World size: `{summary.get('config', {}).get('nproc_per_node')}`",
        f"- Input hashes consistent: `{summary.get('input_hashes_consistent')}`",
        f"- Comparisons: `{summary.get('comparison_count')}`",
        f"- Divergent comparisons: `{summary.get('divergent_comparison_count')}`",
    ]
    if first:
        cmp = first.get("comparison", {})
        lines.extend(
            [
                "",
                "## First divergence",
                "",
                f"- run_a: `{first.get('run_a')}`",
                f"- run_b: `{first.get('run_b')}`",
                f"- call: `{first.get('call_index')}`",
                f"- rank: `{first.get('rank')}`",
                f"- element: `{cmp.get('first_differing_element')}`",
                f"- byte offset: `{cmp.get('first_differing_byte')}`",
                f"- bit offset: `{cmp.get('first_differing_bit')}`",
                f"- changed elements: `{cmp.get('changed_elements')}`",
                f"- changed bytes: `{cmp.get('changed_bytes')}`",
                f"- changed bits: `{cmp.get('changed_bits')}`",
                f"- max abs error: `{cmp.get('max_abs_error')}`",
                f"- max rel error: `{cmp.get('max_rel_error')}`",
                f"- max ordered ULP: `{cmp.get('max_ordered_ulp')}`",
            ]
        )
    else:
        lines.extend(["", "No run-to-run bitwise divergence was observed in this experiment."])
    lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")
