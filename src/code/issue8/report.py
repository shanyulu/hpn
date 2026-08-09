"""Generate an evidence-classified Issue8 decision table."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    return float(value) if value not in {"", "N/A", "null", "None", "unavailable"} else None


def _is_multi_rank_measurement(row: dict[str, str]) -> bool:
    try:
        return (
            row.get("completion_time_evidence") == "GPU_event_max_rank_completion"
            and int(row.get("world_size", "0")) > 1
        )
    except (TypeError, ValueError):
        return False


def decision_rows(
    rows: Iterable[dict[str, str]], precision_thresholds: tuple[float | None, ...] = (None,)
) -> list[dict[str, str]]:
    """Produce latency recommendations only from real multi-rank samples."""
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if _is_multi_rank_measurement(row):
            groups[(row["duplicate_bucket"], row["message_bucket"])].append(row)

    result = []
    for (dup, message), group in sorted(groups.items()):
        valid = [r for r in group if _float(r, "measured_combine_completion_median_ms") is not None]
        if not valid:
            continue
        for threshold in precision_thresholds:
            if threshold is None:
                eligible = valid
                label = "no_error_limit"
            else:
                eligible = [
                    r for r in valid
                    if _float(r, "max_abs_error") is not None
                    and _float(r, "max_abs_error") <= threshold
                ]
                label = f"max_abs_error<={threshold:g}"
            if eligible:
                best = min(
                    eligible,
                    key=lambda r: _float(r, "measured_combine_completion_median_ms") or float("inf"),
                )
                basis = "measured multi-rank max-rank completion time among precision-eligible modes"
            else:
                best = min(valid, key=lambda r: _float(r, "max_abs_error") or float("inf"))
                basis = "no mode met the error limit; minimum measured max_abs_error selected"
            result.append(
                {
                    "evidence": "measured_multi_rank",
                    "duplicate_bucket": dup,
                    "message_bucket": message,
                    "precision_requirement": label,
                    "recommended_mode": best["mode"],
                    "basis": basis,
                }
            )
    return result


def source_derived_rows() -> list[dict[str, str]]:
    """Guidance that deliberately makes no completion-time claim."""
    return [
        {
            "evidence": "source_derived_and_L1_analytical",
            "duplicate_bucket": "zero",
            "message_bucket": "any",
            "precision_requirement": "no host-validated latency threshold",
            "recommended_mode": "A",
            "basis": "direct non-expanded source path; this is not a measured latency recommendation",
        },
        {
            "evidence": "source_derived_and_L1_analytical",
            "duplicate_bucket": "nonzero; expanded layout required",
            "message_bucket": "any",
            "precision_requirement": "logical return-payload priority",
            "recommended_mode": "B",
            "basis": "merges destination-rank collisions before return; no multi-rank latency validation",
        },
        {
            "evidence": "source_derived",
            "duplicate_bucket": "any",
            "message_bucket": "any",
            "precision_requirement": "independent replica return semantics required",
            "recommended_mode": "C",
            "basis": "all valid top-k slots reach the epilogue; no host-validated precision or latency ranking",
        },
    ]


def write_report(summary_csv: Path, output_dir: Path) -> Path:
    measured = decision_rows(load_rows(summary_csv), precision_thresholds=(1e-3, 1e-2, None))
    source_derived = source_derived_rows()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "decision_table.md"
    lines = [
        "# Decision Table",
        "",
        "The tested host has no measured multi-rank DeepEP V2 combine completion time.",
        "No completion-time threshold or latency advantage is inferred below.",
        "",
        "| Evidence | Duplicate bucket | Message bucket | Precision requirement | Recommended mode | Basis |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in source_derived + measured:
        lines.append(
            "| {evidence} | {duplicate_bucket} | {message_bucket} | {precision_requirement} | "
            "{recommended_mode} | {basis} |".format(**row)
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "decision_table.json").write_text(
        json.dumps({"source_derived": source_derived, "measured_multi_rank": measured}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return output
