"""Generate a conservative decision table from recorded results."""

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
    return float(value) if value not in {"", "null", "None"} else None


def decision_rows(
    rows: Iterable[dict[str, str]], precision_thresholds: tuple[float | None, ...] = (None,)
) -> list[dict[str, str]]:
    """Return data-backed recommendations only when measured data exists.

    This intentionally refuses to manufacture latency thresholds from logical
    traffic estimates. A hardware-supported sweep fills the table later.
    """
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("latency_evidence") == "GPU_event_max_rank_completion":
            groups[(row["duplicate_bucket"], row["message_bucket"])].append(row)

    result = []
    for (dup, message), group in sorted(groups.items()):
        valid = [r for r in group if _float(r, "median_ms") is not None]
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
                best = min(eligible, key=lambda r: _float(r, "median_ms") or float("inf"))
                basis = "minimum measured max-rank median completion time among precision-eligible modes"
            else:
                best = min(valid, key=lambda r: _float(r, "max_abs_error") or float("inf"))
                basis = "no measured mode met the error limit; minimum measured max_abs_error selected"
            result.append(
                {
                    "duplicate_bucket": dup,
                    "message_bucket": message,
                    "precision_requirement": label,
                    "recommended_mode": best["mode"],
                    "basis": basis,
                }
            )
    return result


def write_report(summary_csv: Path, output_dir: Path) -> Path:
    rows = load_rows(summary_csv)
    decisions = decision_rows(rows, precision_thresholds=(1e-3, 1e-2, None))
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "decision_table.md"
    lines = [
        "# Decision Table",
        "",
        "Only rows with `GPU_event_max_rank_completion` latency are allowed to produce a recommendation.",
        "Logical payload estimates are never promoted to measured latency.",
        "",
        "| Duplicate bucket | Message bucket | Precision requirement | Recommended mode | Basis |",
        "| --- | --- | --- | --- | --- |",
    ]
    if decisions:
        for row in decisions:
            lines.append(
                "| {duplicate_bucket} | {message_bucket} | {precision_requirement} | "
                "{recommended_mode} | {basis} |".format(**row)
            )
    else:
        lines.append("| pending | pending | pending | pending | No supported multi-rank measurement is present. |")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "decision_table.json").write_text(
        json.dumps(decisions, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output
