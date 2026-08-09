#!/usr/bin/env python3
"""Generate compact Issue6 SVG figures from result CSV files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def bar_svg(rows: list[dict[str, str]], label_col: str, value_col: str, title: str) -> str:
    width, height = 760, 320
    margin_l, margin_b, margin_t = 120, 50, 40
    vals = [float(r[value_col]) for r in rows]
    max_v = max(vals) * 1.15 if vals else 1.0
    bar_w = (width - margin_l - 30) / max(1, len(rows)) * 0.6
    gap = (width - margin_l - 30) / max(1, len(rows))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="24" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">{title}</text>',
        f'<line x1="{margin_l}" y1="{height-margin_b}" x2="{width-20}" y2="{height-margin_b}" stroke="#333"/>',
        f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{height-margin_b}" stroke="#333"/>',
    ]
    for i, row in enumerate(rows):
        v = float(row[value_col])
        x = margin_l + gap * i + gap * 0.2
        h = (height - margin_b - margin_t) * v / max_v
        y = height - margin_b - h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="#2F6B9A"/>')
        parts.append(f'<text x="{x+bar_w/2:.1f}" y="{y-6:.1f}" text-anchor="middle" font-family="Arial" font-size="12">{v:.3f}</text>')
        parts.append(f'<text x="{x+bar_w/2:.1f}" y="{height-28}" text-anchor="middle" font-family="Arial" font-size="12">{row[label_col]}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = [r for r in read_rows(args.benchmark_csv) if r.get("component") == "fused"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "latency.svg").write_text(
        bar_svg(rows, "method", "median_ms", "Issue6 official shape latency (ms)")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
