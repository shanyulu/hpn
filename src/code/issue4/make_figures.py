#!/usr/bin/env python3
# Copyright (c) 2025, TENCENT CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information.

"""Generate ISSUE4 reviewer figures from canonical CSV/JSON evidence."""

from __future__ import annotations

import argparse
import csv
import json
from html import escape
from pathlib import Path


def _text(x: float, y: float, value: object, size: int = 13, weight: str = "400", anchor: str = "start", fill: str = "#1f2937") -> str:
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" '
        f'text-anchor="{anchor}" fill="{fill}" font-family="Arial, sans-serif">'
        f"{escape(str(value))}</text>"
    )


def _status_color(status: str) -> str:
    return {"pass": "#16a34a", "unsupported": "#f59e0b", "not_tested": "#9ca3af"}.get(status, "#dc2626")


def _status_label(status: str) -> str:
    return {"pass": "✓", "unsupported": "unsupported", "not_tested": "n/t"}.get(status, status)


def _size_label(size: int) -> str:
    if size >= 1024 * 1024:
        return f"{size // (1024 * 1024)}M"
    if size >= 1024:
        return f"{size // 1024}K"
    return str(size)


def reproducibility_matrix(results_dir: Path, out_dir: Path) -> None:
    rows = list(csv.DictReader((results_dir / "same_config_matrix.csv").open()))
    cols = ["algo_default", "algo_ring", "algo_tree", "proto_default", "proto_simple", "proto_ll", "proto_ll128"]
    labels = ["Algo\ndefault", "Algo\nRing", "Algo\nTree", "Proto\ndefault", "Proto\nSimple", "Proto\nLL", "Proto\nLL128"]
    width, height = 980, 250
    x0, y0, cell_w, row_h = 180, 85, 105, 48

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        _text(30, 32, "Same-configuration reproducibility matrix", 18, "700"),
        _text(30, 55, "4× RTX PRO 6000 Blackwell, FP32, 1 MiB algorithm/protocol matrix", 12, fill="#4b5563"),
        _text(30, y0 + 31, "Collective", 13, "700"),
    ]
    for idx, label in enumerate(labels):
        for line_idx, line in enumerate(label.split("\n")):
            parts.append(_text(x0 + idx * cell_w + cell_w / 2, y0 - 16 + line_idx * 15, line, 12, "700", "middle"))
    for row_idx, row in enumerate(rows):
        y = y0 + row_idx * row_h
        fill = "#f9fafb" if row_idx % 2 == 0 else "#ffffff"
        parts.append(f'<rect x="20" y="{y}" width="940" height="{row_h}" fill="{fill}" stroke="#e5e7eb"/>')
        parts.append(_text(30, y + 31, row["collective"], 13, "700"))
        for col_idx, col in enumerate(cols):
            status = row[col]
            x = x0 + col_idx * cell_w
            color = _status_color(status)
            parts.append(f'<rect x="{x + 8}" y="{y + 9}" width="{cell_w - 16}" height="30" rx="6" fill="{color}" opacity="0.14" stroke="{color}"/>')
            parts.append(_text(x + cell_w / 2, y + 30, _status_label(status), 13, "700", "middle", color))
    parts.extend([
        _text(30, 220, "✓ = 0 divergent comparisons. unsupported = NCCL reported no supported algorithm/protocol for this configuration.", 12, fill="#4b5563"),
        "</svg>",
    ])
    (out_dir / "reproducibility_matrix.svg").write_text("\n".join(parts) + "\n")


def performance_chart(results_dir: Path, out_dir: Path) -> None:
    rows = [r for r in csv.DictReader((results_dir / "nccl_tests_performance.csv").open()) if r["dataset"] == "default_size_sweep"]
    selected_sizes = {1024, 65536, 1048576, 16777216, 134217728}
    rows = [r for r in rows if int(r["size_bytes"]) in selected_sizes]
    sizes = sorted({int(r["size_bytes"]) for r in rows})
    max_bw = max(float(r["oop_busbw_GBps"]) for r in rows)

    width, height = 920, 380
    left, top, plot_w, plot_h = 70, 70, 780, 230
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        _text(30, 32, "nccl-tests default-path performance", 18, "700"),
        _text(30, 55, "Out-of-place bus bandwidth from PyTorch NCCL 2.27.3 nccl-tests; Python capture timings are not used.", 12, fill="#4b5563"),
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#9ca3af"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#9ca3af"/>',
    ]
    for idx, size in enumerate(sizes):
        x = left + plot_w * (idx / (len(sizes) - 1 if len(sizes) > 1 else 1))
        parts.append(f'<line x1="{x}" y1="{top + plot_h}" x2="{x}" y2="{top + plot_h + 5}" stroke="#9ca3af"/>')
        parts.append(_text(x, top + plot_h + 22, _size_label(size), 11, anchor="middle", fill="#4b5563"))

    colors = {"all_reduce": "#2563eb", "reduce_scatter": "#dc2626"}
    for collective in ("all_reduce", "reduce_scatter"):
        points: list[tuple[float, float, float]] = []
        for idx, size in enumerate(sizes):
            match = [r for r in rows if r["collective"] == collective and int(r["size_bytes"]) == size]
            if not match:
                continue
            bw = float(match[0]["oop_busbw_GBps"])
            x = left + plot_w * (idx / (len(sizes) - 1 if len(sizes) > 1 else 1))
            y = top + plot_h - (bw / max_bw) * plot_h
            points.append((x, y, bw))
        if not points:
            continue
        parts.append('<polyline fill="none" stroke="{}" stroke-width="2.5" points="{}"/>'.format(colors[collective], " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in points)))
        for x, y, bw in points:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{colors[collective]}"/>')
            parts.append(_text(x, y - 8, f"{bw:.1f}", 10, anchor="middle", fill=colors[collective]))
    parts.extend([
        _text(left, top - 10, "busbw GB/s", 11, "700", fill="#4b5563"),
        _text(left + plot_w, top + plot_h + 45, "message size", 11, "700", "end", fill="#4b5563"),
        '<rect x="650" y="72" width="12" height="12" fill="#2563eb"/><text x="668" y="83" font-size="12" font-family="Arial" fill="#1f2937">AllReduce</text>',
        '<rect x="650" y="92" width="12" height="12" fill="#dc2626"/><text x="668" y="103" font-size="12" font-family="Arial" fill="#1f2937">ReduceScatter</text>',
        _text(30, 350, "CSV source: results/nccl_tests_performance.csv. Values come from nccl-tests, not the Python forensic capture path.", 11, fill="#4b5563"),
        "</svg>",
    ])
    (out_dir / "nccl_tests_performance.svg").write_text("\n".join(parts) + "\n")


def ring_tree_chart(results_dir: Path, out_dir: Path) -> None:
    data = json.loads((results_dir / "ring_vs_tree.json").read_text())
    ring = data["ring_same_configuration"]
    tree = data["tree_same_configuration"]
    cross = data["ring_vs_tree_cross_configuration"]
    items = [
        ("Ring same-config", ring["divergent_comparison_count"], ring["comparison_count"], "#16a34a"),
        ("Tree same-config", tree["divergent_comparison_count"], tree["comparison_count"], "#16a34a"),
        ("Ring vs Tree cross-config", cross["divergent_comparison_count"], cross["comparison_count"], "#dc2626"),
    ]
    max_value = max(value for _, value, _, _ in items)
    width, height = 820, 300
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        _text(30, 34, "Ring vs Tree: reproducibility vs numerical sensitivity", 18, "700"),
        _text(30, 58, "Same configuration stayed bitwise stable; changing algorithm changed outputs.", 12, fill="#4b5563"),
    ]
    for idx, (label, value, total, color) in enumerate(items):
        y = 95 + idx * 58
        bar_w = 0 if max_value == 0 else 430 * value / max_value
        parts.extend([
            _text(35, y + 20, label, 13, "700"),
            f'<rect x="230" y="{y}" width="430" height="28" rx="6" fill="#f3f4f6"/>',
            f'<rect x="230" y="{y}" width="{bar_w:.1f}" height="28" rx="6" fill="{color}" opacity="0.85"/>',
            _text(680, y + 20, f"{value} / {total} divergent", 13, "700", fill=color),
        ])
    parts.extend([
        _text(35, 270, "JSON source: results/ring_vs_tree.json. Cross-config difference is not classified as NCCL nondeterminism.", 12, fill="#4b5563"),
        "</svg>",
    ])
    (out_dir / "ring_vs_tree.svg").write_text("\n".join(parts) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    out_dir = args.results_dir / "figures"
    out_dir.mkdir(exist_ok=True)
    reproducibility_matrix(args.results_dir, out_dir)
    performance_chart(args.results_dir, out_dir)
    ring_tree_chart(args.results_dir, out_dir)


if __name__ == "__main__":
    main()
