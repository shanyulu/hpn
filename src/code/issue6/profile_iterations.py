#!/usr/bin/env python3
"""Extract iteration-level communication/GEMM overlap from Nsight Systems SQLite exports."""

from __future__ import annotations

import argparse
import csv
import sqlite3
import statistics
from pathlib import Path


SCATTER_KERNEL_MARKERS = (
    "calc_gather_index_kernel",
    "calc_gather_index_kernel_v2",
    "AgScatterSortOpV2",
    "ag_scatter_sort_topk1_ep_fused_kernel",
    "sort_scatter_index_to_per_expert",
    "prepare_workspace_kernel",
)


def union_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        elif end > merged[-1][1]:
            merged[-1][1] = end
    return [(start, end) for start, end in merged]


def interval_duration_ns(intervals: list[tuple[int, int]]) -> int:
    return sum(end - start for start, end in union_intervals(intervals))


def intersection_duration_ns(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
    left = union_intervals(a)
    right = union_intervals(b)
    i = j = 0
    total = 0
    while i < len(left) and j < len(right):
        start = max(left[i][0], right[j][0])
        end = min(left[i][1], right[j][1])
        if end > start:
            total += end - start
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return total


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[index]


def rank_from_global_tid(global_tid: int, python_processes: list[tuple[int, int]]) -> int | None:
    candidates = [
        (global_tid - global_pid, rank)
        for rank, (_pid, global_pid) in enumerate(python_processes)
        if global_tid >= global_pid and global_tid - global_pid < 1_000_000
    ]
    if not candidates:
        return None
    return min(candidates)[1]


def extract_rows(method: str, sqlite_path: Path, warmup_iters: int) -> list[dict[str, object]]:
    con = sqlite3.connect(sqlite_path)
    con.row_factory = sqlite3.Row
    labels = {row["id"]: row["value"] for row in con.execute("select id, value from StringIds")}
    python_processes = sorted(
        (row["pid"], row["globalPid"])
        for row in con.execute("select pid, globalPid from PROCESSES where name = 'python'")
    )

    loop_by_rank: dict[int, tuple[int, int]] = {}
    for row in con.execute("select start, end, globalTid, text, textId from NVTX_EVENTS where end is not null"):
        label = row["text"] or labels.get(row["textId"])
        if label != "ISSUE6_MEASURED_LOOP":
            continue
        rank = rank_from_global_tid(row["globalTid"], python_processes)
        if rank is not None:
            loop_by_rank[rank] = (row["start"], row["end"])

    rows: list[dict[str, object]] = []
    for rank, (_pid, global_pid) in enumerate(python_processes):
        loop_start, loop_end = loop_by_rank[rank]
        comm_intervals = [
            (row["start"], row["end"])
            for row in con.execute(
                """
                select start, end
                from CUPTI_ACTIVITY_KIND_MEMCPY
                where globalPid = ? and copyKind = 10 and start >= ? and start <= ?
                order by start
                """,
                (global_pid, loop_start, loop_end),
            )
        ]

        gemm_intervals: list[tuple[int, int]] = []
        scatter_intervals: list[tuple[int, int]] = []
        for row in con.execute(
            """
            select k.start, k.end, demangled.value as demangled, short.value as short
            from CUPTI_ACTIVITY_KIND_KERNEL k
            left join StringIds demangled on k.demangledName = demangled.id
            left join StringIds short on k.shortName = short.id
            where k.globalPid = ? and k.start >= ? and k.start <= ?
            order by k.start
            """,
            (global_pid, loop_start, loop_end),
        ):
            name = f"{row['demangled'] or ''} {row['short'] or ''}"
            lower = name.lower()
            if "gemmgroupedv2" in lower and "cutlass::kernel" in lower:
                gemm_intervals.append((row["start"], row["end"]))
            elif any(marker in name for marker in SCATTER_KERNEL_MARKERS):
                scatter_intervals.append((row["start"], row["end"]))

        if not gemm_intervals:
            raise RuntimeError(f"No grouped GEMM kernels found for rank {rank} in {sqlite_path}")
        copies_per_iter = len(comm_intervals) // len(gemm_intervals)
        scatter_per_iter = len(scatter_intervals) // len(gemm_intervals)
        if copies_per_iter == 0 or scatter_per_iter == 0:
            raise RuntimeError(
                f"Could not infer per-iteration activity counts for rank {rank}: "
                f"comm={len(comm_intervals)}, scatter={len(scatter_intervals)}, gemm={len(gemm_intervals)}"
            )

        measured_count = len(gemm_intervals) - warmup_iters
        for measured_idx in range(measured_count):
            source_idx = warmup_iters + measured_idx
            comm = comm_intervals[copies_per_iter * source_idx : copies_per_iter * (source_idx + 1)]
            scatter = scatter_intervals[
                scatter_per_iter * source_idx : scatter_per_iter * (source_idx + 1)
            ]
            gemm = [gemm_intervals[source_idx]]
            comm_ns = interval_duration_ns(comm)
            gemm_ns = interval_duration_ns(gemm)
            scatter_ns = interval_duration_ns(scatter)
            comm_gemm_ns = intersection_duration_ns(comm, gemm)
            comm_compute_ns = intersection_duration_ns(comm, scatter + gemm)
            first_comm_start = min(start for start, _end in comm)
            first_gemm_start = gemm[0][0]
            rows.append(
                {
                    "method": method,
                    "rank": rank,
                    "iteration": measured_idx,
                    "comm_active_us": comm_ns / 1_000.0,
                    "gemm_active_us": gemm_ns / 1_000.0,
                    "scatter_prep_active_us": scatter_ns / 1_000.0,
                    "comm_gemm_overlap_us": comm_gemm_ns / 1_000.0,
                    "comm_compute_overlap_us": comm_compute_ns / 1_000.0,
                    "comm_gemm_overlap_pct": (comm_gemm_ns / comm_ns * 100.0) if comm_ns else 0.0,
                    "comm_compute_overlap_pct": (comm_compute_ns / comm_ns * 100.0)
                    if comm_ns
                    else 0.0,
                    "first_gemm_delay_us": (first_gemm_start - first_comm_start) / 1_000.0,
                    "p2p_memcpy_count": copies_per_iter,
                    "scatter_prep_kernel_count": scatter_per_iter,
                    "source_sqlite": str(sqlite_path),
                }
            )
    return rows


def summarize(method: str, rows: list[dict[str, object]]) -> dict[str, object]:
    out: dict[str, object] = {"method": method, "rows": len(rows)}
    for field in [
        "comm_active_us",
        "gemm_active_us",
        "scatter_prep_active_us",
        "comm_gemm_overlap_us",
        "comm_compute_overlap_us",
        "comm_gemm_overlap_pct",
        "comm_compute_overlap_pct",
        "first_gemm_delay_us",
        "p2p_memcpy_count",
        "scatter_prep_kernel_count",
    ]:
        values = [float(row[field]) for row in rows]
        out[f"{field}_median"] = statistics.median(values)
        out[f"{field}_p25"] = percentile(values, 0.25)
        out[f"{field}_p75"] = percentile(values, 0.75)
        out[f"{field}_p95"] = percentile(values, 0.95)
        out[f"{field}_min"] = min(values)
        out[f"{field}_max"] = max(values)
    out["source_sqlite"] = rows[0]["source_sqlite"] if rows else ""
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        action="append",
        required=True,
        metavar="METHOD=SQLITE",
        help="Method label and Nsight Systems SQLite export path.",
    )
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--iterations-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()

    all_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for item in args.profile:
        method, sep, path_text = item.partition("=")
        if not sep:
            raise ValueError(f"--profile must be METHOD=SQLITE, got {item!r}")
        rows = extract_rows(method, Path(path_text), args.warmup_iters)
        all_rows.extend(rows)
        summaries.append(summarize(method, rows))

    args.iterations_output.parent.mkdir(parents=True, exist_ok=True)
    with args.iterations_output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_rows)

    with args.summary_output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
