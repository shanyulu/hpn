#!/usr/bin/env python3
# Copyright (c) 2025, TENCENT CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information.

"""Parse nccl-tests text output into CSV."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


DATA_RE = re.compile(
    r"^\s*(?P<size>\d+)\s+(?P<count>\d+)\s+(?P<dtype>\S+)\s+(?P<redop>\S+)\s+(?P<root>\S+)\s+"
    r"(?P<oop_time_us>[-+0-9.]+)\s+(?P<oop_algbw>[-+0-9.]+)\s+(?P<oop_busbw>[-+0-9.]+)\s+(?P<oop_wrong>\d+)\s+"
    r"(?P<ip_time_us>[-+0-9.]+)\s+(?P<ip_algbw>[-+0-9.]+)\s+(?P<ip_busbw>[-+0-9.]+)\s+(?P<ip_wrong>\d+)"
)


def parse_log(path: Path, collective: str, algo: str, proto: str, nccl_link: str) -> list[dict[str, str]]:
    rows = []
    command = ""
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("COMMAND:"):
            command = line.removeprefix("COMMAND:").strip()
            continue
        match = DATA_RE.match(line)
        if not match:
            continue
        row = match.groupdict()
        row.update(
            {
                "source_log": str(path),
                "collective": collective,
                "algo": algo,
                "proto": proto,
                "nccl_link": nccl_link,
                "command": command,
            }
        )
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse nccl-tests logs into CSV")
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--collective", required=True)
    parser.add_argument("--algo", default="default")
    parser.add_argument("--proto", default="default")
    parser.add_argument("--nccl-link", default="unknown")
    args = parser.parse_args()
    rows = []
    for item in args.input:
        rows.extend(parse_log(Path(item), args.collective, args.algo, args.proto, args.nccl_link))
    fieldnames = [
        "collective",
        "algo",
        "proto",
        "size",
        "count",
        "dtype",
        "redop",
        "root",
        "oop_time_us",
        "oop_algbw",
        "oop_busbw",
        "oop_wrong",
        "ip_time_us",
        "ip_algbw",
        "ip_busbw",
        "ip_wrong",
        "nccl_link",
        "command",
        "source_log",
    ]
    with Path(args.output).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
