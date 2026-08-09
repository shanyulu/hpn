#!/usr/bin/env python3
# Copyright (c) 2025, TENCENT CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information.

"""Dependency-free CPU test runner for ISSUE4.

The test files are also pytest-compatible, but the challenge environment may not
have pytest installed.  This runner executes top-level ``test_*`` functions with
the Python standard library only.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import sys
import traceback
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ISSUE4 CPU unit tests without pytest")
    parser.add_argument("--tests-dir", default=str(Path(__file__).resolve().parent / "tests"))
    args = parser.parse_args()

    tests_dir = Path(args.tests_dir).resolve()
    code_root = Path(__file__).resolve().parents[1]
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))

    failures: list[tuple[str, BaseException, str]] = []
    total = 0
    for path in sorted(tests_dir.glob("test_*.py")):
        module_name = f"issue4.tests.{path.stem}"
        module = importlib.import_module(module_name)
        for name, fn in sorted(vars(module).items()):
            if name.startswith("test_") and inspect.isfunction(fn):
                total += 1
                case_name = f"{module_name}.{name}"
                try:
                    fn()
                    print(f"PASS {case_name}")
                except BaseException as exc:  # noqa: BLE001 - test runner should catch assertion failures
                    failures.append((case_name, exc, traceback.format_exc()))
                    print(f"FAIL {case_name}: {exc}")

    print(f"SUMMARY total={total} passed={total - len(failures)} failed={len(failures)}")
    if failures:
        for case_name, _, tb in failures:
            print(f"\n--- {case_name} ---")
            print(tb)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
