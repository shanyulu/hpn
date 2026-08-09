#!/usr/bin/env python3
# Copyright (c) 2025, TENCENT CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information.

"""Environment snapshot utilities for ISSUE4 experiments."""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _run_command(cmd: list[str], timeout: int = 30) -> dict[str, Any]:
    try:
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=False)
        return {
            "cmd": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except FileNotFoundError as exc:
        return {"cmd": cmd, "returncode": None, "stdout": "", "stderr": repr(exc)}
    except subprocess.TimeoutExpired as exc:
        return {
            "cmd": cmd,
            "returncode": None,
            "stdout": exc.stdout or "",
            "stderr": f"timeout after {timeout}s: {exc}",
        }


def _git_snapshot(path: Path) -> dict[str, Any]:
    if not (path / ".git").exists():
        return {"path": str(path), "present": path.exists(), "is_git": False}
    head = _run_command(["git", "-C", str(path), "rev-parse", "HEAD"], timeout=10)
    branch = _run_command(["git", "-C", str(path), "status", "--short", "--branch"], timeout=10)
    return {
        "path": str(path),
        "present": True,
        "is_git": True,
        "head": head.get("stdout", "").strip(),
        "status_short_branch": branch.get("stdout", ""),
        "head_command": head,
        "status_command": branch,
    }


def _torch_snapshot() -> dict[str, Any]:
    try:
        import torch

        snapshot: dict[str, Any] = {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "torch_nccl": list(torch.cuda.nccl.version()) if hasattr(torch.cuda, "nccl") else None,
            "cudnn": torch.backends.cudnn.version(),
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "distributed_available": torch.distributed.is_available(),
            "distributed_nccl_available": torch.distributed.is_nccl_available(),
            "devices": [],
            "p2p_matrix": [],
        }
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            snapshot["devices"].append(
                {
                    "index": i,
                    "name": props.name,
                    "compute_capability": f"{props.major}.{props.minor}",
                    "total_memory": int(props.total_memory),
                    "multi_processor_count": int(props.multi_processor_count),
                    "pci_bus_id_decimal": getattr(props, "pci_bus_id", None),
                }
            )
        for i in range(torch.cuda.device_count()):
            row = []
            for j in range(torch.cuda.device_count()):
                row.append(bool(torch.cuda.can_device_access_peer(i, j)) if i != j else False)
            snapshot["p2p_matrix"].append(row)
        return snapshot
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"error": repr(exc)}


def _relevant_env() -> dict[str, str]:
    prefixes = ("NCCL", "CUDA", "CUBLAS", "TORCH", "LD_LIBRARY_PATH")
    return {key: os.environ[key] for key in sorted(os.environ) if key.startswith(prefixes)}


def collect_environment(workspace_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(workspace_root).resolve() if workspace_root else Path.cwd().resolve()
    nvcc_candidates = ["nvcc", "/usr/local/cuda-12.8/bin/nvcc", "/usr/local/cuda/bin/nvcc"]
    commands = {
        "nvidia_smi": _run_command(["nvidia-smi"], timeout=30),
        "nvidia_smi_topo_m": _run_command(["nvidia-smi", "topo", "-m"], timeout=30),
        "nvidia_smi_topo_p2p_w": _run_command(["nvidia-smi", "topo", "-p2p", "w"], timeout=30),
        "nvidia_smi_topo_p2p_r": _run_command(["nvidia-smi", "topo", "-p2p", "r"], timeout=30),
        "nvidia_smi_q": _run_command(["nvidia-smi", "-q"], timeout=60),
        "python_pip_show_torch": _run_command([sys.executable, "-m", "pip", "show", "torch"], timeout=30),
    }
    for candidate in nvcc_candidates:
        result = _run_command([candidate, "--version"], timeout=30)
        commands[f"nvcc_{candidate.replace('/', '_')}"] = result
        if result.get("returncode") == 0:
            break

    return {
        "schema_version": 1,
        "timestamp_unix": time.time(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "cwd": str(Path.cwd().resolve()),
        "workspace_root": str(root),
        "env": _relevant_env(),
        "torch": _torch_snapshot(),
        "git": {
            "workspace": _git_snapshot(root),
            "hpn": _git_snapshot(root / "hpn"),
            "nccl": _git_snapshot(root / "nccl"),
            "nccl_tests": _git_snapshot(root / "nccl-tests"),
        },
        "commands": commands,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture ISSUE4 environment snapshot")
    parser.add_argument("--workspace-root", default=str(Path(__file__).resolve().parents[4]))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    snapshot = collect_environment(args.workspace_root)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
