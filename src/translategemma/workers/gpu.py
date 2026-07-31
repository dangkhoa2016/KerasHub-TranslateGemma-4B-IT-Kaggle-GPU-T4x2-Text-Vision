"""GPU detection via nvidia-smi."""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Any, Dict

_CANDIDATE_PATHS = [
    "/opt/bin/nvidia-smi",
    "/usr/bin/nvidia-smi",
    "/usr/local/bin/nvidia-smi",
]


def _nvidia_smi() -> list[str] | None:
    for candidate in _CANDIDATE_PATHS:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return [candidate]
    found = shutil.which("nvidia-smi")
    return [found] if found else None


def detect_gpus(*, sort_by_free_memory: bool = True) -> list[Dict[str, Any]]:
    smi = _nvidia_smi()
    if smi is None:
        return []
    try:
        output = subprocess.check_output(
            smi
            + [
                "--query-gpu=index,name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            timeout=10,
            text=True,
        )
    except Exception:
        return []

    gpus: list[Dict[str, Any]] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            gpus.append(
                {
                    "id": parts[0],
                    "name": parts[1],
                    "memory_total_mb": int(parts[2]),
                    "memory_free_mb": int(parts[3]),
                }
            )
        except ValueError:
            continue
    if sort_by_free_memory:
        return sorted(gpus, key=lambda gpu: gpu["memory_free_mb"], reverse=True)
    return sorted(gpus, key=lambda gpu: int(gpu["id"]))


def gpu_metrics_by_id() -> Dict[str, Dict[str, Any]]:
    """Return current GPU metrics keyed by physical GPU id."""
    return {str(gpu["id"]): gpu for gpu in detect_gpus(sort_by_free_memory=False)}
