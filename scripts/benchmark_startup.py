#!/usr/bin/env python3
"""Benchmark worker startup strategies on Kaggle T4x2.

Uses an isolated compilation-cache directory so the normal serving cache is not
cleared. Scenarios: cold stagger, warm stagger, warm parallel. The normal server
is restored at the end.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "log"
STATE_DIR = ROOT / "state"
BENCH_CACHE = Path(
    os.environ.get(
        "STARTUP_BENCHMARK_CACHE_DIR",
        "/kaggle/working/.cache/translategemma-jax-startup-benchmark",
    )
)
OUTPUT = LOG_DIR / "startup-benchmark.json"
PORT = int(os.environ.get("PORT", "7860"))
BASE = f"http://127.0.0.1:{PORT}"
TIMEOUT = int(os.environ.get("STARTUP_BENCHMARK_TIMEOUT", "900"))


def run_script(name: str, env: dict[str, str] | None = None, check: bool = True) -> None:
    subprocess.run(
        ["bash", str(ROOT / "scripts" / name)],
        cwd=ROOT,
        env=env,
        check=check,
    )


def api_key() -> str:
    value = os.environ.get("API_KEY", "").strip()
    if value:
        return value
    path = ROOT / "data" / "api_key.txt"
    return path.read_text(encoding="utf-8").strip() if path.exists() else ""


def get_json(path: str, *, authenticated: bool = False) -> dict | None:
    request = urllib.request.Request(BASE + path)
    if authenticated:
        key = api_key()
        if key:
            request.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(request, timeout=3) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                pass
    return total


def wait_ready(started: float) -> tuple[float, dict]:
    deadline = time.monotonic() + TIMEOUT
    last: dict = {}
    while time.monotonic() < deadline:
        health = get_json("/health/ready?all=1&details=1", authenticated=True)
        if health:
            last = health
            if health.get("state") == "ready":
                return time.monotonic() - started, health
        time.sleep(0.5)
    raise RuntimeError(f"Server did not become fully ready; last health={last}")


def scenario(name: str, mode: str, *, clear_cache: bool) -> dict:
    run_script("stop_tunnel.sh", check=False)
    run_script("stop.sh", check=False)
    if clear_cache:
        shutil.rmtree(BENCH_CACHE, ignore_errors=True)
    BENCH_CACHE.mkdir(parents=True, exist_ok=True)
    try:
        BENCH_CACHE.chmod(0o700)
    except OSError:
        pass

    env = os.environ.copy()
    env["WORKER_START_MODE_OVERRIDE"] = mode
    env["JAX_COMPILATION_CACHE_DIR_OVERRIDE"] = str(BENCH_CACHE)
    env["MODEL_DTYPE_OVERRIDE"] = env.get("MODEL_DTYPE", "bfloat16")
    before = directory_size(BENCH_CACHE)
    started_wall = time.time()
    started = time.monotonic()
    run_script("start.sh", env=env)
    ready_seconds, health = wait_ready(started)
    after = directory_size(BENCH_CACHE)
    workers = health.get("workers") or []
    worker_ready_offsets = {}
    for worker in workers:
        ready_at = worker.get("ready_at")
        if isinstance(ready_at, (int, float)):
            worker_ready_offsets[str(worker.get("worker_id"))] = ready_at - started_wall
    result = {
        "name": name,
        "requested_mode": mode,
        "ready_seconds": ready_seconds,
        "cache_bytes_before": before,
        "cache_bytes_after": after,
        "worker_ready_offsets_seconds": worker_ready_offsets,
        "worker_startup": health.get("worker_startup") or {},
        "detected_gpus": health.get("detected_gpus") or [],
    }
    run_script("stop.sh", check=False)
    return result


def restore_normal_server() -> dict:
    env = os.environ.copy()
    env.pop("WORKER_START_MODE_OVERRIDE", None)
    env.pop("JAX_COMPILATION_CACHE_DIR_OVERRIDE", None)
    env.pop("MODEL_DTYPE_OVERRIDE", None)
    started = time.monotonic()
    run_script("start.sh", env=env)
    ready_seconds, health = wait_ready(started)
    try:
        run_script("run_tunnel.sh", env=env, check=False)
    except Exception:
        pass
    return {"ready_seconds": ready_seconds, "worker_startup": health.get("worker_startup") or {}}


def main() -> int:
    if os.environ.get("RUN_STARTUP_BENCHMARK", "0") != "1":
        print("Refusing long startup benchmark unless RUN_STARTUP_BENCHMARK=1")
        return 2
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark_cache_dir": str(BENCH_CACHE),
        "scenarios": [],
        "restored_default_server": None,
    }
    try:
        payload["scenarios"].append(scenario("cold_stagger", "stagger", clear_cache=True))
        payload["scenarios"].append(scenario("warm_stagger", "stagger", clear_cache=False))
        payload["scenarios"].append(scenario("warm_parallel", "parallel", clear_cache=False))
    finally:
        try:
            payload["restored_default_server"] = restore_normal_server()
        finally:
            OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Saved: {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
