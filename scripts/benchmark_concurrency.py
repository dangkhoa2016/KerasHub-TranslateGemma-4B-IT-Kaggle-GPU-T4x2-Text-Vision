#!/usr/bin/env python3
"""Concurrent T4x2 API benchmark with separate prime/hot resource sampling.

The benchmark records both phases:
1. PRIME: first request pair for the benchmark workload.  This exposes any
   residual JAX/XLA compilation cost and related CPU spikes.
2. HOT: an identical second pair.  This measures steady-state T4x2 throughput.

A strict pass requires each pair to occupy two distinct worker/GPU IDs.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "log"


def _simple_env(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if raw and not raw.startswith("#") and "=" in raw:
            key, value = raw.split("=", 1)
            out[key] = value.strip().strip('"').strip("'")
    return out


def _api_key() -> str:
    value = os.environ.get("API_KEY", "").strip()
    if value:
        return value
    path = ROOT / "data/api_key.txt"
    return path.read_text(encoding="utf-8").strip() if path.is_file() else ""


def _request(url: str, key: str, payload: dict, timeout: float) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
    result["client_wall_seconds"] = round(time.monotonic() - started, 3)
    result["http_status"] = response.status
    return result


def _read_cpu() -> tuple[int, int]:
    fields = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()[1:]
    values = [int(value) for value in fields]
    total = sum(values)
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    return total, idle


def _memory() -> tuple[float, float]:
    values: dict[str, float] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, value = line.split(":", 1)
        values[key] = float(value.strip().split()[0]) / 1024.0
    total = values.get("MemTotal", 0.0)
    available = values.get("MemAvailable", 0.0)
    return total - available, total


def _gpus() -> list[dict]:
    try:
        raw = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    out = []
    for line in raw.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 5:
            out.append(
                {
                    "index": parts[0],
                    "gpu_util": float(parts[1]),
                    "mem_util": float(parts[2]),
                    "mem_used_mib": float(parts[3]),
                    "mem_total_mib": float(parts[4]),
                }
            )
    return out


def _monitor(
    stop: threading.Event,
    rows: list[dict],
    interval: float,
    phase: str,
) -> None:
    prev_total, prev_idle = _read_cpu()
    # Take the first sample after one interval so CPU percentage has a real
    # denominator rather than an arbitrary process-start baseline.
    while not stop.wait(interval):
        total, idle = _read_cpu()
        delta_total, delta_idle = total - prev_total, idle - prev_idle
        cpu = 0.0 if delta_total <= 0 else 100.0 * (delta_total - delta_idle) / delta_total
        prev_total, prev_idle = total, idle
        ram_used, ram_total = _memory()
        now = time.time()
        gpu_rows = _gpus() or [
            {
                "index": "",
                "gpu_util": 0.0,
                "mem_util": 0.0,
                "mem_used_mib": 0.0,
                "mem_total_mib": 0.0,
            }
        ]
        for gpu in gpu_rows:
            rows.append(
                {
                    "phase": phase,
                    "timestamp": now,
                    "cpu_percent": round(cpu, 2),
                    "ram_used_mib": round(ram_used, 1),
                    "ram_total_mib": round(ram_total, 1),
                    **gpu,
                }
            )


def _run_pair(
    *,
    phase: str,
    base_url: str,
    key: str,
    payloads: list[dict],
    timeout: float,
    sample_interval: float,
) -> tuple[list[dict], float, list[dict]]:
    samples: list[dict] = []
    stop = threading.Event()
    monitor = threading.Thread(
        target=_monitor,
        args=(stop, samples, sample_interval, phase),
        daemon=True,
    )
    monitor.start()
    wall_started = time.monotonic()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(_request, base_url + "/translate", key, payload, timeout)
                for payload in payloads
            ]
            results = [future.result() for future in futures]
    finally:
        wall_seconds = time.monotonic() - wall_started
        stop.set()
        monitor.join(timeout=max(2.0, sample_interval * 2))
    return results, wall_seconds, samples


def _phase_summary(results: list[dict], wall_seconds: float, samples: list[dict]) -> dict:
    workers = sorted({str(result.get("worker_id")) for result in results if result.get("worker_id")})
    gpus = sorted(
        {
            str(result.get("gpu_id"))
            for result in results
            if result.get("gpu_id") is not None
        }
    )
    cpu_values = [float(row["cpu_percent"]) for row in samples]
    max_gpu: dict[str, float] = {}
    avg_gpu_accum: dict[str, list[float]] = {}
    for row in samples:
        idx = str(row.get("index", ""))
        value = float(row.get("gpu_util", 0.0))
        max_gpu[idx] = max(max_gpu.get(idx, 0.0), value)
        avg_gpu_accum.setdefault(idx, []).append(value)
    avg_gpu = {
        idx: round(sum(values) / len(values), 2)
        for idx, values in avg_gpu_accum.items()
        if values
    }
    ram_values = [float(row["ram_used_mib"]) for row in samples]
    return {
        "wall_seconds": round(wall_seconds, 3),
        "workers_used": workers,
        "gpus_used": gpus,
        "request_results": results,
        "samples": len(samples),
        "max_cpu_percent": round(max(cpu_values), 2) if cpu_values else None,
        "avg_cpu_percent": round(sum(cpu_values) / len(cpu_values), 2) if cpu_values else None,
        "max_ram_used_mib": round(max(ram_values), 1) if ram_values else None,
        "max_gpu_util_percent": max_gpu,
        "avg_gpu_util_percent": avg_gpu,
    }


def _directory_size(path: Path) -> int:
    if not path.is_dir():
        return 0
    total = 0
    for child in path.rglob("*"):
        try:
            if child.is_file():
                total += child.stat().st_size
        except OSError:
            pass
    return total


def _strict_pair_ok(summary: dict) -> bool:
    return len(summary["workers_used"]) == 2 and len(summary["gpus_used"]) == 2


def main() -> int:
    cfg = _simple_env(ROOT / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=f"http://127.0.0.1:{cfg.get('PORT', '7860')}")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=int(os.environ.get("BENCHMARK_MAX_NEW_TOKENS", "128")),
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=float(os.environ.get("BENCHMARK_SAMPLE_INTERVAL", "0.5")),
    )
    parser.add_argument("--timeout", type=float, default=float(cfg.get("REQUEST_TIMEOUT", "600")))
    parser.add_argument("--label", default=os.environ.get("BENCHMARK_LABEL", "t4x2-concurrency"))
    parser.add_argument("--no-strict-two-gpu", action="store_true")
    parser.add_argument("--skip-prime", action="store_true", help="measure only the hot pair")
    args = parser.parse_args()

    key = _api_key() if cfg.get("API_AUTH_REQUIRED", "true").lower() != "false" else ""
    if cfg.get("API_AUTH_REQUIRED", "true").lower() != "false" and not key:
        raise SystemExit("API key is unavailable")

    base_payload = {
        "source_lang": "English",
        "target_lang": "Vietnamese",
        "text": "The little fox found a warm lantern beside the old oak tree.",
        "max_new_tokens": args.max_new_tokens,
    }
    payloads = [dict(base_payload), dict(base_payload)]

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    safe_label = "".join(char if char.isalnum() or char in "-_" else "-" for char in args.label)
    csv_path = LOG_DIR / f"benchmark-{safe_label}-{stamp}.csv"
    json_path = LOG_DIR / f"benchmark-{safe_label}-{stamp}.json"

    all_samples: list[dict] = []
    prime_summary = None
    if not args.skip_prime:
        print("PRIME: measuring the first exact benchmark pair, including compile/cache effects...")
        prime_results, prime_wall, prime_samples = _run_pair(
            phase="prime",
            base_url=args.base_url,
            key=key,
            payloads=payloads,
            timeout=args.timeout,
            sample_interval=args.sample_interval,
        )
        all_samples.extend(prime_samples)
        prime_summary = _phase_summary(prime_results, prime_wall, prime_samples)
        print(json.dumps({"prime": prime_summary}, ensure_ascii=False, indent=2))
        if any(result.get("status") != "completed" for result in prime_results):
            print("FAIL: at least one PRIME request did not complete")
            return 2
        if not args.no_strict_two_gpu and not _strict_pair_ok(prime_summary):
            print("FAIL: PRIME pair did not occupy two distinct GPU workers")
            return 4

    print("HOT: measuring an identical second pair...")
    hot_results, hot_wall, hot_samples = _run_pair(
        phase="hot",
        base_url=args.base_url,
        key=key,
        payloads=payloads,
        timeout=args.timeout,
        sample_interval=args.sample_interval,
    )
    all_samples.extend(hot_samples)
    hot_summary = _phase_summary(hot_results, hot_wall, hot_samples)

    fields = [
        "phase",
        "timestamp",
        "cpu_percent",
        "ram_used_mib",
        "ram_total_mib",
        "index",
        "gpu_util",
        "mem_util",
        "mem_used_mib",
        "mem_total_mib",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_samples)

    cache_raw = os.environ.get("JAX_COMPILATION_CACHE_DIR") or cfg.get(
        "JAX_COMPILATION_CACHE_DIR", ""
    )
    cache_path = Path(cache_raw).expanduser() if cache_raw else None
    cache_bytes = _directory_size(cache_path) if cache_path else 0

    summary = {
        "label": args.label,
        "max_new_tokens": args.max_new_tokens,
        "generation_bucketing": cfg.get("GENERATION_BUCKETING", "true"),
        "generation_length_buckets": cfg.get("GENERATION_LENGTH_BUCKETS", ""),
        "jax_compilation_cache_dir": str(cache_path) if cache_path else None,
        "jax_compilation_cache_bytes": cache_bytes,
        "prime": prime_summary,
        "hot": hot_summary,
        # Backward-compatible aliases now point to HOT steady-state metrics.
        "wall_seconds": hot_summary["wall_seconds"],
        "workers_used": hot_summary["workers_used"],
        "gpus_used": hot_summary["gpus_used"],
        "request_results": hot_summary["request_results"],
        "max_cpu_percent": hot_summary["max_cpu_percent"],
        "avg_cpu_percent": hot_summary["avg_cpu_percent"],
        "max_gpu_util_percent": hot_summary["max_gpu_util_percent"],
        "sample_csv": str(csv_path.relative_to(ROOT)),
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Resource samples: {csv_path}")
    print(f"Summary: {json_path}")

    if any(result.get("status") != "completed" for result in hot_results):
        print("FAIL: at least one HOT request did not complete")
        return 2
    if not args.no_strict_two_gpu and not _strict_pair_ok(hot_summary):
        print(
            "FAIL: expected two distinct workers/GPUs in HOT phase, got "
            f"workers={hot_summary['workers_used']}, gpus={hot_summary['gpus_used']}"
        )
        return 3
    print("PASS: PRIME/HOT benchmark completed on two distinct GPU workers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
