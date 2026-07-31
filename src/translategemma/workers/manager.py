"""Multi-GPU translation coordinator managing isolated model worker processes."""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import queue
import signal
import threading
import time
import uuid
from typing import Any, Dict, Optional

from translategemma.core.config import Config
from translategemma.core.errors import (
    QueueFullError,
    ServiceUnavailableError,
    WorkerNotReadyError,
)
from translategemma.core.paths import STATE_DIR
from translategemma.jobs.models import Job
from translategemma.jobs.store import JobStore
from translategemma.workers.gpu import detect_gpus, gpu_metrics_by_id
from translategemma.workers.worker import model_worker_main

logger = logging.getLogger("translategemma")


class TranslationManager:
    def __init__(self, config: Config):
        self.config = config
        self.store = JobStore(config.max_store_size, config.result_ttl_seconds)
        self.ctx = mp.get_context("spawn")
        self.task_queue = self.ctx.Queue(maxsize=config.max_queue_size)
        self.result_queue = self.ctx.Queue()
        self.shutdown_event = self.ctx.Event()

        self._workers: Dict[str, tuple[Any, int, Optional[str]]] = {}
        self._worker_status: Dict[str, Dict[str, Any]] = {}
        self._worker_lock = threading.RLock()
        self._worker_condition = threading.Condition(self._worker_lock)
        self._collector_stop = threading.Event()
        self._monitor_stop = threading.Event()
        self._startup_thread: Optional[threading.Thread] = None
        self._collector_thread: Optional[threading.Thread] = None
        self._monitor_thread: Optional[threading.Thread] = None
        self._accepting = True
        self._shutting_down = threading.Event()
        self._selected_gpus: list[Dict[str, Any]] = []
        self._startup_strategy: Dict[str, Any] = {}
        self._target_worker_count = 0
        self._worker_registry_file = STATE_DIR / "workers.json"

    def start_async(self) -> None:
        if self._startup_thread is not None:
            return
        self.cleanup_stale_workers()
        self._collector_thread = threading.Thread(
            target=self._collect_results,
            name="result-collector",
            daemon=True,
        )
        self._collector_thread.start()
        self._monitor_thread = threading.Thread(
            target=self._monitor_workers,
            name="worker-monitor",
            daemon=True,
        )
        self._monitor_thread.start()
        self._startup_thread = threading.Thread(
            target=self._startup_workers,
            name="worker-startup",
            daemon=True,
        )
        self._startup_thread.start()

    def _load_worker_registry(self) -> Dict[str, Any]:
        try:
            data = json.loads(self._worker_registry_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _proc_start_ticks(pid: int) -> Optional[int]:
        try:
            with open(f"/proc/{pid}/stat", encoding="utf-8") as handle:
                fields = handle.read().split()
            return int(fields[21])
        except (OSError, ValueError, IndexError):
            return None

    def _write_worker_registry(self) -> None:
        self._worker_registry_file.parent.mkdir(parents=True, exist_ok=True)
        workers = [
            {
                "worker_id": worker_id,
                "pid": process.pid,
                "start_ticks": self._proc_start_ticks(int(process.pid)),
            }
            for worker_id, (process, _generation, _gpu_id) in self._workers.items()
            if process.pid is not None
        ]
        payload = {"coordinator_pid": os.getpid(), "workers": workers}
        try:
            self._worker_registry_file.write_text(
                json.dumps(payload), encoding="utf-8"
            )
        except OSError as exc:
            logger.warning("Could not write worker registry: %s", exc)

    def _remove_worker_registry(self) -> None:
        try:
            self._worker_registry_file.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Could not remove worker registry: %s", exc)

    @staticmethod
    def _kill_process(pid: Optional[int], start_ticks: Optional[int]) -> bool:
        if not pid or int(pid) <= 0:
            return False
        pid = int(pid)
        if start_ticks is not None:
            current_ticks = TranslationManager._proc_start_ticks(pid)
            if current_ticks is not None and current_ticks != int(start_ticks):
                return False
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            return False
        for _ in range(20):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return True
            time.sleep(0.1)
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        return True

    def cleanup_stale_workers(self) -> int:
        data = self._load_worker_registry()
        if not data or data.get("coordinator_pid") == os.getpid():
            self._remove_worker_registry()
            return 0
        workers = data.get("workers")
        if not isinstance(workers, list):
            self._remove_worker_registry()
            return 0
        killed = 0
        for entry in workers:
            if isinstance(entry, dict):
                if self._kill_process(entry.get("pid"), entry.get("start_ticks")):
                    killed += 1
        if killed:
            logger.info(
                "Cleaned up %d stale worker process(es) from a previous run",
                killed,
            )
        self._remove_worker_registry()
        return killed

    def _resolve_worker_devices(self) -> list[Optional[str]]:
        detected = detect_gpus()
        self._selected_gpus = detected
        if self.config.gpu_ids:
            requested = [
                item.strip()
                for item in self.config.gpu_ids.split(",")
                if item.strip()
            ]
            if not requested:
                raise RuntimeError("GPU_IDS is set but contains no GPU IDs")
            if len(set(requested)) != len(requested):
                raise RuntimeError("GPU_IDS must not contain duplicate GPU IDs")
            detected_ids = {str(gpu["id"]) for gpu in detected}
            if not detected_ids:
                raise RuntimeError("GPU_IDS is set but nvidia-smi detected no GPUs")
            missing = [gpu_id for gpu_id in requested if gpu_id not in detected_ids]
            if missing:
                raise RuntimeError(
                    "GPU_IDS requested unavailable GPU(s): " + ",".join(missing)
                )
            return requested[: self.config.max_gpu_workers]
        if detected:
            return [gpu["id"] for gpu in detected[: self.config.max_gpu_workers]]
        if self.config.allow_cpu_fallback:
            return [None]
        return []

    @staticmethod
    def _directory_size(path: str | None) -> int:
        if not path:
            return 0
        root = os.path.expanduser(path)
        total = 0
        try:
            for dirpath, _dirnames, filenames in os.walk(root):
                for name in filenames:
                    try:
                        total += os.path.getsize(os.path.join(dirpath, name))
                    except OSError:
                        continue
        except OSError:
            return 0
        return total

    @staticmethod
    def _available_ram_mb() -> int:
        try:
            with open("/proc/meminfo", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) // 1024
        except (OSError, ValueError, IndexError):
            pass
        return 0

    def _resolve_startup_mode(self, worker_count: int) -> str:
        requested = self.config.worker_start_mode
        cache_bytes = self._directory_size(self.config.jax_compilation_cache_dir)
        available_ram_mb = self._available_ram_mb()
        cache_warm = cache_bytes >= self.config.worker_parallel_cache_min_bytes
        ram_ok = (
            self.config.worker_parallel_min_available_ram_mb <= 0
            or available_ram_mb >= self.config.worker_parallel_min_available_ram_mb
        )

        if worker_count <= 1:
            resolved = "parallel"
            reason = "single-worker"
        elif requested == "parallel":
            resolved = "parallel"
            reason = "forced-parallel"
        elif requested == "stagger":
            resolved = "stagger"
            reason = "forced-stagger"
        elif cache_warm and ram_ok:
            resolved = "parallel"
            reason = "warm-cache-and-sufficient-ram"
        else:
            resolved = "stagger"
            missing = []
            if not cache_warm:
                missing.append("cold-cache")
            if not ram_ok:
                missing.append("low-available-ram")
            reason = "+".join(missing) or "auto-safety"

        self._startup_strategy = {
            "requested_mode": requested,
            "resolved_mode": resolved,
            "reason": reason,
            "jax_cache_bytes": cache_bytes,
            "cache_warm": cache_warm,
            "cache_min_bytes": self.config.worker_parallel_cache_min_bytes,
            "available_ram_mb": available_ram_mb,
            "parallel_min_available_ram_mb": (
                self.config.worker_parallel_min_available_ram_mb
            ),
        }
        logger.info(
            "Worker startup mode: requested=%s resolved=%s reason=%s "
            "cache=%dB available_ram=%dMB",
            requested, resolved, reason, cache_bytes, available_ram_mb,
        )
        return resolved

    def _startup_workers(self) -> None:
        devices = self._resolve_worker_devices()
        self._target_worker_count = len(devices)
        if not devices:
            logger.error("No usable GPU found and CPU fallback is disabled")
            return

        mode = self._resolve_startup_mode(len(devices))
        logger.info(
            "Starting %d model worker(s) on devices %s using %s startup",
            len(devices), devices, mode,
        )
        for index, gpu_id in enumerate(devices):
            worker_id = f"gpu:{gpu_id}" if gpu_id is not None else "cpu:0"
            self._start_worker(worker_id, gpu_id)
            if mode == "stagger" and index < len(devices) - 1:
                self._wait_for_worker_load(worker_id, self.config.worker_load_timeout)

    def _start_worker(self, worker_id: str, gpu_id: Optional[str]) -> None:
        with self._worker_lock:
            previous = self._worker_status.get(worker_id, {})
            generation = int(previous.get("generation", 0)) + 1
            restarts = max(0, generation - 1)
            status = {
                "worker_id": worker_id,
                "gpu_id": "cpu" if gpu_id is None else str(gpu_id),
                "state": "starting",
                "pid": None,
                "generation": generation,
                "restarts": restarts,
                "active_job": None,
                "last_error": None,
                "metadata": None,
                "started_at": time.time(),
                "ready_at": None,
            }
            self._worker_status[worker_id] = status

            process = self.ctx.Process(
                target=model_worker_main,
                name=f"translategemma-{worker_id}",
                args=(
                    worker_id,
                    generation,
                    gpu_id,
                    self.task_queue,
                    self.result_queue,
                    self.shutdown_event,
                    self.config.worker_payload(),
                ),
                daemon=False,
            )
            process.start()
            self._workers[worker_id] = (process, generation, gpu_id)
            status["pid"] = process.pid
            self._write_worker_registry()
            logger.info(
                "Started worker %s generation=%d pid=%s",
                worker_id,
                generation,
                process.pid,
            )
            self._worker_condition.notify_all()

    def _wait_for_worker_load(self, worker_id: str, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        terminal_states = {"ready", "load_error", "crashed", "stopped"}
        with self._worker_condition:
            while time.monotonic() < deadline:
                status = self._worker_status.get(worker_id, {})
                if status.get("state") in terminal_states:
                    return status.get("state") == "ready"
                self._worker_condition.wait(timeout=min(1.0, deadline - time.monotonic()))
        logger.error("Worker %s did not finish loading before timeout", worker_id)
        return False

    def _collect_results(self) -> None:
        while not self._collector_stop.is_set():
            try:
                message = self.result_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            except (EOFError, OSError):
                break
            self._handle_worker_message(message)

    def _handle_worker_message(self, message: Dict[str, Any]) -> None:
        worker_id = str(message.get("worker_id"))
        generation = int(message.get("generation", 0))
        message_type = message.get("type")

        with self._worker_condition:
            status = self._worker_status.get(worker_id)
            if status is None or status.get("generation") != generation:
                return
            status["pid"] = message.get("pid", status.get("pid"))
            status["gpu_id"] = message.get("gpu_id", status.get("gpu_id"))

            if message_type == "worker_state":
                status["state"] = message.get("state", "loading")
            elif message_type == "worker_ready":
                status["state"] = "ready"
                status["metadata"] = message.get("metadata")
                status["ready_at"] = time.time()
                status["last_error"] = None
            elif message_type == "worker_load_error":
                status["state"] = "load_error"
                status["last_error"] = message.get("error")
            elif message_type == "job_started":
                status["state"] = "busy"
                status["active_job"] = message.get("job_id")
                self.store.mark_processing(
                    str(message.get("job_id")),
                    worker_id,
                    str(status.get("gpu_id")),
                )
            elif message_type == "job_completed":
                job_id = str(message.get("job_id"))
                self.store.mark_completed(
                    job_id,
                    str(message.get("result", "")),
                    float(message.get("inference_seconds", 0.0)),
                )
                status["state"] = "ready"
                status["active_job"] = None
            elif message_type == "job_failed":
                job_id = str(message.get("job_id"))
                internal_error = str(message.get("error", "Unknown inference error"))
                self.store.mark_failed(
                    job_id,
                    "Translation failed on the GPU worker",
                    internal_error,
                )
                status["state"] = "ready"
                status["active_job"] = None
                status["last_error"] = internal_error
            elif message_type == "worker_stopped":
                status["state"] = "stopped"
                status["active_job"] = None

            self._worker_condition.notify_all()

    def _monitor_workers(self) -> None:
        while not self._monitor_stop.wait(2.0):
            with self._worker_lock:
                items = list(self._workers.items())
            for worker_id, (process, generation, gpu_id) in items:
                if process.is_alive() or process.exitcode is None:
                    continue

                should_restart = False
                with self._worker_condition:
                    current = self._workers.get(worker_id)
                    status = self._worker_status.get(worker_id)
                    if current is None or current[1] != generation or status is None:
                        continue
                    if status.get("generation") != generation:
                        continue
                    previous_state = status.get("state")
                    if previous_state != "stopped":
                        status["state"] = (
                            "load_error" if previous_state == "load_error" else "crashed"
                        )
                        status["last_error"] = status.get("last_error") or (
                            f"Worker exited with code {process.exitcode}"
                        )
                    status["active_job"] = None
                    failed = self.store.fail_active_for_worker(
                        worker_id,
                        f"Worker exited with code {process.exitcode}",
                    )
                    if failed:
                        logger.error(
                            "Marked %d active job(s) failed after %s exited",
                            failed,
                            worker_id,
                        )
                    should_restart = (
                        not self._shutting_down.is_set()
                        and generation - 1 < self.config.max_worker_restarts
                    )
                    self._worker_condition.notify_all()

                if should_restart:
                    logger.warning("Restarting failed worker %s", worker_id)
                    time.sleep(1.0)
                    self._start_worker(worker_id, gpu_id)

    def has_ready_worker(self) -> bool:
        with self._worker_lock:
            return any(
                status.get("state") in {"ready", "busy"}
                for status in self._worker_status.values()
            )

    def ready_worker_count(self) -> int:
        with self._worker_lock:
            return sum(
                status.get("state") in {"ready", "busy"}
                for status in self._worker_status.values()
            )

    def expected_worker_count(self) -> int:
        with self._worker_lock:
            return self._target_worker_count

    def submit(self, payload: Dict[str, Any]) -> Job:
        if not self._accepting or self._shutting_down.is_set():
            raise ServiceUnavailableError("Server is shutting down")
        if not self.has_ready_worker():
            raise WorkerNotReadyError(
                "No GPU worker is ready yet",
                health=self.health(),
            )

        job = Job(
            id=f"job-{uuid.uuid4().hex[:16]}",
            text=payload["text"],
            src=payload["src"],
            tgt=payload["tgt"],
            max_tokens=payload["max_tokens"],
            image=payload.get("image"),
        )
        self.store.put(job)
        task = {
            "job_id": job.id,
            "text": job.text,
            "src": job.src,
            "tgt": job.tgt,
            "max_tokens": job.max_tokens,
            "image": job.image,
        }
        try:
            self.task_queue.put_nowait(task)
        except queue.Full as exc:
            self.store.delete(job.id)
            raise QueueFullError("Translation queue is full") from exc
        # The multiprocessing queue now owns the payload. Do not retain the
        # decoded image a second time in the result store for the whole TTL.
        job.image = None
        logger.info("Queued %s", job.id)
        return job

    def wait_idle(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.store.pending_count() == 0:
                return True
            time.sleep(0.25)
        return self.store.pending_count() == 0

    def health(self) -> Dict[str, Any]:
        with self._worker_lock:
            workers = [dict(status) for status in self._worker_status.values()]
        ready = sum(w["state"] in {"ready", "busy"} for w in workers)
        expected = self._target_worker_count
        if ready == 0:
            state = "loading" if (
                expected > len(workers)
                or any(w["state"] in {"starting", "loading"} for w in workers)
            ) else "unavailable"
        elif ready < expected:
            state = "degraded"
        else:
            state = "ready"
        live_gpus = gpu_metrics_by_id()
        detected_gpus = []
        for startup_gpu in self._selected_gpus:
            gpu_id = str(startup_gpu.get("id"))
            live = live_gpus.get(gpu_id)
            if live is None:
                detected_gpus.append(dict(startup_gpu))
                continue
            current = dict(live)
            current["startup_memory_free_mb"] = startup_gpu.get("memory_free_mb")
            detected_gpus.append(current)

        return {
            "state": state,
            "ready": ready > 0,
            "ready_workers": ready,
            "expected_workers": expected,
            "accepting_jobs": self._accepting and not self._shutting_down.is_set(),
            "jobs": self.store.stats(),
            "workers": workers,
            "detected_gpus": detected_gpus,
            "worker_startup": dict(self._startup_strategy),
        }

    def shutdown(self, wait_for_jobs: bool, timeout: float) -> bool:
        if self._shutting_down.is_set():
            return self.store.pending_count() == 0
        self._shutting_down.set()
        self._accepting = False
        logger.info(
            "Stopping translation manager; wait_for_jobs=%s timeout=%.1fs",
            wait_for_jobs,
            timeout,
        )

        idle = True
        if wait_for_jobs:
            idle = self.wait_idle(timeout)
            if not idle:
                logger.warning("Graceful wait timed out with pending jobs")

        self.shutdown_event.set()
        with self._worker_lock:
            live_workers = [
                process
                for process, _generation, _gpu_id in self._workers.values()
                if process.is_alive()
            ]

        for _ in live_workers:
            try:
                self.task_queue.put_nowait(None)
            except queue.Full:
                break

        join_deadline = time.monotonic() + min(timeout, 30.0)
        for process in live_workers:
            remaining = max(0.0, join_deadline - time.monotonic())
            process.join(timeout=remaining)
        for process in live_workers:
            if process.is_alive():
                logger.warning("Terminating worker pid=%s", process.pid)
                process.terminate()
                process.join(timeout=5.0)

        self._monitor_stop.set()
        self._collector_stop.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        if self._collector_thread:
            self._collector_thread.join(timeout=2.0)
        self._remove_worker_registry()
        return idle
