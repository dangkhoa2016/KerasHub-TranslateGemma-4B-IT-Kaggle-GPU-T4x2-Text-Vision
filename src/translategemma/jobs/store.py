"""In-memory store for job state with bounded size and result TTL."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Optional

from translategemma.core.errors import StoreFullError
from translategemma.jobs.models import Job


class JobStore:
    def __init__(self, max_size: int, result_ttl_seconds: float):
        self._store: "OrderedDict[str, Job]" = OrderedDict()
        self._lock = threading.RLock()
        self._max_size = max_size
        self._result_ttl_seconds = result_ttl_seconds

    def _cleanup_locked(self, now: Optional[float] = None) -> int:
        now = now or time.time()
        removed = 0
        for job_id, job in list(self._store.items()):
            if (
                job.status in {"completed", "failed"}
                and job.completed_at is not None
                and now - job.completed_at >= self._result_ttl_seconds
            ):
                del self._store[job_id]
                removed += 1
        return removed

    def cleanup(self) -> int:
        with self._lock:
            return self._cleanup_locked()

    def put(self, job: Job) -> None:
        with self._lock:
            self._cleanup_locked()
            while len(self._store) >= self._max_size:
                removable = next(
                    (
                        job_id
                        for job_id, old_job in self._store.items()
                        if old_job.status in {"completed", "failed"}
                    ),
                    None,
                )
                if removable is None:
                    raise StoreFullError("Job result store is full")
                del self._store[removable]
            self._store[job.id] = job

    def delete(self, job_id: str) -> None:
        with self._lock:
            self._store.pop(job_id, None)

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            self._cleanup_locked()
            return self._store.get(job_id)

    def mark_processing(self, job_id: str, worker_id: str, gpu_id: str) -> None:
        with self._lock:
            job = self._store.get(job_id)
            if job is None or job.status != "queued":
                return
            job.status = "processing"
            job.started_at = time.time()
            job.worker_id = worker_id
            job.gpu_id = gpu_id

    def mark_completed(
        self, job_id: str, result: str, inference_seconds: float
    ) -> None:
        with self._lock:
            job = self._store.get(job_id)
            if job is None:
                return
            job.status = "completed"
            job.result = result
            job.inference_seconds = inference_seconds
            job.image = None
            job.completed_at = time.time()
            job.done.set()

    def mark_failed(
        self,
        job_id: str,
        public_error: str,
        internal_error: Optional[str] = None,
    ) -> None:
        with self._lock:
            job = self._store.get(job_id)
            if job is None:
                return
            job.status = "failed"
            job.public_error = public_error
            job.internal_error = internal_error
            job.image = None
            job.completed_at = time.time()
            job.done.set()

    def fail_active_for_worker(self, worker_id: str, reason: str) -> int:
        count = 0
        with self._lock:
            for job in self._store.values():
                if job.status == "processing" and job.worker_id == worker_id:
                    job.status = "failed"
                    job.public_error = "GPU worker stopped during inference"
                    job.internal_error = reason
                    job.image = None
                    job.completed_at = time.time()
                    job.done.set()
                    count += 1
        return count

    def pending_count(self) -> int:
        with self._lock:
            return sum(
                job.status in {"queued", "processing"}
                for job in self._store.values()
            )

    def stats(self) -> Dict[str, int]:
        with self._lock:
            self._cleanup_locked()
            counts = {"queued": 0, "processing": 0, "completed": 0, "failed": 0}
            for job in self._store.values():
                counts[job.status] = counts.get(job.status, 0) + 1
            counts["total"] = len(self._store)
            return counts
