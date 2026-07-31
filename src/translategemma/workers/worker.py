"""Entry point for one isolated model worker process."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import queue
import time
from typing import Any, Dict, Optional

from translategemma.core.paths import LOG_DIR, configure_logging
from translategemma.workers.engine import TranslateGemmaEngine

logger = logging.getLogger("translategemma")


def model_worker_main(
    worker_id: str,
    generation: int,
    gpu_id: Optional[str],
    task_queue: Any,
    result_queue: Any,
    shutdown_event: Any,
    worker_config: Dict[str, Any],
) -> None:
    visible_gpu = "cpu" if gpu_id is None else str(gpu_id)
    worker_log = LOG_DIR / f"worker-{worker_id.replace(':', '-')}.log"
    configure_logging(f"worker={worker_id}", worker_log)

    if gpu_id is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
    else:
        # This must happen before importing Keras, KerasHub, or JAX.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        os.environ.pop("JAX_PLATFORMS", None)

    os.environ["KERAS_BACKEND"] = "jax"
    if worker_config["jax_preallocate"]:
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(
            worker_config["jax_mem_fraction"]
        )
    else:
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        os.environ.pop("XLA_PYTHON_CLIENT_MEM_FRACTION", None)

    # JAX persistent compilation cache is deliberately outside the repository working tree.
    # It is trusted executable material and should not be committed or distributed.
    cache_dir = worker_config.get("jax_compilation_cache_dir")
    if cache_dir:
        cache_path = Path(str(cache_dir)).expanduser()
        cache_path.mkdir(parents=True, exist_ok=True)
        try:
            cache_path.chmod(0o700)
        except OSError:
            pass
        os.environ["JAX_COMPILATION_CACHE_DIR"] = str(cache_path)
        os.environ["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = str(
            worker_config["jax_persistent_cache_min_compile_time_secs"]
        )
        os.environ["JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"] = str(
            worker_config["jax_persistent_cache_min_entry_size_bytes"]
        )

    def emit(message_type: str, **payload: Any) -> None:
        result_queue.put(
            {
                "type": message_type,
                "worker_id": worker_id,
                "generation": generation,
                "gpu_id": visible_gpu,
                "pid": os.getpid(),
                **payload,
            }
        )

    emit("worker_state", state="loading")
    try:
        engine = TranslateGemmaEngine(
            preset_path=worker_config["model_path"],
            dtype=worker_config["model_dtype"],
            vision_enabled=worker_config.get("vision_enabled", False),
            generation_bucketing=worker_config.get("generation_bucketing", True),
            generation_length_buckets=worker_config.get(
                "generation_length_buckets", (256, 512, 1024, 1536, 2048)
            ),
            generation_bucket_step=worker_config.get("generation_bucket_step", 512),
            warmup_output_tokens=worker_config.get("warmup_output_tokens", 128),
            warmup_text_buckets=worker_config.get("warmup_text_buckets", (256,)),
            warmup_vision_buckets=worker_config.get("warmup_vision_buckets", (512,)),
            compilation_cache_dir=str(cache_dir) if cache_dir else None,
        )
        metadata = engine.load(worker_config["warmup_enabled"])
    except Exception as exc:
        logger.exception("Model load failed")
        emit("worker_load_error", error=repr(exc))
        return

    emit("worker_ready", metadata=metadata)

    while not shutdown_event.is_set():
        try:
            task = task_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        if task is None:
            break

        job_id = task["job_id"]
        emit("job_started", job_id=job_id)
        started = time.time()
        try:
            if task.get("image") is not None:
                result = engine.translate_image(
                    image=task["image"],
                    src=task["src"],
                    tgt=task["tgt"],
                    max_tokens=task["max_tokens"],
                )
            else:
                result = engine.translate(
                    text=task["text"],
                    src=task["src"],
                    tgt=task["tgt"],
                    max_tokens=task["max_tokens"],
                )
            elapsed = time.time() - started
            emit(
                "job_completed",
                job_id=job_id,
                result=result,
                inference_seconds=elapsed,
            )
            logger.info("Job %s completed in %.2fs", job_id, elapsed)
        except Exception as exc:
            logger.exception("Job %s failed", job_id)
            emit("job_failed", job_id=job_id, error=repr(exc))

    emit("worker_stopped", state="stopped")
