#!/usr/bin/env python3
"""TranslateGemma 4B IT REST API for Kaggle GPU T4x2.

Architecture:
- The main process runs Flask only and never imports JAX/Keras.
- One isolated model process is created per GPU (up to MAX_GPU_WORKERS).
- Each model process sets CUDA_VISIBLE_DEVICES before importing JAX/Keras.
- A bounded shared queue distributes jobs to available GPU workers.
- API authentication, strict validation, result TTL, readiness checks, and
  graceful restart/shutdown are built in.

The implementation lives in the ``translategemma`` package:

- translategemma.core     -- paths, logging, configuration, errors, validation
- translategemma.jobs     -- Job model and JobStore
- translategemma.workers  -- GPU detection, inference engine, worker process,
                             multi-GPU TranslationManager
- translategemma.api      -- Flask application and Runtime lifecycle

This module is the command-line entry point and re-exports the public API.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import signal
import sys
import threading
from typing import Any

from translategemma.api.app import Runtime, create_app
from translategemma.core.config import Config
from translategemma.core.errors import (
    QueueFullError,
    ServiceUnavailableError,
    StoreFullError,
    ValidationError,
    WorkerNotReadyError,
)
from translategemma.core.paths import DATA_DIR, LOG_DIR, configure_logging
from translategemma.core.validation import (
    parse_image_translation_payload,
    parse_translation_payload,
)
from translategemma.jobs.models import Job
from translategemma.jobs.store import JobStore
from translategemma.workers.manager import TranslationManager

__all__ = [
    "Config",
    "Job",
    "JobStore",
    "QueueFullError",
    "Runtime",
    "ServiceUnavailableError",
    "StoreFullError",
    "ValidationError",
    "WorkerNotReadyError",
    "create_app",
    "parse_image_translation_payload",
    "parse_translation_payload",
]

logger = logging.getLogger("translategemma")


def main() -> int:
    mp.freeze_support()
    configure_logging("api", LOG_DIR / "server.log")

    try:
        config = Config.from_env()
    except Exception:
        logger.exception("Invalid configuration")
        return 2

    logger.info("Starting TranslateGemma API on %s:%d", config.host, config.port)
    logger.info("Model path: %s", config.model_path)
    logger.info(
        "Security files: API key=%s, restart secret=%s",
        DATA_DIR / "api_key.txt",
        DATA_DIR / "restart_secret.txt",
    )

    manager = TranslationManager(config)
    runtime = Runtime(config=config, manager=manager)
    app = create_app(runtime)
    from werkzeug.serving import make_server

    runtime.server = make_server(config.host, config.port, app, threaded=True)
    manager.start_async()

    def request_shutdown(signum: int, _frame: Any) -> None:
        if runtime.shutdown_started.is_set():
            return
        runtime.shutdown_started.set()
        logger.info("Received signal %d", signum)

        def stop() -> None:
            manager.shutdown(
                wait_for_jobs=True,
                timeout=config.shutdown_timeout,
            )
            if runtime.server is not None:
                runtime.server.shutdown()

        threading.Thread(target=stop, name="signal-shutdown", daemon=False).start()

    signal.signal(signal.SIGTERM, request_shutdown)
    signal.signal(signal.SIGINT, request_shutdown)

    try:
        runtime.server.serve_forever()
    finally:
        if not runtime.shutdown_started.is_set():
            manager.shutdown(wait_for_jobs=False, timeout=10.0)
        runtime.server.server_close()
        logger.info("Server stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
