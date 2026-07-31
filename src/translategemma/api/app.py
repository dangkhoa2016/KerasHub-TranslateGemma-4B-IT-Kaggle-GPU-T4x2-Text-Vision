"""Flask HTTP application and runtime lifecycle dataclass."""

from __future__ import annotations

import hmac
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Dict, Optional

from translategemma.core.config import Config
from translategemma.core.errors import (
    QueueFullError,
    ServiceUnavailableError,
    StoreFullError,
    ValidationError,
    WorkerNotReadyError,
)
from translategemma.core.validation import (
    parse_image_translation_payload,
    parse_translation_payload,
)
from translategemma.workers.manager import TranslationManager

logger = logging.getLogger("translategemma")


def _friendly_unavailable(exc: Exception) -> tuple[Dict[str, Any], Dict[str, str]]:
    """Build a user-friendly 503 body and headers when the service is not ready."""
    health = getattr(exc, "health", None)
    if isinstance(exc, WorkerNotReadyError) and isinstance(health, dict):
        state = health.get("state")
        ready = int(health.get("ready_workers", 0) or 0)
        expected = int(health.get("expected_workers", 0) or 0)
        if state == "loading":
            message = (
                f"Model is still loading ({ready}/{expected} worker(s) ready); "
                "please wait and retry"
            )
        elif state == "unavailable":
            message = "No GPU worker is available right now; please retry in a moment"
        else:
            message = "No GPU worker is ready yet; please retry in a moment"
        retry_after = "30"
        return (
            {
                "error": message,
                "state": state,
                "ready_workers": ready,
                "expected_workers": expected,
                "retry_after_seconds": int(retry_after),
                "health_url": "/health/ready",
            },
            {"Retry-After": retry_after},
        )
    return {"error": str(exc)}, {}


@dataclass
class Runtime:
    config: Config
    manager: TranslationManager
    server: Optional[Any] = None
    restart_lock: threading.Lock = field(default_factory=threading.Lock)
    shutdown_started: threading.Event = field(default_factory=threading.Event)


def create_app(runtime: Runtime) -> Any:
    from flask import Flask, jsonify, request
    from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

    config = runtime.config
    manager = runtime.manager
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = config.max_request_bytes

    def _provided_api_key() -> str:
        provided = request.headers.get("X-API-Key", "").strip()
        auth = request.headers.get("Authorization", "")
        if auth.lower().startswith("bearer "):
            provided = auth[7:].strip()
        return provided

    def _valid_api_key() -> bool:
        if not config.api_auth_required:
            return True
        provided = _provided_api_key()
        return bool(provided and hmac.compare_digest(provided, config.api_key))

    def require_api_key(view: Any) -> Any:
        @wraps(view)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if not _valid_api_key():
                return jsonify({"error": "Unauthorized"}), 401
            return view(*args, **kwargs)

        return wrapped

    def read_json_object() -> Dict[str, Any]:
        if not request.is_json:
            raise ValidationError("Content-Type must be application/json")
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            raise ValidationError("Request body must be a JSON object")
        return data

    @app.errorhandler(RequestEntityTooLarge)
    def request_too_large(_exc: RequestEntityTooLarge) -> Any:
        return jsonify({"error": "Request body is too large"}), 413

    @app.errorhandler(404)
    def not_found(_exc: Any) -> Any:
        return jsonify({"error": "Endpoint not found"}), 404

    @app.errorhandler(405)
    def method_not_allowed(_exc: Any) -> Any:
        return jsonify({"error": "Method not allowed"}), 405

    @app.errorhandler(Exception)
    def unhandled_error(exc: Exception) -> Any:
        if isinstance(exc, HTTPException):
            return jsonify({"error": exc.description}), exc.code
        logger.exception("Unhandled HTTP error")
        return jsonify({"error": "Internal server error"}), 500

    @app.route("/", methods=["GET"])
    def index() -> Any:
        return jsonify(
            {
                "service": "TranslateGemma 4B IT",
                "architecture": "one isolated model process per GPU",
                "api_auth_required": config.api_auth_required,
                "endpoints": {
                    "liveness": "GET /health/live",
                    "readiness": "GET /health/ready",
                    "translate_sync": "POST /translate",
                    "translate_async": "POST /translate/async",
                    "translate_image_sync": "POST /translate/image",
                    "translate_image_async": "POST /translate/image/async",
                    "result": "GET /result/<job_id>",
                    "restart": "POST /restart",
                },
            }
        )

    @app.route("/health/live", methods=["GET"])
    def health_live() -> Any:
        return jsonify({"status": "alive", "pid": os.getpid()})

    @app.route("/health", methods=["GET"])
    @app.route("/health/ready", methods=["GET"])
    def health_ready() -> Any:
        health = manager.health()
        require_all = request.args.get("all", "").lower() in {"1", "true", "yes"}
        details = request.args.get("details", "").lower() in {"1", "true", "yes"}
        ready = health["ready"] and (not require_all or health["state"] == "ready")
        status_code = 200 if ready else 503
        if details:
            if not _valid_api_key():
                return jsonify({"error": "Unauthorized"}), 401
            payload = health
        else:
            payload = {
                "state": health["state"],
                "ready": health["ready"],
                "ready_workers": health["ready_workers"],
                "expected_workers": health["expected_workers"],
                "accepting_jobs": health["accepting_jobs"],
                "jobs": health["jobs"],
            }
        return jsonify(payload), status_code

    @app.route("/translate", methods=["POST"])
    @require_api_key
    def translate_sync() -> Any:
        try:
            payload = parse_translation_payload(read_json_object(), config)
            job = manager.submit(payload)
        except ValidationError as exc:
            return jsonify({"error": str(exc)}), 400
        except QueueFullError as exc:
            return jsonify({"error": str(exc)}), 429
        except (ServiceUnavailableError, StoreFullError) as exc:
            body, headers = _friendly_unavailable(exc)
            return jsonify(body), 503, headers

        completed = job.done.wait(timeout=config.request_timeout)
        if not completed:
            return (
                jsonify(
                    {
                        "job_id": job.id,
                        "status": "processing",
                        "message": "Synchronous wait timed out; query the result endpoint",
                        "result_url": f"/result/{job.id}",
                    }
                ),
                202,
            )
        if job.status == "failed":
            return jsonify(job.public_dict()), 500
        return jsonify(job.public_dict()), 200

    @app.route("/translate/async", methods=["POST"])
    @require_api_key
    def translate_async() -> Any:
        try:
            payload = parse_translation_payload(read_json_object(), config)
            job = manager.submit(payload)
        except ValidationError as exc:
            return jsonify({"error": str(exc)}), 400
        except QueueFullError as exc:
            return jsonify({"error": str(exc)}), 429
        except (ServiceUnavailableError, StoreFullError) as exc:
            body, headers = _friendly_unavailable(exc)
            return jsonify(body), 503, headers
        return (
            jsonify(
                {
                    "job_id": job.id,
                    "status": job.status,
                    "result_url": f"/result/{job.id}",
                }
            ),
            202,
        )

    @app.route("/translate/image", methods=["POST"])
    @require_api_key
    def translate_image_sync() -> Any:
        try:
            payload = parse_image_translation_payload(read_json_object(), config)
            job = manager.submit(payload)
        except ValidationError as exc:
            return jsonify({"error": str(exc)}), 400
        except QueueFullError as exc:
            return jsonify({"error": str(exc)}), 429
        except (ServiceUnavailableError, StoreFullError) as exc:
            body, headers = _friendly_unavailable(exc)
            return jsonify(body), 503, headers

        completed = job.done.wait(timeout=config.request_timeout)
        if not completed:
            return (
                jsonify(
                    {
                        "job_id": job.id,
                        "status": "processing",
                        "message": "Synchronous wait timed out; query the result endpoint",
                        "result_url": f"/result/{job.id}",
                    }
                ),
                202,
            )
        if job.status == "failed":
            return jsonify(job.public_dict()), 500
        return jsonify(job.public_dict()), 200

    @app.route("/translate/image/async", methods=["POST"])
    @require_api_key
    def translate_image_async() -> Any:
        try:
            payload = parse_image_translation_payload(read_json_object(), config)
            job = manager.submit(payload)
        except ValidationError as exc:
            return jsonify({"error": str(exc)}), 400
        except QueueFullError as exc:
            return jsonify({"error": str(exc)}), 429
        except (ServiceUnavailableError, StoreFullError) as exc:
            body, headers = _friendly_unavailable(exc)
            return jsonify(body), 503, headers
        return (
            jsonify(
                {
                    "job_id": job.id,
                    "status": job.status,
                    "result_url": f"/result/{job.id}",
                }
            ),
            202,
        )

    @app.route("/result/<job_id>", methods=["GET"])
    @require_api_key
    def get_result(job_id: str) -> Any:
        job = manager.store.get(job_id)
        if job is None:
            return jsonify({"error": "Job not found or result expired"}), 404
        if job.status in {"queued", "processing"}:
            return jsonify(job.public_dict(include_result=False)), 202
        if job.status == "failed":
            return jsonify(job.public_dict()), 500
        return jsonify(job.public_dict()), 200

    @app.route("/restart", methods=["POST"])
    def restart_server() -> Any:
        provided = request.headers.get("X-Restart-Secret", "").strip()
        if not provided or not hmac.compare_digest(provided, config.restart_secret):
            return jsonify({"error": "Unauthorized"}), 401

        try:
            data = request.get_json(silent=True) or {}
            if not isinstance(data, dict):
                raise ValidationError("Request body must be a JSON object")
            wait_for_jobs = data.get("wait_for_jobs", data.get("wait_empty", True))
            if not isinstance(wait_for_jobs, bool):
                raise ValidationError("Field 'wait_for_jobs' must be boolean")
            timeout = float(data.get("timeout", config.shutdown_timeout))
            timeout = max(1.0, min(timeout, 900.0))
        except (ValidationError, TypeError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400

        if not runtime.restart_lock.acquire(blocking=False):
            return jsonify({"error": "Restart already in progress"}), 409

        def do_restart() -> None:
            try:
                time.sleep(0.2)
                runtime.shutdown_started.set()
                manager.shutdown(wait_for_jobs=wait_for_jobs, timeout=timeout)
                if runtime.server is not None:
                    runtime.server.shutdown()
                    runtime.server.server_close()
                logger.info("Executing a fresh server process")
                os.execv(sys.executable, [sys.executable] + sys.argv)
            finally:
                runtime.restart_lock.release()

        threading.Thread(target=do_restart, name="http-restart", daemon=False).start()
        return jsonify({"status": "restarting", "pid": os.getpid()}), 202

    return app
