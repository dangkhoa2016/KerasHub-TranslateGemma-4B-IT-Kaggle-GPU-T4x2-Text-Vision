"""Runtime configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from translategemma.core.paths import DATA_DIR, LOG_DIR, STATE_DIR
from translategemma.core.secrets import _load_or_create_secret


_MODEL_REQUIRED_FILES = (
    "config.json",
    "preprocessor.json",
    "model.weights.h5",
    "assets/tokenizer/vocabulary.spm",
)


def _model_path_is_complete(path: str) -> bool:
    base = Path(path)
    return base.is_dir() and all((base / item).is_file() for item in _MODEL_REQUIRED_FILES)


def _discover_model_path(configured: str) -> str:
    """Return configured path when valid, otherwise discover a Kaggle model version."""
    configured = configured.strip()
    if configured and _model_path_is_complete(configured):
        return configured
    if not _env_bool("MODEL_AUTO_DISCOVER", True):
        return configured or (
            "/kaggle/input/models/keras/translategemma/keras/translategemma_4b_it/1"
        )

    base = Path(
        "/kaggle/input/models/keras/translategemma/keras/translategemma_4b_it"
    )
    candidates = []
    if base.is_dir():
        for child in base.iterdir():
            if child.is_dir() and _model_path_is_complete(str(child)):
                try:
                    key = (1, int(child.name))
                except ValueError:
                    key = (0, child.name)
                candidates.append((key, child))
    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        return str(candidates[0][1])
    return configured or str(base / "1")


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: Optional[int] = None) -> int:
    value = int(os.environ.get(name, str(default)))
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _env_float(name: str, default: float, minimum: Optional[float] = None) -> float:
    value = float(os.environ.get(name, str(default)))
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _env_int_tuple(
    name: str,
    default: str,
    *,
    minimum: int = 1,
    allow_empty: bool = False,
) -> Tuple[int, ...]:
    raw = os.environ.get(name, default).strip()
    if not raw:
        if allow_empty:
            return ()
        raise ValueError(f"{name} must not be empty")
    values = []
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        value = int(item)
        if value < minimum:
            raise ValueError(f"{name} entries must be >= {minimum}")
        values.append(value)
    normalized = tuple(sorted(set(values)))
    if not normalized and not allow_empty:
        raise ValueError(f"{name} must contain at least one integer")
    return normalized


@dataclass(frozen=True)
class Config:
    model_path: str
    host: str
    port: int
    max_gpu_workers: int
    gpu_ids: Optional[str]
    allow_cpu_fallback: bool
    stagger_worker_start: bool
    worker_start_mode: str
    worker_parallel_cache_min_bytes: int
    worker_parallel_min_available_ram_mb: int
    worker_load_timeout: float
    max_worker_restarts: int
    max_queue_size: int
    max_store_size: int
    result_ttl_seconds: float
    max_input_chars: int
    vision_enabled: bool
    max_image_bytes: int
    max_image_width: int
    max_image_height: int
    max_image_pixels: int
    default_output_tokens: int
    max_output_tokens: int
    request_timeout: float
    shutdown_timeout: float
    max_request_bytes: int
    model_dtype: str
    jax_preallocate: bool
    jax_mem_fraction: float
    jax_compilation_cache_dir: Optional[str]
    jax_persistent_cache_min_compile_time_secs: float
    jax_persistent_cache_min_entry_size_bytes: int
    generation_bucketing: bool
    generation_length_buckets: Tuple[int, ...]
    generation_bucket_step: int
    warmup_enabled: bool
    warmup_output_tokens: int
    warmup_text_buckets: Tuple[int, ...]
    warmup_vision_buckets: Tuple[int, ...]
    api_auth_required: bool
    api_key: str
    restart_secret: str

    @classmethod
    def from_env(cls) -> "Config":
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        STATE_DIR.mkdir(parents=True, exist_ok=True)

        api_auth_required = _env_bool("API_AUTH_REQUIRED", True)
        api_key = _load_or_create_secret(
            env_name="API_KEY",
            file_path=DATA_DIR / "api_key.txt",
            required=api_auth_required,
        )
        restart_secret = _load_or_create_secret(
            env_name="RESTART_SECRET",
            file_path=DATA_DIR / "restart_secret.txt",
            required=True,
        )
        default_output_tokens = _env_int("DEFAULT_OUTPUT_TOKENS", 128, 1)

        cache_dir = (
            os.environ.get("JAX_COMPILATION_CACHE_DIR_OVERRIDE")
            or os.environ.get(
                "JAX_COMPILATION_CACHE_DIR",
                "/kaggle/working/.cache/translategemma-jax",
            )
        ).strip()

        config = cls(
            model_path=_discover_model_path(
                os.environ.get(
                    "MODEL_PATH",
                    "/kaggle/input/models/keras/translategemma/keras/"
                    "translategemma_4b_it/1",
                )
            ),
            host=os.environ.get("HOST", "0.0.0.0"),
            port=_env_int("PORT", 7860, 1),
            max_gpu_workers=_env_int("MAX_GPU_WORKERS", 2, 1),
            gpu_ids=os.environ.get("GPU_IDS") or None,
            allow_cpu_fallback=_env_bool("ALLOW_CPU_FALLBACK", False),
            stagger_worker_start=_env_bool("STAGGER_WORKER_START", True),
            worker_start_mode=(
                os.environ.get("WORKER_START_MODE_OVERRIDE")
                or os.environ.get("WORKER_START_MODE", "auto")
            ).strip().lower(),
            worker_parallel_cache_min_bytes=_env_int(
                "WORKER_PARALLEL_CACHE_MIN_BYTES", 1_000_000, 0
            ),
            worker_parallel_min_available_ram_mb=_env_int(
                "WORKER_PARALLEL_MIN_AVAILABLE_RAM_MB", 24_576, 0
            ),
            worker_load_timeout=_env_float("WORKER_LOAD_TIMEOUT", 900.0, 1.0),
            max_worker_restarts=_env_int("MAX_WORKER_RESTARTS", 1, 0),
            max_queue_size=_env_int("MAX_QUEUE_SIZE", 32, 1),
            max_store_size=_env_int("MAX_STORE_SIZE", 1000, 1),
            result_ttl_seconds=_env_float("RESULT_TTL_SECONDS", 3600.0, 1.0),
            max_input_chars=_env_int("MAX_INPUT_CHARS", 20_000, 1),
            vision_enabled=_env_bool("VISION_ENABLED", False),
            max_image_bytes=_env_int("MAX_IMAGE_BYTES", 5_242_880, 1024),
            max_image_width=_env_int("MAX_IMAGE_WIDTH", 8_192, 8),
            max_image_height=_env_int("MAX_IMAGE_HEIGHT", 8_192, 8),
            max_image_pixels=_env_int("MAX_IMAGE_PIXELS", 20_000_000, 64),
            default_output_tokens=default_output_tokens,
            max_output_tokens=_env_int("MAX_OUTPUT_TOKENS", 1024, 1),
            request_timeout=_env_float("REQUEST_TIMEOUT", 600.0, 1.0),
            shutdown_timeout=_env_float("SHUTDOWN_TIMEOUT", 300.0, 1.0),
            max_request_bytes=_env_int("MAX_REQUEST_BYTES", 8_388_608, 1024),
            model_dtype=(
                os.environ.get("MODEL_DTYPE_OVERRIDE")
                or os.environ.get("MODEL_DTYPE", "bfloat16")
            ),
            jax_preallocate=_env_bool("JAX_PREALLOCATE", True),
            jax_mem_fraction=_env_float("JAX_MEM_FRACTION", 0.90, 0.01),
            jax_compilation_cache_dir=cache_dir or None,
            jax_persistent_cache_min_compile_time_secs=_env_float(
                "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", 1.0, 0.0
            ),
            jax_persistent_cache_min_entry_size_bytes=_env_int(
                "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", -1, -1
            ),
            generation_bucketing=_env_bool("GENERATION_BUCKETING", True),
            generation_length_buckets=_env_int_tuple(
                "GENERATION_LENGTH_BUCKETS", "256,512,1024,1536,2048"
            ),
            generation_bucket_step=_env_int("GENERATION_BUCKET_STEP", 512, 0),
            warmup_enabled=_env_bool("WARMUP_ENABLED", True),
            warmup_output_tokens=_env_int(
                "WARMUP_OUTPUT_TOKENS", default_output_tokens, 1
            ),
            warmup_text_buckets=_env_int_tuple(
                "WARMUP_TEXT_BUCKETS", "256", allow_empty=True
            ),
            warmup_vision_buckets=_env_int_tuple(
                "WARMUP_VISION_BUCKETS", "512", allow_empty=True
            ),
            api_auth_required=api_auth_required,
            api_key=api_key,
            restart_secret=restart_secret,
        )
        if config.worker_start_mode not in {"auto", "stagger", "parallel"}:
            raise ValueError(
                "WORKER_START_MODE must be one of: auto, stagger, parallel"
            )
        if config.default_output_tokens > config.max_output_tokens:
            raise ValueError("DEFAULT_OUTPUT_TOKENS must be <= MAX_OUTPUT_TOKENS")
        if config.warmup_output_tokens > config.max_output_tokens:
            raise ValueError("WARMUP_OUTPUT_TOKENS must be <= MAX_OUTPUT_TOKENS")
        if config.jax_mem_fraction > 1.0:
            raise ValueError("JAX_MEM_FRACTION must be <= 1.0")
        if config.vision_enabled and config.jax_mem_fraction < 0.97:
            raise ValueError(
                "VISION_ENABLED requires JAX_MEM_FRACTION >= 0.97 "
                "(the multimodal model needs ~14.6GB of the 15.36GB T4 VRAM)"
            )
        if config.generation_bucketing:
            if not config.generation_length_buckets and config.generation_bucket_step <= 0:
                raise ValueError(
                    "GENERATION_BUCKETING requires GENERATION_LENGTH_BUCKETS "
                    "or a positive GENERATION_BUCKET_STEP"
                )
            configured = set(config.generation_length_buckets)
            for name, warm_buckets in (
                ("WARMUP_TEXT_BUCKETS", config.warmup_text_buckets),
                ("WARMUP_VISION_BUCKETS", config.warmup_vision_buckets),
            ):
                missing = [bucket for bucket in warm_buckets if bucket not in configured]
                if missing:
                    raise ValueError(
                        f"{name} must use configured GENERATION_LENGTH_BUCKETS; "
                        f"missing={missing}"
                    )
        return config

    def worker_payload(self) -> Dict[str, Any]:
        return {
            "model_path": self.model_path,
            "model_dtype": self.model_dtype,
            "jax_preallocate": self.jax_preallocate,
            "jax_mem_fraction": self.jax_mem_fraction,
            "jax_compilation_cache_dir": self.jax_compilation_cache_dir,
            "jax_persistent_cache_min_compile_time_secs": (
                self.jax_persistent_cache_min_compile_time_secs
            ),
            "jax_persistent_cache_min_entry_size_bytes": (
                self.jax_persistent_cache_min_entry_size_bytes
            ),
            "generation_bucketing": self.generation_bucketing,
            "generation_length_buckets": self.generation_length_buckets,
            "generation_bucket_step": self.generation_bucket_step,
            "warmup_enabled": self.warmup_enabled,
            "warmup_output_tokens": self.warmup_output_tokens,
            "warmup_text_buckets": self.warmup_text_buckets,
            "warmup_vision_buckets": self.warmup_vision_buckets,
            "vision_enabled": self.vision_enabled,
        }
