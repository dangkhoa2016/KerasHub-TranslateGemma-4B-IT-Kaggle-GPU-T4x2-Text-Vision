"""Secret loading and generation for API keys and restart secrets."""

from __future__ import annotations

import os
import secrets
from pathlib import Path


def _load_or_create_secret(env_name: str, file_path: Path, required: bool) -> str:
    env_value = os.environ.get(env_name, "").strip()
    if env_value:
        return env_value

    try:
        existing = file_path.read_text(encoding="utf-8").strip()
        if existing:
            return existing
    except FileNotFoundError:
        pass

    if not required:
        return ""

    value = secrets.token_urlsafe(32)
    temp_path = file_path.with_suffix(file_path.suffix + ".tmp")
    temp_path.write_text(value + "\n", encoding="utf-8")
    os.chmod(temp_path, 0o600)
    temp_path.replace(file_path)
    os.chmod(file_path, 0o600)
    return value
