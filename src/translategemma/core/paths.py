"""Filesystem paths and shared logging setup for the TranslateGemma API."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "log"
STATE_DIR = BASE_DIR / "state"


def configure_logging(log_name: str, log_file: Path) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    target = logging.getLogger("translategemma")
    target.handlers.clear()
    target.setLevel(logging.INFO)
    target.propagate = False

    formatter = logging.Formatter(
        f"%(asctime)s | %(levelname)-7s | {log_name} | %(message)s"
    )
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    target.addHandler(stream)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    target.addHandler(file_handler)
