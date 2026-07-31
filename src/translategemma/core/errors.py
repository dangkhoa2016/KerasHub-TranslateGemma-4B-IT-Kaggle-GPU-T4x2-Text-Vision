"""Shared error types used across the TranslateGemma API."""

from __future__ import annotations

from typing import Any, Dict, Optional


class ValidationError(ValueError):
    pass


class ServiceUnavailableError(RuntimeError):
    pass


class WorkerNotReadyError(ServiceUnavailableError):
    def __init__(self, message: str, health: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.health = health


class QueueFullError(RuntimeError):
    pass


class StoreFullError(RuntimeError):
    pass
