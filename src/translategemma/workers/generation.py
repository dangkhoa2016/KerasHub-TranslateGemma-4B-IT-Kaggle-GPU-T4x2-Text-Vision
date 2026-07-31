"""Pure helpers for stable generation-length planning.

KerasHub ``CausalLM.generate()`` compiles around a total ``max_length``.  If the
server passes a different total for every prompt, JAX/XLA can compile many near-
identical executables.  These helpers quantize total sequence lengths into a
small set of buckets while preserving the caller's requested *new-token* budget
at the API boundary (the engine trims any bucket slack after generation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple


@dataclass(frozen=True)
class GenerationPlan:
    prompt_tokens: int
    vision_tokens: int
    requested_new_tokens: int
    safety_tokens: int
    exact_max_length: int
    compiled_max_length: int
    bucketed: bool


def normalize_buckets(values: Iterable[int]) -> Tuple[int, ...]:
    """Return positive, sorted, de-duplicated length buckets."""
    return tuple(sorted({int(value) for value in values if int(value) > 0}))


def choose_compiled_max_length(
    required_length: int,
    buckets: Iterable[int],
    fallback_step: int,
) -> int:
    """Choose the smallest configured bucket that fits ``required_length``.

    Above the largest configured bucket, round up to ``fallback_step``.  A
    non-positive step disables fallback rounding and returns the exact length.
    """
    required = max(1, int(required_length))
    normalized = normalize_buckets(buckets)
    for bucket in normalized:
        if bucket >= required:
            return bucket
    step = int(fallback_step)
    if step <= 0:
        return required
    return ((required + step - 1) // step) * step


def plan_generation(
    *,
    prompt_tokens: int,
    requested_new_tokens: int,
    vision_tokens: int = 0,
    safety_tokens: int = 8,
    bucketing_enabled: bool = True,
    buckets: Iterable[int] = (),
    fallback_step: int = 0,
) -> GenerationPlan:
    prompt = max(0, int(prompt_tokens))
    vision = max(0, int(vision_tokens))
    requested = max(1, int(requested_new_tokens))
    safety = max(0, int(safety_tokens))
    exact = prompt + vision + requested + safety
    compiled = (
        choose_compiled_max_length(exact, buckets, fallback_step)
        if bucketing_enabled
        else exact
    )
    return GenerationPlan(
        prompt_tokens=prompt,
        vision_tokens=vision,
        requested_new_tokens=requested,
        safety_tokens=safety,
        exact_max_length=exact,
        compiled_max_length=compiled,
        bucketed=compiled != exact,
    )
