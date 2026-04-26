"""Shared recovery-cycle constants and helpers for DeepAlign experiments."""

from __future__ import annotations

from typing import Iterable

DEFAULT_DEBUG_LIED_IDS = (
    "D911-07",
    "D911-18",
    "D911-20",
    "D911-22",
    "D911-24",
    "D911-17",
    "D911-06",
    "D911-02",
)


def normalize_debug_lied_ids(values: Iterable[str] | None) -> tuple[str, ...]:
    """Normalize lied ids while preserving order."""
    if values is None:
        return DEFAULT_DEBUG_LIED_IDS
    normalized: list[str] = []
    for value in values:
        text = str(value).strip().upper()
        if not text:
            continue
        normalized.append(text)
    return tuple(normalized)
