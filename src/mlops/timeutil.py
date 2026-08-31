"""Timestamp helpers without Redis."""
from __future__ import annotations

PLACEHOLDER_TS = "2020-01-01"


def is_placeholder_ts(stamp: str | None) -> bool:
    if not stamp:
        return True
    return str(stamp).startswith(PLACEHOLDER_TS)
