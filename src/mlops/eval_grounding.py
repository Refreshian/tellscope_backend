"""CPU-only: answers must not invent numbers missing from the context."""
from __future__ import annotations

import re

_NUM = re.compile(r"(?<![A-Za-zА-Яа-я_])\d+(?:[.,]\d+)?%?")


def extract_numbers(text: str) -> set[str]:
    found: set[str] = set()
    for raw in _NUM.findall(text or ""):
        token = raw.replace(",", ".")
        found.add(token)
        if token.endswith("%"):
            found.add(token[:-1])
        elif "." in token:
            found.add(token.split(".", 1)[0])
    return found


def invented_numbers(answer: str, context: str) -> list[str]:
    """Numbers present in the answer but not in the supplied context."""
    extra = extract_numbers(answer) - extract_numbers(context)
    return sorted(extra)
