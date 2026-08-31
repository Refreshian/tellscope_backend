"""Load versioned prompt templates. Placeholders are {name} replaced literally."""
from __future__ import annotations

import hashlib
from pathlib import Path

from .lock import load_lock

_PROMPTS = Path(__file__).resolve().parent / "prompts"
_FALLBACK = Path("/home/dev/tellscope_app/tellscope_backend/mlops/prompts")


def load_prompt(prompt_id: str) -> str:
    name = f"{prompt_id}.txt"
    for folder in (_PROMPTS, _FALLBACK):
        path = folder / name
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
    raise FileNotFoundError(f"prompt not found: {prompt_id}")


def render_prompt(prompt_id: str, **fields) -> str:
    template = load_prompt(prompt_id)
    for key, value in fields.items():
        template = template.replace("{" + key + "}", str(value))
    return template


def list_prompts() -> list[dict]:
    used: dict[str, list[str]] = {}
    for product, prompt_id in (load_lock().get("prompts") or {}).items():
        used.setdefault(str(prompt_id), []).append(str(product))
    seen: set[str] = set()
    items: list[dict] = []
    for folder in (_PROMPTS, _FALLBACK):
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.txt")):
            if path.name in seen:
                continue
            seen.add(path.name)
            text = path.read_text(encoding="utf-8")
            preview = next((line.strip() for line in text.splitlines() if line.strip()), "")
            items.append(
                {
                    "id": path.stem,
                    "bytes": len(text.encode("utf-8")),
                    "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
                    "used_by": used.get(path.stem, []),
                    "preview": preview[:160],
                }
            )
    return items
