"""Serving lock: pinned models, images, prompt ids. No secrets."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

_PKG = Path(__file__).resolve().parent
_LOCK_CANDIDATES = [
    Path(os.environ["MLOPS_LOCK"]) if os.environ.get("MLOPS_LOCK") else None,
    _PKG / "lock.yaml",
    Path("/home/dev/tellscope_app/tellscope_backend/mlops/lock.yaml"),
]
_LOCK_CANDIDATES = [p for p in _LOCK_CANDIDATES if p is not None]


@lru_cache(maxsize=1)
def load_lock() -> dict:
    for path in _LOCK_CANDIDATES:
        if path and path.exists():
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            data["_lock_path"] = str(path)
            return data
    return {"version": 0, "generate": {}, "embed": {}, "external": {}, "prompts": {}}


def generate_cfg() -> dict:
    lock = load_lock().get("generate") or {}
    return {
        **lock,
        "model": os.environ.get("VLLM_MODEL") or lock.get("model") or "Qwen/Qwen3-32B-FP8",
        "base_url": (
            os.environ.get("VLLM_BASE_URL") or lock.get("base_url") or "http://127.0.0.1:8000"
        ).rstrip("/"),
        "revision": os.environ.get("VLLM_REVISION") or lock.get("revision") or "",
    }


def embed_cfg() -> dict:
    lock = load_lock().get("embed") or {}
    return {
        **lock,
        "model": os.environ.get("EMBED_MODEL") or lock.get("model") or "deepvk/USER2-base",
    }


def external_cfg(profile: str = "dashboard_qa") -> dict:
    lock = load_lock().get("external") or {}
    profiles = lock.get("profiles") or {}
    env_profile = os.environ.get(f"AITUNNEL_MODEL_{profile.upper()}")
    env_model = os.environ.get("AITUNNEL_MODEL")
    model = (
        env_profile
        or profiles.get(profile)
        or env_model
        or profiles.get("dashboard_qa")
        or "gpt-4.1-mini"
    )
    return {
        **lock,
        "model": model,
        "profile": profile,
        "base_url": (
            os.environ.get("AITUNNEL_BASE_URL") or lock.get("base_url") or "https://api.aitunnel.ru/v1"
        ).rstrip("/"),
    }


def prompt_id(name: str, default: str) -> str:
    prompts = load_lock().get("prompts") or {}
    return str(prompts.get(name) or default)


def public_lock() -> dict:
    """Safe to return from an API — no env secrets."""
    lock = dict(load_lock())
    lock.pop("_lock_path", None)
    gen = generate_cfg()
    emb = embed_cfg()
    ext = load_lock().get("external") or {}
    return {
        "version": lock.get("version"),
        "generate": {
            "provider": gen.get("provider"),
            "model": gen.get("model"),
            "revision": gen.get("revision") or None,
            "image": gen.get("image"),
            "image_digest": gen.get("image_digest"),
            "pipeline_parallel_size": gen.get("pipeline_parallel_size"),
            "max_model_len": gen.get("max_model_len"),
            "gpu": gen.get("gpu"),
        },
        "embed": {
            "provider": emb.get("provider"),
            "model": emb.get("model"),
            "gpu": emb.get("gpu"),
        },
        "external": {
            "provider": ext.get("provider"),
            "profiles": ext.get("profiles") or {},
        },
        "prompts": lock.get("prompts") or {},
    }
