"""Run lineage written next to artifacts (run.json)."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

from .lock import embed_cfg, generate_cfg, load_lock
from .prompts import load_prompt


def git_sha() -> str:
    if os.environ.get("GIT_SHA"):
        return os.environ["GIT_SHA"]
    roots = [
        Path("/home/dev/tellscope_app/tellscope_backend"),
        Path.cwd(),
    ]
    for root in roots:
        try:
            out = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=root,
                timeout=2,
                stderr=subprocess.DEVNULL,
            )
            return out.decode().strip()
        except Exception:
            continue
    return ""


def file_hash(path: Path | None, length: int = 16) -> str:
    if not path or not Path(path).exists():
        return ""
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return digest[:length]


def prompt_hash(prompt_id: str) -> str:
    try:
        text = load_prompt(prompt_id)
    except FileNotFoundError:
        return ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def cache_fingerprint(model: str, prompt_id: str, catalog_hash: str) -> str:
    raw = f"{model}|{prompt_id}|{prompt_hash(prompt_id)}|{catalog_hash}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def build_run(
    *,
    product: str,
    route: str,
    job_id: str = "",
    prompt_id: str = "",
    catalog_version: str = "",
    catalog_hash: str = "",
    cache_key: str = "",
    extra: dict | None = None,
) -> dict:
    gen = generate_cfg()
    emb = embed_cfg()
    lock = load_lock()
    run = {
        "product": product,
        "route": route,
        "job_id": job_id,
        "git_sha": git_sha(),
        "started_at": datetime.now().isoformat(),
        "provider": "vllm",
        "model_id": gen.get("model"),
        "model_rev": gen.get("revision") or None,
        "image_digest": gen.get("image_digest"),
        "max_model_len": gen.get("max_model_len"),
        "pipeline_parallel_size": gen.get("pipeline_parallel_size"),
        "embed_model": emb.get("model"),
        "prompt_id": prompt_id,
        "prompt_hash": prompt_hash(prompt_id) if prompt_id else "",
        "catalog_version": catalog_version,
        "catalog_hash": catalog_hash,
        "cache_key": cache_key,
        "lock_version": lock.get("version"),
    }
    if extra:
        run.update(extra)
    return run


def write_run(path: Path, run: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**run, "finished_at": datetime.now().isoformat()}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
