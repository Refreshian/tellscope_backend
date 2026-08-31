"""Unified job index across products. Redis hashes tellscope:job:{id}."""
from __future__ import annotations

import json
from datetime import datetime

import redis

from .timeutil import is_placeholder_ts

INDEX = "tellscope:jobs"
PREFIX = "tellscope:job:"


_REDIS = None


def _r() -> redis.Redis:
    global _REDIS
    if _REDIS is None:
        _REDIS = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
    return _REDIS


def _key(job_id: str) -> str:
    return f"{PREFIX}{job_id}"


def register(job_id: str, **fields) -> None:
    if not job_id:
        return
    now = datetime.now().isoformat()
    payload = {k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v)) for k, v in fields.items()}
    payload["job_id"] = job_id
    try:
        client = _r()
        key = _key(job_id)
        existing_created = client.hget(key, "created_at") or ""
        incoming_created = payload.get("created_at") or ""
        real_existing = "" if is_placeholder_ts(existing_created) else existing_created
        real_incoming = "" if is_placeholder_ts(incoming_created) else incoming_created
        empty_ok = bool(fields.get("_empty_ts"))
        payload.pop("_empty_ts", None)
        payload["created_at"] = real_existing or real_incoming or ("" if empty_ok else now)
        incoming_updated = payload.get("updated_at") or ""
        if is_placeholder_ts(incoming_updated):
            payload["updated_at"] = payload["created_at"] if empty_ok else now
        client.hset(key, mapping=payload)
        client.sadd(INDEX, job_id)
    except Exception:
        pass


def update(job_id: str, **fields) -> None:
    register(job_id, **fields)


def get(job_id: str) -> dict:
    try:
        return _r().hgetall(_key(job_id)) or {}
    except Exception:
        return {}


def list_jobs() -> list[dict]:
    backfill_mosinform()
    try:
        from .runtime import backfill_llm_tasks

        backfill_llm_tasks()
    except Exception:
        pass
    try:
        ids = list(_r().smembers(INDEX) or [])
    except Exception:
        return []
    jobs = []
    for job_id in ids:
        item = get(job_id)
        if item:
            jobs.append(item)
    jobs.sort(key=lambda item: item.get("created_at") or item.get("updated_at") or "", reverse=True)
    return jobs


def backfill_mosinform() -> None:
    try:
        client = _r()
        for key in client.keys("mosinform:*"):
            if key in {"mosinform:jobs", "mosinform:__archive__"}:
                continue
            if key.count(":") != 1:
                continue
            job_id = key.split(":", 1)[1]
            if client.exists(_key(job_id)):
                continue
            data = client.hgetall(key) or {}
            register(
                job_id,
                product="mosinform",
                route="/mosinform-rating",
                status=data.get("status") or "",
                message=data.get("message") or "",
                period=data.get("period") or "",
                files=data.get("files") or "",
                created_at=data.get("created_at") or "",
                updated_at=data.get("updated_at") or "",
            )
    except Exception:
        pass
