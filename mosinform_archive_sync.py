"""Сводка задач Mosinform в Redis и job.json — без перезапуска FastAPI."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import redis

DATA_ROOT = Path("/home/dev/tellscope_app/tellscope_backend/data/mosinform")
REDIS = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
INDEX = "mosinform:jobs"
ARCHIVE_ID = "__archive__"
SUMMARY_KEYS = ("messages", "objects", "untagged", "top", "missing", "notes")


def _job_key(job_id: str) -> str:
    return f"mosinform:{job_id}"


def _parse_summary(raw) -> dict:
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def _public_job(job_id: str) -> dict | None:
    if job_id == ARCHIVE_ID:
        return None
    data = REDIS.hgetall(_job_key(job_id)) or {}
    disk = DATA_ROOT / job_id / "job.json"
    if disk.exists():
        try:
            saved = json.loads(disk.read_text(encoding="utf-8"))
            if isinstance(saved, dict):
                for key, value in saved.items():
                    if not data.get(key):
                        data[key] = value
        except Exception:
            pass
    pptx = DATA_ROOT / job_id / "out" / "mosinform_rating.pptx"
    xlsx = DATA_ROOT / job_id / "out" / "mosinform_rating.xlsx"
    has_pptx = bool((data.get("pptx") and Path(data["pptx"]).exists()) or pptx.exists())
    has_xlsx = bool((data.get("xlsx") and Path(data["xlsx"]).exists()) or xlsx.exists())
    if not data and not (DATA_ROOT / job_id).is_dir():
        return None
    summary = _parse_summary(data.get("summary"))
    return {
        "job_id": job_id,
        "status": data.get("status") or ("done" if has_pptx else "unknown"),
        "message": data.get("message") or "",
        "progress": data.get("progress") or "",
        "period": data.get("period") or "",
        "files": data.get("files") or "",
        "created_at": data.get("created_at") or "",
        "updated_at": data.get("updated_at") or "",
        "summary": {key: summary[key] for key in SUMMARY_KEYS if key in summary},
        "has_pptx": has_pptx,
        "has_xlsx": has_xlsx,
    }


def _known_ids() -> set[str]:
    ids: set[str] = set()
    try:
        ids.update(REDIS.smembers(INDEX) or [])
        for key in REDIS.keys("mosinform:*"):
            if key in {INDEX, _job_key(ARCHIVE_ID)}:
                continue
            if key.startswith("mosinform:") and key.count(":") == 1:
                ids.add(key.split(":", 1)[1])
    except Exception:
        pass
    if DATA_ROOT.exists():
        for path in DATA_ROOT.iterdir():
            if path.is_dir() and path.name != ARCHIVE_ID:
                ids.add(path.name)
    ids.discard(ARCHIVE_ID)
    return ids


def sync_once() -> int:
    jobs = []
    for job_id in _known_ids():
        item = _public_job(job_id)
        if not item:
            continue
        jobs.append(item)
        REDIS.sadd(INDEX, job_id)
        raw = REDIS.hgetall(_job_key(job_id)) or {}
        raw["job_id"] = job_id
        path = DATA_ROOT / job_id / "job.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(raw or item, ensure_ascii=False), encoding="utf-8")
    jobs.sort(key=lambda item: item.get("created_at") or item.get("updated_at") or "", reverse=True)
    REDIS.hset(
        _job_key(ARCHIVE_ID),
        mapping={
            "status": "done",
            "message": "archive",
            "progress": "100",
            "summary": json.dumps({"notes": jobs}, ensure_ascii=False),
        },
    )
    return len(jobs)


def main() -> None:
    once = "--once" in sys.argv
    while True:
        count = sync_once()
        print(f"mosinform archive: {count} jobs", flush=True)
        if once:
            return
        time.sleep(5)


if __name__ == "__main__":
    main()
