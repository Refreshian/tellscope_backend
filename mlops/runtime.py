"""Runtime policy: GPU drain, unified job heartbeats, artifact lineage."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .jobs import list_jobs, register
from .lineage import build_run, write_run
from .lock import generate_cfg
from .timeutil import is_placeholder_ts

BATCH_PRODUCTS = ("mosinform", "llm-run", "smart-agent")
ACTIVE = {"running", "queued", "pending", "in_progress", "llm_processing", "initializing"}
STALE_SEC = 2 * 3600


class GpuBusy(Exception):
    def __init__(self, holders: list[dict]):
        self.holders = holders
        first = holders[0] if holders else {}
        super().__init__(
            f"GPU занят задачей {first.get('product')} {first.get('job_id')}"
        )


def _age_seconds(job: dict) -> float | None:
    stamp = job.get("updated_at") or job.get("created_at") or ""
    if is_placeholder_ts(stamp):
        return None
    try:
        return (datetime.now() - datetime.fromisoformat(stamp)).total_seconds()
    except ValueError:
        return None


def running_batches() -> list[dict]:
    holders = []
    for job in list_jobs():
        if job.get("product") not in BATCH_PRODUCTS:
            continue
        if (job.get("status") or "") not in ACTIVE:
            continue
        age = _age_seconds(job)
        if age is None or age > STALE_SEC:
            continue
        holders.append(job)
    return holders


def stale_jobs() -> list[dict]:
    stale = []
    for job in list_jobs():
        if (job.get("status") or "") not in ACTIVE:
            continue
        age = _age_seconds(job)
        if age is None or age > STALE_SEC:
            stale.append(job)
    return stale


def reap_stale_jobs() -> list[dict]:
    """Mark leftover running jobs without a fresh heartbeat as stale. Does not touch live GPU holders."""
    from .jobs import register, _r

    reaped: list[dict] = []
    live_ids = {job.get("job_id") for job in running_batches()}
    for job in stale_jobs():
        job_id = job.get("job_id") or ""
        if not job_id or job_id in live_ids:
            continue
        note = (job.get("message") or "")[:240]
        register(job_id, status="stale", message=note)
        try:
            client = _r()
            product = job.get("product") or ""
            if product == "mosinform":
                client.hset(f"mosinform:{job_id}", mapping={"status": "stale"})
            elif product == "llm-run":
                client.hset(f"task:{job_id}", mapping={"status": "stale"})
        except Exception:
            pass
        reaped.append({"job_id": job_id, "product": job.get("product"), "was": job.get("status")})
    return reaped


def assert_can_start(product: str) -> None:
    holders = running_batches()
    if holders:
        raise GpuBusy(holders)


def register_llm_run(task: dict, status: str | None = None) -> None:
    task_id = str(task.get("task_id") or "")
    if not task_id:
        return
    now = datetime.now().isoformat()
    raw_created = task.get("created_at") or ""
    created = "" if is_placeholder_ts(raw_created) else raw_created
    register(
        task_id,
        product="llm-run",
        route="/ai-analytics",
        status=status or task.get("status") or "pending",
        message=(task.get("promt_question") or "")[:240],
        files=task.get("folder_name") or "",
        period="",
        user_id=task.get("user_id") or "",
        progress=str(task.get("progress") or "0"),
        prompt_id="llm_run_user",
        model_id=(generate_cfg().get("model") or ""),
        created_at=created or now,
        updated_at=now,
    )


def finish_llm_run(task: dict, status: str, artifact_dir: str | None = None) -> None:
    register_llm_run(task, status=status)
    if not artifact_dir:
        return
    try:
        run = build_run(
            product="llm-run",
            route="/ai-analytics",
            job_id=str(task.get("task_id") or ""),
            prompt_id="llm_run_user",
            extra={
                "folder_name": task.get("folder_name") or "",
                "user_id": task.get("user_id") or "",
                "system_prompt": (task.get("system_prompt") or "")[:500],
                "status": status,
            },
        )
        write_run(Path(artifact_dir) / "run.json", run)
    except Exception:
        pass


def register_smart_agent(task_id: str, user_query: str, status: str, user_id: str = "") -> None:
    register(
        task_id,
        product="smart-agent",
        route="/smart-agent",
        status=status,
        message=(user_query or "")[:240],
        files="",
        user_id=user_id or "",
        model_id=(generate_cfg().get("model") or ""),
        prompt_id="smart_agent_plan",
        created_at=datetime.now().isoformat(),
    )


def finish_smart_agent(task_id: str, user_query: str, status: str, user_id: str = "", artifact_path: str = "") -> None:
    register_smart_agent(task_id, user_query, status, user_id=user_id)
    if not artifact_path:
        return
    try:
        run = build_run(
            product="smart-agent",
            route="/smart-agent",
            job_id=task_id,
            prompt_id="smart_agent_plan",
            extra={"user_id": user_id or "", "status": status, "query": (user_query or "")[:500]},
        )
        write_run(Path(artifact_path).parent / "run.json", run)
    except Exception:
        pass


def register_upload(task_id: str, filename: str, status: str, user_id: str = "", folder: str = "") -> None:
    register(
        task_id,
        product="upload",
        route="/data-set",
        status=status,
        message=filename or "",
        files=folder or filename or "",
        user_id=user_id or "",
        created_at=datetime.now().isoformat(),
    )


def _bertopic_creation(user_id: str, task_id: str) -> str:
    if not user_id or not task_id:
        return ""
    try:
        from .jobs import _r

        raw = _r().hget(str(user_id), "bertopic_files_directory") or ""
        data = json.loads(raw) if raw else {}
    except Exception:
        return ""
    folders = data.values() if isinstance(data, dict) else []
    for items in folders:
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            if str(item.get("task_id") or "") != str(task_id):
                continue
            stamp = str(item.get("creation_date") or "").strip()
            if stamp:
                return stamp.replace(" ", "T", 1)
    return ""


def backfill_llm_tasks() -> None:
    try:
        from .jobs import _key, _r

        client = _r()
        for key in client.keys("task:*"):
            if key.startswith("task:lock") or "lock:task" in key:
                continue
            data = client.hgetall(key) or {}
            task_id = data.get("task_id") or (key.split(":", 1)[1] if ":" in key else "")
            if not task_id or not data.get("folder_name"):
                continue
            recovered = _bertopic_creation(data.get("user_id") or "", task_id)
            if client.exists(_key(task_id)):
                job = client.hgetall(_key(task_id)) or {}
                if recovered and is_placeholder_ts(job.get("created_at") or ""):
                    register(
                        task_id,
                        created_at=recovered,
                        updated_at=recovered,
                    )
                continue
            # Leftover Redis tasks after FastAPI restart have no heartbeat.
            # Do not stamp them as "now" or QoS will think GPU is busy.
            if recovered:
                data = {**data, "created_at": recovered, "updated_at": recovered}
                register_llm_run(data, status=data.get("status") or "unknown")
            else:
                register(
                    task_id,
                    product="llm-run",
                    route="/ai-analytics",
                    status=data.get("status") or "unknown",
                    message=(data.get("promt_question") or "")[:240],
                    files=data.get("folder_name") or "",
                    user_id=data.get("user_id") or "",
                    progress=str(data.get("progress") or "0"),
                    prompt_id="llm_run_user",
                    model_id=(generate_cfg().get("model") or ""),
                    created_at="",
                    updated_at="",
                    _empty_ts=True,
                )
    except Exception:
        pass
