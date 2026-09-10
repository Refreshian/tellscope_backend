# -*- coding: utf-8 -*-
"""Агентные запуски: состояние, журнал шагов, подписки WebSocket, персистентность."""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

from .context import AGENT_RUNS_ROOT, AgentContext, to_unix
from .loop import DEFAULT_CHOICE, MODEL_CHOICES, run_agent
from .registry import resolve_tools

MAX_EVENTS = 800
RUN_BUDGET_SEC = 1800
MAX_ACTIVE_PER_USER = 1
MAX_ACTIVE_TOTAL = 3
MAX_RUNS_PER_DAY = 60

RUNS: Dict[str, Dict[str, Any]] = {}
_SUBS: Dict[str, List[asyncio.Queue]] = {}
_ACTIVE_TASKS: Dict[str, Any] = {}

FILES_ROOT = os.path.join(AGENT_RUNS_ROOT, "files")

_WORKER_LOCK = threading.Lock()
_WORKER_LOOP: Optional[asyncio.AbstractEventLoop] = None


def worker_loop() -> asyncio.AbstractEventLoop:
    """Отдельный цикл событий в фоновом потоке: тяжёлые ES-запросы не блокируют API."""
    global _WORKER_LOOP
    with _WORKER_LOCK:
        if _WORKER_LOOP is None or _WORKER_LOOP.is_closed():
            loop = asyncio.new_event_loop()
            thread = threading.Thread(target=loop.run_forever, name="agent-mode-worker", daemon=True)
            thread.start()
            _WORKER_LOOP = loop
    return _WORKER_LOOP


def _run_file(run_id: str) -> str:
    return os.path.join(AGENT_RUNS_ROOT, f"{run_id}.json")


def artifacts_dir(run_id: str) -> str:
    path = os.path.join(FILES_ROOT, run_id)
    os.makedirs(path, exist_ok=True)
    return path


def _persist(run: Dict[str, Any]) -> None:
    try:
        os.makedirs(AGENT_RUNS_ROOT, exist_ok=True)
        payload = {k: v for k, v in run.items() if k != "events"}
        payload["events"] = (run.get("events") or [])[-200:]
        with open(_run_file(run["run_id"]), "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, default=str)
    except Exception:
        pass


def _load_persisted(user_id: str, limit: int = 30) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if not os.path.isdir(AGENT_RUNS_ROOT):
        return items
    for name in sorted(os.listdir(AGENT_RUNS_ROOT), reverse=True):
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(AGENT_RUNS_ROOT, name), "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            continue
        if str(data.get("user_id")) != str(user_id):
            continue
        data.pop("events", None)
        items.append(data)
        if len(items) >= limit:
            break
    return items


def active_runs_for_user(user_id: Any) -> List[Dict[str, Any]]:
    out = []
    for run in RUNS.values():
        if str(run.get("user_id")) == str(user_id) and run.get("status") in ("queued", "running"):
            out.append(run)
    return out


def runs_today_for_user(user_id: Any) -> int:
    today = time.strftime("%Y-%m-%d")
    count = 0
    for run in RUNS.values():
        if str(run.get("user_id")) == str(user_id) and str(run.get("created_at") or "").startswith(today):
            count += 1
    if count:
        return count
    for item in _load_persisted(str(user_id), limit=200):
        if str(item.get("created_at") or "").startswith(today):
            count += 1
    return count


def create_run(
    *,
    user: Any,
    user_id: str,
    task: str,
    dataset_index: Optional[int],
    dataset_name: Optional[str] = None,
    dataset_label: str = "",
    min_date: Any = None,
    max_date: Any = None,
    tools: Optional[List[str]] = None,
    model_choice: str = DEFAULT_CHOICE,
    folder: str = "Агент",
) -> Dict[str, Any]:
    run_id = str(uuid.uuid4())
    run = {
        "run_id": run_id,
        "user_id": str(user_id),
        "task": task,
        "status": "queued",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "started_at": None,
        "finished_at": None,
        "model_choice": model_choice if model_choice in MODEL_CHOICES else DEFAULT_CHOICE,
        "model_label": (MODEL_CHOICES.get(model_choice) or MODEL_CHOICES[DEFAULT_CHOICE]).get("label"),
        "tools": resolve_tools(tools),
        "dataset_index": dataset_index,
        "dataset_name": dataset_name,
        "dataset_label": dataset_label,
        "min_date": to_unix(min_date),
        "max_date": to_unix(max_date),
        "folder": folder or "Агент",
        "answer": "",
        "stats": {},
        "tool_calls": [],
        "artifacts": [],
        "error": None,
        "events": [],
    }
    RUNS[run_id] = run
    _persist(run)
    return run


def get_run(run_id: str) -> Optional[Dict[str, Any]]:
    run = RUNS.get(run_id)
    if run:
        return run
    path = _run_file(run_id)
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            RUNS[run_id] = data
            return data
        except Exception:
            return None
    return None


def list_runs(user_id: Any, limit: int = 25) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for item in _load_persisted(str(user_id), limit=limit):
        merged[item.get("run_id")] = item
    for run in RUNS.values():
        if str(run.get("user_id")) == str(user_id):
            trimmed = {k: v for k, v in run.items() if k != "events"}
            merged[run["run_id"]] = trimmed
    items = sorted(merged.values(), key=lambda r: str(r.get("created_at") or ""), reverse=True)
    return items[:limit]


def subscribe(run_id: str) -> asyncio.Queue:
    queue: asyncio.Queue = asyncio.Queue(maxsize=500)
    _SUBS.setdefault(run_id, []).append(queue)
    return queue


def unsubscribe(run_id: str, queue: asyncio.Queue) -> None:
    subs = _SUBS.get(run_id) or []
    if queue in subs:
        subs.remove(queue)
    if not subs:
        _SUBS.pop(run_id, None)


async def _emit(run_id: str, event: Dict[str, Any]) -> None:
    run = RUNS.get(run_id)
    if run is not None:
        events = run.setdefault("events", [])
        events.append(event)
        if len(events) > MAX_EVENTS:
            del events[: len(events) - MAX_EVENTS]
    for queue in list(_SUBS.get(run_id) or []):
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            pass


def buffered_events(run_id: str) -> List[Dict[str, Any]]:
    run = RUNS.get(run_id)
    return list((run or {}).get("events") or [])


def artifact_path(run_id: str, name: str) -> Optional[str]:
    safe = os.path.basename(name)
    path = os.path.join(FILES_ROOT, run_id, safe)
    if os.path.isfile(path):
        return path
    return None


def _register_job(run: Dict[str, Any], status: str) -> None:
    try:
        from mlops.runtime import register as register_job

        register_job(
            run["run_id"],
            product="agent-mode",
            route="/agent-mode",
            status=status,
            message=(run.get("task") or "")[:240],
            files=str(run.get("dataset_name") or ""),
            user_id=str(run.get("user_id") or ""),
            progress="0",
            prompt_id="agent_system_v1",
            model_id=str(run.get("model_label") or ""),
        )
    except Exception:
        pass


def active_runs_total() -> int:
    return len([run for run in RUNS.values() if run.get("status") in ("queued", "running")])


def _bridge_emit(run_id: str, main_loop: asyncio.AbstractEventLoop):
    """Пробрасывает события из рабочего потока в основной цикл (там живут WebSocket-подписки)."""
    async def emit(event: Dict[str, Any]) -> None:
        try:
            future = asyncio.run_coroutine_threadsafe(_emit(run_id, event), main_loop)
            await asyncio.wrap_future(future)
        except Exception:
            pass

    return emit


async def execute_run(run_id: str, main_loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
    run = RUNS.get(run_id)
    if run is None:
        return
    run["status"] = "running"
    run["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _register_job(run, "running")
    _persist(run)

    user = run.get("_user")
    emit = _bridge_emit(run_id, main_loop) if main_loop is not None else (lambda event: _emit(run_id, event))
    ctx = AgentContext(
        run_id=run_id,
        user=user,
        user_id=str(run.get("user_id") or ""),
        task=str(run.get("task") or ""),
        dataset_index=run.get("dataset_index"),
        dataset_name=run.get("dataset_name"),
        dataset_label=str(run.get("dataset_label") or ""),
        min_date=run.get("min_date"),
        max_date=run.get("max_date"),
        allowed_tools=set(run.get("tools") or []),
        model_choice=str(run.get("model_choice") or DEFAULT_CHOICE),
        folder=str(run.get("folder") or "Агент"),
        emit=emit,
        artifacts_dir=artifacts_dir(run_id),
        deadline=time.time() + RUN_BUDGET_SEC,
    )
    try:
        result = await run_agent(ctx)
        run["answer"] = result.get("answer") or ""
        run["stats"] = result.get("stats") or {}
        run["tool_calls"] = result.get("tool_calls") or []
        run["artifacts"] = result.get("artifacts") or []
        run["status"] = "completed" if run["answer"] else "failed"
        if not run["answer"]:
            run["error"] = "агент не сформировал ответ"
    except Exception as exc:
        run["status"] = "failed"
        run["error"] = f"{type(exc).__name__}: {exc}"
        await emit({"type": "error", "message": run["error"]})
    finally:
        run.pop("_user", None)
        run["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        _persist(run)
        _register_job(run, str(run.get("status") or "completed"))
        await emit({"type": "done", "status": run.get("status"), "error": run.get("error")})
        for queue in list(_SUBS.get(run_id) or []):
            try:
                queue.put_nowait({"type": "__close__"})
            except asyncio.QueueFull:
                pass
        _ACTIVE_TASKS.pop(run_id, None)


def start_run(run: Dict[str, Any], user: Any) -> Any:
    """Ставит запуск в очередь рабочего потока: агент не блокирует API Tellscope."""
    run["_user"] = user
    main_loop = asyncio.get_running_loop()
    worker = worker_loop()
    future = asyncio.run_coroutine_threadsafe(execute_run(run["run_id"], main_loop), worker)
    _ACTIVE_TASKS[run["run_id"]] = future
    return future
