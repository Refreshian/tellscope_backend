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

from .context import AGENT_RUNS_ROOT, AgentContext, to_unix_end, to_unix_start
from .loop import DEFAULT_CHOICE, DEFAULT_TOKEN_BUDGET, MODEL_CHOICES, run_agent
from .progress import ProgressTracker, human_duration
from .registry import resolve_tools

MAX_EVENTS = 800
RUN_BUDGET_SEC = 1800
MAX_ACTIVE_PER_USER = 1
MAX_ACTIVE_TOTAL = 3
MAX_RUNS_PER_DAY = 60
MAX_TOKENS_PER_DAY = 1_500_000
MIN_TOKEN_BUDGET = 20_000
MAX_TOKEN_BUDGET = 2_000_000

# Запуск без событий дольше этого времени считается прерванным (сервер перезапускался,
# процесс убит): heartbeat во время длинных операций идёт каждые 10 секунд.
STALE_AFTER_SEC = 240

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


def tokens_today_for_user(user_id: Any) -> int:
    """Сколько токенов пользователь уже сжёг агентными запусками за сегодня (учёт расхода)."""
    today = time.strftime("%Y-%m-%d")
    total = 0
    for run in RUNS.values():
        if str(run.get("user_id")) == str(user_id) and str(run.get("created_at") or "").startswith(today):
            total += int((run.get("stats") or {}).get("tokens") or 0)
    for item in _load_persisted(str(user_id), limit=200):
        if str(item.get("created_at") or "").startswith(today):
            stats = item.get("stats") or {}
            if stats:
                total += int(stats.get("tokens") or 0)
    return total


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
    token_budget: Optional[int] = None,
    steps: Optional[List[Dict[str, Any]]] = None,
    mode: str = "",
) -> Dict[str, Any]:
    run_id = str(uuid.uuid4())
    try:
        budget = int(token_budget) if token_budget else DEFAULT_TOKEN_BUDGET
    except Exception:
        budget = DEFAULT_TOKEN_BUDGET
    budget = max(MIN_TOKEN_BUDGET, min(budget, MAX_TOKEN_BUDGET))
    run = {
        "run_id": run_id,
        "user_id": str(user_id),
        "task": task,
        "mode": str(mode or ""),
        "status": "queued",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "created_ts": time.time(),
        "started_at": None,
        "started_ts": None,
        "finished_at": None,
        "finished_ts": None,
        "duration_sec": None,
        "model_choice": model_choice if model_choice in MODEL_CHOICES else DEFAULT_CHOICE,
        "model_label": (MODEL_CHOICES.get(model_choice) or MODEL_CHOICES[DEFAULT_CHOICE]).get("label"),
        "token_budget": budget,
        "cost_usd": 0.0,
        "tools": resolve_tools(tools),
        "steps": list(steps or []),
        "dataset_index": dataset_index,
        "dataset_name": dataset_name,
        "dataset_label": dataset_label,
        "min_date": to_unix_start(min_date),
        "max_date": to_unix_end(max_date),
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
        # Свежесть запуска: по этим полям фронтенд показывает «последнее обновление N с назад»,
        # а список задач понимает, что запуск без событий — прерванный.
        run["last_event_ts"] = time.time()
        if event.get("type") in ("progress", "heartbeat"):
            run["progress"] = dict(event)
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

        progress = "100" if status in ("completed", "failed") else str((run.get("progress") or {}).get("percent") or 0)
        register_job(
            run["run_id"],
            product="agent-mode",
            route="/agent-mode",
            status=status,
            message=(run.get("task") or "")[:240],
            files=str(run.get("dataset_name") or ""),
            user_id=str(run.get("user_id") or ""),
            progress=progress,
            prompt_id="agent_system_v1",
            model_id=str(run.get("model_label") or ""),
        )
    except Exception:
        pass


def active_runs_total() -> int:
    return len([run for run in RUNS.values() if run.get("status") in ("queued", "running")])


# -------------------------------------------------------- прогресс, время, статусы задач

def _parse_local_ts(value: Any) -> float:
    """'2026-09-12 20:15:03' (локальное время сервера) → unix-секунды."""
    text = str(value or "").strip()
    if not text:
        return 0.0
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return time.mktime(time.strptime(text[:19], fmt))
        except Exception:
            continue
    return 0.0


def run_last_activity(run: Dict[str, Any]) -> float:
    """Когда запуск последний раз подавал признаки жизни.

    Сначала — фактические события (progress/heartbeat), затем старт, затем создание записи.
    """
    for key in ("last_event_ts",):
        value = run.get(key)
        if value:
            try:
                return float(value)
            except Exception:
                pass
    progress = run.get("progress") or {}
    events = run.get("events") or []
    if events:
        stamp = str((events[-1] or {}).get("ts") or "")
        if stamp:
            try:
                from datetime import datetime

                return datetime.fromisoformat(stamp).timestamp()
            except Exception:
                pass
    if progress.get("elapsed"):
        started = run.get("started_ts")
        if started:
            try:
                return float(started) + float(progress.get("elapsed") or 0)
            except Exception:
                pass
    for key in ("started_ts", "finished_ts"):
        value = run.get(key)
        if value:
            try:
                return float(value)
            except Exception:
                pass
    for key in ("started_at", "finished_at", "created_at"):
        parsed = _parse_local_ts(run.get(key))
        if parsed:
            return parsed
    return 0.0


def run_is_stale(run: Dict[str, Any], now: Optional[float] = None) -> bool:
    """Запуск числится активным, но событий нет дольше STALE_AFTER_SEC."""
    if str(run.get("status") or "") not in ("queued", "running"):
        return False
    last = run_last_activity(run)
    if not last:
        return False
    return (now or time.time()) - last > STALE_AFTER_SEC


def history_step_seconds(user_id: Any, mode: str = "") -> float:
    """Среднее время одного шага по последним удачным запускам — база для первой оценки ETA.

    Простая оценка: берём завершённые запуски того же режима (или все, если режим не совпал),
    считаем длительность / число шагов и усредняем по последним восьми.
    """
    samples: List[float] = []
    for item in _load_persisted(str(user_id), limit=40):
        if str(item.get("status")) != "completed":
            continue
        if mode and str(item.get("mode") or "") and str(item.get("mode")) != mode:
            continue
        duration = item.get("duration_sec")
        steps = int((item.get("stats") or {}).get("steps") or 0)
        try:
            duration = float(duration)
        except Exception:
            duration = 0.0
        if not duration and item.get("started_at") and item.get("finished_at"):
            duration = _parse_local_ts(item.get("finished_at")) - _parse_local_ts(item.get("started_at"))
        if duration > 0 and steps > 0:
            samples.append(duration / float(steps))
        if len(samples) >= 8:
            break
    if not samples:
        return 0.0
    return sum(samples) / float(len(samples))


def history_run_seconds(user_id: Any, mode: str = "") -> float:
    """Средняя длительность прошлых запусков того же режима — подсказка «обычно это ~N минут»."""
    samples: List[float] = []
    for item in _load_persisted(str(user_id), limit=40):
        if str(item.get("status")) != "completed":
            continue
        if mode and str(item.get("mode") or "") and str(item.get("mode")) != mode:
            continue
        try:
            duration = float(item.get("duration_sec") or 0)
        except Exception:
            duration = 0.0
        if not duration and item.get("started_at") and item.get("finished_at"):
            duration = _parse_local_ts(item.get("finished_at")) - _parse_local_ts(item.get("started_at"))
        if duration > 5:
            samples.append(duration)
        if len(samples) >= 8:
            break
    if not samples:
        return 0.0
    return sum(samples) / float(len(samples))


def task_patch_from_run(run: Dict[str, Any]) -> Dict[str, Any]:
    """Приводит запись задачи Центра ИИ-задач в соответствие с её запуском."""
    ok = str(run.get("status")) == "completed"
    answer = str(run.get("answer") or "")
    return {
        "status": "done" if ok else "failed",
        "status_label": "выполнено" if ok else "неуспешно",
        "run_status": str(run.get("status") or ""),
        "finished_at": run.get("finished_at"),
        "duration_sec": run.get("duration_sec"),
        "answer": answer[:2000],
        "error": run.get("error"),
        "run_progress": dict(run.get("progress") or {}),
    }


def sync_harness_task(run: Dict[str, Any]) -> int:
    """По завершении запуска обновляет статус задачи в data/<user>/harness_tasks.json.

    Без этого задачи навсегда оставались в статусе running, хотя запуск уже completed.
    """
    try:
        from . import harness as harness_store

        user_id = run.get("user_id")
        run_id = str(run.get("run_id") or "")
        if not user_id or not run_id:
            return 0
        changed = 0
        for task in harness_store.list_tasks(user_id, limit=200):
            if str(task.get("run_id") or "") != run_id:
                continue
            harness_store.update_task(user_id, task.get("id"), task_patch_from_run(run))
            changed += 1
        return changed
    except Exception:
        return 0


def reconcile_harness_tasks(user_id: Any) -> int:
    """Чинит зависшие задачи: статус в хранилище приводится к состоянию запуска.

    * запуск завершён, а задача всё ещё running → done/failed с текстом ошибки;
    * запуск активен, но событий нет дольше STALE_AFTER_SEC → «прерван»;
    * записи запуска нет вовсе (сервер перезапускался) → «прерван».
    """
    try:
        from . import harness as harness_store

        tasks = harness_store.list_tasks(user_id, limit=200)
    except Exception:
        return 0
    now = time.time()
    changed = 0
    for task in tasks:
        if str(task.get("status") or "") not in ("running", "new", "queued"):
            continue
        run_id = str(task.get("run_id") or "")
        patch: Optional[Dict[str, Any]] = None
        if not run_id:
            created = _parse_local_ts(task.get("created_at"))
            if created and now - created > STALE_AFTER_SEC:
                patch = {
                    "status": "failed",
                    "status_label": "прерван",
                    "run_status": "interrupted",
                    "error": "задача осталась без запуска (сервер перезапускался) — запустите её снова",
                }
        else:
            run = get_run(run_id)
            if run is None:
                created = _parse_local_ts(task.get("created_at"))
                if created and now - created > STALE_AFTER_SEC:
                    patch = {
                        "status": "failed",
                        "status_label": "прерван",
                        "run_status": "interrupted",
                        "error": "запись запуска не найдена: задача прервана, запустите её снова",
                    }
            elif str(run.get("status")) in ("completed", "failed"):
                patch = task_patch_from_run(run)
            elif run_is_stale(run, now):
                idle = int((now - run_last_activity(run)) / 60.0)
                patch = {
                    "status": "failed",
                    "status_label": "прерван",
                    "run_status": "interrupted",
                    "run_progress": dict(run.get("progress") or {}),
                    "error": f"запуск прерван: событий нет {idle} мин (сервер перезапускался или процесс убит)",
                }
        if patch:
            try:
                harness_store.update_task(user_id, task.get("id"), patch)
                changed += 1
            except Exception:
                continue
    return changed


def _bridge_emit(run_id: str, main_loop: asyncio.AbstractEventLoop):
    """Пробрасывает события из рабочего потока в основной цикл (там живут WebSocket-подписки)."""
    async def emit(event: Dict[str, Any]) -> None:
        try:
            future = asyncio.run_coroutine_threadsafe(_emit(run_id, event), main_loop)
            await asyncio.wrap_future(future)
        except Exception:
            pass

    return emit


async def _resolve_orchestrator_meta(run: Dict[str, Any], emit: Any) -> Dict[str, Any]:
    """Выбирает модель-оркестратор на этот запуск и кладёт решение в метаданные запуска.

    Настройка (agent.orchestrator + orchestrator_fallbacks) читается заново на каждый запуск,
    поэтому смена lock.yaml или переменной окружения действует без перезапуска приложения.
    """
    run["orchestrator_used"] = False
    try:
        from mlops.orchestrator import resolve_orchestrator

        state = await resolve_orchestrator()
    except Exception as exc:  # noqa: BLE001
        run["orchestrator_resolve_error"] = f"{type(exc).__name__}: {exc}"
        await emit({"type": "log", "level": "error", "message": f"Не удалось определить оркестратора: {exc}"})
        return {}
    meta = state.as_dict()
    run.update(
        {
            "orchestrator": state.choice,
            "orchestrator_label": state.label,
            "orchestrator_requested": state.requested,
            "orchestrator_chain": list(state.chain),
            "orchestrator_source": state.source,
            "orchestrator_probe": state.probe_mode,
            "orchestrator_available": state.available,
            "orchestrator_fallback_used": state.fallback_used,
            "orchestrator_reason": state.reason_text(),
        }
    )
    await emit(
        {
            "type": "log",
            "level": "error" if state.fallback_used else "info",
            "message": (
                f"Оркестратор: {state.label} (цепочка: {', '.join(state.chain)}; "
                f"проверка: {state.probe_mode}; источник настройки: {state.source})"
                + (f" — откат, потому что {state.reason_text()}" if state.reason_text() else "")
            ),
        }
    )
    return meta


def _merge_orchestrator_result(run: Dict[str, Any], result: Dict[str, Any]) -> None:
    """После прогона фиксируем ту модель, которая реально оркестрировала (могла смениться на ходу)."""
    meta = result.get("orchestrator") or {}
    stats = result.get("stats") or {}
    final_key = meta.get("orchestrator_final") or stats.get("orchestrator") or run.get("orchestrator")
    if final_key:
        run["orchestrator"] = final_key
    final_label = meta.get("orchestrator_final_label") or stats.get("orchestrator_label")
    if final_label:
        run["orchestrator_label"] = final_label
    for field_name in ("orchestrator_requested", "orchestrator_source", "orchestrator_probe"):
        if meta.get(field_name):
            run[field_name] = meta[field_name]
    if meta.get("orchestrator_chain") or stats.get("orchestrator_chain"):
        run["orchestrator_chain"] = list(meta.get("orchestrator_chain") or stats.get("orchestrator_chain"))
    reason = meta.get("orchestrator_reason") or stats.get("orchestrator_reason")
    if reason:
        run["orchestrator_reason"] = reason
    if meta.get("orchestrator_fallback_used") or stats.get("orchestrator_fallback_used"):
        run["orchestrator_fallback_used"] = True
    if meta.get("analysis_model"):
        run["analysis_model"] = meta["analysis_model"]
        run["analysis_model_label"] = meta.get("analysis_model_label")
    if meta.get("answer_from_analysis") is not None:
        run["answer_from_analysis"] = bool(meta.get("answer_from_analysis"))
    if stats.get("models_used"):
        run["models_used"] = stats["models_used"]


async def execute_run(run_id: str, main_loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
    run = RUNS.get(run_id)
    if run is None:
        return
    run["status"] = "running"
    run["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    run["started_ts"] = time.time()
    run["last_event_ts"] = run["started_ts"]
    _register_job(run, "running")
    _persist(run)

    user = run.get("_user")
    emit = _bridge_emit(run_id, main_loop) if main_loop is not None else (lambda event: _emit(run_id, event))
    orchestrator_meta = await _resolve_orchestrator_meta(run, emit)
    _persist(run)
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
        orchestrator_choice=str(orchestrator_meta.get("orchestrator") or ""),
        orchestrator_chain=list(orchestrator_meta.get("orchestrator_chain") or []),
        orchestrator_info=dict(orchestrator_meta),
        orchestrator_probe_mode=str(orchestrator_meta.get("orchestrator_probe") or "tools"),
        folder=str(run.get("folder") or "Агент"),
        emit=emit,
        artifacts_dir=artifacts_dir(run_id),
        deadline=time.time() + RUN_BUDGET_SEC,
        token_budget=int(run.get("token_budget") or DEFAULT_TOKEN_BUDGET),
    )
    # Прогресс запуска: сколько шагов в плане (если план есть), сколько сделано и сколько осталось.
    steps_plan = run.get("steps") or []
    tracker = ProgressTracker(
        ctx,
        total=len(steps_plan),
        history_step_sec=history_step_seconds(run.get("user_id"), str(run.get("mode") or "")),
        history_run_sec=history_run_seconds(run.get("user_id"), str(run.get("mode") or "")),
        mode=str(run.get("mode") or ""),
    )
    ctx.progress = tracker
    try:
        steps = steps_plan
        if steps:
            # конструктор шагов: детерминированная цепочка, LLM планирует только текст выводов,
            # поэтому модель-оркестратор здесь не участвует
            from .pipeline import run_pipeline

            result = await run_pipeline(ctx, steps)
            run["orchestrator_used"] = False
        else:
            result = await run_agent(ctx)
            run["orchestrator_used"] = True
        run["answer"] = result.get("answer") or ""
        run["stats"] = result.get("stats") or {}
        _merge_orchestrator_result(run, result)
        run["tool_calls"] = result.get("tool_calls") or []
        run["artifacts"] = result.get("artifacts") or []
        run["cost_usd"] = float((result.get("stats") or {}).get("cost_usd") or 0.0)
        run["no_data"] = bool(result.get("no_data"))
        run["text_gap"] = str(result.get("text_gap") or "")
        if run.get("no_data"):
            run["status"] = "failed"
            run["error"] = "данные за период не найдены — отчёт не сформирован"
        elif run.get("text_gap"):
            # Негатив в срезе без конкретной темы и цитаты: отчёт неполный
            run["status"] = "failed"
            run["error"] = run["text_gap"]
        else:
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
        run["finished_ts"] = time.time()
        started_ts = run.get("started_ts") or run["finished_ts"]
        run["duration_sec"] = round(max(0.0, run["finished_ts"] - float(started_ts)), 1)
        # Финальное событие прогресса: «выполнено за 3 мин 20 с» (или причина ошибки).
        status = str(run.get("status") or "completed")
        if status == "completed":
            detail = f"выполнено за {human_duration(run['duration_sec'])}"
        else:
            detail = f"неуспешно за {human_duration(run['duration_sec'])}: {run.get('error') or 'причина не указана'}"
        try:
            await tracker.finish_run(status, detail=detail)
        except Exception:
            pass
        run["progress_detail"] = detail
        _persist(run)
        _register_job(run, status)
        # Статус задачи в хранилище приводим в соответствие с завершившимся запуском.
        sync_harness_task(run)
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
