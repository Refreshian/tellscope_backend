# -*- coding: utf-8 -*-
"""Происхождение файла: ``GET /file-origin``.

Отвечает на вопрос «по какому запросу (задаче) появился этот файл» для страницы
папки/датасета и списка отчётов.

Источники (только данные того же пользователя):

* ``data/<user_id>/harness_tasks.json`` — задачи центра ИИ-задач
  (поля ``text``, ``id``, ``created_at``, ``run_id``, ``status``, ``result``, ``answer``);
* ``data/agent_runs/<run_id>.json`` — записи запусков агента
  (``task``, ``artifacts``, ``result``, ``model_label``, ``status``);
* ``data/ba_archive/imports.jsonl`` — реестр импортов Brand Analytics
  (файл датасета создан экспортом темы BA).

Сопоставление идёт по имени файла (basename) и по нормализованному заголовку отчёта
(без расширения и без суффикса даты ``_ГГГГММДД_ЧЧММ``), поэтому отчёт в формате
``Имя_ГГГГММДД_ЧЧММ.docx``/``.pdf`` находится, даже если у него сменилось расширение.

Модуль самостоятельный (собственный ``APIRouter``), в ``main.py`` подключается одной
строкой ``app.include_router(file_origin_router)``.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException, Query, Request

router = APIRouter(tags=["file origin"])

BACKEND_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BACKEND_ROOT, "data")
AGENT_RUNS_ROOT = os.path.join(DATA_ROOT, "agent_runs")
BA_REGISTRY = os.path.join(DATA_ROOT, "ba_archive", "imports.jsonl")

NOT_FOUND_REASON = "для этого файла запрос не сохранён (создан до появления функции)"
BAD_USER_REASON = "нет доступа к задачам другого пользователя"

# Сколько свежих запусков просматриваем и насколько долго кэшируем ответ.
MAX_RUNS_SCAN = 400
STOP_SCORE = 95
CACHE_TTL_SEC = 120
CACHE_MAX = 400

FILE_EXTS = (
    ".docx", ".doc", ".pdf", ".xlsx", ".xls", ".csv", ".tsv", ".json",
    ".txt", ".md", ".html", ".png", ".jpg", ".jpeg", ".pptx", ".zip",
)

_DATE_SUFFIX_RE = re.compile(r"[_\- ]?\d{8}[_\-]\d{4,6}(?!\d)")
_DATE_DOTTED_RE = re.compile(r"[_\- ]?\d{2}[.\-]\d{2}[.\-]\d{4}(?!\d)")

_CACHE: Dict[Tuple[str, str, str], Tuple[float, Dict[str, Any]]] = {}
_CACHE_LOCK = threading.Lock()


# --------------------------------------------------------------------------- #
# авторизация
# --------------------------------------------------------------------------- #
def _extract_token(request: Request) -> str:
    auth = request.headers.get("authorization") or ""
    if auth[:7].lower() == "bearer ":
        return auth[7:].strip()
    for name in ("token", "access_token", "tellscope_refresh_token"):
        value = request.cookies.get(name)
        if value:
            return value.strip()
    return ""


def _service_token_ok(request: Request) -> bool:
    token = (request.headers.get("x-service-token") or "").strip()
    if not token:
        return False
    try:
        from agent_engine.service_api import match_service_token

        return bool(match_service_token(token))
    except Exception:
        return False


def _authorized_user(request: Request) -> Optional[str]:
    """user_id из JWT (Bearer или cookie) либо ``None`` для служебного токена."""
    token = _extract_token(request)
    if token:
        try:
            import jwt as _jwt
            from auth.auth import SECRET as _SECRET

            payload = _jwt.decode(
                token, _SECRET, algorithms=["HS256"], options={"verify_aud": False}
            )
            sub = payload.get("sub")
            if sub:
                return str(sub)
        except Exception:
            pass
    if _service_token_ok(request):
        return ""
    raise HTTPException(status_code=401, detail="Unauthorized")


# --------------------------------------------------------------------------- #
# нормализация имён
# --------------------------------------------------------------------------- #
def _basename(value: Any) -> str:
    return os.path.basename(str(value or "").replace("\\", "/").strip())


def _norm(value: str) -> str:
    text = (value or "").lower().replace("ё", "е")
    return re.sub(r"[^0-9a-zа-я]+", "", text)


def _stem(name: str) -> str:
    base = _basename(name)
    root, ext = os.path.splitext(base)
    return root or base


def _norm_title(name: str) -> str:
    """Заголовок отчёта: без расширения, без суффикса даты, без регистра/пробелов."""
    root = _stem(name)
    root = _DATE_SUFFIX_RE.sub("", root)
    root = _DATE_DOTTED_RE.sub("", root)
    return _norm(root)


def _looks_like_file(value: str) -> bool:
    low = value.lower()
    return any(low.endswith(ext) for ext in FILE_EXTS)


def _walk_strings(obj: Any, acc: List[str], depth: int = 0) -> None:
    if depth > 8 or len(acc) >= 500:
        return
    if isinstance(obj, str):
        if _looks_like_file(obj):
            acc.append(obj)
        return
    if isinstance(obj, dict):
        for value in obj.values():
            _walk_strings(value, acc, depth + 1)
        return
    if isinstance(obj, (list, tuple)):
        for value in obj:
            _walk_strings(value, acc, depth + 1)


def _score(name: str, base_target: str, norm_target: str, title_target: str,
           folder: str, path: str = "") -> Tuple[int, str]:
    """Насколько строка ``name`` похожа на искомый файл. Возвращает (балл, способ)."""
    base = _basename(name)
    if not base:
        return 0, ""
    norm_base = _norm(base)
    score, how = 0, ""
    if base == base_target:
        score, how = 100, "имя файла"
    elif _stem(base) == _stem(base_target):
        score, how = 96, "имя файла без расширения"
    elif norm_base and norm_base == norm_target:
        score, how = 92, "имя файла (нормализовано)"
    else:
        title_base = _norm_title(base)
        if title_target and title_base and title_base == title_target:
            score, how = 86, "заголовок отчёта"
        elif title_target and len(title_target) >= 10 and title_base:
            short, long_ = sorted((title_target, title_base), key=len)
            if short in long_ and len(short) / max(len(long_), 1) >= 0.7:
                score, how = 70, "заголовок отчёта (частично)"
        if not score and len(norm_target) >= 8 and norm_base:
            short, long_ = sorted((norm_target, norm_base), key=len)
            if short in long_ and len(short) / max(len(long_), 1) >= 0.62:
                score, how = 62, "имя файла (частично)"
    if score and folder and path:
        parts = [p for p in str(path).replace("\\", "/").split("/") if p]
        if folder in parts:
            score += 6
            how += " + папка"
    return score, how


# --------------------------------------------------------------------------- #
# источники
# --------------------------------------------------------------------------- #
def _harness_path(user_id: str) -> str:
    return os.path.join(DATA_ROOT, str(user_id), "harness_tasks.json")


def _load_harness_tasks(user_id: str) -> List[Dict[str, Any]]:
    path = _harness_path(user_id)
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return []
    tasks = data.get("tasks") if isinstance(data, dict) else data
    if not isinstance(tasks, list):
        return []
    return [t for t in tasks if isinstance(t, dict)]


def _task_payload(task: Dict[str, Any], how: str, source: str) -> Dict[str, Any]:
    return {
        "found": True,
        "task_text": str(task.get("text") or "").strip(),
        "task_id": task.get("id"),
        "run_id": task.get("run_id"),
        "created_at": task.get("created_at"),
        "model": task.get("model") or task.get("model_label"),
        "status": task.get("status"),
        "source": source,
        "matched_by": how,
        "reason": "",
    }


def _run_model(run: Dict[str, Any]) -> Optional[str]:
    for key in ("model_label", "analysis_model_label", "model_choice", "model"):
        value = run.get(key)
        if value:
            return str(value)
    return None


def _run_payload(run: Dict[str, Any], how: str) -> Dict[str, Any]:
    return {
        "found": True,
        "task_text": str(run.get("task") or run.get("task_text") or "").strip(),
        "task_id": run.get("task_id"),
        "run_id": run.get("run_id"),
        "created_at": run.get("created_at"),
        "model": _run_model(run),
        "status": run.get("status"),
        "source": "agent_run",
        "matched_by": how,
        "reason": "",
    }


def _iter_run_files() -> List[str]:
    if not os.path.isdir(AGENT_RUNS_ROOT):
        return []
    items: List[Tuple[float, str]] = []
    try:
        for name in os.listdir(AGENT_RUNS_ROOT):
            if not name.endswith(".json"):
                continue
            path = os.path.join(AGENT_RUNS_ROOT, name)
            try:
                items.append((os.path.getmtime(path), path))
            except OSError:
                continue
    except OSError:
        return []
    items.sort(reverse=True)
    return [path for _, path in items[:MAX_RUNS_SCAN]]


def _match_runs(user_id: str, base_target: str, norm_target: str, title_target: str,
                folder: str) -> Optional[Tuple[int, Dict[str, Any], str]]:
    best: Optional[Tuple[int, Dict[str, Any], str]] = None
    for path in _iter_run_files():
        try:
            with open(path, "r", encoding="utf-8") as fh:
                run = json.load(fh)
        except Exception:
            continue
        if not isinstance(run, dict):
            continue
        if str(run.get("user_id")) != str(user_id):
            continue
        strings: List[str] = []
        _walk_strings(run.get("artifacts"), strings)
        _walk_strings(run.get("result"), strings)
        _walk_strings(run.get("files"), strings)
        _walk_strings(run.get("steps"), strings)
        seen = set()
        run_score, run_how = 0, ""
        for candidate in strings:
            key = _basename(candidate)
            if key in seen:
                continue
            seen.add(key)
            score, how = _score(candidate, base_target, norm_target, title_target,
                                folder, candidate)
            if score > run_score:
                run_score, run_how = score, how
        if run_score and (best is None or run_score > best[0]):
            best = (run_score, run, run_how)
            if run_score >= STOP_SCORE:
                break
    return best


def _match_harness(tasks: List[Dict[str, Any]], base_target: str, norm_target: str,
                   title_target: str, folder: str) -> Optional[Tuple[int, Dict[str, Any], str]]:
    best: Optional[Tuple[int, Dict[str, Any], str]] = None
    for task in tasks:
        strings: List[str] = []
        _walk_strings(task.get("result"), strings)
        _walk_strings(task.get("answer"), strings)
        _walk_strings(task.get("text"), strings)
        seen = set()
        task_score, task_how = 0, ""
        for candidate in strings:
            key = _basename(candidate)
            if key in seen:
                continue
            seen.add(key)
            score, how = _score(candidate, base_target, norm_target, title_target,
                                folder, candidate)
            if score > task_score:
                task_score, task_how = score, how
        if task_score and (best is None or task_score > best[0]):
            best = (task_score, task, task_how)
    return best


def _match_ba(user_id: str, base_target: str, norm_target: str, title_target: str,
              folder: str) -> Optional[Dict[str, Any]]:
    """Файл датасета, полученный экспортом темы Brand Analytics.

    На странице датасета файл показан как имя индекса (``ba_озон_отзывы_20260912_150434``),
    поэтому сверяем и с ``index_name``, и с именем файла в реестре — без учёта регистра,
    разделителей и расширения.
    """
    if not os.path.isfile(BA_REGISTRY):
        return None
    record: Optional[Dict[str, Any]] = None
    best = 0
    try:
        with open(BA_REGISTRY, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                if str(item.get("user_id")) != str(user_id):
                    continue
                candidates = [
                    item.get("file"),
                    item.get("index_name"),
                    _stem(item.get("file") or ""),
                ]
                score = 0
                for candidate in candidates:
                    if not candidate:
                        continue
                    value, _how = _score(candidate, base_target, norm_target, title_target, "", "")
                    score = max(score, value)
                if score >= 92 and score >= best:
                    best = score
                    record = item
    except Exception:
        return None
    if not record:
        return None


    def _day(value: Any) -> str:
        try:
            return time.strftime("%d.%m.%Y", time.localtime(float(value)))
        except Exception:
            return str(value or "")

    theme = record.get("theme") or folder or "Brand Analytics"
    period = ""
    if record.get("date_from") and record.get("date_to"):
        period = ", период %s–%s" % (_day(record.get("date_from")), _day(record.get("date_to")))
    return {
        "found": True,
        "task_text": "Импорт из Brand Analytics: тема «%s»%s (индекс %s)" % (
            theme, period, record.get("index_name") or record.get("index_key") or "—",
        ),
        "task_id": record.get("job_id"),
        "run_id": None,
        "created_at": record.get("created"),
        "model": "Brand Analytics (экспорт темы)",
        "status": "completed",
        "source": "ba_import",
        "matched_by": "имя файла датасета + реестр импортов",
        "reason": "",
    }


def _cache_get(key: Tuple[str, str, str]) -> Optional[Dict[str, Any]]:
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
    if not hit:
        return None
    stamp, payload = hit
    if time.time() - stamp > CACHE_TTL_SEC:
        with _CACHE_LOCK:
            _CACHE.pop(key, None)
        return None
    return payload


def _cache_put(key: Tuple[str, str, str], payload: Dict[str, Any]) -> None:
    with _CACHE_LOCK:
        if len(_CACHE) > CACHE_MAX:
            _CACHE.clear()
        _CACHE[key] = (time.time(), payload)


# --------------------------------------------------------------------------- #
# endpoint
# --------------------------------------------------------------------------- #
@router.get("/file-origin")
async def file_origin(
    request: Request,
    user_id: str = Query(..., description="владелец файла"),
    file: str = Query(..., description="имя файла, например Отчёт_20260912_2054.docx"),
    folder: str = Query("", description="папка отчёта/датасета"),
):
    """По какому запросу (задаче) появился файл.

    Без токена — 401, при чужом ``user_id`` — 403. Если происхождение не сохранено
    (файл создан раньше появления функции) — ``found: false`` с причиной.
    """
    caller = _authorized_user(request)
    if caller and str(caller) != str(user_id):
        raise HTTPException(status_code=403, detail=BAD_USER_REASON)

    base_target = _basename(file)
    if not base_target:
        raise HTTPException(status_code=400, detail="Не указано имя файла")
    folder_clean = _basename(folder.replace("\\", "/").rstrip("/")) if folder else ""
    if folder and not folder_clean:
        folder_clean = str(folder).strip()

    cache_key = (str(user_id), folder_clean, base_target)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    norm_target = _norm(base_target)
    title_target = _norm_title(base_target)

    payload: Dict[str, Any]
    run_match = _match_runs(str(user_id), base_target, norm_target, title_target, folder_clean)
    tasks = _load_harness_tasks(str(user_id))

    if run_match and run_match[0] >= 62:
        score, run, how = run_match
        task_for_run = None
        run_id = run.get("run_id")
        if run_id:
            for task in tasks:
                if task.get("run_id") and str(task.get("run_id")) == str(run_id):
                    task_for_run = task
                    break
        if task_for_run is not None:
            payload = _task_payload(task_for_run, how + " + запуск", "harness_task")
            payload["run_id"] = run_id
            payload["status"] = task_for_run.get("status") or run.get("status")
            payload["model"] = task_for_run.get("model") or _run_model(run)
            payload["created_at"] = task_for_run.get("created_at") or run.get("created_at")
        else:
            payload = _run_payload(run, how)
    else:
        task_match = _match_harness(tasks, base_target, norm_target, title_target, folder_clean)
        if task_match and task_match[0] >= 62:
            payload = _task_payload(task_match[1], task_match[2] + " + задача", "harness_task")
        else:
            ba = _match_ba(str(user_id), base_target, norm_target, title_target, folder_clean)
            if ba is not None:
                payload = ba
            else:
                payload = {
                    "found": False,
                    "task_text": "",
                    "task_id": None,
                    "run_id": None,
                    "created_at": None,
                    "model": None,
                    "status": None,
                    "source": "none",
                    "matched_by": "",
                    "reason": NOT_FOUND_REASON,
                }

    payload["file"] = base_target
    payload["folder"] = folder_clean
    _cache_put(cache_key, payload)
    return payload
