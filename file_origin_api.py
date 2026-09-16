# -*- coding: utf-8 -*-
"""Происхождение файла: ``GET /file-origin``.

Отвечает на вопрос «по какому запросу (задаче) появился этот файл» для страницы
папки/датасета и списка отчётов.

Источники (только данные того же пользователя):

* ``data/<user_id>/harness_tasks.json`` — задачи центра ИИ-задач
  (поля ``text``, ``id``, ``created_at``, ``run_id``, ``status``, ``result``, ``answer``);
* ``data/agent_runs/<run_id>.json`` — записи запусков агента
  (``task``, ``artifacts``, ``result``, ``model_label``, ``status``);
* ``data/<user_id>/reports_directory/**/*_summary.json`` — структурные итоги отчётов
  (``report.title``, ``report.files``, ``report.folder``, ``period.key``). Это надёжный
  источник для отчётов, собранных детерминированным сборщиком, а не агентом;
* ``data/ba_archive/imports.jsonl`` — реестр импортов Brand Analytics
  (файл датасета создан экспортом темы BA).

Сопоставление идёт по имени файла (basename) и по нормализованному заголовку отчёта
(без расширения и без суффикса даты ``_ГГГГММДД_ЧЧММ``), поэтому отчёт в формате
``Имя_ГГГГММДД_ЧЧММ.docx``/``.pdf`` находится, даже если у него сменилось расширение,
имя-заголовок или папка: **папка участвует только как бонус к оценке, никогда как
обязательное условие**, поиск идёт по всем папкам пользователя.

Ответ всегда содержит ``found``. Если происхождение известно, но отдельного текстового
запроса нет (структурный итог сборщика), возвращается ``found: true`` с ``kind: "summary"``
и честным описанием (заголовок отчёта, период, дата сборки) — интерфейс показывает такие
файлы отдельным заголовком и не выдаёт описание за запрос пользователя.

Модуль самостоятельный (собственный ``APIRouter``), в ``main.py`` подключается двумя
строками: ``from file_origin_api import router as file_origin_router`` и
``app.include_router(file_origin_router)``.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

from fastapi import APIRouter, HTTPException, Query, Request

router = APIRouter(tags=["file origin"])

BACKEND_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BACKEND_ROOT, "data")
AGENT_RUNS_ROOT = os.path.join(DATA_ROOT, "agent_runs")
BA_REGISTRY = os.path.join(DATA_ROOT, "ba_archive", "imports.jsonl")

NOT_FOUND_REASON = "для этого файла запрос не сохранён (создан до появления учёта запросов)"
BAD_USER_REASON = "нет доступа к задачам другого пользователя"

# Сколько свежих запусков просматриваем и насколько долго кэшируем ответ.
MAX_RUNS_SCAN = 600
STOP_SCORE = 95
CACHE_TTL_SEC = 120
CACHE_MAX = 400
# Разобранные записи запусков: файл читается один раз на версию (mtime+size).
RUN_CACHE_MAX = 900
SUMMARY_TTL_SEC = 90
SUMMARY_CACHE_MAX = 64

# Порог «уверенного» совпадения: слабее — считаем, что запрос не найден.
MIN_QUERY_SCORE = 62
MIN_SUMMARY_SCORE = 80
MIN_SUMMARY_EXACT = 96

FILE_EXTS = (
    ".docx", ".doc", ".pdf", ".xlsx", ".xls", ".csv", ".tsv", ".json",
    ".txt", ".md", ".html", ".png", ".jpg", ".jpeg", ".pptx", ".zip",
)

DATE_SUFFIX_RE = re.compile(r"[_\- ]?\d{8}[_\-]\d{4,6}(?!\d)")
DATE_DOTTED_RE = re.compile(r"[_\- ]?\d{2}[.\-]\d{2}[.\-]\d{4}(?!\d)")
SUMMARY_SUFFIX = "_summary.json"

MONTHS_NOM = {
    1: "январь", 2: "февраль", 3: "март", 4: "апрель", 5: "май", 6: "июнь",
    7: "июль", 8: "август", 9: "сентябрь", 10: "октябрь", 11: "ноябрь", 12: "декабрь",
}
MONTHS_GEN = {
    1: "января", 2: "февраля", 3: "марта", 4: "апреля", 5: "мая", 6: "июня",
    7: "июля", 8: "августа", 9: "сентября", 10: "октября", 11: "ноября", 12: "декабря",
}

_CACHE: Dict[Tuple[str, str, str], Tuple[float, Dict[str, Any]]] = {}
_CACHE_LOCK = threading.Lock()
_RUN_CACHE: Dict[str, Tuple[float, int, Dict[str, Any]]] = {}
_RUN_CACHE_LOCK = threading.Lock()
_SUMMARY_CACHE: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}
_SUMMARY_LOCK = threading.Lock()


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
    root = DATE_SUFFIX_RE.sub("", root)
    root = DATE_DOTTED_RE.sub("", root)
    return _norm(root)


def _looks_like_file(value: str) -> bool:
    low = value.lower()
    return any(low.endswith(ext) for ext in FILE_EXTS)


def _parse_stamp(value: Any) -> Optional[float]:
    text = str(value or "").strip()
    if not text:
        return None
    head = text[:19]
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return time.mktime(time.strptime(head[: len(fmt)], fmt))
        except Exception:
            continue
    return None


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
    """Насколько строка ``name`` похожа на искомый файл. Возвращает (балл, способ).

    Папка никогда не требуется: она только добавляет ``+6`` к уже найденному совпадению,
    поэтому переезд/переименование папки совпадение не ломает.
    """
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
# источники: запуски
# --------------------------------------------------------------------------- #
def _load_run(path: str) -> Optional[Dict[str, Any]]:
    """Разобранная запись запуска с кэшем по (mtime, size) — иначе каждый вызов
    перечитывал бы сотни JSON-файлов (кнопка проверяет запрос для каждой строки)."""
    try:
        stat = os.stat(path)
    except OSError:
        return None
    with _RUN_CACHE_LOCK:
        hit = _RUN_CACHE.get(path)
    if hit and hit[0] == stat.st_mtime and hit[1] == stat.st_size:
        return hit[2]
    try:
        with open(path, "r", encoding="utf-8") as fh:
            run = json.load(fh)
    except Exception:
        return None
    if not isinstance(run, dict):
        return None
    with _RUN_CACHE_LOCK:
        if len(_RUN_CACHE) > RUN_CACHE_MAX:
            _RUN_CACHE.clear()
        _RUN_CACHE[path] = (stat.st_mtime, stat.st_size, run)
    return run


def _iter_run_paths() -> List[str]:
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


def _iter_runs(user_id: str) -> Iterator[Dict[str, Any]]:
    for path in _iter_run_paths():
        run = _load_run(path)
        if not run:
            continue
        if str(run.get("user_id")) != str(user_id):
            continue
        yield run


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


def _run_model(run: Dict[str, Any]) -> Optional[str]:
    for key in ("model_label", "analysis_model_label", "model_choice", "model"):
        value = run.get(key)
        if value:
            return str(value)
    return None


def _task_payload(task: Dict[str, Any], how: str, source: str) -> Dict[str, Any]:
    return {
        "found": True,
        "kind": "query",
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


def _run_payload(run: Dict[str, Any], how: str) -> Dict[str, Any]:
    return {
        "found": True,
        "kind": "query",
        "task_text": str(run.get("task") or run.get("task_text") or "").strip(),
        "task_id": run.get("task_id"),
        "run_id": run.get("run_id"),
        "created_at": run.get("created_at") or run.get("started_at"),
        "model": _run_model(run),
        "status": run.get("status"),
        "source": "agent_run",
        "matched_by": how,
        "reason": "",
    }


def _task_for_run(tasks: List[Dict[str, Any]], run_id: Any) -> Optional[Dict[str, Any]]:
    if not run_id:
        return None
    for task in tasks:
        if task.get("run_id") and str(task.get("run_id")) == str(run_id):
            return task
    return None


def _match_runs(user_id: str, base_target: str, norm_target: str, title_target: str,
                folder: str) -> Optional[Tuple[int, Dict[str, Any], str]]:
    best: Optional[Tuple[int, Dict[str, Any], str]] = None
    for run in _iter_runs(user_id):
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


def _artifact_folder(run: Dict[str, Any], base_target: str) -> str:
    """Папка, в которой файл был создан (по артефактам запуска) — для подсказки о переезде."""
    target_norm = _norm(base_target)
    found: List[str] = []

    def walk(node: Any, depth: int = 0) -> None:
        if depth > 8 or found:
            return
        if isinstance(node, dict):
            name = _basename(node.get("name") or node.get("title") or node.get("path")
                             or node.get("url") or "")
            if name and _norm(name) == target_norm:
                meta = node.get("meta") if isinstance(node.get("meta"), dict) else {}
                folder = str(meta.get("folder") or "").strip()
                if not folder:
                    path = str(node.get("path") or node.get("url") or "").replace("\\", "/")
                    if "/" in path:
                        folder = _basename(path.rsplit("/", 1)[0])
                if folder:
                    found.append(folder)
                    return
            for value in node.values():
                walk(value, depth + 1)
            return
        if isinstance(node, (list, tuple)):
            for value in node:
                walk(value, depth + 1)

    walk(run.get("artifacts"))
    if not found:
        walk(run.get("files"))
    return found[0] if found else ""


# --------------------------------------------------------------------------- #
# источники: структурные итоги отчётов (<ГГГГ-ММ>_summary.json)
# --------------------------------------------------------------------------- #
def _reports_root(user_id: str) -> str:
    return os.path.join(DATA_ROOT, str(user_id), "reports_directory")


def _period_label(period: Dict[str, Any]) -> str:
    if not isinstance(period, dict):
        return ""
    start = str(period.get("from") or "").strip()
    end = str(period.get("to") or "").strip()
    key = str(period.get("key") or "").strip()
    if start and end:
        if start[:10] == end[:10]:
            return start[:10]
        return "%s – %s" % (start[:16], end[:16])
    return key


def _period_keywords(key: str) -> List[str]:
    """Ключевые слова для поиска запуска по периоду («2026-08» → «август2026»)."""
    text = str(key or "").strip()
    out: List[str] = []
    match = re.match(r"^(\d{4})-(\d{2})$", text)
    if match:
        year, month = int(match.group(1)), int(match.group(2))
        name = MONTHS_NOM.get(month)
        if name:
            out.append(_norm("%s %d" % (name, year)))
            out.append(_norm("%s %d" % (MONTHS_GEN[month], year)))
    return [item for item in out if item]


def _summary_entries(user_id: str) -> List[Dict[str, Any]]:
    """Все структурные итоги пользователя во всех его папках отчётов."""
    now = time.time()
    with _SUMMARY_LOCK:
        hit = _SUMMARY_CACHE.get(str(user_id))
    if hit and now - hit[0] <= SUMMARY_TTL_SEC:
        return hit[1]

    entries: List[Dict[str, Any]] = []
    root = _reports_root(user_id)
    if os.path.isdir(root):
        for dirpath, _dirnames, filenames in os.walk(root):
            for name in filenames:
                if not name.lower().endswith(SUMMARY_SUFFIX):
                    continue
                path = os.path.join(dirpath, name)
                try:
                    with open(path, "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                except Exception:
                    continue
                if not isinstance(data, dict):
                    continue
                report = data.get("report") if isinstance(data.get("report"), dict) else {}
                period = data.get("period") if isinstance(data.get("period"), dict) else {}
                dataset = data.get("dataset") if isinstance(data.get("dataset"), dict) else {}
                raw_files = report.get("files")
                entries.append({
                    "name": name,
                    "key": str(period.get("key") or "").strip(),
                    "period": period,
                    "title": str(report.get("title") or "").strip(),
                    "folder": str(report.get("folder") or "").strip() or os.path.basename(dirpath),
                    "location": os.path.basename(dirpath),
                    "files": [str(f) for f in raw_files if f] if isinstance(raw_files, list) else [],
                    "dataset": str(dataset.get("label") or dataset.get("name") or "").strip(),
                    "generated_at": data.get("generated_at"),
                    "generated_ts": _parse_stamp(data.get("generated_at")),
                })
    entries.sort(key=lambda item: item.get("generated_ts") or 0, reverse=True)
    with _SUMMARY_LOCK:
        if len(_SUMMARY_CACHE) > SUMMARY_CACHE_MAX:
            _SUMMARY_CACHE.clear()
        _SUMMARY_CACHE[str(user_id)] = (now, entries)
    return entries


def _query_for_report(user_id: str, entry: Dict[str, Any],
                      tasks: List[Dict[str, Any]]
                      ) -> Optional[Tuple[Dict[str, Any], Optional[Dict[str, Any]], str]]:
    """Запуск/задача, породившие отчёт, которому принадлежит структурный итог."""
    folder = entry.get("folder") or ""

    # 1) точное имя файла из итога — самый надёжный путь
    for fname in entry.get("files") or []:
        base = _basename(fname)
        if not base:
            continue
        match = _match_runs(user_id, base, _norm(base), _norm_title(base), folder)
        if match and match[0] >= MIN_SUMMARY_EXACT:
            return match[1], _task_for_run(tasks, match[1].get("run_id")), "точное имя файла отчёта"

    # 2) заголовок отчёта
    title = entry.get("title") or ""
    if title:
        probe = title + ".docx"
        match = _match_runs(user_id, _basename(probe), _norm(probe), _norm_title(probe), folder)
        if match and match[0] >= 70:
            return match[1], _task_for_run(tasks, match[1].get("run_id")), "заголовок отчёта"

    # 3) период отчёта («август 2026» в тексте задачи), ближайший к дате сборки итога
    keywords = _period_keywords(entry.get("key") or "")
    if keywords:
        candidates: List[Dict[str, Any]] = []
        for run in _iter_runs(user_id):
            text = _norm(str(run.get("task") or ""))
            if text and any(word in text for word in keywords):
                candidates.append(run)
        if candidates:
            if len(candidates) > 1 and entry.get("generated_ts"):
                def distance(run: Dict[str, Any]) -> float:
                    stamp = _parse_stamp(run.get("created_at") or run.get("started_at")) or 0.0
                    return abs(stamp - float(entry["generated_ts"]))

                candidates.sort(key=distance)
            run = candidates[0]
            return run, _task_for_run(tasks, run.get("run_id")), "период отчёта"
    return None


def _match_summary_self(user_id: str, base_target: str, folder: str,
                        tasks: List[Dict[str, Any]]
                        ) -> Optional[Tuple[int, Dict[str, Any], str]]:
    """Файл сам является структурным итогом (``2026-08_summary.json``)."""
    if not base_target.lower().endswith(SUMMARY_SUFFIX):
        return None
    base_norm = _norm(base_target)
    entries = [item for item in _summary_entries(user_id)
               if _norm(item.get("name")) == base_norm]
    if not entries:
        return None
    entry = entries[0]
    if folder:
        for item in entries:
            if item.get("location") == folder:
                entry = item
                break
    return 97, entry, "структурный итог %s" % entry.get("name")


def _match_summary_structure(user_id: str, base_target: str, norm_target: str,
                             title_target: str, folder: str
                             ) -> Optional[Tuple[int, Dict[str, Any], str]]:
    """Файл перечислен в ``report.files`` или равен ``report.title`` структурного итога."""
    best: Optional[Tuple[int, Dict[str, Any], str]] = None
    for entry in _summary_entries(user_id):
        score, how = 0, ""
        for fname in entry.get("files") or []:
            base = _basename(fname)
            if not base:
                continue
            if base == base_target:
                score, how = 99, "файл из %s" % entry.get("name")
                break
            if _norm(base) == norm_target:
                score, how = 95, "файл из %s (нормализовано)" % entry.get("name")
                break
        if not score and entry.get("title"):
            entry_title = _norm_title(entry["title"])
            if entry_title and entry_title == title_target:
                score, how = 88, "заголовок отчёта из %s" % entry.get("name")
        if not score:
            continue
        if folder and entry.get("location") == folder:
            score += 3
            how += " + папка"
        if best is None or score > best[0]:
            best = (score, entry, how)
    return best


def _summary_payload(entry: Dict[str, Any], how: str) -> Dict[str, Any]:
    """Происхождение известно, отдельного текстового запроса нет (итог сборщика)."""
    period = _period_label(entry.get("period") or {})
    parts = ["Структурный итог отчёта «%s»" % (entry.get("title") or "без названия")]
    if period:
        parts.append("период %s" % period)
    if entry.get("dataset"):
        parts.append("датасет «%s»" % entry["dataset"])
    where = entry.get("location") or entry.get("folder")
    if where:
        parts.append("папка «%s»" % where)
    if entry.get("generated_at"):
        parts.append("собран %s" % entry["generated_at"])
    if entry.get("files"):
        parts.append("описывает файлы: %s" % ", ".join(entry["files"][:4]))
    text = "; ".join(parts) + ". Текстовый запрос пользователя для этого файла не сохранён."
    return {
        "found": True,
        "kind": "summary",
        "task_text": text,
        "task_id": None,
        "run_id": None,
        "created_at": entry.get("generated_at"),
        "model": None,
        "status": None,
        "source": "report_summary",
        "matched_by": how,
        "report_title": entry.get("title") or "",
        "period": entry.get("key") or "",
        "period_label": period,
        "reason": "",
    }


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
        "kind": "query",
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
# поиск происхождения
# --------------------------------------------------------------------------- #
def resolve_origin(user_id: str, file: str, folder: str = "") -> Dict[str, Any]:
    """Собрать ответ «по какому запросу появился файл».

    Порядок источников — от самого точного к самому общему: структурный итог того же
    файла → артефакты запуска агента → запись в структурном итоге → задача центра задач
    → реестр импортов Brand Analytics → честное ``found: false``.
    """
    base_target = _basename(file)
    folder_clean = _basename(folder.replace("\\", "/").rstrip("/")) if folder else ""
    if folder and not folder_clean:
        folder_clean = str(folder).strip()

    cache_key = (str(user_id), folder_clean, base_target)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    norm_target = _norm(base_target)
    title_target = _norm_title(base_target)
    tasks = _load_harness_tasks(str(user_id))

    payload: Optional[Dict[str, Any]] = None
    moved_from = ""

    # 1) файл сам является структурным итогом отчёта — самый точный сигнал
    if payload is None:
        self_match = _match_summary_self(str(user_id), base_target, folder_clean, tasks)
        if self_match:
            score, entry, how = self_match
            query = _query_for_report(str(user_id), entry, tasks)
            if query:
                run, task, query_how = query
                if task is not None:
                    payload = _task_payload(task, "%s → %s" % (how, query_how), "harness_task")
                    payload["run_id"] = run.get("run_id")
                    payload["status"] = task.get("status") or run.get("status")
                    payload["model"] = task.get("model") or _run_model(run)
                    payload["created_at"] = task.get("created_at") or run.get("created_at")
                else:
                    payload = _run_payload(run, "%s → %s" % (how, query_how))
                payload["report_title"] = entry.get("title") or ""
                payload["period"] = entry.get("key") or ""
                payload["period_label"] = _period_label(entry.get("period") or {})
            else:
                payload = _summary_payload(entry, how)

    # 2) имя файла встречается в артефактах запуска агента
    if payload is None:
        run_match = _match_runs(str(user_id), base_target, norm_target, title_target, folder_clean)
        if run_match and run_match[0] >= MIN_QUERY_SCORE:
            score, run, how = run_match
            task = _task_for_run(tasks, run.get("run_id"))
            if task is not None:
                payload = _task_payload(task, how + " + запуск", "harness_task")
                payload["run_id"] = run.get("run_id")
                payload["status"] = task.get("status") or run.get("status")
                payload["model"] = task.get("model") or _run_model(run)
                payload["created_at"] = task.get("created_at") or run.get("created_at")
            else:
                payload = _run_payload(run, how)
            moved_from = _artifact_folder(run, base_target)

    # 3) файл назван в структурном итоге (в т.ч. после переезда/переименования)
    if payload is None:
        summary_match = _match_summary_structure(
            str(user_id), base_target, norm_target, title_target, folder_clean
        )
        if summary_match and summary_match[0] >= MIN_SUMMARY_SCORE:
            score, entry, how = summary_match
            query = _query_for_report(str(user_id), entry, tasks)
            if query:
                run, task, query_how = query
                if task is not None:
                    payload = _task_payload(task, "%s → %s" % (how, query_how), "harness_task")
                    payload["run_id"] = run.get("run_id")
                    payload["status"] = task.get("status") or run.get("status")
                    payload["model"] = task.get("model") or _run_model(run)
                    payload["created_at"] = task.get("created_at") or run.get("created_at")
                else:
                    payload = _run_payload(run, "%s → %s" % (how, query_how))
                payload["report_title"] = entry.get("title") or ""
                payload["period"] = entry.get("key") or ""
                payload["period_label"] = _period_label(entry.get("period") or {})
            else:
                payload = _summary_payload(entry, how)

    # 4) имя файла встречается в задаче центра задач
    if payload is None:
        task_match = _match_harness(tasks, base_target, norm_target, title_target, folder_clean)
        if task_match and task_match[0] >= MIN_QUERY_SCORE:
            payload = _task_payload(task_match[1], task_match[2] + " + задача", "harness_task")

    # 5) файл датасета из Brand Analytics
    if payload is None:
        payload = _match_ba(str(user_id), base_target, norm_target, title_target, folder_clean)

    # 6) ничего не сохранено — честный отказ
    if payload is None:
        payload = {
            "found": False,
            "kind": "none",
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
    if moved_from:
        payload["created_in_folder"] = moved_from
        payload["moved"] = bool(folder_clean) and _norm(moved_from) != _norm(folder_clean)
    _cache_put(cache_key, payload)
    return payload


@router.get("/file-origin")
async def file_origin(
    request: Request,
    user_id: str = Query(..., description="владелец файла"),
    file: str = Query(..., description="имя файла, например Отчёт_20260912_2054.docx"),
    folder: str = Query("", description="папка отчёта/датасета (только подсказка для оценки)"),
):
    """По какому запросу (задаче) появился файл.

    Без токена — 401, при чужом ``user_id`` — 403. Если происхождение не сохранено
    (файл создан раньше появления функции) — ``found: false`` с причиной.
    """
    caller = _authorized_user(request)
    if caller and str(caller) != str(user_id):
        raise HTTPException(status_code=403, detail=BAD_USER_REASON)

    if not _basename(file):
        raise HTTPException(status_code=400, detail="Не указано имя файла")
    return resolve_origin(str(user_id), file, folder)
