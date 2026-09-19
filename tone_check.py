# -*- coding: utf-8 -*-
"""Проверка тональности загруженных данных локальной моделью на vLLM.

Что делает модуль
-----------------
Кнопка «Проверить тональность» в интерфейсе запускает фоновую разметку выбранного
датасета двумя проходами:

1. быстрая модель ``qwen3-4b-fast`` (профиль ``texts_bulk`` из ``mlops/lock.yaml``,
   127.0.0.1:8001) размечает сообщения пачками по 20 — строгий JSON
   ``{"results": [{"id": ..., "tone": "negative|neutral|positive", "confidence": 0..1,
   "reason": "до 100 знаков"}]}``;
2. спорные случаи (уверенность ниже порога ИЛИ расхождение с полем источника
   ``toneMark``) уходят вторым проходом на ``Qwen/Qwen3-32B-FP8`` (127.0.0.1:8000) —
   она решает окончательно.

Результат на сообщение — отдельные поля, исходное ``toneMark`` НЕ изменяется:
``tone_llm`` (-1/0/1), ``tone_llm_conf`` (0..1), ``tone_llm_by`` ("4b"/"32b"/"none"),
``tone_llm_reason`` (до 100 знаков), ``tone_llm_at`` (unix-время разметки).

Если 32B недоступна, перепроверка честно помечается как невыполненная: 4B-решение
остаётся в ``tone_llm``, но в отчёте это видно, и оно не выдаётся за окончательное.

Фон и возобновление
-------------------
Работа идёт в фоновом потоке процесса FastAPI (как BA-импорт), состояние — в
``data/tone_check/<job_id>.json`` плюс журнал ``<job_id>.jsonl``. Прогресс пишется в
Redis (``tone:job:<job_id>``) и отдаётся эндпоинтом статуса, поэтому закрытие браузера
проход не прерывает. Если сервис перезапустили в середине, запись помечается
``interrupted``, а при старте приложения (или при первом обращении к API) проход
ВОЗОБНОВЛЯЕТСЯ с места остановки: выборка пачки исключает уже размеченные документы
(``must_not exists tone_llm``), а курсор ``search_after`` хранится в файле задачи.

Эндпоинты
---------
* ``POST   /tone-check``                       — запуск (index, mode, sample_size, даты, tone);
* ``GET    /tone-check/jobs``                  — свои задачи проверки;
* ``GET    /tone-check/datasets``              — доступные пользователю датасеты;
* ``GET    /tone-check/{job_id}``              — статус, прогресс, этап, статистика;
* ``GET    /tone-check/{job_id}/report``       — итоговый отчёт (JSON);
* ``GET    /tone-check/{job_id}/report/file``  — отчёт DOCX/PDF;
* ``POST   /tone-check/{job_id}/cancel``       — кооперативная остановка.

Модуль самостоятельный (собственный ``APIRouter``), в ``main.py`` подключается одной
строкой ``app.include_router(tone_check_router)``. ``main`` не импортируется.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import redis
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from access_guard import current_user_any

# --------------------------------------------------------------------------- #
# Константы
# --------------------------------------------------------------------------- #

BACKEND_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BACKEND_ROOT, "data")
INDEXES_PKL = os.path.join(DATA_DIR, "indexes.pkl")
STATE_DIR = os.path.join(DATA_DIR, "tone_check")
REPORTS_DIR_NAME = "reports_directory"
REPORT_FOLDER = "Проверка тональности"

TONE_FIELDS = ("tone_llm", "tone_llm_conf", "tone_llm_by", "tone_llm_reason", "tone_llm_at")

TONE_TO_INT = {"negative": -1, "neutral": 0, "positive": 1}
TONE_RU = {-1: "негатив", 0: "нейтрал", 1: "позитив"}
CLASSES = (-1, 0, 1)

# Человеческие подписи происхождения вердикта — только для отчёта, который читает
# пользователь. В полях Elasticsearch (``tone_llm_by`` / ``tone_aspect_by``) остаются
# технические значения «4b»/«32b»/«none».
BY_RU = {
    "4b": "первичная автоматическая разметка",
    "32b": "уточняющая проверка",
    "none": "без вердикта автоматической разметки",
}

# Модели: быстрая 4B (порт 8001, профиль texts_bulk) и общая 32B (порт 8000).
DEFAULT_BATCH = 20
DEFAULT_PARALLEL = 12
DEFAULT_PASS2_BATCH = 10
DEFAULT_PARALLEL_PASS2 = 3
DEFAULT_CONF = 0.6
PASS1_MESSAGE_CHARS = 400
PASS2_MESSAGE_CHARS = 600
PASS1_MAX_TOKENS = 2500
PASS2_MAX_TOKENS = 2000
PASS1_CHAR_BUDGET = 9000
PASS2_CHAR_BUDGET = 6500
PAGE_SIZE = 200
BULK_CHUNK = 500
MAX_EXAMPLES = 25

# Полная разметка очень большого индекса — это часы работы и сотни мегабайт журнала.
# По умолчанию разрешаем «полностью» до 300 000 сообщений: дальше нужен срез по датам.
MAX_FULL_DOCS = int(os.environ.get("TELLSCOPE_TONE_FULL_LIMIT") or 300000)

# Перепроверка на 32B в разы медленнее первого прохода, поэтому её доля ограничена: если
# спорных больше, чем PASS2_MAX_SHARE от объёма задачи, перепроверяем самые неуверенные
# (сортировка по возрастанию уверенности), остальные остаются с решением 4B. Факт и размер
# лимита попадают в отчёт (pass2_capped / pass2_max_share), молча ничего не отбрасывается.
PASS2_MAX_SHARE = float(os.environ.get("TELLSCOPE_TONE_PASS2_MAX_SHARE") or 0.5)

# Жёсткий предел на один вызов модели: зависший запрос не должен держать пачку и всю
# страницу минутами. По таймауту пачка повторяется, затем делится пополам.
BATCH_TIMEOUT = float(os.environ.get("TELLSCOPE_TONE_BATCH_TIMEOUT") or 180.0)

# Через сколько секунд без прироста счётчиков честно писать «нет новых результатов».
STALE_AFTER = float(os.environ.get("TELLSCOPE_TONE_STALE_AFTER") or 120.0)

# Шкала общего процента: подготовка → первый проход → перепроверка спорных → отчёт.
WORK_PREPARE = 1
WORK_PASS1_END = 62
WORK_PASS2_END = 93
WORK_REPORT_END = 96

# Версия схемы расчёта прогресса. Задачи, записанные прежней формулой (постоянные 63% на
# шаге записи в ES), помечены другой версией — их залипший процент пересчитывается честно.
PROGRESS_SCHEMA = 2

# Потолок полей индекса: у датасетов с динамическим маппингом 3000 полей уже выбраны
# на загрузке данных, и без поднятия лимита наши поля физически не создать.
TOTAL_FIELDS_LIMIT = int(os.environ.get("TELLSCOPE_TONE_FIELD_LIMIT") or 4000)

STAGE_LABELS = {
    "preparing": "читаю сообщения",
    "pass1": "определяю тональность",
    "writing": "сохраняю результат",
    "pass2": "перепроверяю спорные случаи",
    "report": "собираю отчёт",
    "done": "готово",
    "cancelled": "остановлено",
    "error": "не получилось",
}

LOG_TAIL = 60

_router_lock = threading.Lock()
_RECOVERY_STARTED = False


# --------------------------------------------------------------------------- #
# Elasticsearch
# --------------------------------------------------------------------------- #

_es_lock = threading.Lock()
_es_client = None
_mapping_cache: Dict[str, Dict[str, Any]] = {}


def _es():
    """Клиент Elasticsearch (тот же узел и креды, что у остального приложения)."""
    global _es_client
    with _es_lock:
        if _es_client is not None:
            return _es_client
        try:
            import config  # noqa: F401 — load_dotenv(): supervisor .env не подставляет
        except Exception:
            pass
        from elasticsearch import Elasticsearch

        host = os.environ.get("ELASTICSEARCH_HOST") or "localhost"
        port = os.environ.get("ELASTICSEARCH_PORT") or "9200"
        user = os.environ.get("ELASTICSEARCH_USER") or ""
        pwd = os.environ.get("ELASTICSEARCH_PASSWORD") or ""
        kwargs: Dict[str, Any] = {
            "hosts": ["http://%s:%s" % (host, port)],
            "verify_certs": False,
            "headers": {"Accept": "application/vnd.elasticsearch+json; compatible-with=9"},
            "request_timeout": 120,
        }
        if user and pwd:
            kwargs["basic_auth"] = (user, pwd)
        _es_client = Elasticsearch(**kwargs)
        return _es_client


def _mapping(name: str) -> Dict[str, Any]:
    """properties индекса (кэш в памяти процесса)."""
    if name in _mapping_cache:
        return _mapping_cache[name]
    props: Dict[str, Any] = {}
    try:
        res = _es().indices.get_mapping(index=name)
        for body in res.values():
            props = ((body.get("mappings") or {}).get("properties") or {})
            break
    except Exception:
        props = {}
    _mapping_cache[name] = props
    return props


def _text_field(name: str) -> str:
    """Фактическое поле текста сообщения: text / msgText / «Текст сообщения»."""
    props = _mapping(name)
    for candidate in ("text", "msgText", "Текст сообщения", "content"):
        if candidate in props:
            return candidate
    return "text"


def _id_field(name: str) -> str:
    """Поле-идентификатор для сортировки (нужно для устойчивого search_after)."""
    props = _mapping(name)
    for candidate in ("id", "docId", "idExternal"):
        if candidate in props:
            return candidate
    return ""


def _ensure_mapping(name: str) -> None:
    """Создаёт поля нашей разметки. Существующие поля не трогаем (additive)."""
    props = {
        "tone_llm": {"type": "long"},
        "tone_llm_conf": {"type": "float"},
        "tone_llm_by": {"type": "keyword"},
        "tone_llm_reason": {"type": "text"},
        "tone_llm_at": {"type": "long"},
        # Аспектная разметка: отношение к объекту, а не тон сообщения целиком.
        "tone_aspect": {"type": "nested", "properties": {
            "object": {"type": "keyword"},
            "tone": {"type": "long"},
            "confidence": {"type": "float"},
            "reason": {"type": "text"},
            "quote": {"type": "text"},
            "by": {"type": "keyword"},
        }},
        "tone_aspect_objects": {"type": "keyword"},
        "tone_aspect_at": {"type": "long"},
        "tone_aspect_by": {"type": "keyword"},
        # Общий тон сообщения из аспектного прохода: отдельные поля, чтобы не смешивать
        # происхождение с tone_llm* и не блокировать обычный режим.
        "tone_msg_llm": {"type": "long"},
        "tone_msg_conf": {"type": "float"},
        "tone_msg_by": {"type": "keyword"},
        "tone_msg_reason": {"type": "text"},
    }
    try:
        _es().indices.put_mapping(index=name, properties=props)
        _mapping_cache.pop(name, None)
        return
    except Exception as exc:  # noqa: BLE001 — маппинг уже мог быть создан или поле конфликтует
        text = str(exc)
        if "Limit of total fields" not in text:
            _log_line("маппинг полей тональности не применён: %s" % text[:200], level="warning")
            return
    # Индексы с динамическим маппингом (например kfc_* с 2,9 млн сообщений) выбирают лимит
    # полей (3000) ещё на загрузке данных, и тогда put_mapping падает целиком: разметка не
    # записалась бы вообще, а проход молча крутился бы на месте. Поднимаем потолок и повторяем.
    try:
        _es().indices.put_settings(
            index=name,
            settings={"index": {"mapping": {"total_fields": {"limit": TOTAL_FIELDS_LIMIT}}}})
        _es().indices.put_mapping(index=name, properties=props)
        _mapping_cache.pop(name, None)
        _log_line("у индекса %s выбран лимит полей — поднял его до %d, чтобы записать поля разметки"
                  % (name, TOTAL_FIELDS_LIMIT), level="warning")
    except Exception as exc2:  # noqa: BLE001
        _log_line("маппинг полей тональности не применён: %s" % str(exc2)[:200], level="error")


# --------------------------------------------------------------------------- #
# Redis и файлы состояния
# --------------------------------------------------------------------------- #

_redis = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)


def _key(jid: str) -> str:
    return "tone:job:%s" % jid


def _redis_put(jid: str, fields: Dict[str, Any]) -> None:
    try:
        _redis.hset(_key(jid), mapping={k: str(v) for k, v in fields.items()})
        _redis.expire(_key(jid), 7 * 24 * 3600)
    except Exception:
        pass


def _redis_get(jid: str) -> Dict[str, str]:
    try:
        return _redis.hgetall(_key(jid)) or {}
    except Exception:
        return {}


def _job_path(jid: str) -> str:
    return os.path.join(STATE_DIR, "%s.json" % jid)


def _journal_path(jid: str) -> str:
    return os.path.join(STATE_DIR, "%s.jsonl" % jid)


def _report_path(jid: str) -> str:
    return os.path.join(STATE_DIR, "%s.report.json" % jid)


def _job_load(jid: str) -> Optional[Dict[str, Any]]:
    path = _job_path(jid)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def _job_save(job: Dict[str, Any]) -> None:
    os.makedirs(STATE_DIR, exist_ok=True)
    job["updated"] = datetime.now().isoformat(timespec="seconds")
    tmp = _job_path(job["id"]) + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(job, handle, ensure_ascii=False, indent=1)
    os.replace(tmp, _job_path(job["id"]))


def _log_line(message: str, level: str = "info") -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    print("[tone-check %s] %s" % (stamp, message), flush=True)


# --------------------------------------------------------------------------- #
# Задача: состояние, журнал, прогресс
# --------------------------------------------------------------------------- #

def _set(jid: str, **fields: Any) -> Dict[str, Any]:
    """Обновляет состояние задачи и в файле, и в Redis (для быстрого статуса)."""
    job = _job_load(jid) or {"id": jid}
    log = fields.pop("log", None)
    if log:
        lines = list(job.get("log") or [])
        lines.append("%s %s" % (datetime.now().strftime("%H:%M:%S"), log))
        job["log"] = lines[-LOG_TAIL:]
    previous = int(job.get("percent") or 0)
    job.update(fields)
    job["id"] = jid
    if "percent" not in fields and any(
            key in fields for key in ("processed", "total", "stage", "pass2_done", "pass2_total")):
        fresh = _percent(job)
        if int(job.get("progress_schema") or 0) == PROGRESS_SCHEMA:
            # Процент только растёт: переходы между шагами не откатывают полосу назад.
            job["percent"] = max(previous, fresh)
        else:
            # Задача из старой схемы: залипшие 63% нельзя тянуть дальше — считаем заново.
            job["percent"] = fresh
    _note_progress(job)
    _job_save(job)
    _redis_put(jid, {
        "status": job.get("status", ""),
        "stage": job.get("stage", ""),
        "stage_label": job.get("stage_label") or STAGE_LABELS.get(job.get("stage", ""), ""),
        "percent": job.get("percent", 0),
        "processed": job.get("processed", 0),
        "total": job.get("total", 0),
        "updated": job.get("updated", ""),
    })
    return job


def _stage_progress(job: Dict[str, Any]) -> Dict[str, Any]:
    """Прогресс по этапам: сколько сделано из скольких ИМЕННО на этом этапе.

    Процент этапа всегда совпадает со счётчиком «N / M», который видит пользователь.
    Запись в Elasticsearch (``writing``) — это под-шаг, а не отдельный этап: у неё нет
    собственного счётчика, поэтому и своего процента быть не должно.
    """
    stage = str(job.get("stage") or "preparing")
    p1_total = int(job.get("total") or 0)
    p1_done = int(job.get("processed") or 0)
    if p1_total:
        p1_done = min(p1_done, p1_total)
    p2_total = int(job.get("pass2_total") or 0)
    p2_done = int(job.get("pass2_done") or 0)
    if p2_total:
        p2_done = min(p2_done, p2_total)
    if stage == "pass2":
        cur_done, cur_total = p2_done, p2_total
    else:
        cur_done, cur_total = p1_done, p1_total
    return {
        "stage": stage,
        "stage_label": job.get("stage_label") or STAGE_LABELS.get(stage, ""),
        "stage_done": cur_done,
        "stage_total": cur_total,
        "stage_percent": int(round(100.0 * cur_done / cur_total)) if cur_total else 0,
        "pass1_done": p1_done,
        "pass1_total": p1_total,
        "pass1_percent": int(round(100.0 * p1_done / p1_total)) if p1_total else 0,
        "pass2_done": p2_done,
        "pass2_total": p2_total,
        "pass2_percent": int(round(100.0 * p2_done / p2_total)) if p2_total else 0,
    }


def _percent(job: Dict[str, Any]) -> int:
    """Общий процент: взвешенная сумма работы обоих проходов.

    Работа измеряется документами: N сообщений первого прохода плюс K спорных второго.
    Пока объём второго прохода неизвестен, первый проход занимает шкалу 2..62%.

    Здесь больше НЕТ постоянного процента для шага записи в ES. Раньше ``writing``
    возвращал жёсткие 63%, а ``_set`` держит процент только растущим — поэтому полоса
    навсегда замирала на 63%, пока счётчик «N / M сообщений» шёл вперёд.
    """
    stage = str(job.get("stage") or "preparing")
    if stage == "writing":
        # Состояние задач, записанных до этой правки: уточняем этап по объёму второго прохода.
        stage = "pass2" if int(job.get("pass2_total") or 0) else "pass1"
    prog = _stage_progress(job)
    if stage == "done":
        return 100
    if stage == "cancelled":
        return int(job.get("percent") or 0)
    if stage in ("preparing", "queued"):
        return WORK_PREPARE
    if stage == "report":
        return WORK_REPORT_END
    p1_done, p1_total = prog["pass1_done"], prog["pass1_total"]
    p2_done, p2_total = prog["pass2_done"], prog["pass2_total"]
    if stage == "pass1":
        if p1_total <= 0:
            return WORK_PREPARE
        span = WORK_PASS1_END - WORK_PREPARE
        return min(WORK_PASS1_END, WORK_PREPARE + int(span * p1_done / p1_total))
    if stage == "pass2":
        if p1_total <= 0:
            return WORK_PASS1_END
        if p2_total <= 0:
            return WORK_PASS1_END + 1
        span = WORK_PASS2_END - WORK_PREPARE
        return min(WORK_PASS2_END,
                   WORK_PREPARE + int(span * (p1_done + p2_done) / max(1, p1_total + p2_total)))
    return WORK_PREPARE


def _note_progress(job: Dict[str, Any]) -> Dict[str, Any]:
    """Запоминает момент реального прироста: без него интерфейс не отличит «идёт» от «висит».

    Работа = размеченные сообщения первого прохода + перепроверенные второго. Точки
    хранятся коротким кольцом, по ним считается живая скорость и оценка остатка.
    """
    now = time.time()
    done = int(job.get("processed") or 0) + int(job.get("pass2_done") or 0)
    samples = [s for s in (job.get("progress_samples") or []) if isinstance(s, (list, tuple)) and len(s) == 2]
    if not samples or int(samples[-1][1]) != done:
        samples.append([round(now, 1), done])
        samples = samples[-60:]
    job["progress_samples"] = samples
    try:
        job["last_progress_at"] = datetime.fromtimestamp(float(samples[-1][0])).isoformat(timespec="seconds")
    except Exception:
        job["last_progress_at"] = job.get("updated")
    return job


def _rate_and_eta(job: Dict[str, Any], prog: Dict[str, Any]) -> Tuple[float, Optional[int]]:
    """Средняя скорость (сообщений/мин) за последние минуты и оценка остатка в секундах."""
    now = time.time()
    window = [s for s in (job.get("progress_samples") or [])
              if isinstance(s, (list, tuple)) and len(s) == 2 and now - float(s[0]) <= 300]
    rate = 0.0
    if len(window) >= 2:
        dt = float(window[-1][0]) - float(window[0][0])
        dd = float(window[-1][1]) - float(window[0][1])
        if dt >= 1.0 and dd > 0:
            rate = dd / dt * 60.0
    remaining = max(0, prog["pass1_total"] - prog["pass1_done"])
    if prog["pass2_total"]:
        remaining += max(0, prog["pass2_total"] - prog["pass2_done"])
    eta = int(remaining / rate * 60.0) if rate > 0 and remaining > 0 else None
    return round(rate, 1), eta


def _journal_append(jid: str, records: List[Dict[str, Any]]) -> None:
    """Журнал результатов: append-only, поэтому сбой на записи не теряет прошлое."""
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(_journal_path(jid), "a", encoding="utf-8", newline="\n") as handle:
        for rec in records:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _journal_load(jid: str) -> Dict[str, Dict[str, Any]]:
    """Результаты задачи из журнала: последняя запись по каждому документу важнее."""
    out: Dict[str, Dict[str, Any]] = {}
    path = _journal_path(jid)
    if not os.path.isfile(path):
        return out
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                key = str(rec.get("_id") or rec.get("id") or "")
                if key:
                    out[key] = rec
    except Exception:
        pass
    return out


def _cancelled(jid: str) -> bool:
    job = _job_load(jid) or {}
    if job.get("cancel"):
        return True
    try:
        return str(_redis.hget(_key(jid), "cancel") or "") in ("1", "True", "true")
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Датасеты и права
# --------------------------------------------------------------------------- #

def _index_map() -> Dict[int, str]:
    try:
        import pickle

        with open(INDEXES_PKL, "rb") as handle:
            raw = pickle.load(handle) or {}
    except Exception:
        return {}
    out: Dict[int, str] = {}
    for key, value in raw.items():
        try:
            out[int(key)] = str(value)
        except Exception:
            continue
    return out


def _allowed_stems(user_id: Any) -> set:
    """Имена датасетов, доступные пользователю: свои папки + расшаренные папки.

    Та же логика, что у ``main._allowed_dataset_stems``, без импорта ``main``.
    """
    stems = set()
    rds = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

    def _add(raw_json: Any, folder: Optional[str] = None) -> None:
        if not raw_json:
            return
        try:
            data = json.loads(raw_json)
        except Exception:
            return
        if folder is not None:
            files = data.get(folder, []) if isinstance(data, dict) else []
        else:
            files = []
            if isinstance(data, dict):
                for value in data.values():
                    files.extend(value or [])
        for name in files or []:
            text = str(name).lower()
            stems.add(text[:-5] if text.endswith(".json") else text)

    try:
        _add(rds.hget(str(user_id), "json_files_directory"))
    except Exception:
        pass
    try:
        for row in _load_shares():
            if str(row.get("user_id")) != str(user_id):
                continue
            _add(rds.hget(str(row.get("owner_user_id")), "json_files_directory"), row.get("folder"))
    except Exception:
        pass
    return stems


def _load_shares() -> List[Dict[str, Any]]:
    """Расшаренные папки из таблицы tellscope_shares (только чтение)."""
    try:
        from config import DB_HOST, DB_NAME, DB_PASS, DB_PORT, DB_USER
        import psycopg2

        conn = psycopg2.connect(host=DB_HOST, port=DB_PORT or 5432, dbname=DB_NAME,
                                user=DB_USER, password=DB_PASS, connect_timeout=5)
        try:
            cur = conn.cursor()
            cur.execute("SELECT owner_user_id, folder, user_id, access FROM tellscope_shares ORDER BY id")
            rows = [{"owner_user_id": r[0], "folder": r[1], "user_id": r[2], "access": r[3]}
                    for r in cur.fetchall()]
            cur.close()
        finally:
            conn.close()
        return rows
    except Exception:
        return []


def _norm_name(value: Any) -> str:
    """Имя индекса ES из значения indexes.pkl: без суффикса .json."""
    text = str(value or "").strip()
    if text.lower().endswith(".json"):
        text = text[:-5]
    return text


def _resolve_dataset(spec: Any) -> Tuple[Optional[int], str]:
    """``index`` — числовой ключ из ``data/indexes.pkl`` либо имя индекса ES."""
    text = str(spec or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Не выбран набор данных")
    mapping = _index_map()
    if re.fullmatch(r"\d+", text):
        key = int(text)
        name = mapping.get(key)
        if not name:
            raise HTTPException(status_code=404, detail="Набор данных не найден")
        return key, _norm_name(name)
    low = _norm_name(text).lower()
    for key, name in mapping.items():
        if _norm_name(name).lower() == low:
            return key, _norm_name(name)
    if _es().indices.exists(index=text):
        return None, text
    raise HTTPException(status_code=404, detail="Набор данных не найден")


def _guard_dataset(user: Any, spec: Any) -> Tuple[Optional[int], str]:
    """Тот же уровень доступа, что у аналитики: чужой датасет — 403."""
    key, name = _resolve_dataset(spec)
    if getattr(user, "is_superuser", False):
        return key, name
    stem = _norm_name(name).lower()
    if stem not in _allowed_stems(getattr(user, "id", None)):
        raise HTTPException(status_code=403, detail="Нет доступа к этому набору данных")
    return key, name


def _pretty_label(name: str) -> str:
    """Человеческое имя датасета: 'ba_озон_отзывы_20260912_150434' → 'Озон отзывы'."""
    try:
        from agent_engine.tools_reports import _pretty_topic_name

        label = _pretty_topic_name(name)
        if label:
            return label
    except Exception:
        pass
    return re.sub(r"[_]+", " ", str(name or "")).strip()


def _tone_filter(tone: Any) -> Optional[int]:
    """Фильтр по тональности источника: -1/0/1, negative/neutral/positive."""
    text = str(tone if tone is not None else "").strip().lower()
    if not text or text in ("all", "any", "все"):
        return None
    if text in TONE_TO_INT:
        return TONE_TO_INT[text]
    if text in ("-1", "1", "0"):
        return int(text)
    raise HTTPException(status_code=400, detail="Тональность фильтра: negative, neutral, positive или -1/0/1")


# --------------------------------------------------------------------------- #
# Запросы к ES
# --------------------------------------------------------------------------- #

def _base_query(job: Dict[str, Any]) -> Dict[str, Any]:
    filters: List[Dict[str, Any]] = []
    lo, hi = job.get("min_date"), job.get("max_date")
    if lo or hi:
        rng: Dict[str, Any] = {}
        if lo:
            rng["gte"] = int(lo)
        if hi:
            rng["lte"] = int(hi)
        filters.append({"range": {"timeCreate": rng}})
    tone = job.get("tone_int")
    if tone is not None:
        filters.append({"term": {"toneMark": int(tone)}})
    # ---- область проверки: объект / инфоповод-тема / площадка / автор ----
    objects_query = _object_query(job.get("objects"))
    if objects_query:
        filters.append(objects_query)
    theme_query = _theme_query(job.get("theme"))
    if theme_query:
        filters.append(theme_query)
    hub = _flat(job.get("hub"))
    if hub:
        filters.append({"term": {"hub": hub}})
    author = _flat(job.get("author"))
    if author:
        filters.append({"match_phrase": {"authorObject.fullname": author}})
    if not filters:
        return {"match_all": {}}
    return {"bool": {"filter": filters}}


def _unlabeled_query(base: Dict[str, Any], marker: str = "tone_llm") -> Dict[str, Any]:
    """Только ещё не размеченные нашей моделью документы — это и есть точка возобновления.

    ``marker`` — поле-отметка прохода: обычный режим пишет ``tone_llm``, аспектный —
    ``tone_aspect_at``. Так два режима не мешают друг другу на одном датасете.
    """
    if not base or "match_all" in base:
        return {"bool": {"must_not": [{"exists": {"field": marker}}]}}
    inner = list(base.get("bool", {}).get("filter") or [])
    return {"bool": {"filter": inner, "must_not": [{"exists": {"field": marker}}]}}


def _active_query(job: Dict[str, Any], marker: str = "tone_llm") -> Dict[str, Any]:
    """Рабочая выборка задачи: при relabel размечаем заново, иначе — только ещё не размеченное."""
    base = _base_query(job)
    if job.get("relabel"):
        return base
    return _unlabeled_query(base, marker)


def _search_page(job: Dict[str, Any], size: int, cursor: Optional[List[Any]],
                 marker: str = "tone_llm", extra_source: Tuple[str, ...] = ()) -> List[Dict[str, Any]]:
    name = job["index_name"]
    query = _active_query(job, marker)
    text_field = _text_field(name)
    id_field = _id_field(name)
    source = ["toneMark", "timeCreate", "hub", "hubtype", "type", "url", "title",
              "review_rating", "id", text_field] + list(extra_source)
    source = sorted(set(source))
    kwargs: Dict[str, Any] = {
        "index": name,
        "size": max(1, int(size)),
        "query": query,
        "source_includes": source,
        "track_total_hits": False,
    }
    full = str(job.get("mode") or "sample") == "full"
    if full and cursor:
        kwargs["sort"] = _sort_spec(id_field)
        kwargs["search_after"] = cursor
    elif full:
        kwargs["sort"] = _sort_spec(id_field)
    else:
        # Выборка: случайный порядок с сохранённым зерном, чтобы срез не был «первыми по времени».
        kwargs["query"] = {
            "function_score": {
                "query": query,
                "random_score": {"seed": int(job.get("seed") or 1), "field": "_seq_no"},
                "boost_mode": "replace",
            }
        }
    try:
        res = _es().search(**kwargs)
    except Exception:
        if not full:
            kwargs["query"] = query
            res = _es().search(**kwargs)
        else:
            kwargs.pop("sort", None)
            res = _es().search(**kwargs)
    hits = ((res.get("hits") or {}).get("hits") or [])
    return hits


def _sort_spec(id_field: str) -> List[Dict[str, Any]]:
    spec = [{"timeCreate": {"order": "asc", "missing": "_first"}}]
    if id_field:
        spec.append({id_field: {"order": "asc", "missing": "_first"}})
    return spec


def _doc_from_hit(hit: Dict[str, Any], job: Dict[str, Any]) -> Dict[str, Any]:
    src = hit.get("_source") or {}
    name = job["index_name"]
    text = str(src.get(_text_field(name)) or src.get("title") or "")
    return {
        "_id": str(hit.get("_id") or ""),
        "id": src.get("id"),
        "text": text,
        "toneMark": src.get("toneMark"),
        "timeCreate": src.get("timeCreate"),
        "hub": src.get("hub") or "",
        "hubtype": src.get("hubtype") or "",
        "type": src.get("type") or "",
        "url": src.get("url") or "",
        "author": _author_name(src.get("authorObject")),
        "sort": list(hit.get("sort") or []),
    }


# --------------------------------------------------------------------------- #
# Промпты и вызов модели
# --------------------------------------------------------------------------- #

PASS1_SYSTEM = ("Ты аналитик тональности текстов о брендах. Отвечай строго JSON, "
                "без пояснений и текста вокруг.")

PASS1_INSTRUCTION = """Определи тональность каждого сообщения: negative, neutral или positive.

Правила:
- negative — жалоба, недовольство, критика, проблема, испорченный опыт, риск, угроза, призыв бойкотировать;
- positive — благодарность, похвала, одобрение, радость, удачный опыт, рекомендация;
- neutral — нейтральное информирование, вопрос без оценки, реклама, анонс, факт, шумовой или бессмысленный текст.
Слабо выраженная оценка, реклама и бессмыслица — neutral с низкой уверенностью.
reason — до 100 знаков, по-русски, коротко: на чём основано решение.

Верни СТРОГО JSON такой формы:
{{"results": [{{"id": "<id>", "tone": "negative|neutral|positive", "confidence": 0.0, "reason": "..."}}]}}
В ответе должны быть ВСЕ {count} сообщений, id — ровно как в списке.

Сообщения:
{messages}"""

PASS2_SYSTEM = ("Ты старший аналитик тональности: перепроверяешь спорные случаи и решаешь "
                "окончательно. Отвечай строго JSON, без пояснений и текста вокруг.")

PASS2_INSTRUCTION = """Спорные сообщения: быстрая модель сомневалась или разошлась с разметкой источника.
Посмотри текст внимательно и реши окончательно: negative, neutral или positive.
Если оценка действительно не выражена — neutral. reason — до 100 знаков, по-русски, что решило дело.

Верни СТРОГО JSON такой формы:
{{"results": [{{"id": "<id>", "tone": "negative|neutral|positive", "confidence": 0.0, "reason": "..."}}]}}
В ответе должны быть ВСЕ {count} сообщений, id — ровно как в списке.

Сообщения:
{messages}"""


class _Ctx:
    """Минимальный контекст для agent_engine.tools_llm._qwen (он умеет только ctx.log)."""

    def __init__(self, jid: str) -> None:
        self.jid = jid

    async def log(self, message: str, level: str = "info") -> None:
        _log_line("[%s] %s" % (self.jid, message))

    def cancelled(self) -> bool:
        return _cancelled(self.jid)


def _bulk_profile() -> Dict[str, Any]:
    """Профиль быстрого чтения (mlops/lock.yaml, секция texts_bulk). Пусто — читаем на 32B."""
    try:
        from mlops.lock import texts_bulk_cfg

        cfg = texts_bulk_cfg() or {}
    except Exception:
        return {}
    if not cfg.get("enabled"):
        return {}

    def _int(key: str, default: int) -> int:
        try:
            return int(cfg.get(key) or default)
        except (TypeError, ValueError):
            return default

    return {
        "vllm_cfg": cfg,
        "model": str(cfg.get("model") or ""),
        "base_url": str(cfg.get("base_url") or ""),
        "max_tokens": _int("max_tokens", PASS1_MAX_TOKENS),
        "batch_size": _int("batch_size", DEFAULT_BATCH),
        "parallel": _int("parallel_batches", DEFAULT_PARALLEL),
    }


def _message_line(doc: Dict[str, Any], limit: int) -> str:
    text = " ".join(str(doc.get("text") or "").split())[:limit]
    date = ""
    try:
        date = datetime.fromtimestamp(float(doc.get("timeCreate") or 0)).strftime("%d.%m.%Y")
    except Exception:
        date = ""
    return "[%s] (%s, %s) %s" % (doc.get("_id"), doc.get("hub") or "источник", date, text)


def _make_batches(docs: List[Dict[str, Any]], batch_size: int, char_budget: int,
                  chars: int) -> List[List[Dict[str, Any]]]:
    batches: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    size = 0
    for doc in docs:
        length = min(len(doc.get("text") or ""), chars) + 60
        if current and (len(current) >= batch_size or size + length > char_budget):
            batches.append(current)
            current, size = [], 0
        current.append(doc)
        size += length
    if current:
        batches.append(current)
    return batches


def _parse_results(text: str, finish: str) -> Optional[Dict[str, Dict[str, Any]]]:
    """Разбирает ответ модели в {_id: {tone, conf, reason}}; None — ответ непригоден."""
    from agent_engine.tools_llm import _extract_json

    if finish == "length":
        return None
    parsed = _extract_json(text)
    if not isinstance(parsed, dict):
        return None
    items = parsed.get("results")
    if not isinstance(items, list) or not items:
        return None
    out: Dict[str, Dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        key = str(item.get("id") or "").strip()
        tone = str(item.get("tone") or "").strip().lower()
        if not key or tone not in TONE_TO_INT:
            continue
        try:
            conf = float(item.get("confidence"))
        except (TypeError, ValueError):
            conf = 0.5
        conf = max(0.0, min(1.0, conf))
        reason = " ".join(str(item.get("reason") or "").split())[:100]
        out[key] = {"tone": tone, "conf": conf, "reason": reason}
    return out or None


async def _read_batch(ctx: _Ctx, docs: List[Dict[str, Any]], *, system: str, instruction: str,
                      chars: int, max_tokens: int, vllm_cfg: Optional[Dict[str, Any]],
                      sem: asyncio.Semaphore) -> Tuple[Dict[str, Dict[str, Any]], str]:
    """Одна пачка: строгий JSON, повторы, деление пачки при обрыве по лимиту генерации.

    Возвращает ({_id: вердикт}, ошибка). Пустой первый элемент — пачка не разобрана.
    """
    from agent_engine.tools_llm import _qwen

    async with sem:
        if ctx.cancelled():
            return {}, "остановлено"

    async def _call(lines: List[str], depth: int) -> Dict[str, Dict[str, Any]]:
        prompt = instruction.format(count=len(lines), messages="\n".join(lines))
        last = "нет ответа модели"
        for attempt in range(2 if depth == 0 else 1):
            meta: Dict[str, Any] = {}
            try:
                # Жёсткий предел на вызов: без него зависший запрос держит пачку (а с ней и
                # всю страницу) до сетевого таймаута — это и есть «голова очереди».
                text, _tokens = await asyncio.wait_for(
                    _qwen(ctx, prompt, system=system, max_tokens=max_tokens,
                          temperature=0.0, vllm_cfg=vllm_cfg, meta=meta),
                    timeout=BATCH_TIMEOUT)
            except asyncio.TimeoutError:
                last = "нет ответа модели за %.0f с" % BATCH_TIMEOUT
                _log_line("пачка из %d не ответила за %.0f с" % (len(lines), BATCH_TIMEOUT),
                          level="warning")
                await asyncio.sleep(1.0 + attempt)
                continue
            except Exception as exc:  # noqa: BLE001 — пачка не должна ломать весь проход
                last = "%s: %s" % (type(exc).__name__, str(exc)[:160])
                await asyncio.sleep(1.0 + attempt)
                continue
            parsed = _parse_results(text, str(meta.get("finish_reason") or ""))
            if parsed:
                return parsed
            finish = str(meta.get("finish_reason") or "")
            last = ("ответ обрезан по лимиту %d токенов" % max_tokens) if finish == "length" \
                else "в ответе нет корректного списка results"
        raise RuntimeError(last)

    async def _split(lines: List[str], depth: int) -> Dict[str, Dict[str, Any]]:
        try:
            return await _call(lines, depth)
        except Exception as exc:  # noqa: BLE001
            if len(lines) <= 2 or depth >= 2:
                raise
            half = len(lines) // 2
            out: Dict[str, Dict[str, Any]] = {}
            for part in (lines[:half], lines[half:]):
                try:
                    out.update(await _split(part, depth + 1))
                except Exception as inner:  # noqa: BLE001
                    _log_line("часть пачки (%d) не разобрана: %s" % (len(part), str(inner)[:120]))
            if not out:
                raise exc
            return out

    lines = [_message_line(doc, chars) for doc in docs]
    try:
        return await _split(lines, 0), ""
    except Exception as exc:  # noqa: BLE001
        return {}, str(exc)[:200]


def _records_from(parsed: Dict[str, Dict[str, Any]], docs: List[Dict[str, Any]],
                  source: str, model: str) -> List[Dict[str, Any]]:
    """Записи для журнала и ES: по одной на сообщение пачки."""
    out: List[Dict[str, Any]] = []
    for doc in docs:
        verdict = parsed.get(doc["_id"])
        base = {
            "_id": doc["_id"],
            "id": doc.get("id"),
            "toneMark": doc.get("toneMark"),
            "timeCreate": doc.get("timeCreate"),
            "hub": doc.get("hub") or "",
            "hubtype": doc.get("hubtype") or "",
            "type": doc.get("type") or "",
            "url": doc.get("url") or "",
            "text": " ".join(str(doc.get("text") or "").split())[:1200],
            "chars": len(str(doc.get("text") or "")),
            "model": model,
        }
        if verdict:
            base.update({
                "tone_llm": TONE_TO_INT[verdict["tone"]],
                "tone_llm_conf": round(float(verdict["conf"]), 3),
                "tone_llm_by": source,
                "tone_llm_reason": verdict["reason"] or "—",
            })
        else:
            # Вердикта нет: не выдаём чужое решение за наше, но и не теряем документ.
            try:
                guess = int(doc.get("toneMark"))
            except (TypeError, ValueError):
                guess = 0
            base.update({
                "tone_llm": guess if guess in CLASSES else 0,
                "tone_llm_conf": 0.0,
                "tone_llm_by": "none",
                "tone_llm_reason": "автоматическая разметка не дала вердикт",
            })
        out.append(base)
    return out


def _bulk_write(job: Dict[str, Any], records: List[Dict[str, Any]]) -> Tuple[int, int]:
    """Пакетная запись нашей разметки в ES. Исходные поля не трогаем.

    ``refresh="wait_for"`` обязателен: следующая страница выборки отсекает уже размеченные
    документы запросом ``must_not exists tone_llm``, а без обновления индекса Elasticsearch
    отдал бы те же самые документы и проход крутился бы на месте.
    """
    name = job["index_name"]
    stamp = int(time.time())
    written = 0
    errors = 0
    for start in range(0, len(records), BULK_CHUNK):
        chunk = records[start:start + BULK_CHUNK]
        operations: List[Dict[str, Any]] = []
        for rec in chunk:
            if not rec.get("_id"):
                continue
            operations.append({"update": {"_index": name, "_id": rec["_id"], "retry_on_conflict": 3}})
            operations.append({"doc": {
                "tone_llm": rec["tone_llm"],
                "tone_llm_conf": rec["tone_llm_conf"],
                "tone_llm_by": rec["tone_llm_by"],
                "tone_llm_reason": rec["tone_llm_reason"],
                "tone_llm_at": stamp,
            }})
        if not operations:
            continue
        try:
            res = _es().bulk(operations=operations, refresh="wait_for")
            failed = 0
            if res.get("errors"):
                for item in res.get("items") or []:
                    if (item.get("update") or {}).get("error"):
                        failed += 1
            errors += failed
            written += len(chunk) - failed
        except Exception as exc:  # noqa: BLE001
            errors += len(chunk)
            _log_line("пакетная запись не удалась: %s" % str(exc)[:200], level="error")
    return written, errors


async def _model_up(base_url: str) -> bool:
    if not base_url:
        return False
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            res = await client.get(base_url.rstrip("/") + "/v1/models")
            return res.status_code == 200
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Воркер
# --------------------------------------------------------------------------- #

def _needs_pass2(rec: Dict[str, Any], threshold: float) -> bool:
    if rec.get("tone_llm_by") == "none":
        return True
    if rec.get("tone_llm_by") == "32b":
        return False
    conf = float(rec.get("tone_llm_conf") or 0.0)
    if conf < threshold:
        return True
    try:
        source = int(rec.get("toneMark"))
    except (TypeError, ValueError):
        return True
    return source != int(rec.get("tone_llm") or 0)


async def _run_job(jid: str) -> None:
    job = _job_load(jid)
    if not job:
        return
    if str(job.get("label_mode") or "message") == "aspect":
        # Аспектная разметка — отдельная ветка: свой промпт, свои поля, свой отчёт.
        return await _run_aspect_job(jid)
    name = job["index_name"]
    ctx = _Ctx(jid)
    started = time.time()
    try:
        _set(jid, status="running", stage="preparing", stage_label=STAGE_LABELS["preparing"],
             started=job.get("started") or datetime.now().isoformat(timespec="seconds"),
             writing=False, progress_schema=PROGRESS_SCHEMA, worker_pid=os.getpid(),
             log="старт разметки: %s (%s)" % (name, job.get("mode")))
        _ensure_mapping(name)
        results = _journal_load(jid)
        # Сколько осталось разметить + сколько уже размечено этой задачей = полный объём задачи.
        # При возобновлении после перезапуска счётчик не начинается заново.
        remaining = int(_es().count(index=name, query=_active_query(job)).get("count") or 0)
        total = len(results) + remaining
        if job.get("mode") == "sample":
            total = min(int(job.get("sample_size") or 1000), total)
        elif total > MAX_FULL_DOCS:
            raise RuntimeError(
                "Полная проверка — это %d сообщений, а за один раз мы проверяем до %d. "
                "Сузьте период или добавьте фильтры (площадка, объект)."
                % (total, MAX_FULL_DOCS))
        _set(jid, total=total, processed=len(results), errors=0)
        _log_line("[%s] к разметке %d сообщений, уже размечено %d" % (jid, total, len(results)))

        # ------------------------------ проход 1: быстрая 4B ------------------------------
        _set(jid, stage="pass1", stage_label=STAGE_LABELS["pass1"])
        bulk = _bulk_profile()
        if not bulk:
            _log_line("[%s] профиль texts_bulk выключен — размечаю на общей модели 32B" % jid)
        batch_size = int(job.get("batch_size") or bulk.get("batch_size") or DEFAULT_BATCH)
        parallel = int(job.get("parallel") or bulk.get("parallel") or DEFAULT_PARALLEL)
        model_label = bulk.get("model") or "Qwen/Qwen3-32B-FP8"
        sem = asyncio.Semaphore(max(1, parallel))
        cursor = job.get("cursor")
        stall = 0
        seen_before = set(results)
        while len(results) < total and not _cancelled(jid):
            want = min(PAGE_SIZE, total - len(results))
            hits = _search_page(job, want, cursor)
            if not hits:
                if cursor:
                    # Курсор мог устареть после перезапуска: один раз начинаем страницу заново,
                    # уже размеченные документы исключает must_not exists tone_llm.
                    cursor = None
                    job["cursor"] = None
                    continue
                break
            docs = [_doc_from_hit(hit, job) for hit in hits]
            batches = _make_batches(docs, batch_size, PASS1_CHAR_BUDGET, PASS1_MESSAGE_CHARS)
            tasks = [
                asyncio.create_task(_read_batch(
                    ctx, batch, system=PASS1_SYSTEM, instruction=PASS1_INSTRUCTION,
                    chars=PASS1_MESSAGE_CHARS,
                    max_tokens=int(bulk.get("max_tokens") or PASS1_MAX_TOKENS),
                    vllm_cfg=bulk.get("vllm_cfg"), sem=sem))
                for batch in batches
            ]
            batch_of = {task: batch for task, batch in zip(tasks, batches)}
            records: List[Dict[str, Any]] = []
            pending = set(tasks)
            while pending:
                finished, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in finished:
                    batch = batch_of[task]
                    parsed, error = task.result()
                    if error and error != "остановлено":
                        _log_line("[%s] пачка из %d не разобрана на %s: %s"
                                  % (jid, len(batch), "4B" if bulk else "32B", error))
                    recs = _records_from(parsed, batch, "4b" if bulk else "32b",
                                         model_label if bulk else "Qwen/Qwen3-32B-FP8")
                    for rec in recs:
                        results[rec["_id"]] = rec
                    records.extend(recs)
                    _journal_append(jid, recs)
                # Прогресс обновляем по готовым пачкам, а не только в конце страницы:
                # иначе на длинном датасете полоса стоит по 20 секунд.
                _set(jid, processed=len(results), total=total)
            fresh = len([rec for rec in records if rec["_id"] not in seen_before])
            if not fresh:
                stall += 1
                _log_line("[%s] страница не дала новых документов (%d-я подряд)"
                          % (jid, stall), level="warning")
                if stall >= 3:
                    _log_line("[%s] выборка перестала давать новые документы — останавливаю проход"
                              % jid, level="warning")
                    break
            else:
                stall = 0
            seen_before.update(rec["_id"] for rec in records)
            if job.get("mode") == "full" and docs and docs[-1].get("sort"):
                cursor = docs[-1]["sort"]
                job["cursor"] = cursor
            # Запись в ES — это под-шаг первого прохода, а не отдельный этап: у неё нет
            # своего счётчика, поэтому этап не подменяется и полоса не скачет.
            _set(jid, writing=True)
            written, werrors = _bulk_write(job, records)
            job["errors"] = int(job.get("errors") or 0) + werrors
            job["bulk_written"] = int(job.get("bulk_written") or 0) + written
            _set(jid, writing=False, stage="pass1", stage_label=STAGE_LABELS["pass1"],
                 processed=len(results), total=total, cursor=job.get("cursor"),
                 bulk_written=job["bulk_written"], errors=job["errors"])
        _set(jid, processed=len(results), total=total)

        # ------------------------- проход 2: перепроверка на 32B -------------------------
        threshold = float(job.get("conf_threshold") or DEFAULT_CONF)
        disputed = [rec for rec in results.values() if _needs_pass2(rec, threshold)]
        pass2 = {"available": True, "skipped": False, "reason": "", "targets": len(disputed),
                 "decided": 0, "failed": 0}
        # 32B в разы медленнее 4B: если спорных слишком много, перепроверка съест всё время.
        # Ограничиваем её долю и перепроверяем самые неуверенные; остальные честно остаются
        # с решением 4B, а факт и размер лимита видны в отчёте (pass2_capped).
        disputed_total = len(disputed)
        pass2_limit = int(max(1, total) * PASS2_MAX_SHARE)
        pass2_capped = 0
        if disputed_total > pass2_limit:
            disputed.sort(key=lambda rec: float(rec.get("tone_llm_conf") or 0.0))
            pass2_capped = disputed_total - max(1, pass2_limit)
            disputed = disputed[:max(1, pass2_limit)]
            _log_line("[%s] спорных %d — больше лимита %.0f%% (%d): перепроверяю %d самых "
                      "неуверенных, остальные %d остаются решением 4B"
                      % (jid, disputed_total, PASS2_MAX_SHARE * 100, pass2_limit,
                         len(disputed), pass2_capped), level="warning")
        pass2["candidates"] = disputed_total
        pass2["capped"] = pass2_capped
        pass2["max_share"] = PASS2_MAX_SHARE
        pass2["share"] = round(disputed_total / float(max(1, total)), 4)
        stopped = _cancelled(jid)
        if not stopped and disputed:
            from mlops.lock import generate_cfg

            gen = generate_cfg() or {}
            base_url = str(gen.get("base_url") or "")
            if not await _model_up(base_url):
                pass2.update({"available": False, "skipped": True,
                              "reason": "сервис уточняющей проверки был недоступен"})
                _log_line("[%s] 32B недоступна: перепроверка не выполнена" % jid, level="warning")
            else:
                _set(jid, stage="pass2", stage_label=STAGE_LABELS["pass2"],
                     pass2_total=len(disputed), pass2_done=0)
                p2_batch = int(job.get("pass2_batch") or DEFAULT_PASS2_BATCH)
                p2_parallel = int(job.get("pass2_parallel") or DEFAULT_PARALLEL_PASS2)
                p2_sem = asyncio.Semaphore(max(1, p2_parallel))
                done = 0
                for start in range(0, len(disputed), PAGE_SIZE):
                    if _cancelled(jid):
                        break
                    window = disputed[start:start + PAGE_SIZE]
                    batches = _make_batches(
                        [{"_id": r["_id"], "text": r.get("text") or "", "hub": r.get("hub") or "",
                          "timeCreate": r.get("timeCreate")} for r in window],
                        p2_batch, PASS2_CHAR_BUDGET, PASS2_MESSAGE_CHARS)
                    by_id = {r["_id"]: r for r in window}
                    tasks = [
                        asyncio.create_task(_read_batch(
                            ctx, batch, system=PASS2_SYSTEM, instruction=PASS2_INSTRUCTION,
                            chars=PASS2_MESSAGE_CHARS, max_tokens=PASS2_MAX_TOKENS,
                            vllm_cfg=None, sem=p2_sem))
                        for batch in batches
                    ]
                    batch_of = {task: batch for task, batch in zip(tasks, batches)}
                    changed: List[Dict[str, Any]] = []
                    seen = 0
                    pending = set(tasks)
                    while pending:
                        finished, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                        for task in finished:
                            batch = batch_of[task]
                            seen += len(batch)
                            parsed, error = task.result()
                            if error and error != "остановлено":
                                pass2["failed"] += len(batch)
                                _log_line("[%s] пачка из %d не разобрана на 32B: %s"
                                          % (jid, len(batch), error))
                            source_docs = [{"_id": d["_id"], "id": by_id[d["_id"]].get("id"),
                                            "toneMark": by_id[d["_id"]].get("toneMark"),
                                            "timeCreate": by_id[d["_id"]].get("timeCreate"),
                                            "hub": by_id[d["_id"]].get("hub"),
                                            "hubtype": by_id[d["_id"]].get("hubtype"),
                                            "type": by_id[d["_id"]].get("type"),
                                            "url": by_id[d["_id"]].get("url"),
                                            "text": by_id[d["_id"]].get("text")}
                                           for d in batch]
                            recs = _records_from(parsed, source_docs, "32b", "Qwen/Qwen3-32B-FP8")
                            batch_recs: List[Dict[str, Any]] = []
                            for rec in recs:
                                if rec["_id"] not in by_id:
                                    continue
                                if rec["tone_llm_by"] != "32b":
                                    # 32B не ответила по этому сообщению — оставляем решение 4B,
                                    # чтобы не подменить его пустышкой.
                                    continue
                                previous = by_id[rec["_id"]]
                                rec["chars"] = previous.get("chars") or rec.get("chars")
                                results[rec["_id"]] = rec
                                batch_recs.append(rec)
                                pass2["decided"] += 1
                            if batch_recs:
                                _journal_append(jid, batch_recs)
                                changed.extend(batch_recs)
                            _set(jid, stage="pass2", stage_label=STAGE_LABELS["pass2"],
                                 pass2_done=done + seen, pass2_total=len(disputed),
                                 processed=len(results), total=total)
                    if changed:
                        _set(jid, writing=True)
                        written, werrors = _bulk_write(job, changed)
                        job["errors"] = int(job.get("errors") or 0) + werrors
                        job["bulk_written"] = int(job.get("bulk_written") or 0) + written
                    done += len(window)
                    _set(jid, writing=False, stage="pass2", stage_label=STAGE_LABELS["pass2"],
                         pass2_done=done, pass2_total=len(disputed),
                         processed=len(results), total=total, errors=job.get("errors") or 0,
                         bulk_written=job.get("bulk_written") or 0)
        # -------------------------------- отчёт --------------------------------
        cancelled = _cancelled(jid)
        _set(jid, status="running", stage="report", stage_label=STAGE_LABELS["report"],
             processed=len(results), total=total, percent_before_report=int(job.get("percent") or 0),
             bulk_written=job.get("bulk_written") or 0, errors=job.get("errors") or 0)
        report = build_report(job, results, pass2)
        report["elapsed_sec"] = round(time.time() - started, 1)
        report["bulk_written"] = int(job.get("bulk_written") or 0)
        report["write_errors"] = int(job.get("errors") or 0)
        with open(_report_path(jid), "w", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=1)
        files = _write_report_files(job, report)
        report["files"] = files
        with open(_report_path(jid), "w", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=1)
        rate = round(len(results) / max(1e-6, report["elapsed_sec"]) * 60.0, 1)
        _set(jid, status="cancelled" if cancelled else "done",
             stage="cancelled" if cancelled else "done",
             stage_label=STAGE_LABELS["cancelled" if cancelled else "done"],
             processed=len(results), total=total,
             percent=(int((_job_load(jid) or {}).get("percent_before_report")
                          or (_job_load(jid) or {}).get("percent") or 0) if cancelled else 100),
             finished=datetime.now().isoformat(timespec="seconds"),
             elapsed_sec=report["elapsed_sec"], rate_per_min=rate,
             agreement=report["summary"]["agreement"], kappa=report["summary"]["kappa"],
             pass2_share=report["summary"]["pass2_share"],
             report_files=files, worker_pid=0,
             log="готово: размечено %d, согласие %.1f%%, каппа %.3f"
                 % (len(results), 100 * report["summary"]["agreement"], report["summary"]["kappa"]))
    except Exception as exc:  # noqa: BLE001
        _log_line("[%s] ошибка: %s" % (jid, str(exc)[:300]), level="error")
        _set(jid, status="error", stage="error", stage_label=STAGE_LABELS["error"],
             error=str(exc)[:500], worker_pid=0,
             finished=datetime.now().isoformat(timespec="seconds"))


_ACTIVE_WORKERS: Dict[str, threading.Thread] = {}
_workers_lock = threading.Lock()


def _worker_alive(jid: str) -> bool:
    """Работает ли воркер этой задачи прямо в текущем процессе."""
    with _workers_lock:
        thread = _ACTIVE_WORKERS.get(jid)
    return bool(thread and thread.is_alive())


def _owner_process_alive(job: Dict[str, Any]) -> bool:
    """Жив ли процесс, который ведёт задачу (pid записан в файле задачи).

    Статус ``running`` сам по себе не значит «прервана»: он значит, что задачу кто-то
    ведёт прямо сейчас. Возобновлять её можно только если владелец действительно умер.
    """
    try:
        pid = int(job.get("worker_pid") or 0)
    except (TypeError, ValueError):
        pid = 0
    if pid <= 0:
        return False
    if pid == os.getpid():
        return _worker_alive(str(job.get("id") or ""))
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        # Процесса нет — задача действительно прервана, её можно поднимать.
        return False
    except PermissionError:
        # Процесс есть, но он чужой (например, root): сигнал не пошёл, значит владелец жив.
        return True
    except OSError:
        # Неизвестная ошибка: безопаснее счесть владельца живым, чем запустить второй проход.
        return True
    return True


def _start_job(jid: str) -> None:
    """Запускает воркер задачи ровно один раз.

    Повторный запуск для уже работающей задачи запрещён: иначе на одну задачу идут два
    прохода сразу — двойные вызовы моделей и счётчики прогресса, «прыгающие» между
    объёмами (это и выглядело как сломанный прогресс).
    """
    with _workers_lock:
        thread = _ACTIVE_WORKERS.get(jid)
        if thread and thread.is_alive():
            _log_line("[%s] воркер уже работает — повторный запуск пропущен" % jid, level="warning")
            return
        thread = threading.Thread(target=lambda: asyncio.run(_run_job(jid)), daemon=True,
                                  name="tone-check-%s" % jid)
        _ACTIVE_WORKERS[jid] = thread
    thread.start()


def _resume_job(jid: str) -> None:
    job = _job_load(jid) or {}
    if _worker_alive(jid) or _owner_process_alive(job):
        # Возобновление «по статусу running» без этой проверки поднимало второй проход на ту
        # же задачу — например, когда модуль импортировал посторонний процесс.
        _log_line("[%s] возобновление не нужно: задачу ведёт живой воркер" % jid, level="warning")
        return
    _set(jid, status="running", stage="preparing", stage_label="продолжаю после перезапуска",
         interrupted=False, cancel=False, log="возобновляю после перезапуска сервиса")
    try:
        _redis.hdel(_key(jid), "cancel")
    except Exception:
        pass
    _start_job(jid)


def _recover() -> List[str]:
    """Возобновляет проходы, прерванные перезапуском сервиса."""
    resumed: List[str] = []
    if not os.path.isdir(STATE_DIR):
        return resumed
    for entry in sorted(os.listdir(STATE_DIR)):
        if not entry.endswith(".json") or entry.endswith(".report.json"):
            continue
        jid = entry[:-5]
        job = _job_load(jid)
        if not job:
            continue
        status = str(job.get("status") or "")
        if status in ("running", "queued", "interrupted"):
            _log_line("возобновляю прерванную задачу %s (статус был %s)" % (jid, status))
            try:
                _resume_job(jid)
                resumed.append(jid)
            except Exception as exc:  # noqa: BLE001
                _log_line("не удалось возобновить %s: %s" % (jid, str(exc)[:200]), level="error")
    return resumed


def _ensure_recovery() -> None:
    global _RECOVERY_STARTED
    with _router_lock:
        if _RECOVERY_STARTED:
            return
        _RECOVERY_STARTED = True
    threading.Thread(target=_recover, daemon=True, name="tone-check-recover").start()


def _delayed_recovery() -> None:
    """Lifespan приложения кастомный, поэтому startup-события роутера не выполняются.

    Ждём, пока приложение поднимется, и добираем прерванные проходы сами.
    """
    time.sleep(float(os.environ.get("TELLSCOPE_TONE_RECOVER_DELAY") or 20))
    try:
        _ensure_recovery()
    except Exception as exc:  # noqa: BLE001
        _log_line("восстановление задач не запустилось: %s" % str(exc)[:200], level="error")


def _looks_like_server() -> bool:
    """Похоже ли, что модуль импортирован самим веб-приложением, а не посторонним скриптом.

    Импорт модуля не должен поднимать фоновые проходы: раньше автовозобновление стартовало
    при любом импорте, и служебный скрипт поднимал вторую копию уже идущей задачи.
    """
    argv = " ".join(sys.argv).lower()
    return any(marker in argv for marker in ("uvicorn", "gunicorn", "hypercorn", "main:app", "main.py"))


if str(os.environ.get("TELLSCOPE_TONE_NO_AUTORESUME") or "") != "1":
    if _looks_like_server():
        threading.Thread(target=_delayed_recovery, daemon=True, name="tone-check-autoresume").start()
    else:
        _log_line("автовозобновление не запускаю: импорт не от веб-приложения (%s)"
                  % " ".join(sys.argv)[:120])


# --------------------------------------------------------------------------- #
# Отчёт о качестве разметки источника
# --------------------------------------------------------------------------- #

def _pct(value: float) -> str:
    return ("%.1f%%" % (100.0 * value)).replace(".", ",")


def _num(value: Any, digits: int = 2) -> str:
    try:
        return ("%." + str(digits) + "f") % float(value)
    except (TypeError, ValueError):
        return "—"


def _plural(count: int, one: str, few: str, many: str) -> str:
    """Русское число с существительным: 1 случай, 2 случая, 5 случаев."""
    value = abs(int(count)) % 100
    if 11 <= value <= 14:
        return "%d %s" % (count, many)
    value %= 10
    if value == 1:
        return "%d %s" % (count, one)
    if 2 <= value <= 4:
        return "%d %s" % (count, few)
    return "%d %s" % (count, many)


def _kappa(pairs: List[Tuple[int, int]]) -> Tuple[float, float]:
    """Каппа Коэна и наблюдаемое согласие по парам (источник, модель)."""
    n = len(pairs)
    if not n:
        return 0.0, 0.0
    diag = sum(1 for a, b in pairs if a == b)
    po = diag / float(n)
    pe = 0.0
    for cls in CLASSES:
        row = sum(1 for a, _ in pairs if a == cls) / float(n)
        col = sum(1 for _, b in pairs if b == cls) / float(n)
        pe += row * col
    if pe >= 1.0:
        return po, 1.0
    return po, (po - pe) / (1.0 - pe)


def _length_bucket(chars: int) -> str:
    if chars < 100:
        return "до 100 знаков"
    if chars < 300:
        return "100–300 знаков"
    if chars < 1000:
        return "300–1000 знаков"
    return "больше 1000 знаков"


def _date_str(ts: Any) -> str:
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%d.%m.%Y %H:%M")
    except Exception:
        return "—"


def build_report(job: Dict[str, Any], results: Dict[str, Dict[str, Any]],
                 pass2: Dict[str, Any]) -> Dict[str, Any]:
    """Считает качество разметки источника: согласие, каппу, матрицу, разбивки, примеры."""
    all_recs = list(results.values())
    evaluated = [r for r in all_recs if r.get("tone_llm_by") != "none"]
    unresolved = [r for r in all_recs if r.get("tone_llm_by") == "none"]
    with_source = [r for r in evaluated if str(r.get("toneMark")) in ("-1", "0", "1")]
    no_source = [r for r in evaluated if str(r.get("toneMark")) not in ("-1", "0", "1")]

    pairs = [(int(r["toneMark"]), int(r["tone_llm"])) for r in with_source]
    agreement, kappa = _kappa(pairs)
    matches = sum(1 for a, b in pairs if a == b)

    matrix = {str(a): {str(b): 0 for b in CLASSES} for a in CLASSES}
    for a, b in pairs:
        matrix[str(a)][str(b)] += 1

    per_class: List[Dict[str, Any]] = []
    for cls in CLASSES:
        tp = sum(1 for a, b in pairs if a == cls and b == cls)
        pred = sum(1 for _, b in pairs if b == cls)
        actual = sum(1 for a, _ in pairs if a == cls)
        precision = tp / pred if pred else 0.0
        recall = tp / actual if actual else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_class.append({"class": cls, "label": TONE_RU[cls], "precision": round(precision, 4),
                          "recall": round(recall, 4), "f1": round(f1, 4),
                          "support": actual, "predicted": pred, "tp": tp})

    def _breakdown(field: str, limit: int = 15, min_total: int = 1) -> List[Dict[str, Any]]:
        buckets: Dict[str, List[int]] = {}
        for rec in with_source:
            key = str(rec.get(field) or "—").strip() or "—"
            cell = buckets.setdefault(key, [0, 0])
            cell[0] += 1
            if int(rec["toneMark"]) != int(rec["tone_llm"]):
                cell[1] += 1
        rows = [{"key": key, "total": total, "mismatch": bad,
                 "share": round(bad / float(total), 4) if total else 0.0}
                for key, (total, bad) in buckets.items() if total >= min_total]
        rows.sort(key=lambda item: (-item["mismatch"], -item["total"]))
        return rows[:limit]

    by_hub = _breakdown("hub", 15)
    by_section = _breakdown("hubtype", 10)
    by_type = _breakdown("type", 10)
    lengths: Dict[str, List[int]] = {}
    for rec in with_source:
        key = _length_bucket(int(rec.get("chars") or len(str(rec.get("text") or ""))))
        cell = lengths.setdefault(key, [0, 0])
        cell[0] += 1
        if int(rec["toneMark"]) != int(rec["tone_llm"]):
            cell[1] += 1
    order = ["до 100 знаков", "100–300 знаков", "300–1000 знаков", "больше 1000 знаков"]
    by_length = [{"key": key, "total": lengths[key][0], "mismatch": lengths[key][1],
                  "share": round(lengths[key][1] / float(lengths[key][0]), 4)}
                 for key in order if key in lengths]

    pass2_share = (sum(1 for r in all_recs if r.get("tone_llm_by") == "32b") / float(len(all_recs))
                   if all_recs else 0.0)

    # примеры расхождений: сначала уверенные решения модели, потом остальные
    mismatches = [r for r in with_source if int(r["toneMark"]) != int(r["tone_llm"])]
    mismatches.sort(key=lambda r: (-float(r.get("tone_llm_conf") or 0.0),
                                   -int(r.get("chars") or 0)))
    examples = []
    for rec in mismatches[:MAX_EXAMPLES]:
        text = " ".join(str(rec.get("text") or "").split())
        examples.append({
            "quote": text[:280] + ("…" if len(text) > 280 else ""),
            "url": rec.get("url") or "",
            "hub": rec.get("hub") or "—",
            "hubtype": rec.get("hubtype") or "",
            "date": _date_str(rec.get("timeCreate")),
            "source": TONE_RU.get(int(rec["toneMark"]), "—"),
            "model": TONE_RU.get(int(rec["tone_llm"]), "—"),
            "source_int": int(rec["toneMark"]),
            "model_int": int(rec["tone_llm"]),
            "confidence": float(rec.get("tone_llm_conf") or 0.0),
            "decided_by": BY_RU.get(rec.get("tone_llm_by") or "", rec.get("tone_llm_by") or ""),
            "reason": rec.get("tone_llm_reason") or "—",
            "chars": int(rec.get("chars") or len(text)),
        })

    # Перекос: где модель строже источника
    model_more_negative = sum(1 for a, b in pairs if b < a)
    model_more_positive = sum(1 for a, b in pairs if b > a)

    summary = {
        "dataset": job.get("dataset_label") or job.get("index_name"),
        "index_name": job.get("index_name"),
        "index_key": job.get("index_key"),
        "mode": job.get("mode"),
        "sample_size": job.get("sample_size"),
        "checked": len(all_recs),
        "evaluated": len(with_source),
        "unresolved": len(unresolved),
        "no_source_tone": len(no_source),
        "agreement": round(agreement, 4),
        "matches": matches,
        "mismatches": len(pairs) - matches,
        "kappa": round(kappa, 4),
        "pass2_share": round(pass2_share, 4),
        "pass2_decided": int(pass2.get("decided") or 0),
        # Сколько спорных было всего и сколько не попало в перепроверку из-за лимита доли.
        "pass2_candidates": int(pass2.get("candidates") or pass2.get("targets") or 0),
        "pass2_capped": int(pass2.get("capped") or 0),
        "pass2_max_share": float(pass2.get("max_share") or PASS2_MAX_SHARE),
        "pass2_available": bool(pass2.get("available", True)),
        "pass2_reason": str(pass2.get("reason") or ""),
        "threshold": float(job.get("conf_threshold") or DEFAULT_CONF),
        "model_more_negative": model_more_negative,
        "model_more_positive": model_more_positive,
        "bulk_written": int(job.get("bulk_written") or 0),
        "write_errors": int(job.get("errors") or 0),
        "elapsed_sec": job.get("elapsed_sec"),
        "rate_per_min": job.get("rate_per_min"),
    }
    conclusions = _conclusions(summary, by_length, by_hub, per_class, matrix)
    recommendation = _recommendation(summary, by_length, by_hub)
    return {
        "job_id": job.get("id"),
        "created": job.get("created"),
        "finished": datetime.now().isoformat(timespec="seconds"),
        "dataset": job.get("dataset_label") or job.get("index_name"),
        "index_name": job.get("index_name"),
        "period": _period_label(job),
        "method": {
            "pass1_model": "первичная автоматическая разметка",
            "pass2_model": "уточняющая проверка",
            "batch_size": int(job.get("batch_size") or DEFAULT_BATCH),
            "parallel": int(job.get("parallel") or DEFAULT_PARALLEL),
            "conf_threshold": float(job.get("conf_threshold") or DEFAULT_CONF),
            "fields": list(TONE_FIELDS),
            "source_field": "разметка источника (не изменяется)",
            "pass2": dict(pass2),
        },
        "summary": summary,
        "matrix": matrix,
        "per_class": per_class,
        "by_hub": by_hub,
        "by_section": by_section,
        "by_type": by_type,
        "by_length": by_length,
        "examples": examples,
        "conclusions": conclusions,
        "recommendation": recommendation,
    }


def _period_label(job: Dict[str, Any]) -> str:
    lo, hi = job.get("min_date"), job.get("max_date")
    if not lo and not hi:
        return "весь период"
    fmt = lambda value: datetime.fromtimestamp(float(value)).strftime("%d.%m.%Y") if value else "…"
    return "%s — %s" % (fmt(lo), fmt(hi))


def _conclusions(summary: Dict[str, Any], by_length: List[Dict[str, Any]],
                 by_hub: List[Dict[str, Any]], per_class: List[Dict[str, Any]],
                 matrix: Optional[Dict[str, Any]] = None) -> List[str]:
    """5–7 выводов человеческим языком — это главное, что читает пользователь."""
    out: List[str] = []
    agreement = float(summary.get("agreement") or 0.0)
    kappa = float(summary.get("kappa") or 0.0)
    level = ("высокое" if agreement >= 0.85 else "среднее" if agreement >= 0.7 else "низкое")
    out.append(
        "Проверено %s: наша оценка совпала с разметкой источника в %s случаев, "
        "каппа Коэна %.2f — согласие %s (каппа выше 0,6 считается хорошей, ниже 0,4 — слабой)."
        % (_plural(int(summary.get("evaluated") or 0), "сообщение", "сообщения", "сообщений"),
           _pct(agreement), kappa, level))

    worst = None
    for row in by_length:
        if row["total"] >= 5 and (worst is None or row["share"] > worst["share"]):
            worst = row
    if worst and worst["share"] > 0.0:
        out.append("Больше всего расхождений на текстах длиной %s: %s при %d сообщениях."
                   % (worst["key"], _pct(float(worst["share"])), worst["total"]))
    if by_length:
        best = min(by_length, key=lambda row: float(row["share"]))
        if best is not worst:
            out.append("Лучше всего разметка источника держится на текстах %s — расхождений %s."
                       % (best["key"], _pct(float(best["share"]))))

    matrix = matrix or {}
    top_pair = None
    for a in CLASSES:
        for b in CLASSES:
            if a == b:
                continue
            count = int((matrix.get(str(a)) or {}).get(str(b)) or 0)
            if count and (top_pair is None or count > top_pair[2]):
                top_pair = (a, b, count)
    if top_pair:
        out.append("Самая частая путаница: наша оценка ставит «%s» там, где источник поставил «%s» — %s."
                   % (TONE_RU[top_pair[1]], TONE_RU[top_pair[0]],
                      _plural(top_pair[2], "случай", "случая", "случаев")))

    diff = int(summary.get("model_more_negative") or 0) - int(summary.get("model_more_positive") or 0)
    if abs(diff) >= max(3, int(0.01 * max(1, summary.get("evaluated") or 1))):
        if diff > 0:
            out.append("Наша оценка строже источника: негатив там, где источник поставил более "
                       "мягкую оценку (%s против %s) — источник часть недовольства относит к нейтрали."
                       % (_plural(int(summary.get("model_more_negative") or 0), "случай", "случая", "случаев"),
                          _plural(int(summary.get("model_more_positive") or 0), "обратного", "обратных", "обратных")))
        else:
            out.append("Наша оценка мягче источника: нейтрал или позитив там, где источник "
                       "видит негатив (%s против %s)."
                       % (_plural(int(summary.get("model_more_positive") or 0), "случай", "случая", "случаев"),
                          _plural(int(summary.get("model_more_negative") or 0), "обратного", "обратных", "обратных")))

    if by_hub:
        bad = by_hub[0]
        out.append("Площадка с наибольшим числом расхождений — %s: %d из %d сообщений (%s)."
                   % (bad["key"], bad["mismatch"], bad["total"], _pct(float(bad["share"]))))

    share = float(summary.get("pass2_share") or 0.0)
    out.append("Спорных случаев, прошедших уточняющую проверку, — %s от выборки (%d сообщений); "
               "по ним решение окончательное." % (_pct(share), int(summary.get("pass2_decided") or 0)))
    if not summary.get("pass2_available", True):
        out.append("ВНИМАНИЕ: уточняющая проверка была недоступна — спорные случаи остались "
                   "результатом первичной автоматической разметки, окончательными их считать нельзя.")
    capped = int(summary.get("pass2_capped") or 0)
    if capped:
        out.append("Спорных оказалось %d — больше лимита %.0f%% от выборки, поэтому уточняющую "
                   "проверку прошли только %d самых неуверенных сообщений; у остальных %d "
                   "осталась первичная автоматическая разметка."
                   % (int(summary.get("pass2_candidates") or 0),
                      float(summary.get("pass2_max_share") or PASS2_MAX_SHARE) * 100,
                      int(summary.get("pass2_decided") or 0), capped))
    if int(summary.get("unresolved") or 0):
        out.append("Для %d сообщений автоматическая разметка не дала вердикт: они исключены "
                   "из расчёта согласия, а не выданы за нашу оценку."
                   % int(summary["unresolved"]))
    return out[:7]


def _recommendation(summary: Dict[str, Any], by_length: List[Dict[str, Any]],
                    by_hub: List[Dict[str, Any]]) -> str:
    agreement = float(summary.get("agreement") or 0.0)
    kappa = float(summary.get("kappa") or 0.0)
    weak = None
    for row in by_length:
        if row["total"] >= 5 and float(row["share"]) >= 0.25:
            weak = row
            break
    if agreement >= 0.85 and kappa >= 0.6:
        text = ("Источнику можно доверять: разметка источника и наша оценка совпадают почти всегда "
                "(согласие %s, каппа %.2f). Для отчётности можно опираться на поле источника."
                % (_pct(agreement), kappa))
        if weak:
            text += (" Оговорка: на текстах %s расхождений заметно больше (%s) — такой подпериод "
                     "лучше считать по нашей оценке." % (weak["key"], _pct(float(weak["share"]))))
        return text
    if agreement >= 0.7:
        text = ("Источник годится для общих трендов, но не для точных цифр: согласие %s, каппа %.2f. "
                "Для решений и публичных цифр считайте по нашим оценкам тональности "
                "(отдельная колонка в данных)."
                % (_pct(agreement), kappa))
        if weak:
            text += (" Основная слабость источника — %s: расхождения %s."
                     % (weak["key"], _pct(float(weak["share"]))))
        if by_hub:
            text += (" Хуже всего дела на площадке %s (%d расхождений из %d)."
                     % (by_hub[0]["key"], by_hub[0]["mismatch"], by_hub[0]["total"]))
        return text
    text = ("Разметке источника доверять нельзя: согласие всего %s, каппа %.2f — источник и наша "
            "оценка расходятся слишком часто. Считайте по нашим оценкам тональности "
            "(отдельная колонка в данных)."
            % (_pct(agreement), kappa))
    if weak:
        text += (" Главная причина — %s: там расхождения %s."
                 % (weak["key"], _pct(float(weak["share"]))))
    if by_hub:
        text += (" Проверьте площадку %s отдельно (%d расхождений из %d)."
                 % (by_hub[0]["key"], by_hub[0]["mismatch"], by_hub[0]["total"]))
    return text


def _sections(report: Dict[str, Any], pass2: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Разделы для платформенного сборщика DOCX/PDF."""
    summary = report["summary"]
    matrix = report["matrix"]
    src_columns = ["Источник \\ наша оценка"] + [TONE_RU[c] for c in CLASSES]
    matrix_rows = []
    for a in CLASSES:
        row = [TONE_RU[a]]
        for b in CLASSES:
            row.append(int((matrix.get(str(a)) or {}).get(str(b)) or 0))
        matrix_rows.append(row)

    pass2_row = "не выполнялась (%s)" % (pass2.get("reason") or "нет спорных случаев") \
        if not pass2.get("available", True) else "выполнена"

    sections: List[Dict[str, Any]] = []
    sections.append({
        "heading": "Как проверяли",
        "text": ("Набор данных: %s. Период: %s. Режим: %s. Проверено сообщений: %d.\n"
                 "Сообщения размечают автоматически в два этапа: сначала идёт первичная "
                 "автоматическая разметка всей выборки, затем спорные случаи перепроверяет более "
                 "точная модель, и её решение окончательное. Спорным считается случай, где "
                 "автоматическая оценка неуверенна (ниже %.2f) или расходится с разметкой источника.\n"
                 "Уточняющая проверка: %s.\n"
                 "Результаты проверки сохраняются отдельно от разметки источника: исходные данные "
                 "не меняются, существующая аналитика продолжает работать по ним."
                 % (report.get("dataset") or "—", report.get("period") or "—",
                    "полная разметка" if summary.get("mode") == "full" else
                    "выборка %s сообщений" % (summary.get("sample_size") or "—"),
                    summary.get("checked") or 0,
                    float(report["method"]["conf_threshold"]), pass2_row)),
        "tables": [{
            "title": "Ключевые цифры",
            "columns": ["Показатель", "Значение"],
            "rows": [
                ["Сообщений проверено", str(summary.get("checked") or 0)],
                ["Учтено в сравнении с источником", str(summary.get("evaluated") or 0)],
                ["Согласие нашей оценки с источником", _pct(float(summary.get("agreement") or 0.0))],
                ["Совпало", str(summary.get("matches") or 0)],
                ["Расхождений", str(summary.get("mismatches") or 0)],
                ["Каппа Коэна", "%.3f" % float(summary.get("kappa") or 0.0)],
                ["Спорных случаев (уточняющая проверка)", _pct(float(summary.get("pass2_share") or 0.0))],
                ["Решений после уточняющей проверки", str(summary.get("pass2_decided") or 0)],
                ["Без вердикта автоматической разметки", str(summary.get("unresolved") or 0)],
                ["Сохранено оценок", str(summary.get("bulk_written") or 0)],
            ],
            "layout": "portrait",
        }],
    })
    sections.append({
        "heading": "Матрица «источник → наша оценка»",
        "text": ("Строки — тональность источника, столбцы — наша оценка тональности. "
                 "Диагональ — совпадения."),
        "tables": [{
            "title": "Распределение «источник → наша оценка»",
            "columns": src_columns,
            "rows": matrix_rows,
            "layout": "portrait",
        }, {
            "title": "Точность и полнота по классам (эталон — источник)",
            "columns": ["Класс", "Точность", "Полнота", "F1", "В источнике", "Наша оценка поставила"],
            "rows": [[row["label"], _num(row["precision"]), _num(row["recall"]), _num(row["f1"]),
                      row["support"], row["predicted"]] for row in report["per_class"]],
            "note": "Точность — доля верных среди наших оценок по классу, полнота — доля "
                    "найденных сообщений этого класса источника.",
            "layout": "portrait",
        }],
    })
    sections.append({
        "heading": "Где расходимся: площадки и разделы",
        "text": "Разбивка расхождений по площадкам, типам источников и разделам набора данных.",
        "tables": [
            {"title": "Площадки", "columns": ["Площадка", "Сообщений", "Расхождений", "Доля"],
             "rows": [[row["key"], row["total"], row["mismatch"], _pct(float(row["share"]))]
                      for row in report["by_hub"]], "layout": "auto"},
            {"title": "Тип источника", "columns": ["Тип", "Сообщений", "Расхождений", "Доля"],
             "rows": [[row["key"], row["total"], row["mismatch"], _pct(float(row["share"]))]
                      for row in report["by_section"]], "layout": "auto"},
            {"title": "Раздел набора", "columns": ["Раздел", "Сообщений", "Расхождений", "Доля"],
             "rows": [[row["key"], row["total"], row["mismatch"], _pct(float(row["share"]))]
                      for row in report["by_type"]], "layout": "auto"},
        ],
    })
    sections.append({
        "heading": "Зависимость от длины текста",
        "text": ("На коротких отзывах и репликах тональность выражена слабее — если расхождения "
                 "растут именно там, источник стоит перепроверять на коротких текстах отдельно."),
        "tables": [{
            "title": "Расхождения по длине текста",
            "columns": ["Длина текста", "Сообщений", "Расхождений", "Доля"],
            "rows": [[row["key"], row["total"], row["mismatch"], _pct(float(row["share"]))]
                     for row in report["by_length"]],
            "layout": "portrait",
        }],
    })
    examples = report.get("examples") or []
    tables = []
    if examples:
        tables.append({
            "title": "Примеры расхождений (до %d)" % MAX_EXAMPLES,
            "columns": ["Площадка", "Дата", "Источник → наша оценка", "Цитата из сообщения",
                        "Пояснение", "Ссылка"],
            "rows": [[item["hub"], item["date"],
                      "%s → %s" % (item["source"], item["model"]),
                      item["quote"], item["reason"], item["url"] or "—"] for item in examples],
            "note": "Ссылки приведены текстом — их можно скопировать и открыть в браузере.",
            "layout": "landscape",
        })
    sections.append({
        "heading": "Примеры расхождений",
        "text": "Показаны самые уверенные наши оценки, разошедшиеся с источником: именно такие "
                "случаи чаще всего означают ошибку разметки источника.",
        "tables": tables,
    })
    sections.append({
        "heading": "Выводы",
        "bullets": report.get("conclusions") or [],
    })
    sections.append({
        "heading": "Рекомендация",
        "text": report.get("recommendation") or "",
    })
    return sections


def _reports_dir(user_id: Any, folder: str) -> str:
    path = os.path.join(DATA_DIR, str(user_id), REPORTS_DIR_NAME, folder)
    os.makedirs(path, exist_ok=True)
    return path


def _write_report_files(job: Dict[str, Any], report: Dict[str, Any]) -> Dict[str, str]:
    """DOCX и PDF платформенным сборщиком в папку отчётов владельца задачи."""
    from agent_engine.tools_reports import _build_docx, _build_pdf, _safe_name

    owner = str(job.get("owner") or "")
    if not owner:
        return {}
    out: Dict[str, str] = {}
    label = _safe_name(report.get("dataset") or job.get("index_name") or "набор", 40)
    stamp = datetime.now().strftime("%Y-%m-%d %H-%M")
    mode = "полная" if (job.get("mode") == "full") else "выборка %s" % (job.get("sample_size") or "")
    title = "Проверка тональности: качество разметки источника"
    subtitle = "%s — %s, %s" % (label, report.get("period") or "", mode)
    meta = {
        "dataset_label": report.get("dataset") or "",
        "period": report.get("period") or "",
        "author": "Tellscope, автоматическая разметка",
        "date": datetime.now().strftime("%d.%m.%Y %H:%M"),
    }
    sections = _sections(report, (report.get("method") or {}).get("pass2") or {})
    folder = _reports_dir(owner, REPORT_FOLDER)
    for kind, builder, ext in (("docx", _build_docx, "docx"), ("pdf", _build_pdf, "pdf")):
        path = os.path.join(folder, "%s %s.%s" % (label, stamp, ext))
        try:
            builder(path, title, subtitle, sections, meta)
            out[kind] = path
        except Exception as exc:  # noqa: BLE001 — без файла отчёт всё равно есть в JSON
            _log_line("[%s] не удалось собрать %s: %s" % (job.get("id"), ext.upper(), str(exc)[:200]),
                      level="error")
    return out


# --------------------------------------------------------------------------- #
# HTTP API
# --------------------------------------------------------------------------- #

router = APIRouter(prefix="/tone-check", tags=["tone check"])


class ToneCheckBody(BaseModel):
    index: str = Field(description="набор данных, по которому идёт проверка")
    mode: str = Field(default="sample", description="sample | full")
    sample_size: int = Field(default=1000, ge=1, le=MAX_FULL_DOCS)
    min_date: Optional[int] = None
    max_date: Optional[int] = None
    tone: str = Field(default="", description="фильтр по тональности источника: negative/neutral/positive")
    conf_threshold: float = Field(default=DEFAULT_CONF, ge=0.0, le=1.0)
    batch_size: int = Field(default=0, ge=0, le=40)
    parallel: int = Field(default=0, ge=0, le=32)
    pass2_parallel: int = Field(default=0, ge=0, le=16)
    relabel: bool = Field(default=False, description="размечать заново, даже если поля уже заполнены")
    # ---- область проверки ----
    label_mode: str = Field(default="message", description="message (тон сообщения) | aspect (отношение к объекту)")
    objects: List[str] = Field(default_factory=list, description="объекты аспектной разметки: бренд, продукт, конкурент")
    theme: str = Field(default="", description="инфоповод или тема, по которой сужаем проверку")
    hub: str = Field(default="", description="площадка (поле hub), точное совпадение")
    author: str = Field(default="", description="автор (authorObject.fullname), поиск по фразе")
    preset: str = Field(default="", description="имя пресета, если запуск из сохранённого набора")


def _human_sec(value: Optional[float]) -> str:
    """Секунды в короткую человеческую подпись («8 с», «3 мин», «1 ч 05 мин»)."""
    if value is None:
        return ""
    total = int(max(0, value))
    if total < 60:
        return "%d с" % total
    if total < 3600:
        return "%d мин" % (total // 60)
    hours, minutes = divmod(total // 60, 60)
    return "%d ч %02d мин" % (hours, minutes)


def _public_status(job: Dict[str, Any]) -> Dict[str, Any]:
    """Статус задачи для интерфейса: прогресс, этап, статистика на ходу."""
    report = None
    if os.path.isfile(_report_path(job["id"])):
        try:
            with open(_report_path(job["id"]), "r", encoding="utf-8") as handle:
                report = json.load(handle)
        except Exception:
            report = None
    prog = _stage_progress(job)
    rate, eta = _rate_and_eta(job, prog)
    now = datetime.now()

    def _age(stamp: Any) -> Optional[float]:
        try:
            return max(0.0, (now - datetime.fromisoformat(str(stamp))).total_seconds())
        except Exception:
            return None

    updated_ago = _age(job.get("updated"))
    stale = _age(job.get("last_progress_at"))
    if stale is None:
        stale = updated_ago
    running = str(job.get("status") or "") == "running"
    writing = bool(job.get("writing"))
    stalled = bool(running and stale is not None and stale > STALE_AFTER)
    note = ""
    if running:
        if stalled:
            note = "нет новых результатов %s — жду ответа модели" % _human_sec(stale)
        elif prog["stage"] == "pass2":
            # Второй проход на 32B заведомо медленнее: подпись объясняет паузу счётчика.
            note = "идёт уточняющая проверка спорных случаев — она медленнее первого этапа"
        if writing:
                note = "сохраняю результат"
    if job.get("error"):
        note = str(job.get("error"))
    out = {
        "job_id": job.get("id"),
        "status": job.get("status"),
        "stage": prog["stage"],
        "stage_label": prog["stage_label"],
        "percent": int(job.get("percent") or 0),
        # Процент текущего этапа — ровно тот же, что у счётчика «stage_done / stage_total».
        "stage_percent": prog["stage_percent"],
        "stage_done": prog["stage_done"],
        "stage_total": prog["stage_total"],
        "pass1_percent": prog["pass1_percent"],
        "pass1_done": prog["pass1_done"],
        "pass1_total": prog["pass1_total"],
        "pass2_percent": prog["pass2_percent"],
        "processed": int(job.get("processed") or 0),
        "total": int(job.get("total") or 0),
        "pass2_done": prog["pass2_done"],
        "pass2_total": prog["pass2_total"],
        "index_name": job.get("index_name"),
        "index_key": job.get("index_key"),
        "dataset_label": job.get("dataset_label"),
        "mode": job.get("mode"),
        "sample_size": job.get("sample_size"),
        "created": job.get("created"),
        "started": job.get("started"),
        "finished": job.get("finished"),
        "updated": job.get("updated"),
        "elapsed_sec": job.get("elapsed_sec"),
        # Живые «часы» интерфейса: без них замерший счётчик выглядит как зависший проход.
        "updated_ago_sec": int(updated_ago) if updated_ago is not None else None,
        "stale_sec": int(stale) if stale is not None else None,
        "stalled": stalled,
        "writing": writing,
        "note": note,
        "rate_per_min": rate or float(job.get("rate_per_min") or 0),
        "rate_messages_per_min": rate or float(job.get("rate_per_min") or 0),
        "eta_sec": eta,
        "eta_text": _human_sec(eta) if eta else "",
        "bulk_written": int(job.get("bulk_written") or 0),
        "write_errors": int(job.get("errors") or 0),
        "error": job.get("error") or "",
        "has_report": bool(report),
        "report_files": job.get("report_files") or {},
        "log": (job.get("log") or [])[-12:],
        "label_mode": str(job.get("label_mode") or "message"),
        "objects": _object_terms(job.get("objects")),
        "theme": job.get("theme") or "",
        "hub": job.get("hub") or "",
        "author": job.get("author") or "",
        "preset": job.get("preset") or "",
        "scope": _scope_public(job),
        "scope_text": _scope_text(job),
    }
    if report:
        summary = report.get("summary") or {}
        out["summary"] = {
            "checked": summary.get("checked"),
            "agreement": summary.get("agreement"),
            "kappa": summary.get("kappa"),
            "mismatches": summary.get("mismatches"),
            "pass2_share": summary.get("pass2_share"),
        }
        if summary.get("objects"):
            # Аспектный отчёт: у него своя сводка — по объектам, а не согласие с источником.
            out["aspect_objects"] = summary.get("objects")
            out["source_note"] = report.get("source_note") or ""
        out["recommendation"] = report.get("recommendation") or ""
        out["conclusions"] = (report.get("conclusions") or [])[:7]
    return out


@router.get("/datasets")
def datasets(user: Any = Depends(current_user_any)):
    """Датасеты, доступные пользователю: ключ, имя, число сообщений, сколько уже размечено."""
    mapping = _index_map()
    if getattr(user, "is_superuser", False):
        names = {_norm_name(name) for name in mapping.values()}
    else:
        stems = _allowed_stems(getattr(user, "id", None))
        names = {_norm_name(name) for name in mapping.values() if _norm_name(name).lower() in stems}
    names = {str(name) for name in names if name}
    counts: Dict[str, int] = {}
    try:
        for row in _es().cat.indices(format="json", h="index,docs.count"):
            counts[str(row.get("index"))] = int(float(row.get("docs.count") or 0))
    except Exception:
        counts = {}
    reverse: Dict[str, int] = {}
    for key, value in mapping.items():
        reverse.setdefault(_norm_name(value), key)
    items = []
    for name in sorted(names):
        if counts.get(name, 0) <= 0:
            continue
        labeled = 0
        try:
            labeled = int(_es().count(index=name, query={"exists": {"field": "tone_llm"}}).get("count") or 0)
        except Exception:
            labeled = 0
        items.append({
            "index": reverse.get(name),
            "name": name,
            "label": _pretty_label(name),
            "docs": counts.get(name, 0),
            "labeled": labeled,
        })
    items.sort(key=lambda item: item["docs"])
    return {"datasets": items}


@router.get("/jobs")
def jobs(user: Any = Depends(current_user_any), limit: int = Query(default=20, ge=1, le=100)):
    """Свои задачи проверки тональности (свежие сверху)."""
    _ensure_recovery()
    me = str(getattr(user, "id", ""))
    admin = bool(getattr(user, "is_superuser", False))
    items = []
    if os.path.isdir(STATE_DIR):
        for entry in os.listdir(STATE_DIR):
            if not entry.endswith(".json") or entry.endswith(".report.json"):
                continue
            job = _job_load(entry[:-5])
            if not job:
                continue
            if not admin and str(job.get("owner") or "") != me:
                continue
            items.append(job)
    items.sort(key=lambda job: str(job.get("created") or ""), reverse=True)
    return {"jobs": [_public_status(job) for job in items[:limit]]}


@router.post("")
def start(body: ToneCheckBody, user: Any = Depends(current_user_any)):
    """Запуск проверки тональности. Работает в фоне и переживает закрытие браузера."""
    _ensure_recovery()
    mode = str(body.mode or "sample").strip().lower()
    if mode not in ("sample", "full"):
        raise HTTPException(status_code=400, detail="Режим: sample или full")
    key, name = _guard_dataset(user, body.index)
    if not _es().indices.exists(index=name):
        raise HTTPException(status_code=404, detail="Данные набора не найдены — возможно, он удалён")
    tone_int = _tone_filter(body.tone)
    label_mode = str(body.label_mode or "message").strip().lower()
    if label_mode not in ("message", "aspect"):
        raise HTTPException(status_code=400, detail="Режим разметки: message или aspect")
    objects = _object_terms(body.objects)
    if label_mode == "aspect" and not objects:
        raise HTTPException(status_code=400, detail="Добавьте хотя бы один объект для проверки по объектам")
    jid = uuid.uuid4().hex[:12]
    job = {
        "id": jid,
        "owner": str(getattr(user, "id", "")),
        "owner_email": str(getattr(user, "email", "") or ""),
        "index_key": key,
        "index_name": name,
        "dataset_label": _pretty_label(name),
        "mode": mode,
        "sample_size": int(body.sample_size),
        "min_date": body.min_date,
        "max_date": body.max_date,
        "tone_int": tone_int,
        "conf_threshold": float(body.conf_threshold),
        "batch_size": int(body.batch_size or 0),
        "parallel": int(body.parallel or 0),
        "pass2_parallel": int(body.pass2_parallel or 0),
        "relabel": bool(body.relabel),
        "label_mode": label_mode,
        "objects": objects,
        "theme": _flat(body.theme),
        "hub": _flat(body.hub),
        "author": _flat(body.author),
        "preset": _flat(body.preset),
        "seed": int(time.time()) % 2147483647,
        "status": "queued",
        "stage": "preparing",
        "stage_label": STAGE_LABELS["preparing"],
        "percent": 0,
        "processed": 0,
        "total": 0,
        "bulk_written": 0,
        "errors": 0,
        "cursor": None,
        "created": datetime.now().isoformat(timespec="seconds"),
        "log": [],
    }
    _job_save(job)
    _redis_put(jid, {"status": "queued", "stage": "preparing", "percent": 0,
                     "stage_label": STAGE_LABELS["preparing"], "owner": job["owner"]})
    if body.relabel:
        # Явная повторная разметка: журнал и курсор обнуляем, иначе проход «продолжится».
        for path in (_journal_path(jid), _report_path(jid)):
            try:
                if os.path.isfile(path):
                    os.remove(path)
            except Exception:
                pass
    _start_job(jid)
    _log_line("[%s] запуск: %s (%s, %s, %s), владелец %s"
              % (jid, name, mode, label_mode, ", ".join(objects) or "без объектов", job["owner"]))
    return {"job_id": jid, "status": "queued", "index_name": name, "mode": mode,
            "label_mode": label_mode, "objects": objects, "scope": _scope_public(job)}


@router.get("/scope")
def scope_preview(index: str = Query(description="набор данных, по которому считаем объём"),
                  objects: str = Query(default="", description="объекты через запятую"),
                  theme: str = Query(default=""), hub: str = Query(default=""),
                  author: str = Query(default=""), tone: str = Query(default=""),
                  min_date: Optional[int] = None, max_date: Optional[int] = None,
                  label_mode: str = Query(default="message"),
                  user: Any = Depends(current_user_any)):
    """Объём проверки под заданной областью — до запуска.

    Интерфейс показывает это число и человеческое описание области («проверяю: объект
    «Rostic's», тема «качество еды», период 01–31.07.2026, площадка 2gis.ru»), чтобы
    запуск был осознанным, а не «на весь датасет».
    """
    _, name = _guard_dataset(user, index)
    job = {
        "index_name": name,
        "label_mode": str(label_mode or "message").strip().lower(),
        "objects": _object_terms(objects),
        "theme": _flat(theme),
        "hub": _flat(hub),
        "author": _flat(author),
        "min_date": min_date,
        "max_date": max_date,
        "tone_int": _tone_filter(tone),
    }
    query = _base_query(job)
    marker = ASPECT_MARKER if job["label_mode"] == "aspect" else "tone_llm"
    try:
        count = int(_es().count(index=name, query=query).get("count") or 0)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Не удалось посчитать объём: %s" % str(exc)[:200])
    done = 0
    try:
        done = int(_es().count(index=name, query={
            "bool": {"filter": [query], "must": [{"exists": {"field": marker}}]}}).get("count") or 0)
    except Exception:
        done = 0
    return {
        "index_name": name,
        "index_key": _resolve_dataset(index)[0],
        "label": _pretty_label(name),
        "label_mode": job["label_mode"],
        "count": count,
        "labeled": done,
        "remaining": max(0, count - done),
        "objects": job["objects"],
        "scope": _scope_public(job),
        "scope_text": _scope_text(job),
        "by_tone": _cat_counts(name, "toneMark", query, 5),
        "by_hub": _cat_counts(name, "hub", query, 8),
    }


@router.get("/scope-options")
def scope_options(index: str = Query(description="набор данных для подсказок"),
                  limit: int = Query(default=30, ge=5, le=80),
                  user: Any = Depends(current_user_any)):
    """Подсказки для области проверки: темы датасета, частые термины, площадки, авторы."""
    _, name = _guard_dataset(user, index)
    base = _base_query({"index_name": name})
    return {
        "index_name": name,
        "index_key": _resolve_dataset(index)[0],
        "label": _pretty_label(name),
        "docs": int(_es().count(index=name).get("count") or 0),
        "themes": _theme_catalog(user, name),
        "objects": _term_suggestions(name, base, limit),
        "hubs": _cat_counts(name, "hub", None, 20),
        "authors": _author_suggestions(name, base, 15),
    }


class PresetBody(BaseModel):
    name: str = Field(min_length=1, max_length=120, description="имя набора, например «KFC → Rostic's, еда, лето 2026»")
    index: str
    label_mode: str = "message"
    mode: str = "sample"
    sample_size: int = Field(default=1000, ge=1, le=MAX_FULL_DOCS)
    min_date: Optional[int] = None
    max_date: Optional[int] = None
    tone: str = ""
    objects: List[str] = Field(default_factory=list)
    theme: str = ""
    hub: str = ""
    author: str = ""
    conf_threshold: float = Field(default=DEFAULT_CONF, ge=0.0, le=1.0)


@router.get("/presets")
def presets_list(user: Any = Depends(current_user_any)):
    """Сохранённые наборы «датасет + объекты + тема + период + площадка + режим»."""
    items = _presets_load(getattr(user, "id", ""))
    for item in items:
        item["note"] = _preset_note(item)
    return {"presets": items}


@router.post("/presets")
def preset_save(body: PresetBody, user: Any = Depends(current_user_any)):
    """Сохранить или перезаписать набор по имени (запуск из списка — в один клик)."""
    uid = getattr(user, "id", "")
    record = {
        "name": _flat(body.name)[:120],
        "index": str(body.index),
        "label_mode": str(body.label_mode or "message").strip().lower(),
        "mode": str(body.mode or "sample").strip().lower(),
        "sample_size": int(body.sample_size),
        "min_date": body.min_date,
        "max_date": body.max_date,
        "tone": _flat(body.tone),
        "objects": _object_terms(body.objects),
        "theme": _flat(body.theme),
        "hub": _flat(body.hub),
        "author": _flat(body.author),
        "conf_threshold": float(body.conf_threshold),
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    items = [item for item in _presets_load(uid) if _norm_key(item.get("name")) != _norm_key(record["name"])]
    items.insert(0, record)
    items = items[:50]
    _presets_save(uid, items)
    record["note"] = _preset_note(record)
    for item in items:
        item["note"] = _preset_note(item)
    return {"saved": record, "presets": items}


@router.delete("/presets/{name}")
def preset_delete(name: str, user: Any = Depends(current_user_any)):
    """Удалить набор по имени."""
    uid = getattr(user, "id", "")
    items = _presets_load(uid)
    left = [item for item in items if _norm_key(item.get("name")) != _norm_key(name)]
    _presets_save(uid, left)
    for item in left:
        item["note"] = _preset_note(item)
    return {"deleted": len(items) - len(left), "presets": left}


@router.get("/{job_id}")
def status(job_id: str, user: Any = Depends(current_user_any)):
    _ensure_recovery()
    job = _job_load(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    if str(job.get("owner") or "") != str(getattr(user, "id", "")) and not getattr(user, "is_superuser", False):
        raise HTTPException(status_code=403, detail="Нет доступа к задаче другого пользователя")
    return _public_status(job)


@router.post("/{job_id}/cancel")
def cancel(job_id: str, user: Any = Depends(current_user_any)):
    """Кооперативная остановка: воркер завершает текущую пачку и останавливается."""
    job = _job_load(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    if str(job.get("owner") or "") != str(getattr(user, "id", "")) and not getattr(user, "is_superuser", False):
        raise HTTPException(status_code=403, detail="Нет доступа к задаче другого пользователя")
    status_now = str(job.get("status") or "")
    if status_now in ("done", "cancelled", "error"):
        return {"job_id": job_id, "status": status_now, "message": "Задача уже завершена"}
    job["cancel"] = True
    _job_save(job)
    try:
        _redis.hset(_key(job_id), "cancel", "1")
    except Exception:
        pass
    _set(job_id, log="получена команда остановки")
    return {"job_id": job_id, "status": "cancelling"}


@router.get("/{job_id}/report")
def report_json(job_id: str, user: Any = Depends(current_user_any)):
    job = _job_load(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    if str(job.get("owner") or "") != str(getattr(user, "id", "")) and not getattr(user, "is_superuser", False):
        raise HTTPException(status_code=403, detail="Нет доступа к задаче другого пользователя")
    if not os.path.isfile(_report_path(job_id)):
        raise HTTPException(status_code=409, detail="Отчёт ещё не готов: статус %s" % job.get("status"))
    with open(_report_path(job_id), "r", encoding="utf-8") as handle:
        return json.load(handle)


@router.get("/{job_id}/report/file")
def report_file(job_id: str, fmt: str = Query(default="docx"), user: Any = Depends(current_user_any)):
    job = _job_load(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    if str(job.get("owner") or "") != str(getattr(user, "id", "")) and not getattr(user, "is_superuser", False):
        raise HTTPException(status_code=403, detail="Нет доступа к задаче другого пользователя")
    fmt = str(fmt or "docx").lower()
    if fmt not in ("docx", "pdf"):
        raise HTTPException(status_code=400, detail="fmt: docx или pdf")
    path = str((job.get("report_files") or {}).get(fmt) or "")
    if not path or not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Файл отчёта не найден")
    media = ("application/pdf" if fmt == "pdf" else
             "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    return FileResponse(path, media_type=media, filename=os.path.basename(path))


# =========================================================================== #
# Область проверки: объект / инфоповод-тема / площадка / автор
#
# Зачем: «весь датасет» — почти всегда не тот вопрос. Маркетингу нужно «что думают
# про Rostic's», «что говорят про крылышки в отзывах 2ГИС за июль». Поэтому у задачи
# есть область проверки, она видна в интерфейсе, показывает объём ДО запуска и попадает
# в отчёт. Фильтры работают в обоих режимах разметки.
# =========================================================================== #

PRESETS_FILE = "tone_check_presets.json"
MAX_OBJECTS = 6
SCOPE_OPTIONS_DOCS = 800

STOPWORDS = frozenset("""
и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только
ее мне было вот от меня еще нет о из ему теперь когда даже ну вдруг ли если уже или ни
быть был него до вас нибудь опять уж вам ведь там потом себя ничего ей может они тут где
есть надо ней для мы тебя их чем была сам чтоб без будто чего раз тоже себе под будет ж
тогда кто этот того потому этого какой совсем ним здесь этом один почти мой тем чтобы нее
сейчас были куда зачем всех никогда можно при наконец два об другой хоть после над больше
тот через эти нас про всего них какая много разве три эту моя впрочем хорошо свою этой
перед иногда лучше чуть том нельзя такой им более всегда конечно всю между это который
также свою есть если того чтобы меня тебе нами вами ими себе весь вся всё оно эти эта
этот таких такой также очень просто либо весь кого кому чему чем тем тех тех этих
""".split())


def _flat(value: Any) -> str:
    """Однострочный текст без лишних пробелов."""
    return " ".join(str(value if value is not None else "").split())


def _norm_key(value: Any) -> str:
    """Ключ сравнения: регистр, пробелы и пунктуация не важны."""
    return re.sub(r"[^0-9a-zа-яё]+", "", _flat(value).lower())


def _author_name(value: Any) -> str:
    """Имя автора из поля authorObject (объект, список или строка)."""
    if isinstance(value, dict):
        for key in ("fullname", "name", "title"):
            if _flat(value.get(key)):
                return _flat(value[key])
        return ""
    if isinstance(value, (list, tuple)) and value:
        return _author_name(value[0])
    return _flat(value)


def _object_terms(raw: Any) -> List[str]:
    """Объекты аспектной разметки из массива или строки «KFC, Rostic's»."""
    if raw is None:
        return []
    items = list(raw) if isinstance(raw, (list, tuple)) else re.split(r"[,\n;]+", str(raw))
    out: List[str] = []
    seen = set()
    for item in items:
        text = _flat(item).strip(" \"'«»")
        key = _norm_key(text)
        if not text or not key or key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out[:MAX_OBJECTS]


def _object_query(objects: Any) -> Optional[Dict[str, Any]]:
    """Сужающий фильтр «в тексте упомянут хотя бы один из объектов».

    Поле ``text`` в части индексов размечено русским анализатором, поэтому склонения
    («крылышки»/«крылышек») находятся одной фразой. Плюс то же по заголовку.
    """
    terms = _object_terms(objects)
    if not terms:
        return None
    should: List[Dict[str, Any]] = []
    for term in terms:
        should.append({"match_phrase": {"text": term}})
        should.append({"match_phrase": {"title": term}})
    return {"bool": {"should": should, "minimum_should_match": 1}}


def _theme_query(theme: Any) -> Optional[Dict[str, Any]]:
    """Инфоповод/тема датасета как текстовый фильтр по значимым словам названия."""
    text = _flat(theme)
    if not text:
        return None
    words = [word for word in re.split(r"[^0-9A-Za-zА-Яа-яЁё'\-]+", text) if len(word) > 2]
    if not words:
        return None
    should: List[Dict[str, Any]] = []
    for word in words[:6]:
        should.append({"match_phrase": {"text": word}})
        should.append({"match_phrase": {"title": word}})
    return {"bool": {"should": should, "minimum_should_match": 1}}


def _fmt_date(value: Any) -> str:
    try:
        return datetime.fromtimestamp(float(value)).strftime("%d.%m.%Y")
    except Exception:
        return ""


def _scope_text(job: Dict[str, Any]) -> str:
    """Человеческое описание области проверки для интерфейса и отчёта."""
    parts: List[str] = []
    objects = _object_terms(job.get("objects"))
    aspect = str(job.get("label_mode") or "message") == "aspect"
    if objects:
        if aspect:
            parts.append("объект%s: %s" % ("ы" if len(objects) > 1 else "",
                                           ", ".join("«%s»" % item for item in objects)))
        else:
            parts.append("в тексте есть: %s" % ", ".join("«%s»" % item for item in objects))
    if _flat(job.get("theme")):
        parts.append("тема «%s»" % _flat(job["theme"]))
    lo, hi = job.get("min_date"), job.get("max_date")
    if lo or hi:
        left = _fmt_date(lo) or "начало"
        right = _fmt_date(hi) or "сегодня"
        parts.append("период %s — %s" % (left, right))
    if _flat(job.get("hub")):
        parts.append("площадка %s" % _flat(job["hub"]))
    if _flat(job.get("author")):
        parts.append("автор %s" % _flat(job["author"]))
    return ", ".join(parts) if parts else "все сообщения набора"


def _scope_public(job: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "label_mode": str(job.get("label_mode") or "message"),
        "objects": _object_terms(job.get("objects")),
        "theme": _flat(job.get("theme")),
        "hub": _flat(job.get("hub")),
        "author": _flat(job.get("author")),
        "min_date": job.get("min_date"),
        "max_date": job.get("max_date"),
        "text": _scope_text(job),
    }


def _cat_counts(name: str, field: str, query: Optional[Dict[str, Any]],
                size: int = 20) -> List[Dict[str, Any]]:
    """Частоты по keyword-полю (площадки, тональность источника)."""
    body: Dict[str, Any] = {"size": 0, "aggs": {"v": {"terms": {"field": field, "size": size}}}}
    if query:
        body["query"] = query
    try:
        res = _es().search(index=name, body=body)
        return [{"key": bucket.get("key"), "count": bucket.get("doc_count")}
                for bucket in res["aggregations"]["v"]["buckets"]]
    except Exception:
        return []


def _theme_catalog(user: Any, index_name: str, limit: int = 60) -> List[Dict[str, Any]]:
    """Темы датасета из уже собранных итогов (``<ГГГГ-ММ>_summary.json``)."""
    import glob

    root = os.path.join(DATA_DIR, str(getattr(user, "id", "")), REPORTS_DIR_NAME)
    stem = _norm_name(index_name).lower()
    items: List[Dict[str, Any]] = []
    seen = set()
    try:
        paths = glob.glob(os.path.join(root, "**", "*_summary.json"), recursive=True)
    except Exception:
        paths = []
    for path in sorted(paths):
        folder = os.path.basename(os.path.dirname(path)).lower()
        if stem and stem not in folder and stem not in os.path.basename(path).lower():
            continue
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            continue
        period = _flat(data.get("period")) or os.path.basename(path)[:7]
        for topic in (data.get("topics") or [])[:25]:
            if not isinstance(topic, dict):
                continue
            name = _flat(topic.get("name"))
            key = (_norm_key(name), period)
            if not name or key in seen:
                continue
            seen.add(key)
            items.append({
                "name": name,
                "count": topic.get("count"),
                "tone": _flat(topic.get("tone")),
                "category": _flat(topic.get("category")),
                "period": period,
                "source": os.path.basename(path),
            })
    items.sort(key=lambda item: -(int(item.get("count") or 0)))
    return items[:limit]


def _term_suggestions(index_name: str, query: Dict[str, Any], limit: int = 30) -> List[Dict[str, Any]]:
    """Частотные слова из сэмпла сообщений — подсказки для объектов и продуктов."""
    try:
        res = _es().search(index=index_name, size=SCOPE_OPTIONS_DOCS, query=query,
                           source_includes=[_text_field(index_name), "title"])
    except Exception:
        return []
    counter: Dict[str, int] = {}
    for hit in ((res.get("hits") or {}).get("hits") or []):
        src = hit.get("_source") or {}
        text = "%s %s" % (_flat(src.get(_text_field(index_name))), _flat(src.get("title")))
        for word in re.findall(r"[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё\-]{3,}", text.lower()):
            word = word.strip("-'")
            if len(word) < 4 or word in STOPWORDS:
                continue
            counter[word] = counter.get(word, 0) + 1
    rows = sorted(counter.items(), key=lambda item: -item[1])[:limit]
    return [{"term": word, "count": count} for word, count in rows if count > 1]


def _author_suggestions(index_name: str, query: Dict[str, Any], limit: int = 15) -> List[Dict[str, Any]]:
    """Частые авторы: агрегация по keyword-подполю authorObject.fullname."""
    rows = _cat_counts(index_name, "authorObject.fullname.keyword", query, limit)
    return [{"name": row["key"], "count": row["count"]} for row in rows if row.get("key")]


# --------------------------------------------------------------------------- #
# Пресеты: «KFC → Rostic's, качество еды, лето 2026»
# --------------------------------------------------------------------------- #

def _presets_path(user_id: Any) -> str:
    return os.path.join(DATA_DIR, str(user_id), PRESETS_FILE)


def _presets_load(user_id: Any) -> List[Dict[str, Any]]:
    path = _presets_path(user_id)
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return [item for item in (data.get("presets") or []) if isinstance(item, dict)]
    except Exception:
        return []


def _presets_save(user_id: Any, items: List[Dict[str, Any]]) -> None:
    path = _presets_path(user_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
        json.dump({"presets": items}, handle, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def _preset_note(record: Dict[str, Any]) -> str:
    """Короткая подпись пресета: «KFC → Rostic's, качество еды, лето 2026»."""
    pieces: List[str] = []
    if record.get("objects"):
        pieces.append(", ".join(record["objects"]))
    if record.get("theme"):
        pieces.append(record["theme"])
    lo, hi = record.get("min_date"), record.get("max_date")
    if lo or hi:
        pieces.append("%s — %s" % (_fmt_date(lo) or "…", _fmt_date(hi) or "…"))
    if record.get("hub"):
        pieces.append(record["hub"])
    if record.get("label_mode") == "aspect":
        pieces.append("по объектам")
    return ", ".join(pieces)


# =========================================================================== #
# Аспектная разметка: отношение к объекту, а не тон сообщения целиком
#
# Обычный режим отвечает на вопрос «совпадает ли разметка источника с моделью».
# Аспектный — на вопрос «что люди думают про ЭТОТ объект»: «сеть ругают, но новый
# продукт хвалят». Поэтому отдельный промпт, отдельные поля в Elasticsearch
# (tone_aspect*, маркер прохода tone_aspect_at) и отдельный отчёт без сравнения
# с источником как с эталоном: toneMark размечен на уровне сообщения.
# =========================================================================== #

ASPECT_MARKER = "tone_aspect_at"
ASPECT_FIELDS = ("tone_aspect", "tone_aspect_objects", "tone_aspect_at", "tone_aspect_by",
                 "tone_msg_llm", "tone_msg_conf", "tone_msg_by", "tone_msg_reason")

ASPECT_BATCH = 8
ASPECT_PARALLEL = 10
ASPECT_MESSAGE_CHARS = 420
ASPECT_CHAR_BUDGET = 6000
ASPECT_MAX_TOKENS = 2600
ASPECT_PASS2_BATCH = 6
ASPECT_PASS2_MAX_TOKENS = 2200

ASPECT_SYSTEM = ("Ты аналитик репутации. Оцениваешь отношение автора К КОНКРЕТНОМУ ОБЪЕКТУ "
                 "(бренду, продукту, конкуренту), а не общий тон сообщения. "
                 "Отвечай строго JSON, без пояснений вокруг.")

ASPECT_INSTRUCTION = """Оцени, как автор каждого сообщения относится ИМЕННО к указанным объектам.

Объекты: {objects}
{theme_line}Правила:
- объект считается упомянутым, только если он реально назван в тексте или однозначно подразумевается (учитывай склонения, латиницу и кириллицу, сокращения); иначе mentioned=false;
- если объект упомянут — оцени отношение автора К ЭТОМУ ОБЪЕКТУ: negative, neutral или positive;
- общий тон сообщения может быть другим: сеть могут ругать в целом, но конкретный продукт хвалить — это нормальный случай, оценивай именно объект;
- реклама, анонс, нейтральный факт без оценки, бессмысленный текст — neutral с низкой уверенностью;
- quote — ДОСЛОВНЫЙ короткий фрагмент (до 120 знаков) из текста, на котором основано решение по этому объекту; объект не упомянут — пустая строка;
- message_tone — тон всего сообщения целиком (отдельное поле, не путать с отношением к объекту);
- reason — до 100 знаков, по-русски, коротко.

Верни СТРОГО JSON такой формы:
{{"results": [{{"id": "<id>", "message_tone": "negative|neutral|positive", "objects": [{{"object": "<объект из списка>", "mentioned": true, "tone": "negative|neutral|positive", "confidence": 0.0, "reason": "...", "quote": "..."}}]}}]}}
В ответе должны быть ВСЕ {count} сообщений, и для каждого — ВСЕ объекты из списка (даже если mentioned=false).

{hint}Сообщения:
{messages}"""


def _parse_aspect_results(text: str, finish: str,
                          objects: List[str]) -> Optional[Dict[str, Dict[str, Any]]]:
    """Разбор ответа аспектной разметки: {_id: {message_tone, objects: {ключ: вердикт}}}."""
    from agent_engine.tools_llm import _extract_json

    if finish == "length":
        return None
    parsed = _extract_json(text)
    if not isinstance(parsed, dict):
        return None
    items = parsed.get("results")
    if not isinstance(items, list) or not items:
        return None
    wanted = {_norm_key(name): name for name in objects}
    out: Dict[str, Dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        key = str(item.get("id") or "").strip()
        if not key:
            continue
        message_tone = str(item.get("message_tone") or "").strip().lower()
        if message_tone not in TONE_TO_INT:
            message_tone = ""
        entries = item.get("objects")
        per: Dict[str, Dict[str, Any]] = {}
        for entry in (entries if isinstance(entries, list) else []):
            if not isinstance(entry, dict):
                continue
            name_key = _norm_key(entry.get("object"))
            if name_key not in wanted:
                # Модель могла вернуть объект в другом падеже/регистре — ищем по вхождению.
                name_key = next((k for k in wanted if k and (k in name_key or name_key in k)), "")
                if not name_key:
                    continue
            tone = str(entry.get("tone") or "").strip().lower()
            mentioned = bool(entry.get("mentioned")) and tone in TONE_TO_INT
            try:
                conf = float(entry.get("confidence"))
            except (TypeError, ValueError):
                conf = 0.5
            conf = max(0.0, min(1.0, conf))
            per[name_key] = {
                "mentioned": mentioned,
                "tone": TONE_TO_INT[tone] if mentioned else None,
                "confidence": round(conf, 3) if mentioned else 0.0,
                "reason": _flat(entry.get("reason"))[:100],
                "quote": _flat(entry.get("quote"))[:160],
            }
        out[key] = {"message_tone": message_tone, "objects": per}
    return out or None


async def _read_aspect_batch(ctx: _Ctx, docs: List[Dict[str, Any]], *, objects: List[str],
                             theme: str, hint: str, sem: asyncio.Semaphore,
                             batch_chars: int, max_tokens: int,
                             vllm_cfg: Optional[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], str]:
    """Одна аспектная пачка: строгий JSON, повторы, деление пачки при обрыве по лимиту."""
    from agent_engine.tools_llm import _qwen

    async with sem:
        if ctx.cancelled():
            return {}, "остановлено"

    theme_line = ("Инфоповод/тема: «%s» — учитывай её, если она относится к объекту.\n" % theme) \
        if theme else ""

    async def _call(lines: List[str], depth: int) -> Dict[str, Dict[str, Any]]:
        prompt = ASPECT_INSTRUCTION.format(count=len(lines), objects=", ".join(objects),
                                           theme_line=theme_line, hint=hint,
                                           messages="\n".join(lines))
        last = "нет ответа модели"
        for attempt in range(2 if depth == 0 else 1):
            meta: Dict[str, Any] = {}
            try:
                text, _tokens = await _qwen(ctx, prompt, system=ASPECT_SYSTEM, max_tokens=max_tokens,
                                            temperature=0.0, vllm_cfg=vllm_cfg, meta=meta)
            except Exception as exc:  # noqa: BLE001 — пачка не должна ломать весь проход
                last = "%s: %s" % (type(exc).__name__, str(exc)[:160])
                await asyncio.sleep(1.0 + attempt)
                continue
            parsed = _parse_aspect_results(text, str(meta.get("finish_reason") or ""), objects)
            if parsed:
                return parsed
            finish = str(meta.get("finish_reason") or "")
            last = ("ответ обрезан по лимиту %d токенов" % max_tokens) if finish == "length" \
                else "в ответе нет корректного списка results"
        raise RuntimeError(last)

    async def _split(lines: List[str], depth: int) -> Dict[str, Dict[str, Any]]:
        try:
            return await _call(lines, depth)
        except Exception as exc:  # noqa: BLE001
            if len(lines) <= 2 or depth >= 2:
                raise
            half = len(lines) // 2
            out: Dict[str, Dict[str, Any]] = {}
            for part in (lines[:half], lines[half:]):
                try:
                    out.update(await _split(part, depth + 1))
                except Exception as inner:  # noqa: BLE001
                    _log_line("аспектная пачка (%d) не разобрана: %s" % (len(part), str(inner)[:120]))
            if not out:
                raise exc
            return out

    lines = [_message_line(doc, batch_chars) for doc in docs]
    try:
        return await _split(lines, 0), ""
    except Exception as exc:  # noqa: BLE001
        return {}, str(exc)[:200]


def _aspect_records_from(parsed: Dict[str, Dict[str, Any]], docs: List[Dict[str, Any]],
                         source: str, model: str,
                         objects: List[str]) -> List[Dict[str, Any]]:
    """Записи для журнала и ES: вердикт по каждому объекту + общий тон сообщения."""
    wanted = [(_norm_key(name), name) for name in objects]
    out: List[Dict[str, Any]] = []
    for doc in docs:
        verdict = parsed.get(doc["_id"])
        aspect: List[Dict[str, Any]] = []
        for key, display in wanted:
            info = ((verdict or {}).get("objects") or {}).get(key) if verdict else None
            aspect.append({
                "object": display,
                "mentioned": bool(info and info.get("mentioned")),
                "tone": (info or {}).get("tone"),
                "confidence": float((info or {}).get("confidence") or 0.0),
                "reason": (info or {}).get("reason") or "",
                "quote": (info or {}).get("quote") or "",
                "by": (source if (info and info.get("mentioned")) else "none"),
            })
        message_tone = None
        if verdict and verdict.get("message_tone"):
            message_tone = TONE_TO_INT.get(verdict["message_tone"])
        by = source if verdict else "none"
        out.append({
            "_id": doc["_id"],
            "id": doc.get("id"),
            "toneMark": doc.get("toneMark"),
            "timeCreate": doc.get("timeCreate"),
            "hub": doc.get("hub") or "",
            "hubtype": doc.get("hubtype") or "",
            "type": doc.get("type") or "",
            "url": doc.get("url") or "",
            "author": doc.get("author") or "",
            "text": " ".join(str(doc.get("text") or "").split())[:1200],
            "chars": len(str(doc.get("text") or "")),
            "model": model,
            "tone_aspect": aspect,
            "tone_aspect_objects": [entry["object"] for entry in aspect if entry["mentioned"]],
            "tone_aspect_by": by,
            "tone_msg_llm": message_tone,
            "tone_msg_conf": round(float(min([entry["confidence"] for entry in aspect
                                              if entry["mentioned"]] or [0.0])), 3),
            "tone_msg_by": by,
            "tone_msg_reason": _flat((verdict or {}).get("message_tone_reason"))[:100],
        })
    return out


def _aspect_needs_pass2(rec: Dict[str, Any], threshold: float) -> bool:
    """Спорный случай аспектного прохода: нет вердикта, низкая уверенность или нет цитаты."""
    if rec.get("tone_aspect_by") in (None, "", "none"):
        return True
    if rec.get("tone_aspect_by") == "32b":
        return False
    if rec.get("tone_msg_llm") is None:
        return True
    for entry in rec.get("tone_aspect") or []:
        if not entry.get("mentioned") or entry.get("by") == "32b":
            continue
        if float(entry.get("confidence") or 0.0) < threshold:
            return True
        if not _flat(entry.get("quote")):
            return True
    return False


def _aspect_worst_conf(rec: Dict[str, Any]) -> float:
    values = [float(entry.get("confidence") or 0.0) for entry in (rec.get("tone_aspect") or [])
              if entry.get("mentioned") and entry.get("by") != "32b"]
    return min(values) if values else 0.0


_LAST_WRITE_ERROR = ""


def _bulk_write_aspect(job: Dict[str, Any], records: List[Dict[str, Any]]) -> Tuple[int, int]:
    """Пакетная запись аспектной разметки. toneMark и поля tone_llm* не трогаем."""
    name = job["index_name"]
    stamp = int(time.time())
    written = 0
    errors = 0
    for start in range(0, len(records), BULK_CHUNK):
        chunk = records[start:start + BULK_CHUNK]
        operations: List[Dict[str, Any]] = []
        for rec in chunk:
            if not rec.get("_id"):
                continue
            mentioned = [entry for entry in (rec.get("tone_aspect") or []) if entry.get("mentioned")]
            operations.append({"update": {"_index": name, "_id": rec["_id"], "retry_on_conflict": 3}})
            operations.append({"doc": {
                "tone_aspect": [{"object": entry["object"], "tone": entry.get("tone"),
                                 "confidence": entry.get("confidence"),
                                 "reason": entry.get("reason") or "",
                                 "quote": entry.get("quote") or "",
                                 "by": entry.get("by") or ""} for entry in mentioned],
                "tone_aspect_objects": rec.get("tone_aspect_objects") or [],
                "tone_aspect_at": stamp,
                "tone_aspect_by": rec.get("tone_aspect_by") or "",
                "tone_msg_llm": rec.get("tone_msg_llm"),
                "tone_msg_conf": rec.get("tone_msg_conf"),
                "tone_msg_by": rec.get("tone_msg_by") or "",
                "tone_msg_reason": rec.get("tone_msg_reason") or "",
            }})
        if not operations:
            continue
        try:
            res = _es().bulk(operations=operations, refresh="wait_for")
            failed = 0
            for item in (res.get("items") or []):
                problem = (item.get("update") or {}).get("error")
                if not problem:
                    continue
                failed += 1
                if failed == 1:
                    # Причина первой ошибки важна: без неё видно только «не записалось».
                    global _LAST_WRITE_ERROR
                    _LAST_WRITE_ERROR = str(json.dumps(problem, ensure_ascii=False))[:300]
                    _log_line("аспектная разметка не записывается: %s" % _LAST_WRITE_ERROR, level="error")
            errors += failed
            written += len(chunk) - failed
        except Exception as exc:  # noqa: BLE001
            errors += len(chunk)
            _LAST_WRITE_ERROR = str(exc)[:300]
            _log_line("пакетная запись аспектной разметки не удалась: %s" % str(exc)[:200], level="error")
    return written, errors


async def _run_aspect_job(jid: str) -> None:
    """Фоновый аспектный проход: 4B по объектам, спорные — на 32B, затем отчёт по объектам."""
    job = _job_load(jid)
    if not job:
        return
    name = job["index_name"]
    objects = _object_terms(job.get("objects"))
    if not objects:
        _set(jid, status="error", stage="error", stage_label=STAGE_LABELS["error"], worker_pid=0,
             error="Аспектная разметка требует хотя бы один объект")
        return
    theme = _flat(job.get("theme"))
    ctx = _Ctx(jid)
    started = time.time()
    try:
        _set(jid, status="running", stage="preparing", stage_label="читаю сообщения",
             progress_schema=PROGRESS_SCHEMA, worker_pid=os.getpid(), writing=False,
             started=job.get("started") or datetime.now().isoformat(timespec="seconds"),
             objects=objects, scope=_scope_public(job),
             log="старт аспектной разметки: %s, объекты %s" % (name, ", ".join(objects)))
        _ensure_mapping(name)
        results = _journal_load(jid)
        remaining = int(_es().count(index=name, query=_active_query(job, ASPECT_MARKER)).get("count") or 0)
        total = len(results) + remaining
        if job.get("mode") == "sample":
            total = min(int(job.get("sample_size") or 1000), total)
        elif total > MAX_FULL_DOCS:
            raise RuntimeError(
                "Полная проверка по объектам — это %d сообщений, а за один раз мы проверяем "
                "до %d. "
                "Сузьте область: период, площадку или список объектов."
                % (total, MAX_FULL_DOCS))
        _set(jid, total=total, processed=len(results), errors=0)
        _log_line("[%s] аспектная разметка: к обработке %d сообщений, уже размечено %d, объекты %s"
                  % (jid, total, len(results), ", ".join(objects)))

        # ------------------------------ проход 1: быстрая 4B ------------------------------
        _set(jid, stage="pass1", stage_label="определяю отношение к объектам", phase="pass1")
        bulk = _bulk_profile()
        batch_size = int(job.get("batch_size") or ASPECT_BATCH)
        parallel = int(job.get("parallel") or ASPECT_PARALLEL)
        model_label = bulk.get("model") or "Qwen/Qwen3-32B-FP8"
        source_label = "4b" if bulk else "32b"
        sem = asyncio.Semaphore(max(1, parallel))
        cursor = job.get("cursor")
        stall = 0
        seen_before = set(results)
        while len(results) < total and not _cancelled(jid):
            want = min(PAGE_SIZE, total - len(results))
            hits = _search_page(job, want, cursor, marker=ASPECT_MARKER,
                                extra_source=("authorObject",))
            if not hits:
                if cursor:
                    cursor = None
                    job["cursor"] = None
                    continue
                break
            docs = [_doc_from_hit(hit, job) for hit in hits]
            batches = _make_batches(docs, batch_size, ASPECT_CHAR_BUDGET, ASPECT_MESSAGE_CHARS)
            tasks = [
                asyncio.create_task(_read_aspect_batch(
                    ctx, batch, objects=objects, theme=theme, hint="",
                    sem=sem, batch_chars=ASPECT_MESSAGE_CHARS,
                    max_tokens=int(bulk.get("max_tokens") or ASPECT_MAX_TOKENS) if bulk else ASPECT_MAX_TOKENS,
                    vllm_cfg=bulk.get("vllm_cfg")))
                for batch in batches
            ]
            batch_of = {task: batch for task, batch in zip(tasks, batches)}
            records: List[Dict[str, Any]] = []
            pending = set(tasks)
            while pending:
                finished, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in finished:
                    batch = batch_of[task]
                    parsed, error = task.result()
                    if error and error != "остановлено":
                        _log_line("[%s] аспектная пачка из %d не разобрана: %s"
                                  % (jid, len(batch), error))
                    recs = _aspect_records_from(parsed, batch, source_label, model_label, objects)
                    for rec in recs:
                        results[rec["_id"]] = rec
                    records.extend(recs)
                    _journal_append(jid, recs)
                _set(jid, processed=len(results), total=total)
            fresh = len([rec for rec in records if rec["_id"] not in seen_before])
            if not fresh:
                stall += 1
                if stall >= 3:
                    _log_line("[%s] выборка перестала давать новые документы — останавливаю проход"
                              % jid, level="warning")
                    break
            else:
                stall = 0
            seen_before.update(rec["_id"] for rec in records)
            if job.get("mode") == "full" and docs and docs[-1].get("sort"):
                cursor = docs[-1]["sort"]
                job["cursor"] = cursor
            _set(jid, writing=True, stage="writing", stage_label=STAGE_LABELS["writing"])
            written, werrors = _bulk_write_aspect(job, records)
            job["errors"] = int(job.get("errors") or 0) + werrors
            job["bulk_written"] = int(job.get("bulk_written") or 0) + written
            if records and not written and werrors:
                # Ничего не записалось — разметка физически не сохраняется. Честно падаем
                # с причиной, а не крутимся на месте до срабатывания детектора застоя.
                raise RuntimeError("не удалось сохранить результат проверки — попробуйте ещё раз "
                                   "или обратитесь к администратору")
            _set(jid, writing=False, stage="pass1", stage_label="определяю отношение к объектам",
                 processed=len(results), total=total, cursor=job.get("cursor"),
                 bulk_written=job["bulk_written"], errors=job["errors"])
        _set(jid, processed=len(results), total=total)

        # ------------------------- проход 2: перепроверка на 32B -------------------------
        threshold = float(job.get("conf_threshold") or DEFAULT_CONF)
        disputed = [rec for rec in results.values() if _aspect_needs_pass2(rec, threshold)]
        pass2 = {"available": True, "skipped": False, "reason": "", "targets": len(disputed),
                 "decided": 0, "failed": 0}
        disputed_total = len(disputed)
        # Тот же лимит доли второго прохода, что и в обычном режиме: 32B в разы медленнее.
        pass2_limit = int(max(1, total) * PASS2_MAX_SHARE)
        pass2_capped = 0
        if disputed_total > pass2_limit:
            disputed.sort(key=_aspect_worst_conf)
            pass2_capped = disputed_total - max(1, pass2_limit)
            disputed = disputed[:max(1, pass2_limit)]
            _log_line("[%s] аспектных спорных %d — больше лимита %.0f%%: перепроверяю %d самых "
                      "неуверенных, остальные %d остаются решением 4B"
                      % (jid, disputed_total, PASS2_MAX_SHARE * 100, len(disputed), pass2_capped),
                      level="warning")
        pass2["candidates"] = disputed_total
        pass2["capped"] = pass2_capped
        pass2["max_share"] = PASS2_MAX_SHARE
        pass2["share"] = round(disputed_total / float(max(1, total)), 4)
        if not _cancelled(jid) and disputed:
            from mlops.lock import generate_cfg

            gen = generate_cfg() or {}
            base_url = str(gen.get("base_url") or "")
            if not await _model_up(base_url):
                pass2.update({"available": False, "skipped": True,
                              "reason": "сервис уточняющей проверки был недоступен"})
                _log_line("[%s] 32B недоступна: перепроверка аспектов не выполнена" % jid, level="warning")
            else:
                _set(jid, stage="pass2", stage_label="уточняю спорные случаи", phase="pass2",
                     pass2_total=len(disputed), pass2_done=0)
                p2_batch = int(job.get("pass2_batch") or ASPECT_PASS2_BATCH)
                p2_parallel = int(job.get("pass2_parallel") or DEFAULT_PARALLEL_PASS2)
                p2_sem = asyncio.Semaphore(max(1, p2_parallel))
                done = 0
                for start in range(0, len(disputed), PAGE_SIZE):
                    if _cancelled(jid):
                        break
                    window = disputed[start:start + PAGE_SIZE]
                    by_id = {rec["_id"]: rec for rec in window}
                    docs = [{"_id": rec["_id"], "id": rec.get("id"), "text": rec.get("text") or "",
                             "hub": rec.get("hub") or "", "timeCreate": rec.get("timeCreate"),
                             "toneMark": rec.get("toneMark"), "hubtype": rec.get("hubtype"),
                             "type": rec.get("type"), "url": rec.get("url"),
                             "author": rec.get("author")} for rec in window]
                    batches = _make_batches(docs, p2_batch, ASPECT_CHAR_BUDGET, ASPECT_MESSAGE_CHARS)
                    tasks = [
                        asyncio.create_task(_read_aspect_batch(
                            ctx, batch, objects=objects, theme=theme,
                            hint="Это спорные случаи: посмотри внимательно и реши окончательно.\n",
                            sem=p2_sem, batch_chars=ASPECT_MESSAGE_CHARS,
                            max_tokens=ASPECT_PASS2_MAX_TOKENS, vllm_cfg=None))
                        for batch in batches
                    ]
                    batch_of = {task: batch for task, batch in zip(tasks, batches)}
                    changed: List[Dict[str, Any]] = []
                    seen = 0
                    pending = set(tasks)
                    while pending:
                        finished, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                        for task in finished:
                            batch = batch_of[task]
                            seen += len(batch)
                            parsed, error = task.result()
                            if error and error != "остановлено":
                                pass2["failed"] += len(batch)
                                _log_line("[%s] аспектная пачка 32B из %d не разобрана: %s"
                                          % (jid, len(batch), error))
                            recs = _aspect_records_from(parsed, batch, "32b",
                                                        "Qwen/Qwen3-32B-FP8", objects)
                            batch_recs: List[Dict[str, Any]] = []
                            for rec in recs:
                                if rec["_id"] not in by_id or rec["tone_aspect_by"] != "32b":
                                    # 32B не ответила — оставляем решение 4B, не подменяем пустышкой.
                                    continue
                                results[rec["_id"]] = rec
                                batch_recs.append(rec)
                                pass2["decided"] += 1
                            if batch_recs:
                                _journal_append(jid, batch_recs)
                                changed.extend(batch_recs)
                            _set(jid, stage="pass2", stage_label="уточняю спорные случаи",
                                 pass2_done=done + seen, pass2_total=len(disputed),
                                 processed=len(results), total=total)
                    if changed:
                        _set(jid, writing=True, stage="writing", stage_label=STAGE_LABELS["writing"])
                        written, werrors = _bulk_write_aspect(job, changed)
                        job["errors"] = int(job.get("errors") or 0) + werrors
                        job["bulk_written"] = int(job.get("bulk_written") or 0) + written
                    done += len(window)
                    _set(jid, writing=False, stage="pass2", stage_label="уточняю спорные случаи",
                         pass2_done=done, pass2_total=len(disputed),
                         processed=len(results), total=total, errors=job.get("errors") or 0,
                         bulk_written=job.get("bulk_written") or 0)

        # -------------------------------- отчёт --------------------------------
        cancelled = _cancelled(jid)
        _set(jid, status="running", stage="report", stage_label="собираю отчёт",
             phase="report", processed=len(results), total=total,
             percent_before_report=int(job.get("percent") or 0),
             bulk_written=job.get("bulk_written") or 0, errors=job.get("errors") or 0)
        report = _build_aspect_report(job, results, pass2)
        report["elapsed_sec"] = round(time.time() - started, 1)
        report["bulk_written"] = int(job.get("bulk_written") or 0)
        report["write_errors"] = int(job.get("errors") or 0)
        with open(_report_path(jid), "w", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=1)
        files = _write_aspect_report_files(job, report)
        report["files"] = files
        with open(_report_path(jid), "w", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=1)
        rate = round(len(results) / max(1e-6, report["elapsed_sec"]) * 60.0, 1)
        _set(jid, status="cancelled" if cancelled else "done",
             stage="cancelled" if cancelled else "done", worker_pid=0, writing=False,
             stage_label=STAGE_LABELS["cancelled" if cancelled else "done"],
             processed=len(results), total=total,
             percent=(int((_job_load(jid) or {}).get("percent_before_report")
                          or (_job_load(jid) or {}).get("percent") or 0) if cancelled else 100),
             finished=datetime.now().isoformat(timespec="seconds"),
             elapsed_sec=report["elapsed_sec"], rate_per_min=rate,
             pass2_share=report["summary"]["pass2_share"], report_files=files,
             log="аспектная разметка готова: сообщений %d, объектов %d"
                 % (len(results), len(objects)))
    except Exception as exc:  # noqa: BLE001
        _log_line("[%s] аспектная разметка: ошибка: %s" % (jid, str(exc)[:300]), level="error")
        _set(jid, status="error", stage="error", stage_label=STAGE_LABELS["error"], worker_pid=0,
             error=str(exc)[:500], finished=datetime.now().isoformat(timespec="seconds"))


# --------------------------------------------------------------------------- #
# Отчёт аспектного режима
# --------------------------------------------------------------------------- #

def _share(part: Any, whole: Any) -> float:
    try:
        whole = float(whole)
        return round(float(part) / whole, 4) if whole else 0.0
    except (TypeError, ValueError):
        return 0.0


def _build_aspect_report(job: Dict[str, Any], results: Dict[str, Dict[str, Any]],
                         pass2: Dict[str, Any]) -> Dict[str, Any]:
    """Распределение отношения по объектам, динамика, площадки, авторы и расхождения."""
    objects = _object_terms(job.get("objects"))
    all_recs = list(results.values())
    usable = [rec for rec in all_recs if rec.get("tone_aspect_by") not in (None, "", "none")]
    unresolved = len(all_recs) - len(usable)

    buckets: Dict[str, Dict[str, Any]] = {
        name: {"object": name, "mentions": 0, "negative": 0, "neutral": 0, "positive": 0,
               "by_32b": 0, "conf_sum": 0.0} for name in objects}
    months: Dict[Tuple[str, str], List[int]] = {}
    hubs: Dict[Tuple[str, str], List[int]] = {}
    authors: Dict[str, Dict[str, List[int]]] = {}
    divergence: List[Dict[str, Any]] = []
    divergence_by_object: Dict[str, int] = {}
    messages_with_mention = 0
    source_mismatch = 0

    for rec in usable:
        try:
            month = datetime.fromtimestamp(float(rec.get("timeCreate"))).strftime("%Y-%m") \
                if rec.get("timeCreate") else "—"
        except Exception:
            month = "—"
        hub = _flat(rec.get("hub")) or "—"
        author = _flat(rec.get("author")) or "—"
        message_tone = rec.get("tone_msg_llm")
        try:
            source_tone = int(rec.get("toneMark"))
        except (TypeError, ValueError):
            source_tone = None
        if message_tone is not None and source_tone is not None and int(message_tone) != source_tone:
            source_mismatch += 1
        mentioned_here = False
        for entry in rec.get("tone_aspect") or []:
            name = entry.get("object")
            bucket = buckets.get(name)
            if bucket is None or not entry.get("mentioned") or entry.get("tone") is None:
                continue
            tone = int(entry["tone"])
            conf = float(entry.get("confidence") or 0.0)
            mentioned_here = True
            bucket["mentions"] += 1
            bucket["conf_sum"] += conf
            if entry.get("by") == "32b":
                bucket["by_32b"] += 1
            if tone < 0:
                bucket["negative"] += 1
            elif tone > 0:
                bucket["positive"] += 1
            else:
                bucket["neutral"] += 1
            cell = months.setdefault((month, name), [0, 0, 0])
            cell[0] += 1
            if tone < 0:
                cell[1] += 1
            elif tone > 0:
                cell[2] += 1
            spot = hubs.setdefault((name, hub), [0, 0, 0])
            spot[0] += 1
            if tone < 0:
                spot[1] += 1
            elif tone > 0:
                spot[2] += 1
            author_cell = authors.setdefault(name, {}).setdefault(author, [0, 0, 0])
            author_cell[0] += 1
            if tone < 0:
                author_cell[1] += 1
            elif tone > 0:
                author_cell[2] += 1
            # Самая ценная находка: общий тон сообщения и отношение к объекту расходятся.
            if message_tone is not None and int(message_tone) != tone:
                text = _flat(rec.get("text"))
                divergence_by_object[name] = divergence_by_object.get(name, 0) + 1
                divergence.append({
                    "object": name,
                    "message_tone": TONE_RU.get(int(message_tone), "—"),
                    "message_tone_int": int(message_tone),
                    "object_tone": TONE_RU.get(tone, "—"),
                    "object_tone_int": tone,
                    "source_tone": TONE_RU.get(source_tone, "—"),
                    "confidence": conf,
                    "decided_by": BY_RU.get(entry.get("by") or "", entry.get("by") or ""),
                    "reason": entry.get("reason") or "—",
                    "quote": _flat(entry.get("quote")) or text[:220],
                    "text": text[:320],
                    "url": rec.get("url") or "",
                    "hub": hub,
                    "date": _fmt_date(rec.get("timeCreate")),
                    "author": author,
                })
        if mentioned_here:
            messages_with_mention += 1

    object_rows: List[Dict[str, Any]] = []
    for name in objects:
        bucket = buckets[name]
        mentions = int(bucket["mentions"])
        row = {
            "object": name,
            "mentions": mentions,
            "share_of_checked": _share(mentions, len(usable)),
            "negative": int(bucket["negative"]),
            "neutral": int(bucket["neutral"]),
            "positive": int(bucket["positive"]),
            "negative_share": _share(bucket["negative"], mentions),
            "neutral_share": _share(bucket["neutral"], mentions),
            "positive_share": _share(bucket["positive"], mentions),
            "mean_confidence": round(bucket["conf_sum"] / mentions, 3) if mentions else 0.0,
            "by_32b": int(bucket["by_32b"]),
        }
        row["tone_index"] = round(row["positive_share"] - row["negative_share"], 4)
        object_rows.append(row)
    object_rows.sort(key=lambda row: (row["tone_index"], -row["mentions"]))

    month_rows: List[Dict[str, Any]] = []
    for (month, name), cell in months.items():
        month_rows.append({"month": month, "object": name, "mentions": cell[0],
                           "negative": cell[1], "positive": cell[2],
                           "neutral": cell[0] - cell[1] - cell[2],
                           "negative_share": _share(cell[1], cell[0]),
                           "positive_share": _share(cell[2], cell[0])})
    month_rows.sort(key=lambda row: (row["month"], row["object"]))

    hub_rows: List[Dict[str, Any]] = []
    for (name, hub), cell in hubs.items():
        hub_rows.append({"object": name, "hub": hub, "mentions": cell[0],
                         "negative": cell[1], "positive": cell[2],
                         "negative_share": _share(cell[1], cell[0]),
                         "positive_share": _share(cell[2], cell[0])})
    hub_rows.sort(key=lambda row: (row["object"], -row["mentions"]))
    by_object_hubs: List[Dict[str, Any]] = []
    for name in objects:
        rows = [row for row in hub_rows if row["object"] == name]
        if not rows:
            continue
        # Одна жалоба на площадке даёт «негатив 100%»: сначала площадки, где упоминаний
        # достаточно для доли, и только если таких нет — все остальные.
        solid = [row for row in rows if row["mentions"] >= 3]
        pool = solid or rows
        worst = sorted(pool, key=lambda row: (-row["negative_share"], -row["mentions"]))[:5]
        best = sorted(pool, key=lambda row: (-row["positive_share"], -row["mentions"]))[:5]
        by_object_hubs.append({"object": name, "worst": worst, "best": best})

    author_rows: List[Dict[str, Any]] = []
    for name in objects:
        rows = [{"author": author, "mentions": cell[0], "negative": cell[1], "positive": cell[2],
                 "negative_share": _share(cell[1], cell[0])}
                for author, cell in (authors.get(name) or {}).items()]
        rows.sort(key=lambda row: (-row["mentions"], -row["negative_share"]))
        author_rows.append({"object": name, "authors": rows[:8]})

    # Всего расхождений считаем ДО отбора примеров: иначе сводка показывала бы размер
    # списка примеров, а не реальное число случаев.
    divergence_total = len(divergence)
    # Для примеров полезнее разнообразие направлений, а не 25 однотипных «нейтрал → позитив».
    seen_pairs = set()
    examples_pool: List[Dict[str, Any]] = []
    for item in sorted(divergence, key=lambda row: (-float(row.get("confidence") or 0),
                                                    -len(row.get("text") or ""))):
        pair = (item.get("object"), item.get("message_tone_int"), item.get("object_tone_int"))
        if pair in seen_pairs and len(examples_pool) < MAX_EXAMPLES - 6:
            continue
        seen_pairs.add(pair)
        examples_pool.append(item)
    examples = examples_pool[:MAX_EXAMPLES]

    pass2_share = (sum(1 for rec in all_recs if rec.get("tone_aspect_by") == "32b")
                   / float(len(all_recs)) if all_recs else 0.0)
    summary = {
        "dataset": job.get("dataset_label") or job.get("index_name"),
        "index_name": job.get("index_name"),
        "index_key": job.get("index_key"),
        "mode": job.get("mode"),
        "label_mode": "aspect",
        "sample_size": job.get("sample_size"),
        "checked": len(all_recs),
        "evaluated": len(usable),
        "unresolved": unresolved,
        "messages_with_mention": messages_with_mention,
        "objects": object_rows,
        "divergence_total": divergence_total,
        "divergence_examples": len(examples),
        "source_message_mismatch": source_mismatch,
        "pass2_share": round(pass2_share, 4),
        "pass2_decided": int(pass2.get("decided") or 0),
        "pass2_available": bool(pass2.get("available", True)),
        "pass2_reason": str(pass2.get("reason") or ""),
        "threshold": float(job.get("conf_threshold") or DEFAULT_CONF),
        "bulk_written": int(job.get("bulk_written") or 0),
        "write_errors": int(job.get("errors") or 0),
        "elapsed_sec": job.get("elapsed_sec"),
        "rate_per_min": job.get("rate_per_min"),
        "scope_text": _scope_text(job),
        "objects_requested": objects,
        "divergence_by_object": divergence_by_object,
    }
    conclusions = _aspect_conclusions(summary, object_rows, by_object_hubs, month_rows)
    recommendation = _aspect_recommendation(summary, object_rows, by_object_hubs)
    return {
        "job_id": job.get("id"),
        "created": job.get("created"),
        "finished": datetime.now().isoformat(timespec="seconds"),
        "dataset": job.get("dataset_label") or job.get("index_name"),
        "index_name": job.get("index_name"),
        "label_mode": "aspect",
        "period": _period_label(job),
        "scope": _scope_public(job),
        "scope_text": _scope_text(job),
        "preset": job.get("preset") or "",
        "source_note": ("Разметка источника сделана на уровне сообщения целиком, а не "
                        "по объекту. Поэтому в аспектном режиме она показана только как контекст "
                        "и НЕ используется как эталон точности: сравнивать отношение к объекту "
                        "с общей разметкой сообщения нельзя."),
        "method": {
            "pass1_model": "первичная автоматическая разметка",
            "pass2_model": "уточняющая проверка",
            "batch_size": int(job.get("batch_size") or ASPECT_BATCH),
            "parallel": int(job.get("parallel") or ASPECT_PARALLEL),
            "conf_threshold": float(job.get("conf_threshold") or DEFAULT_CONF),
            "objects": objects,
            "fields": list(ASPECT_FIELDS),
            "source_field": "разметка источника (не изменяется, только контекст)",
            "pass2": dict(pass2),
        },
        "summary": summary,
        "by_object": object_rows,
        "by_month": month_rows,
        "by_object_hubs": by_object_hubs,
        "top_authors": author_rows,
        "examples": examples,
        "conclusions": conclusions,
        "recommendation": recommendation,
    }


def _aspect_conclusions(summary: Dict[str, Any], rows: List[Dict[str, Any]],
                        by_hubs: List[Dict[str, Any]],
                        month_rows: List[Dict[str, Any]]) -> List[str]:
    """5–8 выводов человеческим языком — главное, что читает пользователь."""
    out: List[str] = []
    checked = int(summary.get("checked") or 0)
    with_mention = int(summary.get("messages_with_mention") or 0)
    out.append("Проверено %s по области: %s. Объект упомянут в %s из них."
               % (_plural(checked, "сообщение", "сообщения", "сообщений"),
                  summary.get("scope_text") or "все сообщения набора",
                  _plural(with_mention, "сообщении", "сообщениях", "сообщениях")))
    mentioned = [row for row in rows if row["mentions"] >= 3]
    if mentioned:
        worst = max(mentioned, key=lambda row: row["negative_share"])
        best = min(mentioned, key=lambda row: row["negative_share"])
        out.append("Хуже всего отношение к «%s»: негатив %s при %s."
                   % (worst["object"], _pct(float(worst["negative_share"])),
                      _plural(int(worst["mentions"]), "упоминании", "упоминаниях", "упоминаниях")))
        if best["object"] != worst["object"]:
            out.append("Лучше всего — «%s»: негатив %s, позитив %s (%s)."
                       % (best["object"], _pct(float(best["negative_share"])),
                          _pct(float(best["positive_share"])),
                          _plural(int(best["mentions"]), "упоминание", "упоминания", "упоминаний")))
    divergent = int(summary.get("divergence_total") or 0)
    if divergent:
        worst_obj = max(rows, key=lambda row: row["mentions"]) if rows else None
        out.append("В %s общий тон сообщения расходится с отношением к объекту — это те случаи, "
                   "где оценка «по сообщению целиком» даёт неверную картину."
                   % _plural(divergent, "случае", "случаях", "случаях"))
        per_object_div = {}
        for item in (summary.get("divergence_by_object") or {}).items():
            per_object_div[item[0]] = item[1]
        if per_object_div:
            top_obj = max(per_object_div.items(), key=lambda pair: pair[1])
            out.append("Больше всего расхождений вокруг «%s»: %s."
                       % (top_obj[0], _plural(int(top_obj[1]), "случай", "случая", "случаев")))
        elif worst_obj:
            out.append("Больше всего таких расхождений вокруг «%s» (%s)."
                       % (worst_obj["object"],
                          _plural(int(worst_obj["mentions"]), "упоминание", "упоминания", "упоминаний")))
    if month_rows:
        months = sorted({row["month"] for row in month_rows})
        if len(months) >= 2:
            last = months[-1]
            rows_last = [row for row in month_rows if row["month"] == last]
            if rows_last:
                top = max(rows_last, key=lambda row: row["mentions"])
                out.append("Свежий срез — %s: у «%s» %s, негатив %s."
                           % (last, top["object"],
                              _plural(int(top["mentions"]), "упоминание", "упоминания", "упоминаний"),
                              _pct(float(top["negative_share"]))))
    if by_hubs:
        worst_hub = None
        for block in by_hubs:
            for row in block["worst"][:1]:
                if row["mentions"] >= 3 and (worst_hub is None or row["negative_share"] > worst_hub[2]["negative_share"]):
                    worst_hub = (block["object"], row["hub"], row)
        if worst_hub:
            out.append("Худшая площадка — %s по объекту «%s»: негатив %s при %s."
                       % (worst_hub[1], worst_hub[0], _pct(float(worst_hub[2]["negative_share"])),
                          _plural(int(worst_hub[2]["mentions"]), "упоминании", "упоминаниях", "упоминаниях")))
    out.append("Спорных случаев, прошедших уточняющую проверку, — %s от проверенного (%s)."
               % (_pct(float(summary.get("pass2_share") or 0.0)),
                  _plural(int(summary.get("pass2_decided") or 0), "сообщение", "сообщения", "сообщений")))
    if not summary.get("pass2_available", True):
        out.append("ВНИМАНИЕ: уточняющая проверка была недоступна — спорные случаи остались "
                   "результатом первичной автоматической разметки, окончательными их считать нельзя.")
    if int(summary.get("unresolved") or 0):
        out.append("Для %s автоматическая разметка не дала вердикт: такие сообщения не попали "
                   "в распределение." % _plural(int(summary["unresolved"]), "сообщения",
                                                       "сообщений", "сообщений"))
    return out[:8]


def _aspect_recommendation(summary: Dict[str, Any], rows: List[Dict[str, Any]],
                           by_hubs: List[Dict[str, Any]]) -> str:
    mentioned = [row for row in rows if row["mentions"] >= 3]
    if not mentioned:
        return ("Упоминаний выбранных объектов в области проверки почти нет — расширьте область "
                "(период, площадки) или проверьте название объекта.")
    worst = max(mentioned, key=lambda row: row["negative_share"])
    best = min(mentioned, key=lambda row: row["negative_share"])
    parts: List[str] = []
    if float(worst["negative_share"]) >= 0.4:
        parts.append("По объекту «%s» негатив доминирует (%s при %d упоминаниях) — это первый "
                     "приоритет для работы с репутацией."
                     % (worst["object"], _pct(float(worst["negative_share"])), worst["mentions"]))
    elif float(worst["negative_share"]) >= 0.2:
        parts.append("По объекту «%s» негатив заметен (%s) — стоит разобрать причины."
                     % (worst["object"], _pct(float(worst["negative_share"]))))
    if best["object"] != worst["object"]:
        parts.append("«%s» держится лучше (негатив %s) — его аргументы можно переносить на "
                     "проблемный объект." % (best["object"], _pct(float(best["negative_share"]))))
    if int(summary.get("divergence_total") or 0) > max(3, int(0.03 * max(1, summary.get("evaluated") or 1))):
        parts.append("Общий тон сообщения часто не совпадает с отношением к объекту (%s): "
                     "для решений опирайтесь на оценку по объекту, а не на общую тональность "
                     "сообщения."
                     % _plural(int(summary.get("divergence_total") or 0),
                                "случай", "случая", "случаев"))
    for block in by_hubs:
        if block["object"] == worst["object"] and block["worst"]:
            top = block["worst"][0]
            if top["mentions"] >= 3:
                parts.append("Основной источник негатива — %s (%s при %d упоминаниях)."
                             % (top["hub"], _pct(float(top["negative_share"])), top["mentions"]))
            break
    if not summary.get("pass2_available", True):
        parts.append("Уточняющая проверка не выполнялась — цифры стоит уточнить повторным прогоном.")
    parts.append("Разметка источника в аспектном режиме не эталон: она сделана по сообщению "
                 "целиком, поэтому итоговые цифры берите из оценки по объекту.")
    return " ".join(parts)


def _aspect_sections(report: Dict[str, Any], pass2: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Разделы аспектного отчёта для платформенного сборщика DOCX/PDF."""
    summary = report["summary"]
    rows = report["by_object"]
    scope = report.get("scope") or {}
    objects = scope.get("objects") or []
    pass2_text = "не выполнялась (%s)" % (pass2.get("reason") or "нет спорных случаев") \
        if not pass2.get("available", True) else "выполнена"

    object_table = {
        "title": "Отношение по объектам",
        "columns": ["Объект", "Упоминаний", "Доля проверенного", "Позитив", "Нейтрал", "Негатив",
                    "Перевес позитива", "Средняя уверенность"],
        "rows": [[row["object"], row["mentions"], _pct(float(row["share_of_checked"])),
                  _pct(float(row["positive_share"])), _pct(float(row["neutral_share"])),
                  _pct(float(row["negative_share"])), _num(row["tone_index"]),
                  _num(row["mean_confidence"])] for row in rows],
        "note": "Перевес позитива = доля позитива минус доля негатива (от −1 до 1). Считается "
                "по упоминаниям, то есть по сообщениям, где объект реально назван.",
        "layout": "landscape",
    }
    absolute_table = {
        "title": "Распределение в абсолютных числах",
        "columns": ["Объект"] + [TONE_RU[c] for c in CLASSES] + ["Упоминаний", "Уточнённых оценок"],
        "rows": [[row["object"], row["negative"], row["neutral"], row["positive"],
                  row["mentions"], row["by_32b"]] for row in rows],
        "layout": "portrait",
    }
    months = sorted({row["month"] for row in report["by_month"]})
    month_table = None
    if months:
        month_table = {
            "title": "Динамика по месяцам: доля негатива по объектам",
            "columns": ["Месяц"] + [row["object"] for row in rows],
            "rows": [[month] + [
                (lambda cell: "%s (%d)" % (_pct(float(cell["negative_share"])), cell["mentions"])
                 if cell else "—")(next((row for row in report["by_month"]
                                         if row["month"] == month and row["object"] == obj), None))
                for obj in [row["object"] for row in rows]] for month in months],
            "note": "В скобках — сколько упоминаний объекта в этом месяце.",
            "layout": "landscape",
        }
    hub_tables = []
    for block in report["by_object_hubs"]:
        hub_tables.append({
            "title": "Площадки по объекту «%s»: где хуже и где лучше" % block["object"],
            "columns": ["Площадка", "Упоминаний", "Негатив", "Позитив"],
            "note": "Сначала площадки с 3+ упоминаниями: при одном-двух сообщениях доля случайна.",
            "rows": [["хуже: " + row["hub"], row["mentions"], _pct(float(row["negative_share"])),
                      _pct(float(row["positive_share"]))] for row in block["worst"]] +
                    [["лучше: " + row["hub"], row["mentions"], _pct(float(row["negative_share"])),
                      _pct(float(row["positive_share"]))] for row in block["best"]],
            "layout": "auto",
        })
    author_tables = []
    for block in report["top_authors"]:
        if not block["authors"]:
            continue
        author_tables.append({
            "title": "Кто чаще всего пишет про «%s»" % block["object"],
            "columns": ["Автор", "Упоминаний", "Негатив", "Позитив"],
            "rows": [[row["author"], row["mentions"], row["negative"], row["positive"]]
                     for row in block["authors"]],
            "layout": "auto",
        })
    examples = report.get("examples") or []
    example_table = None
    if examples:
        example_table = {
            "title": "Где общий тон сообщения расходится с отношением к объекту (%d)" % len(examples),
            "columns": ["Объект", "Площадка", "Дата", "Тон сообщения", "Отношение к объекту",
                        "Цитата-доказательство", "Пояснение", "Ссылка"],
            "rows": [[item["object"], item["hub"], item["date"], item["message_tone"],
                      item["object_tone"], item["quote"], item["reason"], item["url"] or "—"]
                     for item in examples],
            "note": "Именно эти случаи ломают оценку «по сообщению целиком»: реакция на бренд "
                    "и реакция на продукт здесь расходятся.",
            "layout": "landscape",
        }

    sections: List[Dict[str, Any]] = [{
        "heading": "Как проверяли",
        "text": ("Набор данных: %s. Область проверки: %s. Что считаем: отношение сообщений "
                 "к выбранным объектам.\n"
                 "Проверено сообщений: %d. Объектов в проверке: %d. Уточняющая проверка спорных "
                 "случаев: %s. Результаты проверки сохраняются отдельно от разметки источника: "
                 "исходные данные не меняются.\n%s"
                 % (report.get("dataset") or "—", report.get("scope_text") or "—",
                    int(summary.get("checked") or 0), len(objects), pass2_text,
                    report.get("source_note") or "")),
        "tables": [{
            "title": "Ключевые цифры",
            "columns": ["Показатель", "Значение"],
            "rows": [
                ["Сообщений проверено", str(summary.get("checked") or 0)],
                ["С вердиктом автоматической разметки", str(summary.get("evaluated") or 0)],
                ["Объект упомянут хотя бы в одном сообщении", str(summary.get("messages_with_mention") or 0)],
                ["Всего упоминаний объектов", str(sum(row["mentions"] for row in rows))],
                ["Расхождений «тон сообщения ≠ отношение к объекту»", str(summary.get("divergence_total") or 0)],
                ["Спорных случаев (уточняющая проверка)", _pct(float(summary.get("pass2_share") or 0.0))],
                ["Решений после уточняющей проверки", str(summary.get("pass2_decided") or 0)],
                ["Без вердикта автоматической разметки", str(summary.get("unresolved") or 0)],
                ["Сохранено оценок", str(summary.get("bulk_written") or 0)],
            ],
            "layout": "portrait",
        }],
    }, {
        "heading": "Отношение к объектам",
        "text": ("Считаются только сообщения, где объект реально назван. Один и тот же текст может "
                 "давать разное отношение к разным объектам — это и есть смысл аспектной разметки."),
        "tables": [object_table, absolute_table] + ([month_table] if month_table else []),
    }]
    if hub_tables:
        sections.append({"heading": "Площадки: где отношение хуже, а где лучше",
                         "text": "Срез по площадкам внутри каждого объекта.",
                         "tables": hub_tables})
    if author_tables:
        sections.append({"heading": "Авторы", "text": "Кто чаще всего пишет про объекты проверки.",
                         "tables": author_tables})
    if example_table:
        sections.append({
            "heading": "Общий тон сообщения ≠ отношение к объекту",
            "text": ("Самая ценная находка аспектного режима: сообщение в целом нейтральное или "
                     "позитивное, а к объекту отношение другое (или наоборот)."),
            "tables": [example_table],
        })
    sections.append({"heading": "Выводы", "bullets": report.get("conclusions") or []})
    sections.append({"heading": "Рекомендация", "text": report.get("recommendation") or ""})
    return sections


def _write_aspect_report_files(job: Dict[str, Any], report: Dict[str, Any]) -> Dict[str, str]:
    """DOCX и PDF аспектного отчёта — в папку отчётов владельца."""
    from agent_engine.tools_reports import _build_docx, _build_pdf, _safe_name

    owner = str(job.get("owner") or "")
    if not owner:
        return {}
    out: Dict[str, str] = {}
    objects = _object_terms(job.get("objects"))
    label = _safe_name(report.get("dataset") or job.get("index_name") or "набор", 30)
    objects_part = _safe_name(", ".join(objects), 40)
    stamp = datetime.now().strftime("%Y-%m-%d %H-%M")
    mode = "полная" if job.get("mode") == "full" else "выборка %s" % (job.get("sample_size") or "")
    title = "Проверка тональности по объектам"
    subtitle = "%s → %s — %s, %s" % (label, objects_part or "объекты не заданы",
                                     report.get("period") or "", mode)
    meta = {
        "dataset_label": report.get("dataset") or "",
        "period": report.get("period") or "",
        "author": "Tellscope, автоматическая разметка",
        "date": datetime.now().strftime("%d.%m.%Y %H:%M"),
    }
    sections = _aspect_sections(report, (report.get("method") or {}).get("pass2") or {})
    folder = _reports_dir(owner, REPORT_FOLDER)
    base = _safe_name("%s — %s %s" % (label, objects_part or "объекты", stamp), 120)
    for ext, builder in (("docx", _build_docx), ("pdf", _build_pdf)):
        path = os.path.join(folder, "%s.%s" % (base, ext))
        try:
            builder(path, title, subtitle, sections, meta)
            out[ext] = path
        except Exception as exc:  # noqa: BLE001 — без файла отчёт всё равно есть в JSON
            _log_line("[%s] не удалось собрать аспектный %s: %s"
                      % (job.get("id"), ext.upper(), str(exc)[:200]), level="error")
    return out
