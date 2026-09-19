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
# лимита попадают в отчёт (pass2_capped / pass2_capped_share), молча ничего не отбрасывается.
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

STAGE_LABELS = {
    "preparing": "подготовка",
    "pass1": "размечаю выборку",
    "writing": "пишу результаты",
    "pass2": "перепроверяю спорные на 32B",
    "report": "собираю отчёт",
    "done": "готово",
    "cancelled": "остановлено",
    "error": "ошибка",
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
    }
    try:
        _es().indices.put_mapping(index=name, properties=props)
        _mapping_cache.pop(name, None)
    except Exception as exc:  # noqa: BLE001 — маппинг уже мог быть создан или поле конфликтует
        _log_line("маппинг полей тональности не применён: %s" % str(exc)[:200], level="warning")


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
        raise HTTPException(status_code=400, detail="Не указан датасет (index)")
    mapping = _index_map()
    if re.fullmatch(r"\d+", text):
        key = int(text)
        name = mapping.get(key)
        if not name:
            raise HTTPException(status_code=404, detail="Датасет не найден: ключ %s" % key)
        return key, _norm_name(name)
    low = _norm_name(text).lower()
    for key, name in mapping.items():
        if _norm_name(name).lower() == low:
            return key, _norm_name(name)
    if _es().indices.exists(index=text):
        return None, text
    raise HTTPException(status_code=404, detail="Датасет не найден: %s" % text)


def _guard_dataset(user: Any, spec: Any) -> Tuple[Optional[int], str]:
    """Тот же уровень доступа, что у аналитики: чужой датасет — 403."""
    key, name = _resolve_dataset(spec)
    if getattr(user, "is_superuser", False):
        return key, name
    stem = _norm_name(name).lower()
    if stem not in _allowed_stems(getattr(user, "id", None)):
        raise HTTPException(status_code=403, detail="Нет доступа к этому датасету")
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
    if not filters:
        return {"match_all": {}}
    return {"bool": {"filter": filters}}


def _unlabeled_query(base: Dict[str, Any]) -> Dict[str, Any]:
    """Только ещё не размеченные нашей моделью документы — это и есть точка возобновления."""
    if not base or "match_all" in base:
        return {"bool": {"must_not": [{"exists": {"field": "tone_llm"}}]}}
    inner = list(base.get("bool", {}).get("filter") or [])
    return {"bool": {"filter": inner, "must_not": [{"exists": {"field": "tone_llm"}}]}}


def _active_query(job: Dict[str, Any]) -> Dict[str, Any]:
    """Рабочая выборка задачи: при relabel размечаем заново, иначе — только ещё не размеченное."""
    base = _base_query(job)
    if job.get("relabel"):
        return base
    return _unlabeled_query(base)


def _search_page(job: Dict[str, Any], size: int, cursor: Optional[List[Any]]) -> List[Dict[str, Any]]:
    name = job["index_name"]
    query = _active_query(job)
    text_field = _text_field(name)
    id_field = _id_field(name)
    source = ["toneMark", "timeCreate", "hub", "hubtype", "type", "url", "title",
              "review_rating", text_field]
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
                "tone_llm_reason": "локальная модель не вернула вердикт",
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
    name = job["index_name"]
    ctx = _Ctx(jid)
    started = time.time()
    try:
        _set(jid, status="running", stage="preparing", stage_label=STAGE_LABELS["preparing"],
             started=job.get("started") or datetime.now().isoformat(timespec="seconds"),
             writing=False, progress_schema=PROGRESS_SCHEMA,
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
                "Полная разметка этого датасета — %d сообщений, это больше лимита %d. "
                "Сузьте период (min_date/max_date) или уменьшите лимит переменной "
                "TELLSCOPE_TONE_FULL_LIMIT." % (total, MAX_FULL_DOCS))
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
                              "reason": "модель перепроверки недоступна (%s)" % (base_url or "не задана")})
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
             report_files=files,
             log="готово: размечено %d, согласие %.1f%%, каппа %.3f"
                 % (len(results), 100 * report["summary"]["agreement"], report["summary"]["kappa"]))
    except Exception as exc:  # noqa: BLE001
        _log_line("[%s] ошибка: %s" % (jid, str(exc)[:300]), level="error")
        _set(jid, status="error", stage="error", stage_label=STAGE_LABELS["error"],
             error=str(exc)[:500], finished=datetime.now().isoformat(timespec="seconds"))


def _start_job(jid: str) -> None:
    threading.Thread(target=lambda: asyncio.run(_run_job(jid)), daemon=True,
                     name="tone-check-%s" % jid).start()


def _resume_job(jid: str) -> None:
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


if str(os.environ.get("TELLSCOPE_TONE_NO_AUTORESUME") or "") != "1":
    threading.Thread(target=_delayed_recovery, daemon=True, name="tone-check-autoresume").start()


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
            "decided_by": rec.get("tone_llm_by") or "",
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
            "pass1_model": "qwen3-4b-fast (vLLM, 127.0.0.1:8001, профиль texts_bulk)",
            "pass2_model": "Qwen/Qwen3-32B-FP8 (vLLM, 127.0.0.1:8000)",
            "batch_size": int(job.get("batch_size") or DEFAULT_BATCH),
            "parallel": int(job.get("parallel") or DEFAULT_PARALLEL),
            "conf_threshold": float(job.get("conf_threshold") or DEFAULT_CONF),
            "fields": list(TONE_FIELDS),
            "source_field": "toneMark (не изменяется)",
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
        return "весь период датасета"
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
        "Проверено %s: модель совпала с разметкой источника в %s случаев, "
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
        out.append("Самая частая путаница: модель ставит «%s» там, где источник поставил «%s» — %s."
                   % (TONE_RU[top_pair[1]], TONE_RU[top_pair[0]],
                      _plural(top_pair[2], "случай", "случая", "случаев")))

    diff = int(summary.get("model_more_negative") or 0) - int(summary.get("model_more_positive") or 0)
    if abs(diff) >= max(3, int(0.01 * max(1, summary.get("evaluated") or 1))):
        if diff > 0:
            out.append("Модель строже источника: она видит негатив там, где источник поставил более "
                       "мягкую оценку (%s против %s) — источник часть недовольства относит к нейтрали."
                       % (_plural(int(summary.get("model_more_negative") or 0), "случай", "случая", "случаев"),
                          _plural(int(summary.get("model_more_positive") or 0), "обратного", "обратных", "обратных")))
        else:
            out.append("Модель мягче источника: она ставит нейтраль или позитив там, где источник "
                       "видит негатив (%s против %s)."
                       % (_plural(int(summary.get("model_more_positive") or 0), "случай", "случая", "случаев"),
                          _plural(int(summary.get("model_more_negative") or 0), "обратного", "обратных", "обратных")))

    if by_hub:
        bad = by_hub[0]
        out.append("Площадка с наибольшим числом расхождений — %s: %d из %d сообщений (%s)."
                   % (bad["key"], bad["mismatch"], bad["total"], _pct(float(bad["share"]))))

    share = float(summary.get("pass2_share") or 0.0)
    out.append("Спорных случаев, которые решала 32B, — %s от выборки (%d сообщений); "
               "по ним решение окончательное." % (_pct(share), int(summary.get("pass2_decided") or 0)))
    if not summary.get("pass2_available", True):
        out.append("ВНИМАНИЕ: модель перепроверки была недоступна — спорные случаи остались "
                   "решением быстрой модели, окончательными их считать нельзя.")
    capped = int(summary.get("pass2_capped") or 0)
    if capped:
        out.append("Спорных оказалось %d — больше лимита %.0f%% от выборки, поэтому на 32B ушли "
                   "только %d самых неуверенных сообщений; у остальных %d осталось решение 4B."
                   % (int(summary.get("pass2_candidates") or 0),
                      float(summary.get("pass2_max_share") or PASS2_MAX_SHARE) * 100,
                      int(summary.get("pass2_decided") or 0), capped))
    if int(summary.get("unresolved") or 0):
        out.append("Для %d сообщений локальная модель не вернула вердикт: они помечены "
                   "tone_llm_by=none и исключены из расчёта согласия, а не выданы за решение модели."
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
        text = ("Источнику можно доверять: разметка источника и наша модель совпадают почти всегда "
                "(согласие %s, каппа %.2f). Для отчётности можно опираться на поле источника."
                % (_pct(agreement), kappa))
        if weak:
            text += (" Оговорка: на текстах %s расхождений заметно больше (%s) — такой подпериод "
                     "лучше считать по нашей разметке." % (weak["key"], _pct(float(weak["share"]))))
        return text
    if agreement >= 0.7:
        text = ("Источник годится для общих трендов, но не для точных цифр: согласие %s, каппа %.2f. "
                "Для решений и публичных цифр считайте по нашей разметке (поля tone_llm*)."
                % (_pct(agreement), kappa))
        if weak:
            text += (" Основная слабость источника — %s: расхождения %s."
                     % (weak["key"], _pct(float(weak["share"]))))
        if by_hub:
            text += (" Хуже всего дела на площадке %s (%d расхождений из %d)."
                     % (by_hub[0]["key"], by_hub[0]["mismatch"], by_hub[0]["total"]))
        return text
    text = ("Разметке источника доверять нельзя: согласие всего %s, каппа %.2f — источник и модель "
            "расходятся слишком часто. Считайте по нашей разметке (tone_llm, tone_llm_conf, tone_llm_by)."
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
    src_columns = ["Источник \\ модель"] + [TONE_RU[c] for c in CLASSES]
    matrix_rows = []
    for a in CLASSES:
        row = [TONE_RU[a]]
        for b in CLASSES:
            row.append(int((matrix.get(str(a)) or {}).get(str(b)) or 0))
        matrix_rows.append(row)

    pass2_row = "перепроверка не выполнялась (%s)" % (pass2.get("reason") or "нет спорных случаев") \
        if not pass2.get("available", True) else "выполнена"

    sections: List[Dict[str, Any]] = []
    sections.append({
        "heading": "Как проверяли",
        "text": ("Датасет: %s. Период: %s. Режим: %s. Проверено сообщений: %d.\n"
                 "Сообщения размечала быстрая модель qwen3-4b-fast (vLLM, порт 8001) пачками по %d "
                 "в строгий JSON; спорные случаи — там, где уверенность ниже %.2f или решение "
                 "разошлось с полем источника toneMark — перепроверяла Qwen3-32B (порт 8000) и "
                 "решала окончательно (%s).\n"
                 "Наша разметка пишется в отдельные поля tone_llm, tone_llm_conf, tone_llm_by, "
                 "tone_llm_reason; исходное toneMark не изменяется, существующая аналитика "
                 "продолжает работать по нему."
                 % (report.get("dataset") or "—", report.get("period") or "—",
                    "полная разметка" if summary.get("mode") == "full" else
                    "выборка %s сообщений" % (summary.get("sample_size") or "—"),
                    summary.get("checked") or 0, report["method"]["batch_size"],
                    float(report["method"]["conf_threshold"]), pass2_row)),
        "tables": [{
            "title": "Ключевые цифры",
            "columns": ["Показатель", "Значение"],
            "rows": [
                ["Сообщений проверено", str(summary.get("checked") or 0)],
                ["Учтено в сравнении с источником", str(summary.get("evaluated") or 0)],
                ["Согласие модели с источником", _pct(float(summary.get("agreement") or 0.0))],
                ["Совпало", str(summary.get("matches") or 0)],
                ["Расхождений", str(summary.get("mismatches") or 0)],
                ["Каппа Коэна", "%.3f" % float(summary.get("kappa") or 0.0)],
                ["Спорных случаев на 32B", _pct(float(summary.get("pass2_share") or 0.0))],
                ["Решений 32B", str(summary.get("pass2_decided") or 0)],
                ["Не разобрано моделью", str(summary.get("unresolved") or 0)],
                ["Записано в Elasticsearch", str(summary.get("bulk_written") or 0)],
            ],
            "layout": "portrait",
        }],
    })
    sections.append({
        "heading": "Матрица «источник → модель»",
        "text": ("Строки — тональность источника (toneMark), столбцы — тональность нашей модели "
                 "(tone_llm). Диагональ — совпадения."),
        "tables": [{
            "title": "Распределение «источник → модель»",
            "columns": src_columns,
            "rows": matrix_rows,
            "layout": "portrait",
        }, {
            "title": "Точность и полнота по классам (эталон — источник)",
            "columns": ["Класс", "Точность", "Полнота", "F1", "В источнике", "Модель поставила"],
            "rows": [[row["label"], _num(row["precision"]), _num(row["recall"]), _num(row["f1"]),
                      row["support"], row["predicted"]] for row in report["per_class"]],
            "note": "Точность — доля верных среди решений модели по классу, полнота — доля "
                    "найденных сообщений этого класса источника.",
            "layout": "portrait",
        }],
    })
    sections.append({
        "heading": "Где расходимся: площадки и разделы",
        "text": "Разбивка расхождений по площадкам, типам источников и разделам датасета.",
        "tables": [
            {"title": "Площадки", "columns": ["Площадка", "Сообщений", "Расхождений", "Доля"],
             "rows": [[row["key"], row["total"], row["mismatch"], _pct(float(row["share"]))]
                      for row in report["by_hub"]], "layout": "auto"},
            {"title": "Тип источника (hubtype)", "columns": ["Тип", "Сообщений", "Расхождений", "Доля"],
             "rows": [[row["key"], row["total"], row["mismatch"], _pct(float(row["share"]))]
                      for row in report["by_section"]], "layout": "auto"},
            {"title": "Раздел датасета (type)", "columns": ["Раздел", "Сообщений", "Расхождений", "Доля"],
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
            "columns": ["Площадка", "Дата", "Источник → модель", "Цитата из сообщения",
                        "Пояснение модели", "Ссылка"],
            "rows": [[item["hub"], item["date"],
                      "%s → %s" % (item["source"], item["model"]),
                      item["quote"], item["reason"], item["url"] or "—"] for item in examples],
            "note": "Ссылки приведены текстом: в JSON-версии отчёта они кликабельны.",
            "layout": "landscape",
        })
    sections.append({
        "heading": "Примеры расхождений",
        "text": "Показаны самые уверенные решения модели, разошедшиеся с источником: именно такие "
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
    label = _safe_name(report.get("dataset") or job.get("index_name") or "датасет", 40)
    stamp = datetime.now().strftime("%Y-%m-%d %H-%M")
    mode = "полная" if (job.get("mode") == "full") else "выборка %s" % (job.get("sample_size") or "")
    title = "Проверка тональности: качество разметки источника"
    subtitle = "%s — %s, %s" % (label, report.get("period") or "", mode)
    meta = {
        "dataset_label": report.get("dataset") or "",
        "period": report.get("period") or "",
        "author": "Tellscope, локальные модели vLLM",
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
    index: str = Field(description="ключ датасета из indexes.pkl или имя индекса Elasticsearch")
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
            note = "идёт перепроверка спорных на 32B — она медленнее первого прохода"
        if writing:
            note = "пишу размеченную пачку в Elasticsearch"
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
        raise HTTPException(status_code=404, detail="Индекс %s не найден в Elasticsearch" % name)
    tone_int = _tone_filter(body.tone)
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
    _log_line("[%s] запуск: %s (%s, %s), владелец %s" % (jid, name, mode, body.sample_size, job["owner"]))
    return {"job_id": jid, "status": "queued", "index_name": name, "mode": mode}


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
