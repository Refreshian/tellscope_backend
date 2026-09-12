# -*- coding: utf-8 -*-
"""Инструменты аналитики агентного режима: датасеты, поиск, тональность, СМИ, голос клиента.

Все обработчики выполняются в процессе FastAPI и переиспользуют функции эндпоинтов
интерфейса, поэтому доступ к датасету проверяется той же функцией
main._guard_index_access(user, index).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi.encoders import jsonable_encoder

from .context import compact, to_unix
from .registry import ToolError, tool

INDEXES_PKL = "/home/dev/tellscope_app/tellscope_backend/data/indexes.pkl"

TONE_VALUES = {-1: "негатив", 0: "нейтрал", 1: "позитив"}


def _m():
    import main

    return main


def _iso(ts: Any) -> str:
    try:
        return datetime.fromtimestamp(int(float(ts)), tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ""


def index_map() -> Dict[int, str]:
    try:
        raw = _m().load_dict_from_pickle(INDEXES_PKL)
    except Exception:
        return {}
    out: Dict[int, str] = {}
    for key, value in (raw or {}).items():
        try:
            out[int(key)] = str(value)
        except Exception:
            continue
    return out


def _stem(name: Any) -> str:
    low = str(name or "").lower()
    return low[:-5] if low.endswith(".json") else low


_TRANSLIT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e", "ж": "zh", "з": "z",
    "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r",
    "с": "s", "т": "t", "у": "u", "ф": "f", "х": "h", "ц": "c", "ч": "ch", "ш": "sh", "щ": "sch",
    "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
}


def _norm_text(value: Any) -> str:
    """Нормализует название темы: нижний регистр, транслит кириллицы, только буквы и цифры."""
    text = _stem(value).lower()
    text = "".join(_TRANSLIT.get(ch, ch) for ch in text)
    return "".join(ch for ch in text if ch.isalnum())


_NICE_WORDS = {
    "kfc": "KFC", "smi": "СМИ", "oiv": "ОИВ", "ba": "БА", "orvi": "ОРВИ", "vk": "VK",
    "tg": "TG", "gis": "2ГИС", "rrb": "РСХБ", "ai": "ИИ",
    "jan": "январь", "feb": "февраль", "mar": "март", "apr": "апрель", "may": "май",
    "jun": "июнь", "jul": "июль", "aug": "август", "sep": "сентябрь", "oct": "октябрь",
    "nov": "ноябрь", "dec": "декабрь",
}


def _nice_word(word: str) -> str:
    """kfc → KFC, may → май, platon → Платон."""
    low = word.lower()
    if low in _NICE_WORDS:
        return _NICE_WORDS[low]
    if word.isupper():
        return word
    return word.capitalize()


def _split_name(name: str) -> "tuple[str, str]":
    """«platon_13.10.2025-30.11.2025» → («Платон», «13.10.2025 — 30.11.2025»)."""
    import re as _re

    text = _stem(name)
    period = ""
    match = _re.search(r"(\d{2}\.\d{2}\.\d{4})\s*[-–]\s*(\d{2}\.\d{2}\.\d{4})", text)
    if match:
        period = f"{match.group(1)} — {match.group(2)}"
        text = (text[: match.start()] + text[match.end():]).strip(" _-.")
    text = _re.sub(r"[0-9a-f]{16,}", "", text).strip(" _-.")
    words = [word for word in text.replace("_", " ").replace("-", " ").split() if word]
    label = " ".join(_nice_word(word) for word in words)
    return (label or _stem(name), period)


def datasets_public(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Уникальные темы для интерфейса и Dify: имя, подпись и период (без дублей индексов)."""
    seen: Dict[str, Dict[str, Any]] = {}
    for idx, name in sorted(index_map().items(), reverse=True):
        key = _stem(name)
        label, period = _split_name(name)
        item = {"index": idx, "name": key, "label": label, "period": period}
        if key not in seen:
            seen[key] = item
        else:  # оставляем самый свежий индекс для этого имени
            seen[key]["index"] = max(seen[key]["index"], idx)
    items = sorted(seen.values(), key=lambda item: item["index"], reverse=True)
    return items[:limit] if limit else items


def _match_dataset(value: Any) -> Optional[int]:
    """Ищет тему по названию, подписи или части строки — с учётом транслита (Платон → platon)."""
    raw = _stem(value).strip().lower()
    text = _norm_text(value)
    if len(text) < 2:
        return None
    # «Платон · 13.10.2025 — 30.11.2025»: сначала подпись, при необходимости — период
    if "·" in str(value):
        head, _, tail = str(value).partition("·")
        head_norm = _norm_text(head)
        tail_norm = _norm_text(tail)
        best: Optional[int] = None
        for idx, name in index_map().items():
            label, period = _split_name(name)
            if _norm_text(label) != head_norm:
                continue
            if tail_norm and tail_norm not in _norm_text(period):
                continue
            best = idx if best is None else max(best, idx)
        if best is not None:
            return best
    mapping = index_map()
    for idx, name in mapping.items():  # 1) точное имя
        if _stem(name).strip().lower() == raw:
            return idx
    candidates: List[Any] = []
    for idx, name in mapping.items():
        label, _period = _split_name(name)
        if _norm_text(label) == text:
            candidates.append((idx, 0, len(_norm_text(label))))
    if candidates:  # 2) точная подпись темы
        candidates.sort(key=lambda item: (-item[0], item[1]))
        return candidates[0][0]
    hits = []
    for idx, name in mapping.items():
        label, _period = _split_name(name)
        for source in (_stem(name), label):
            norm = _norm_text(source)
            if text and text in norm:
                hits.append((idx, len(norm)))
                break
    if hits:  # 3) запрос — часть названия или подписи (свежий индекс при дублях)
        hits.sort(key=lambda item: (item[1], -item[0]))
        return hits[0][0]
    hits = []
    for idx, name in mapping.items():
        label, _period = _split_name(name)
        for source in (_stem(name), label):
            norm = _norm_text(source)
            if len(norm) > 3 and norm in text:
                hits.append((idx, len(norm)))
                break
    if hits:  # 4) название темы — часть запроса («отчёт по Платону»)
        hits.sort(key=lambda item: (-item[1], -item[0]))
        return hits[0][0]
    return None


def available_datasets(limit: int = 12) -> str:
    names = sorted({_stem(name) for name in index_map().values()})
    shown = ", ".join(f"«{name}»" for name in names[:limit])
    tail = " и другие" if len(names) > limit else ""
    return shown + tail + ". Если нужной темы нет — загрузите её из Brand Analytics инструментом fetch_dataset"


def resolve_index(ctx: Any, index: Any = None) -> int:
    """Принимает номер датасета или название темы («Платон», «kfc_may_01-14»)."""
    idx = index if index is not None else ctx.dataset_index
    if isinstance(idx, str):
        text = idx.strip()
        if text and not text.lstrip("-").isdigit():
            found = _match_dataset(text)
            if found is None:
                raise ToolError(
                    f"Тема «{text}» не найдена. Доступные темы: {available_datasets()}"
                )
            return found
        idx = text or None
    if idx is None:
        raise ToolError(
            "Тема не выбрана: укажите название датасета (например «Платон») или его номер"
        )
    try:
        return int(idx)
    except Exception as exc:
        raise ToolError(f"Некорректный индекс датасета: {index!r}") from exc


def guard(ctx: Any, index: Optional[int] = None):
    """Проверяет доступ пользователя к датасету и возвращает (index, имя индекса ES)."""
    idx = resolve_index(ctx, index)
    _m()._guard_index_access(ctx.user, idx)
    name = index_map().get(idx)
    if not name:
        raise ToolError(f"Индекс {idx} не найден в справочнике датасетов")
    return idx, name


def dates(ctx: Any, min_date: Any = None, max_date: Any = None):
    lo = to_unix(min_date) if min_date is not None else ctx.min_date
    hi = to_unix(max_date) if max_date is not None else ctx.max_date
    return lo, hi


def _es():
    return _m().es


def _terms(index_name: str, field: str, size: int = 12, query: Optional[dict] = None) -> List[Dict[str, Any]]:
    body: Dict[str, Any] = {"size": 0, "aggs": {"v": {"terms": {"field": field, "size": size}}}}
    if query:
        body["query"] = query
    try:
        res = _es().search(index=index_name, body=body)
        return [{"key": b.get("key"), "count": b.get("doc_count")} for b in res["aggregations"]["v"]["buckets"]]
    except Exception:
        return []


def _monthly(index_name: str, query: Optional[dict] = None, limit: int = 40) -> List[Dict[str, Any]]:
    """Динамика по месяцам. timeCreate — unix-секунды (long), поэтому считаем через runtime-поле."""
    body: Dict[str, Any] = {
        "size": 0,
        "runtime_mappings": {"__month": {"type": "date", "script": {"source": "emit(doc['timeCreate'].value * 1000)"}}},
        "aggs": {
            "d": {
                "date_histogram": {
                    "field": "__month",
                    "calendar_interval": "month",
                    "format": "yyyy-MM",
                    "min_doc_count": 1,
                }
            }
        },
    }
    if query:
        body["query"] = query
    try:
        res = _es().search(index=index_name, body=body)
        buckets = res["aggregations"]["d"]["buckets"]
        rows = []
        for bucket in buckets:
            month = str(bucket.get("key_as_string") or "")
            if month.startswith("1970-01"):
                # timeCreate не заполнен в индексе — «январь 1970» это нулевые метки, а не данные
                continue
            rows.append({"month": month, "count": bucket.get("doc_count")})
        return rows[-limit:]
    except Exception:
        return []


def _period(index_name: str):
    """Реальный период датасета по timeCreate. (None, None), если поле в индексе пустое."""
    try:
        res = _es().search(
            index=index_name,
            body={"size": 0, "aggs": {"lo": {"min": {"field": "timeCreate"}}, "hi": {"max": {"field": "timeCreate"}}}},
        )
        agg = res.get("aggregations") or {}
        lo = int((agg.get("lo") or {}).get("value") or 0)
        hi = int((agg.get("hi") or {}).get("value") or 0)
    except Exception:
        return None, None
    if lo <= 0 or hi <= 0:
        return None, None
    return lo, hi


def _period_or_none(lo, hi):
    if lo and hi and hi > 0:
        return int(lo), int(hi)
    return None, None


def _tone_rows(buckets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for item in buckets:
        key = item.get("key")
        try:
            key = int(key)
        except Exception:
            pass
        rows.append({"tone": TONE_VALUES.get(key, key), "count": item.get("count")})
    return rows


NO_TIME_NOTE = (
    "В этом индексе не заполнено поле времени (timeCreate равно 0), поэтому доступны только метрики "
    "без разбивки по датам: общие счётчики, тональность, площадки, поиск сообщений."
)


def _require_period(index_name: str, lo, hi):
    """Период для инструментов, которым он нужен. Бросает понятную ошибку, если времени нет."""
    lo, hi = _period_or_none(lo, hi)
    if lo and hi:
        return lo, hi
    lo2, hi2 = _period(index_name)
    if lo2 and hi2:
        return lo2, hi2
    raise ToolError(NO_TIME_NOTE)


def _exact_count(index_name: str, query: Optional[dict] = None) -> int:
    try:
        if query is None:
            res = _es().count(index=index_name)
        else:
            res = _es().count(index=index_name, body={"query": query})
        return int(res.get("count") or 0)
    except Exception:
        return 0


def _tone_filter(tone: Optional[str]) -> Optional[dict]:
    value = (tone or "any").lower()
    if value in ("any", "", "all", "любая"):
        return None
    if value.startswith("нег") or value == "negative":
        return {"term": {"toneMark": -1}}
    if value.startswith("поз") or value == "positive":
        return {"term": {"toneMark": 1}}
    if value.startswith("ней") or value == "neutral":
        return {"term": {"toneMark": 0}}
    return None


def _query(phrase: Optional[str], lo: Optional[int], hi: Optional[int], tone: Optional[str] = None) -> dict:
    must: List[dict] = []
    if phrase and str(phrase).strip():
        must.append({"match_phrase": {"text": str(phrase).strip()}})
    filters: List[dict] = []
    if lo or hi:
        rng: Dict[str, Any] = {}
        if lo:
            rng["gte"] = int(lo)
        if hi:
            rng["lte"] = int(hi)
        filters.append({"range": {"timeCreate": rng}})
    tone_q = _tone_filter(tone)
    if tone_q:
        filters.append(tone_q)
    query: Dict[str, Any] = {"bool": {}}
    if must:
        query["bool"]["must"] = must
    if filters:
        query["bool"]["filter"] = filters
    if not query["bool"]:
        query = {"match_all": {}}
    return query


def _sample(hit: Dict[str, Any]) -> Dict[str, Any]:
    src = hit.get("_source") or {}
    author = src.get("authorObject") or {}
    return {
        "date": _iso(src.get("timeCreate")),
        "text": (src.get("text") or src.get("title") or "")[:400],
        "hub": src.get("hub") or "",
        "author": author.get("fullname") or "",
        "author_type": author.get("author_type") or "",
        "city": src.get("city") or "",
        "tone": TONE_VALUES.get(src.get("toneMark"), src.get("toneMark")),
        "likes": src.get("likesCount"),
        "comments": src.get("commentsCount"),
        "url": src.get("url") or "",
        "es_id": hit.get("_id"),
    }


@tool(
    "list_datasets",
    title="Список тем (датасетов)",
    description=(
        "Список доступных тем соцмедиа и СМИ: подпись, период и название датасета. "
        "Если пользователь назвал тему словами («Платон», «KFC за май») — сначала вызови этот инструмент, "
        "чтобы подобрать нужную тему, и передавай дальше её название или индекс."
    ),
    parameters={
        "type": "object",
        "properties": {
            "include_shared": {"type": "boolean", "description": "включать общие датасеты"},
        },
    },
    group="analytics",
)
async def list_datasets(ctx, include_shared: bool = False):
    """Уникальные темы с читаемыми подписями — для выбора темы человеком и моделью."""
    items = datasets_public()
    visible = []
    for item in items:
        try:
            _m()._guard_index_access(ctx.user, item["index"])
        except Exception:
            continue
        visible.append(item)
    return {
        "datasets": compact(visible, max_items=60),
        "total": len(visible),
        "hint": "index можно не указывать: инструменты принимают и название темы, и её подпись",
    }


@tool(
    "dataset_overview",
    title="Обзор датасета",
    description=(
        "Ключевые характеристики датасета: число сообщений, период, распределение тональности, "
        "площадки-источники, города и динамика по месяцам. Хороший первый шаг перед детальным анализом."
    ),
    parameters={
        "type": "object",
        "properties": {
            "index": {"type": "string", "description": "тема: название датасета (например «Платон») или его номер; по умолчанию — выбранный в интерфейсе"},
            "top_n": {"type": "integer", "description": "сколько значений в топах, по умолчанию 10"},
        },
    },
    group="analytics",
)
async def dataset_overview(ctx, index: Optional[int] = None, top_n: int = 10):
    idx, index_name = guard(ctx, index)
    top_n = max(3, min(int(top_n or 10), 25))
    total = _exact_count(index_name)
    lo, hi = _period(index_name)
    tone = _tone_rows(_terms(index_name, "toneMark", 5))
    period = {
        "from": _iso(lo) if lo else None,
        "to": _iso(hi) if hi else None,
        "days": int((hi - lo) / 86400) if lo and hi else None,
    }
    notes = ["toneMark: -1 негатив, 0 нейтрал, 1 позитив"]
    if not lo:
        notes.append(NO_TIME_NOTE)
    return {
        "index": idx,
        "index_name": index_name,
        "messages_total": total,
        "period": period,
        "tonality": tone,
        "hubs": _terms(index_name, "hub", top_n),
        "cities": _terms(index_name, "city", top_n),
        "monthly_dynamics": _monthly(index_name, limit=48),
        "note": "; ".join(notes),
    }


@tool(
    "search_messages",
    title="Поиск сообщений",
    description=(
        "Поиск сообщений по теме с фильтрами по датам и тональности. Поиск смысловой: учитываются формы слов и "
        "синонимы («отравление» находит «траванулся», «отравился», «пищевое отравление»), поэтому в ответе есть "
        "search_terms и per_term_counts — по каким формулировкам и сколько найдено. Возвращает точное число "
        "совпадений, распределение по площадкам и тональности, динамику по месяцам и примеры сообщений со ссылками. "
        "Основной инструмент доказательной базы отчёта."
    ),
    parameters={
        "type": "object",
        "properties": {
            "phrase": {"type": "string", "description": "тема поиска; пусто — все сообщения периода"},
            "index": {"type": "string", "description": "тема: название датасета или его номер"},
            "min_date": {"type": "string", "description": "начало периода: YYYY-MM-DD или unix-секунды"},
            "max_date": {"type": "string", "description": "конец периода: YYYY-MM-DD или unix-секунды"},
            "tone": {"type": "string", "enum": ["any", "negative", "positive", "neutral"], "description": "фильтр тональности"},
            "limit": {"type": "integer", "description": "сколько примеров вернуть (по умолчанию 10, максимум 30)"},
            "sort": {"type": "string", "enum": ["date_desc", "date_asc", "relevance"], "description": "порядок примеров"},
            "expand": {"type": "boolean", "description": "расширять запрос формами слов и синонимами (по умолчанию да)"},
        },
        "required": ["phrase"],
    },
    group="analytics",
)
async def search_messages(
    ctx,
    phrase: str = "",
    index: Optional[int] = None,
    min_date: Any = None,
    max_date: Any = None,
    tone: str = "any",
    limit: int = 10,
    sort: str = "date_desc",
    expand: bool = True,
):
    idx, index_name = guard(ctx, index)
    lo, hi = dates(ctx, min_date, max_date)
    limit = max(1, min(int(limit or 10), 30))

    variants: List[str] = []
    per_term: Dict[str, int] = {}
    expanded_info: Dict[str, Any] = {}
    if expand and phrase and str(phrase).strip():
        from . import semantic

        expanded_info = await semantic.expand_terms(ctx, phrase, use_llm=False)
        variants = expanded_info.get("variants") or []

    query = semantic.build_query(variants, []) if variants else _query(phrase, lo, hi, tone)
    if variants:
        # период и тональность добавляем фильтрами, чтобы не терять расширенные формулировки
        filters: List[dict] = []
        if lo or hi:
            rng: Dict[str, Any] = {}
            if lo:
                rng["gte"] = int(lo)
            if hi:
                rng["lte"] = int(hi)
            filters.append({"range": {"timeCreate": rng}})
        filters.extend(_query("", None, None, tone).get("bool", {}).get("filter") or [])
        if filters:
            query.setdefault("bool", {})["filter"] = filters
        per_term = semantic.count_terms(index_name, variants, filters)

    sort_spec = [{"timeCreate": {"order": "desc"}}]
    if sort == "date_asc":
        sort_spec = [{"timeCreate": {"order": "asc"}}]
    elif sort == "relevance":
        sort_spec = ["_score", {"timeCreate": {"order": "desc"}}]
    body = {
        "size": limit,
        "_source": ["text", "title", "timeCreate", "hub", "city", "toneMark", "url", "likesCount", "commentsCount", "authorObject"],
        "query": query,
        "sort": sort_spec,
    }
    try:
        res = _es().search(index=index_name, body=body)
    except Exception as exc:
        raise ToolError(f"Ошибка поиска в Elasticsearch: {exc}") from exc
    hits = (res.get("hits") or {}).get("hits") or []
    total = _exact_count(index_name, query)
    notes = []
    if not (lo or hi):
        if not _period(index_name)[0]:
            notes.append(NO_TIME_NOTE)
    if variants:
        notes.append("Поиск расширен формами слов и синонимами: см. search_terms и per_term_counts — только эти формулировки дают данное число сообщений.")

    result = {
        "index": idx,
        "index_name": index_name,
        "phrase": phrase,
        "period": {"from": _iso(lo) if lo else None, "to": _iso(hi) if hi else None},
        "messages_found": total,
        "tonality": _tone_rows(_terms(index_name, "toneMark", 5, query)),
        "hubs": _terms(index_name, "hub", 12, query),
        "cities": _terms(index_name, "city", 8, query),
        "monthly_dynamics": _monthly(index_name, query, limit=48),
        "examples": [_sample(h) for h in hits],
        "notes": notes,
    }
    if variants:
        result["search_terms"] = variants
        result["search_sources"] = expanded_info.get("sources") or []
        result["per_term_counts"] = compact(per_term, max_items=20, max_str=80)
        result["exact_phrase_messages"] = _exact_count(index_name, _query(phrase, lo, hi, tone))
    return result


def _guard_call(method, *args, **kwargs):
    try:
        return method(*args, **kwargs)
    except Exception as exc:
        try:
            from fastapi import HTTPException

            if isinstance(exc, HTTPException):
                raise ToolError(f"HTTP {exc.status_code}: {exc.detail}") from exc
        except ImportError:
            pass
        raise ToolError(f"{type(exc).__name__}: {exc}") from exc


@tool(
    "tonality_summary",
    title="Тональность и авторы",
    description=(
        "Тональный ландшафт датасета: распределение тональности по площадкам, авторы-источники "
        "негатива и позитива. Считается агрегациями по Elasticsearch, работает быстро на больших датасетах. "
        "Используй для оценки репутационного фона и поиска драйверов негатива."
    ),
    parameters={
        "type": "object",
        "properties": {
            "index": {"type": "string", "description": "тема: название датасета или её номер"},
            "min_date": {"type": "string", "description": "YYYY-MM-DD или unix-секунды"},
            "max_date": {"type": "string", "description": "YYYY-MM-DD или unix-секунды"},
        },
    },
    group="analytics",
    timeout=300.0,
)
async def tonality_summary(ctx, index: Optional[int] = None, min_date: Any = None, max_date: Any = None):
    idx, index_name = guard(ctx, index)
    lo, hi = dates(ctx, min_date, max_date)
    lo, hi = _require_period(index_name, lo, hi)
    base = _query(None, lo, hi, None)
    hubs = _terms(index_name, "hub", 12, base)
    by_hub = []
    for item in hubs[:8]:
        hub_query = {
            "bool": {
                "must": [{"term": {"hub": item.get("key")}}],
                "filter": [{"range": {"timeCreate": {"gte": int(lo), "lte": int(hi)}}}],
            }
        }
        by_hub.append({"hub": item.get("key"), "count": item.get("count"), "tonality": _tone_rows(_terms(index_name, "toneMark", 5, hub_query))})

    def top_authors(tone: str, limit: int = 10) -> List[Dict[str, Any]]:
        query = _query(None, lo, hi, tone)
        body = {
            "size": 1000,
            "_source": ["authorObject", "hub"],
            "query": query,
            "sort": [{"likesCount": {"order": "desc"}}],
        }
        try:
            res = _es().search(index=index_name, body=body)
        except Exception:
            return []
        counter: Dict[str, int] = {}
        for hit in (res.get("hits") or {}).get("hits") or []:
            src = hit.get("_source") or {}
            author = (src.get("authorObject") or {}).get("fullname") or src.get("hub") or ""
            if author:
                counter[author] = counter.get(author, 0) + 1
        top = sorted(counter.items(), key=lambda kv: -kv[1])[:limit]
        return [{"author": name, "posts_in_top1000": count} for name, count in top]

    return {
        "index": idx,
        "index_name": index_name,
        "period": {"from": _iso(lo), "to": _iso(hi)},
        "tonality_total": _tone_rows(_terms(index_name, "toneMark", 5, base)),
        "tonality_by_hub": by_hub,
        "top_negative_authors": top_authors("negative"),
        "top_positive_authors": top_authors("positive"),
        "note": "Авторы считаются по 1000 самым вовлекающим сообщениям соответствующей тональности.",
    }


SMI_FILTER = {"match_phrase": {"hubtype": "Онлайн-СМИ"}}


@tool(
    "media_rating",
    title="Рейтинг СМИ",
    description=(
        "Рейтинг онлайн-СМИ (hubtype = «Онлайн-СМИ») по датасету: сколько негативных и позитивных публикаций "
        "у каждого издания, плюс свежая лента публикаций СМИ со ссылками. Считается агрегациями Elasticsearch — быстро."
    ),
    parameters={
        "type": "object",
        "properties": {
            "index": {"type": "string", "description": "тема: название датасета или её номер"},
            "min_date": {"type": "string", "description": "YYYY-MM-DD или unix-секунды"},
            "max_date": {"type": "string", "description": "YYYY-MM-DD или unix-секунды"},
            "limit": {"type": "integer", "description": "сколько изданий вернуть в каждом рейтинге (по умолчанию 12)"},
        },
    },
    group="analytics",
    timeout=240.0,
)
async def media_rating(ctx, index: Optional[int] = None, min_date: Any = None, max_date: Any = None, limit: int = 12):
    idx, index_name = guard(ctx, index)
    lo, hi = dates(ctx, min_date, max_date)
    lo, hi = _require_period(index_name, lo, hi)
    limit = max(3, min(int(limit or 12), 30))
    query = {
        "bool": {
            "must": [SMI_FILTER],
            "filter": [{"range": {"timeCreate": {"gte": int(lo), "lte": int(hi)}}}],
        }
    }
    body = {
        "size": 0,
        "query": query,
        "aggs": {
            "hubs": {
                "terms": {"field": "hub", "size": 80},
                "aggs": {"tone": {"terms": {"field": "toneMark", "size": 5}}},
            }
        },
    }
    try:
        res = _es().search(index=index_name, body=body)
    except Exception as exc:
        raise ToolError(f"Ошибка агрегации по СМИ: {exc}") from exc
    buckets = ((res.get("aggregations") or {}).get("hubs") or {}).get("buckets") or []
    rows = []
    for bucket in buckets:
        tones = {b.get("key"): b.get("doc_count") for b in ((bucket.get("tone") or {}).get("buckets") or [])}
        rows.append(
            {
                "media": bucket.get("key"),
                "messages": bucket.get("doc_count"),
                "negative": int(tones.get(-1) or 0),
                "positive": int(tones.get(1) or 0),
                "neutral": int(tones.get(0) or 0),
            }
        )
    feed_body = {
        "size": 20,
        "_source": ["text", "title", "timeCreate", "hub", "toneMark", "url", "likesCount", "commentsCount", "authorObject"],
        "query": query,
        "sort": [{"timeCreate": {"order": "desc"}}],
    }
    feed = []
    try:
        feed_res = _es().search(index=index_name, body=feed_body)
        feed = [_sample(h) for h in (feed_res.get("hits") or {}).get("hits") or []]
    except Exception:
        feed = []
    return {
        "index": idx,
        "index_name": index_name,
        "period": {"from": _iso(lo), "to": _iso(hi)},
        "smi_messages_total": _exact_count(index_name, query),
        "negative_smi": sorted(rows, key=lambda r: -r["negative"])[:limit],
        "positive_smi": sorted(rows, key=lambda r: -r["positive"])[:limit],
        "media_feed": feed,
        "note": "СМИ определяется по hubtype «Онлайн-СМИ»; рейтинг — по числу публикаций соответствующей тональности.",
    }


@tool(
    "voice_of_customer",
    title="Голос клиента",
    description=(
        "Отзывы и жалобы клиентов: группировка сообщений по темам с тональностью и метриками вовлечения. "
        "Используй, когда нужно понять причины недовольства продуктом, сервисом или точками."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query_str": {"type": "string", "description": "ключевая фраза/тема, например «Ростикс» или «бургер»"},
            "index": {"type": "string", "description": "тема: название датасета или её номер"},
            "min_date": {"type": "string", "description": "YYYY-MM-DD или unix-секунды"},
            "max_date": {"type": "string", "description": "YYYY-MM-DD или unix-секунды"},
        },
    },
    group="analytics",
    timeout=300.0,
)
async def voice_of_customer(ctx, query_str: Optional[str] = None, index: Optional[int] = None, min_date: Any = None, max_date: Any = None):
    m = _m()
    idx, index_name = guard(ctx, index)
    lo, hi = dates(ctx, min_date, max_date)
    lo, hi = _require_period(index_name, lo, hi)
    data = await _guard_call(m.voice_analize, user=ctx.user, index=idx, min_date=lo, max_date=hi, query_str=query_str)
    payload = jsonable_encoder(data)
    values = payload.get("values") or []
    return {
        "index": idx,
        "query_str": query_str,
        "period": {"from": _iso(lo), "to": _iso(hi)},
        "topics": compact(values, max_items=12, max_str=200),
        "topics_total": len(values),
    }


@tool(
    "ai_analytics",
    title="ИИ-аналитика датасета",
    description=(
        "Подборка наиболее показательных сообщений датасета по запросу (встроенный ИИ-анализ Tellscope): "
        "тексты, площадки, вовлечение и ссылки. Запрос формулируйте как ключевую фразу темы "
        "(например «отравились», «бургер», «Ростикс») — по общим формулировкам выборка может быть пустой."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query_str": {"type": "string", "description": "что искать: тема, бренд, проблема"},
            "index": {"type": "string", "description": "тема: название датасета или её номер"},
            "min_date": {"type": "string", "description": "YYYY-MM-DD или unix-секунды"},
            "max_date": {"type": "string", "description": "YYYY-MM-DD или unix-секунды"},
        },
        "required": ["query_str"],
    },
    group="analytics",
    timeout=420.0,
)
async def ai_analytics(ctx, query_str: str, index: Optional[int] = None, min_date: Any = None, max_date: Any = None):
    m = _m()
    idx, index_name = guard(ctx, index)
    lo, hi = dates(ctx, min_date, max_date)
    lo, hi = _require_period(index_name, lo, hi)
    data = await _guard_call(m.ai_analytics_get, user=ctx.user, index=idx, min_date=lo, max_date=hi, query_str=query_str)
    payload = jsonable_encoder(data)
    items = payload.get("data") or []
    rows = []
    for item in items[:20]:
        rows.append(
            {
                "date": _iso(item.get("timeCreate")),
                "text": (item.get("text") or "")[:300],
                "hub": item.get("hub"),
                "audience": item.get("audienceCount"),
                "comments": item.get("commentsCount"),
                "url": item.get("url"),
            }
        )
    return {"index": idx, "query_str": query_str, "total_rows": payload.get("total_rows"), "examples": rows}
