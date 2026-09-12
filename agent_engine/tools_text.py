# -*- coding: utf-8 -*-
"""Чтение текстов датасета локальной моделью Qwen (vLLM): темы, цитаты, важные сообщения.

Зачем это нужно: статистика (графики, счётчики, распределение тональности) показывает,
СКОЛЬКО сообщений и какой тональности, но не объясняет, ПОЧЕМУ люди недовольны.
`analyze_texts` читает сами тексты сообщений за период и возвращает:

  * topics     — темы с числом сообщений, долей, средней тональностью и 2–3 цитатами
                 (цитата всегда дословная, с датой, площадкой, автором и ссылкой);
  * highlights — ключевые сообщения по вовлечённости (с откатом на рейтинг отзыва и объём текста);
  * categories — сопоставление тем с фиксированным каркасом категорий;
  * summary    — аналитический текст по темам и цитатам;
  * stats      — сколько сообщений прочитано, сколько батчей, время, модель, токены.

Разбор идёт пачками по 15–20 сообщений с повторами при ошибке; ответ модели — строго JSON.
Число сообщений по теме считается по идентификаторам, которые вернула модель, а не по её
арифметике, поэтому частоты и доли всегда сходятся с выборкой.

Вызов LLM идёт через существующий путь mlops.gateway (через _qwen из tools_llm) — свой
HTTP-клиент не создаётся.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from .context import compact
from .registry import ToolError, tool
from .tools_llm import _extract_json, _qwen

# ---------------------------------------------------------------- параметры разбора
BATCH_SIZE = 15          # сообщений в одной пачке (допустимо 15–25)
MIN_BATCH_SIZE = 15
MAX_BATCH_SIZE = 25
MAX_BATCHES = 6          # ограничение по времени запуска инструмента
MAX_MESSAGES = 120
DEFAULT_LIMIT = 60

MESSAGE_CHARS = 480      # обрезка текста сообщения в промпте (max_model_len vLLM = 8192)
BATCH_CHAR_BUDGET = 9000  # страховка по длине промпта пачки
BATCH_MAX_TOKENS = 1500
SUMMARY_MAX_TOKENS = 1100
PARALLEL_BATCHES = 3

MODEL_LABEL = "Qwen3-32B-FP8 (vLLM, локально, без оплаты)"

# Каркас категорий: темы модели приводятся к нему, чтобы отчёты были сопоставимы между датасетами.
CATEGORY_FRAMEWORK = [
    "доставка",
    "качество товара",
    "цена",
    "возврат/деньги",
    "поддержка",
    "упаковка",
    "сроки",
    "другое",
]

# Ключевые слова для приведения темы к каркасу, если модель не назвала категорию точно.
CATEGORY_KEYWORDS: List[Tuple[str, Tuple[str, ...]]] = [
    ("доставка", ("доставк", "курьер", "пункт выдачи", "пвз", "привез", "привоз", "достав", "почт", "курьерск")),
    ("качество товара", ("качеств", "брак", "поврежд", "сломан", "некачеств", "дефект", "порван", "разбит", "грязн")),
    ("цена", ("цен", "дорог", "переплат", "стоимост", "скидк", "наценк", "дешевл")),
    ("возврат/деньги", ("возврат", "вернул", "деньг", "денеж", "refund", "компенсац", "оплат", "списан", "не вернул", "возмещ")),
    ("поддержка", ("поддержк", "оператор", "служб", "консультант", "ответил", "чат", "горяч", "обращени")),
    ("упаковка", ("упаковк", "коробк", "пакет", "запечатан", "тара", "плёнк")),
    ("сроки", ("срок", "задержк", "опозда", "вовремя", "просроч", "долго ждал", "неделю", "месяц ждал")),
]

# Поля вовлечённости. В датасетах Brand Analytics они есть, но заполнены не всегда —
# поэтому ниже есть честный откат на рейтинг отзыва и объём текста.
ENGAGEMENT_WEIGHTS = {
    "commentsCount": 3.0,
    "repostsCount": 4.0,
    "likesCount": 2.0,
    "audienceCount": 1.0,
    "er": 1.0,
    "massMediaAudience": 1.0,
    "viewsCount": 0.02,
    "duplicateCount": 0.5,
}
# Числовые поля, по которым можно считать max-агрегацию в Elasticsearch (viewsCount — text).
ENGAGEMENT_AGG_FIELDS = ["commentsCount", "likesCount", "repostsCount", "audienceCount", "er", "massMediaAudience"]

TONE_LABELS = {-1: "негатив", 0: "нейтрал", 1: "позитив"}

SOURCE_FIELDS = [
    "text", "title", "timeCreate", "hub", "hubtype", "url", "toneMark", "city",
    "likesCount", "commentsCount", "repostsCount", "viewsCount", "audienceCount",
    "er", "massMediaAudience", "duplicateCount", "review_rating", "authorObject", "idExternal",
]

BATCH_SYSTEM = (
    "Ты аналитик отзывов и сообщений соцмедиа и СМИ. Ты читаешь реальные тексты сообщений "
    "и группируешь их по темам строго по фактам из текста, ничего не додумывая. "
    "Отвечай ТОЛЬКО JSON, без пояснений вокруг."
)

BATCH_INSTRUCTION = """Разбери сообщения{scope}{focus}.
Сгруппируй их в темы, которые реально видны в текстах.

Верни JSON строго такого вида:
{{
  "topics": [
    {{"name": "название темы, 3-6 слов",
      "category": "категория из списка ниже",
      "msg_ids": ["m1", "m4"],
      "essence": "суть темы одним предложением",
      "tone": "негатив | нейтрал | позитив | смешанно",
      "quotes": [{{"msg_id": "m1", "quote": "дословная выдержка из сообщения"}}]}}
  ],
  "highlights": [{{"msg_id": "m3", "why": "почему это сообщение важно"}}],
  "summary": "2-3 предложения: что происходит в этих сообщениях"
}}

Категории (ровно одна на тему): {categories}
Жёсткие правила:
- msg_ids — только идентификаторы из квадратных скобок; каждый id попадает ровно в одну тему;
- quote — дословная выдержка до 200 символов, без пересказа и без правок;
- в каждой теме 1-2 цитаты, у каждой цитаты указан msg_id;
- highlights — 2-4 самых значимых сообщения (сильные претензии, массовая проблема, показательный позитив);
- тем не меньше 2 и не больше 8.

Сообщения:
{messages}"""

SUMMARY_SYSTEM = (
    "Ты аналитик соцмедиа и СМИ. Пиши деловым русским языком, только текст, без вводных фраз "
    "вида «в данном отчёте». Опирайся строго на переданные темы, цитаты и числа."
)

SUMMARY_INSTRUCTION = """Ниже результат чтения текстов датасета «{dataset}» за период {period}.
Прочитано сообщений: {count}. Негативных в срезе: {negative}. Тональность среза: {tone}.

Темы (тема — сообщений — доля — тональность — суть):
{topics}

Цитаты из сообщений:
{quotes}

Ключевые сообщения (по вовлечённости):
{highlights}

Напиши аналитический текст 3-5 абзацами: 1) какие темы доминируют, сколько это сообщений и долей;
2) на что конкретно жалуются — с опорой на приведённые цитаты; 3) что хвалят; 4) какие темы
требуют действий в первую очередь и почему. Числа бери только из данных выше. Без JSON и без заголовков."""


# --------------------------------------------------------------------- утилиты

def _num(value: Any) -> Optional[float]:
    """Приводит значение из Elasticsearch к числу (viewsCount приходит строкой)."""
    if value in (None, "", [], {}):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        match = re.search(r"-?\d+(?:[.,]\d+)?", str(value))
        if not match:
            return None
        try:
            number = float(match.group(0).replace(",", "."))
        except ValueError:
            return None
    return number


def _flat(text: Any) -> str:
    return " ".join(str(text or "").split())


def _snippet(text: Any, limit: int = 220) -> str:
    """Дословная выдержка из текста сообщения: без пробельных «дыр», обрезка по границе слова."""
    clean = _flat(text)
    if len(clean) <= limit:
        return clean
    cut = clean[:limit]
    space = cut.rfind(" ")
    return (cut[:space] if space > limit * 0.6 else cut).rstrip(" ,;:.-") + "…"


def _norm_key(text: Any) -> str:
    """Ключ для склейки одинаковых тем между пачками (регистр, пробелы, пунктуация)."""
    return re.sub(r"[^0-9a-zа-яё]+", "", _flat(text).lower())


def _gateway_model_label() -> str:
    try:
        from mlops.lock import generate_cfg

        model = str((generate_cfg() or {}).get("model") or "").strip()
        if model:
            return f"{model} (vLLM, локально, без оплаты)"
    except Exception:
        pass
    return MODEL_LABEL


# ------------------------------------------------------------------ выборка сообщений

def _doc_from_hit(hit: Dict[str, Any]) -> Dict[str, Any]:
    """Нормализует документ Elasticsearch: текст, атрибуция и все метрики вовлечённости."""
    src = hit.get("_source") or {}
    author = src.get("authorObject") or {}
    raw_text = _flat(src.get("text") or src.get("title") or "")
    doc: Dict[str, Any] = {
        "es_id": hit.get("_id"),
        "text": raw_text,
        "title": _flat(src.get("title")),
        "time": src.get("timeCreate"),
        "hub": _flat(src.get("hub")),
        "hubtype": _flat(src.get("hubtype")),
        "url": _flat(src.get("url")),
        "city": _flat(src.get("city")),
        "author": _flat(author.get("fullname")),
        "author_type": _flat(author.get("author_type")),
        "tone_mark": src.get("toneMark"),
        "rating": _num(src.get("review_rating")),
        "chars": len(raw_text),
    }
    engagement = 0.0
    for field, weight in ENGAGEMENT_WEIGHTS.items():
        value = _num(src.get(field))
        doc[field] = value
        if value:
            engagement += value * weight
    doc["engagement"] = round(engagement, 2)
    return doc


def _importance(doc: Dict[str, Any], engagement_available: bool) -> Tuple[float, str]:
    """Важность сообщения.

    Основной сигнал — вовлечённость (комментарии, репосты, лайки, просмотры, аудитория).
    Если в срезе она не заполнена (типично для выгрузок отзывов Brand Analytics), используется
    рейтинг отзыва и объём текста: чем ниже оценка и длиннее текст, тем информативнее отзыв.
    """
    if engagement_available and doc.get("engagement"):
        return float(doc["engagement"]), "вовлечённость (комментарии/репосты/лайки/просмотры)"
    rating = doc.get("rating")
    if rating is not None:
        return round((5.0 - rating) * 20.0 + min(doc.get("chars") or 0, 3000) / 300.0, 2), "рейтинг отзыва + объём текста"
    return round((doc.get("chars") or 0) / 100.0, 2), "объём текста (вовлечённость в датасете не заполнена)"


def _fetch_slice(index_name: str, lo, hi, tone: str, limit: int) -> Dict[str, Any]:
    """Берёт сообщения среза из Elasticsearch и определяет, заполнена ли вовлечённость."""
    from .tools_data import _es, _exact_count, _iso, _query

    query = _query(None, lo, hi, tone)
    total = _exact_count(index_name, query)

    body = {
        "size": 0,
        "query": query,
        "aggs": {field: {"max": {"field": field}} for field in ENGAGEMENT_AGG_FIELDS},
    }
    engagement_available = False
    try:
        agg = (_es().search(index=index_name, body=body).get("aggregations") or {})
        engagement_available = any(float((agg.get(field) or {}).get("value") or 0) > 0 for field in ENGAGEMENT_AGG_FIELDS)
    except Exception:
        agg = {}

    # Отбор: при заполненной вовлечённости берём самое вовлекающее, иначе — самые свежие
    # сообщения среза (в отзывах Brand Analytics счётчики обычно нулевые).
    if engagement_available:
        sort_spec = [
            {"commentsCount": {"order": "desc"}},
            {"repostsCount": {"order": "desc"}},
            {"likesCount": {"order": "desc"}},
            {"timeCreate": {"order": "desc"}},
        ]
    else:
        sort_spec = [{"timeCreate": {"order": "desc"}}]
    search_body = {
        "size": int(limit),
        "_source": SOURCE_FIELDS,
        "query": query,
        "sort": sort_spec,
    }
    try:
        res = _es().search(index=index_name, body=search_body)
    except Exception as exc:
        raise ToolError(f"Ошибка выборки сообщений: {exc}") from exc

    hits = (res.get("hits") or {}).get("hits") or []
    docs: List[Dict[str, Any]] = []
    for hit in hits:
        doc = _doc_from_hit(hit)
        doc["date"] = _iso(doc.get("time"))
        doc["tone"] = TONE_LABELS.get(doc.get("tone_mark"), doc.get("tone_mark"))
        score, basis = _importance(doc, engagement_available)
        doc["importance"] = score
        doc["importance_basis"] = basis
        docs.append(doc)

    # m1 — самое важное сообщение среза: модель видит важное первым, ключевые сообщения
    # отчёта отбираются по этой же величине.
    docs.sort(key=lambda item: -float(item.get("importance") or 0))
    lines: List[str] = []
    docs_by_id: Dict[str, Dict[str, Any]] = {}
    for pos, doc in enumerate(docs, start=1):
        msg_id = f"m{pos}"
        doc["msg_id"] = msg_id
        docs_by_id[msg_id] = doc
        lines.append(_message_line(msg_id, doc))
    return {
        "docs": docs,
        "docs_by_id": docs_by_id,
        "lines": lines,
        "messages_in_slice": total,
        "engagement_available": engagement_available,
        "engagement_max": {field: (agg.get(field) or {}).get("value") for field in ENGAGEMENT_AGG_FIELDS},
    }


def _message_line(msg_id: str, doc: Dict[str, Any]) -> str:
    """Строка сообщения для промпта: id, площадка, дата, метрики и текст."""
    metrics = []
    if doc.get("rating") is not None:
        metrics.append(f"рейтинг {doc['rating']:g}")
    if doc.get("commentsCount"):
        metrics.append(f"комментариев {doc['commentsCount']:g}")
    if doc.get("likesCount"):
        metrics.append(f"лайков {doc['likesCount']:g}")
    if doc.get("repostsCount"):
        metrics.append(f"репостов {doc['repostsCount']:g}")
    tail = (", " + ", ".join(metrics)) if metrics else ""
    return "[%s] (%s, %s, %s%s) %s" % (
        msg_id, doc.get("hub") or "источник", doc.get("date") or "", doc.get("tone") or "—", tail,
        (doc.get("text") or "")[:MESSAGE_CHARS],
    )


def _chunks(docs: List[Dict[str, Any]], batch_size: int) -> List[List[Dict[str, Any]]]:
    """Режет сообщения на пачки: по количеству и по бюджету символов промпта."""
    batches: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    size = 0
    for doc in docs:
        length = min(len(doc.get("text") or ""), MESSAGE_CHARS) + 60
        if current and (len(current) >= batch_size or size + length > BATCH_CHAR_BUDGET):
            batches.append(current)
            current, size = [], 0
        current.append(doc)
        size += length
    if current:
        batches.append(current)
    return batches


# ------------------------------------------------------------- разбор пачек моделью

async def _run_batch(ctx, number: int, total: int, scope: str, focus: str, batch: List[str]) -> Dict[str, Any]:
    """Один вызов локальной модели по пачке сообщений. Две попытки, ответ — строго JSON."""
    prompt = BATCH_INSTRUCTION.format(
        scope=scope,
        focus=focus,
        categories=", ".join(CATEGORY_FRAMEWORK),
        messages="\n".join(batch),
    )
    tokens = 0
    last_error = ""
    for attempt in (1, 2):
        try:
            text, used = await _qwen(ctx, prompt, system=BATCH_SYSTEM, max_tokens=BATCH_MAX_TOKENS)
            tokens += used
            parsed = _extract_json(text)
            if parsed:
                await ctx.log(f"Пачка {number}/{total}: разобрано тем {len(parsed.get('topics') or [])}")
                return {"ok": True, "parsed": parsed, "tokens": tokens}
            last_error = "модель вернула ответ без JSON"
            prompt = prompt + "\n\nНапоминаю: ответ — только JSON, без текста вокруг."
        except Exception as exc:  # noqa: BLE001 — пачка не должна ломать весь разбор
            last_error = f"{type(exc).__name__}: {exc}"
    await ctx.log(f"Пачка {number}/{total} не разобрана: {last_error}", level="error")
    return {"ok": False, "error": last_error, "tokens": tokens}


# ------------------------------------------------------------------ склейка результата

def _match_category(raw: Any, topic_name: str, quotes: List[str]) -> str:
    """Приводит категорию темы к фиксированному каркасу."""
    candidate = _flat(raw).lower().strip()
    for known in CATEGORY_FRAMEWORK:
        if candidate == known:
            return known
    if candidate:
        for known in CATEGORY_FRAMEWORK:
            head = known.split("/")[0]
            if head and (head in candidate or candidate in known):
                return known
    haystack = (_flat(topic_name) + " " + " ".join(_flat(q) for q in quotes)).lower()
    best, best_score = "другое", 0
    for known, keywords in CATEGORY_KEYWORDS:
        score = sum(1 for word in keywords if word in haystack)
        if score > best_score:
            best, best_score = known, score
    return best


def _quote_entry(doc: Optional[Dict[str, Any]], raw_quote: str) -> Optional[Dict[str, Any]]:
    """Собирает цитату с атрибуцией. Цитата обязана быть дословной — иначе берём выдержку из текста."""
    if doc is None:
        return None
    text = _flat(doc.get("text"))
    quote = _flat(raw_quote)
    verified = False
    if quote and text:
        probe = _flat(quote).strip("«»\"' ").lower()
        if len(probe) > 12:
            flat_text = text.lower()
            if probe in flat_text:
                verified = True
            else:  # модель могла склеить несколько предложений — проверяем по началу цитаты
                head = probe[:60]
                verified = bool(head) and head in flat_text
    if not verified:
        quote = _snippet(text)  # гарантированно дословная выдержка из сообщения
    return {
        "text": quote,
        "verified": verified,
        "date": doc.get("date") or "",
        "hub": doc.get("hub") or "",
        "author": doc.get("author") or "",
        "url": doc.get("url") or "",
        "tone": doc.get("tone"),
        "rating": doc.get("rating"),
        "msg_id": doc.get("msg_id"),
    }


def _merge(parsed_batches: List[Dict[str, Any]], docs_by_id: Dict[str, Dict[str, Any]], total_read: int) -> Dict[str, Any]:
    """Склеивает темы из пачек. Частоты считаются по msg_id, а не по арифметике модели."""
    topics: Dict[str, Dict[str, Any]] = {}
    highlight_votes: Dict[str, str] = {}
    for parsed in parsed_batches:
        for item in parsed.get("topics") or []:
            if not isinstance(item, dict):
                continue
            name = _flat(item.get("name"))
            if not name:
                continue
            key = _norm_key(name) or name.lower()
            slot = topics.setdefault(
                key,
                {"name": name, "msg_ids": set(), "raw_quotes": {}, "essence": "", "category_raw": "", "tone_raw": ""},
            )
            if len(name) > len(slot["name"]) and slot["name"].lower() in name.lower():
                slot["name"] = name  # предпочитаем более развёрнутое название темы
            for msg_id in item.get("msg_ids") or []:
                msg_id = _flat(msg_id)
                if msg_id in docs_by_id:
                    slot["msg_ids"].add(msg_id)
            for quote in item.get("quotes") or []:
                if not isinstance(quote, dict):
                    continue
                msg_id = _flat(quote.get("msg_id"))
                if msg_id in docs_by_id and _flat(quote.get("quote")):
                    slot["raw_quotes"].setdefault(msg_id, _flat(quote.get("quote")))
            if not slot["essence"] and _flat(item.get("essence")):
                slot["essence"] = _flat(item.get("essence"))
            if not slot["category_raw"] and _flat(item.get("category")):
                slot["category_raw"] = _flat(item.get("category"))
            if not slot["tone_raw"] and _flat(item.get("tone")):
                slot["tone_raw"] = _flat(item.get("tone"))
        for item in parsed.get("highlights") or []:
            if isinstance(item, dict):
                msg_id = _flat(item.get("msg_id"))
                if msg_id in docs_by_id:
                    highlight_votes.setdefault(msg_id, _flat(item.get("why")))

    rows: List[Dict[str, Any]] = []
    for slot in topics.values():
        msg_ids = sorted(slot["msg_ids"], key=lambda mid: -float(docs_by_id[mid].get("importance") or 0))
        if not msg_ids:
            continue
        tones = [docs_by_id[mid].get("tone_mark") for mid in msg_ids if isinstance(docs_by_id[mid].get("tone_mark"), (int, float))]
        tone_avg = round(sum(tones) / len(tones), 2) if tones else None
        # Тема с негативом и позитивом одновременно — «смешанная», иначе средняя тональность темы
        if tones and min(tones) < 0 < max(tones):
            tone_label = "смешанная"
        elif tone_avg is not None:
            tone_label = TONE_LABELS.get(int(round(tone_avg)), "смешанная")
        else:
            tone_label = "—"
        # 2–3 цитаты на тему: сначала те, что назвала модель, затем — из самых важных сообщений темы
        picked: List[str] = [mid for mid in msg_ids if mid in slot["raw_quotes"]][:3]
        for msg_id in msg_ids:
            if len(picked) >= 3:
                break
            if msg_id not in picked:
                picked.append(msg_id)
        quotes = [q for q in (_quote_entry(docs_by_id[mid], slot["raw_quotes"].get(mid, "")) for mid in picked[:3]) if q]
        rows.append(
            {
                "topic": slot["name"],
                "count": len(msg_ids),
                "share": round(len(msg_ids) / total_read, 4) if total_read else 0.0,
                "share_pct": round(100.0 * len(msg_ids) / total_read, 1) if total_read else 0.0,
                "tone": tone_label,
                "tone_avg": tone_avg,
                "model_tone": slot["tone_raw"],
                "essence": slot["essence"],
                "category": _match_category(slot["category_raw"], slot["name"], [q["text"] for q in quotes]),
                "msg_ids": msg_ids,
                "quotes": quotes,
                "importance": round(sum(float(docs_by_id[mid].get("importance") or 0) for mid in msg_ids) / len(msg_ids), 2),
            }
        )
    rows.sort(key=lambda item: (-item["count"], -item["importance"]))

    categories: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        bucket = categories.setdefault(row["category"], {"category": row["category"], "count": 0, "topics": [], "share": 0.0})
        bucket["count"] += row["count"]
        bucket["topics"].append(row["topic"])
    for bucket in categories.values():
        bucket["share"] = round(bucket["count"] / total_read, 4) if total_read else 0.0
        bucket["share_pct"] = round(100.0 * bucket["count"] / total_read, 1) if total_read else 0.0
    ordered_categories = sorted(categories.values(), key=lambda item: -item["count"])

    highlights: List[Dict[str, Any]] = []
    for msg_id, why in highlight_votes.items():
        doc = docs_by_id[msg_id]
        highlights.append(_highlight_entry(doc, why, model_picked=True))
    listed = {item["msg_id"] for item in highlights}
    for doc in sorted(docs_by_id.values(), key=lambda item: -float(item.get("importance") or 0)):
        if len(highlights) >= 8:
            break
        if doc.get("msg_id") in listed:
            continue
        highlights.append(_highlight_entry(doc, "", model_picked=False))
        listed.add(doc.get("msg_id"))
    highlights.sort(key=lambda item: (-float(item.get("importance") or 0)))
    return {"topics": rows, "categories": ordered_categories, "highlights": highlights[:8]}


def _highlight_entry(doc: Dict[str, Any], why: str, *, model_picked: bool) -> Dict[str, Any]:
    return {
        "msg_id": doc.get("msg_id"),
        "text": _snippet(doc.get("text"), 320),
        "why": why,
        "model_picked": model_picked,
        "date": doc.get("date") or "",
        "hub": doc.get("hub") or "",
        "author": doc.get("author") or "",
        "url": doc.get("url") or "",
        "tone": doc.get("tone"),
        "rating": doc.get("rating"),
        "importance": doc.get("importance"),
        "importance_basis": doc.get("importance_basis"),
    }


def _fallback_summary(topics: List[Dict[str, Any]], total_read: int, negative: int, highlight: List[Dict[str, Any]]) -> str:
    """Текст раздела без модели: темы, доли и цитаты всё равно должны попасть в отчёт."""
    if not topics:
        return f"Прочитано {total_read} сообщений, устойчивых тем из текстов выделить не удалось."
    top = topics[0]
    lines = [
        f"Прочитано {total_read} сообщений среза, негативных среди них {negative}. "
        f"Самая частая тема — «{top['topic']}»: {top['count']} сообщений ({top['share_pct']}% выборки).",
        "Распределение по темам: " + "; ".join(f"«{row['topic']}» — {row['count']} ({row['share_pct']}%)" for row in topics[:6]) + ".",
    ]
    for row in topics[:3]:
        if row.get("quotes"):
            quote = row["quotes"][0]
            lines.append(f"Тема «{row['topic']}»: «{quote['text']}» ({quote['hub']}, {quote['date']}, {quote['author']}).")
    if highlight:
        lines.append(f"Самое вовлекающее сообщение среза ({highlight[0].get('importance_basis')}): «{highlight[0]['text']}».")
    return "\n\n".join(lines)


# --------------------------------------------------- компактные формы для ответа
# Списки отдаём обычными массивами объектов: результат читает и модель, и build_report,
# поэтому служебных элементов внутри списков быть не должно.

def _public_quote(quote: Dict[str, Any], limit: int = 300) -> Dict[str, Any]:
    return {
        "text": _snippet(quote.get("text"), limit),
        "date": quote.get("date") or "",
        "hub": quote.get("hub") or "",
        "author": quote.get("author") or "",
        "url": quote.get("url") or "",
        "tone": quote.get("tone"),
        "rating": quote.get("rating"),
        "verified": bool(quote.get("verified")),
    }


def _public_topics(topics: List[Dict[str, Any]], limit: int = 12) -> List[Dict[str, Any]]:
    rows = []
    for row in topics[:limit]:
        rows.append(
            {
                "topic": row["topic"],
                "count": row["count"],
                "share": row["share"],
                "share_pct": row["share_pct"],
                "tone": row["tone"],
                "category": row["category"],
                "essence": row["essence"],
                "importance": row["importance"],
                "msg_ids": row["msg_ids"][:20],
                "quotes": [_public_quote(q) for q in (row.get("quotes") or [])[:2]],
            }
        )
    return rows


def _public_categories(categories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "category": row["category"],
            "count": row["count"],
            "share": row["share"],
            "share_pct": row["share_pct"],
            "topics": row["topics"][:8],
        }
        for row in categories
    ]


def _public_highlights(highlights: List[Dict[str, Any]], limit: int = 8) -> List[Dict[str, Any]]:
    return [
        {
            "text": item["text"],
            "why": item.get("why") or "",
            "date": item.get("date") or "",
            "hub": item.get("hub") or "",
            "author": item.get("author") or "",
            "url": item.get("url") or "",
            "tone": item.get("tone"),
            "rating": item.get("rating"),
            "importance": item.get("importance"),
            "importance_basis": item.get("importance_basis") or "",
        }
        for item in highlights[:limit]
    ]


# ------------------------------------------------------------------------- инструмент

@tool(
    "analyze_texts",
    title="Читать тексты датасета (локальная модель)",
    description=(
        "Читает САМИ ТЕКСТЫ сообщений датасета за период локальной моделью Qwen и отдаёт темы с числом "
        "сообщений, долями, средней тональностью и цитатами (с датой, площадкой, автором и ссылкой), "
        "сопоставление тем с каркасом категорий (доставка, качество товара, цена, возврат/деньги, поддержка, "
        "упаковка, сроки, другое), ключевые сообщения по вовлечённости и аналитический текст. "
        "Вызывай этот инструмент ПЕРЕД build_report, чтобы выводы и пояснения к графикам опирались на темы "
        "и цитаты из текстов, а не только на статистику. Обязателен, когда в срезе есть негатив: "
        "причины жалоб нужно подтверждать цитатами."
    ),
    parameters={
        "type": "object",
        "properties": {
            "index": {"type": "string", "description": "тема: название датасета или её номер; по умолчанию — выбранный в интерфейсе"},
            "min_date": {"type": "string", "description": "начало периода: YYYY-MM-DD, ISO или unix-секунды"},
            "max_date": {"type": "string", "description": "конец периода: YYYY-MM-DD, ISO или unix-секунды"},
            "tone": {"type": "string", "enum": ["all", "negative", "positive", "neutral"], "description": "какие сообщения читать, по умолчанию все"},
            "limit": {"type": "integer", "description": "сколько сообщений прочитать (по умолчанию 60, максимум 120)"},
            "batch_size": {"type": "integer", "description": "сообщений в одной пачке для модели, 15–25 (по умолчанию 15)"},
            "focus": {"type": "string", "description": "на что смотреть в первую очередь (например «причины возвратов»)"},
        },
    },
    group="analytics",
    timeout=600.0,
)
async def analyze_texts(
    ctx,
    index: Optional[int] = None,
    min_date: Any = None,
    max_date: Any = None,
    tone: str = "all",
    limit: int = DEFAULT_LIMIT,
    batch_size: int = BATCH_SIZE,
    focus: Optional[str] = None,
):
    from .tools_data import _iso, _require_period, dates, guard

    started = time.time()
    idx, index_name = guard(ctx, index)
    lo, hi = dates(ctx, min_date, max_date)
    lo, hi = _require_period(index_name, lo, hi)

    limit = max(1, min(int(limit or DEFAULT_LIMIT), MAX_MESSAGES))
    batch_size = max(MIN_BATCH_SIZE, min(int(batch_size or BATCH_SIZE), MAX_BATCH_SIZE))
    focus_text = f" Особое внимание: {_flat(focus)}." if _flat(focus) else ""
    scope = f" за период {_iso(lo)} — {_iso(hi)}" + (f" (тональность: {tone})" if str(tone).lower() not in ("all", "any", "") else "")

    found = _fetch_slice(index_name, lo, hi, tone, limit)
    docs, docs_by_id, lines = found["docs"], found["docs_by_id"], found["lines"]
    if not docs:
        return {
            "index": idx,
            "index_name": index_name,
            "messages_analyzed": 0,
            "messages_in_slice": int(found.get("messages_in_slice") or 0),
            "topics": [],
            "highlights": [],
            "categories": [],
            "summary": "",
            "stats": {"messages_processed": 0, "messages_in_slice": int(found.get("messages_in_slice") or 0), "batches": 0, "seconds": round(time.time() - started, 1)},
            "note": "За указанный период сообщений нет — читать нечего. Проверьте период и тональность.",
        }

    batches = _chunks(docs, batch_size)[:MAX_BATCHES]
    # Дальше работают только те сообщения, которые реально ушли в модель.
    batches = [batch for batch in batches if batch]
    read_ids = {doc["msg_id"] for batch in batches for doc in batch}
    docs_by_id = {mid: doc for mid, doc in found["docs_by_id"].items() if mid in read_ids}
    selected = len(read_ids)
    if selected < len(docs):
        await ctx.log(
            f"Читаю {selected} сообщений из {len(docs)} отобранных "
            f"(пачек не больше {MAX_BATCHES} по {batch_size} сообщений)"
        )
    if not docs_by_id:
        raise ToolError("Не удалось подготовить сообщения для чтения — повторите запуск")

    negative = sum(1 for doc in docs_by_id.values() if doc.get("tone_mark") == -1)
    positive = sum(1 for doc in docs_by_id.values() if doc.get("tone_mark") == 1)
    neutral = sum(1 for doc in docs_by_id.values() if doc.get("tone_mark") == 0)
    model_label = _gateway_model_label()
    await ctx.log(
        f"Чтение текстов: {selected} сообщений, {len(batches)} пачек, модель {model_label}. "
        f"Тональность среза: негатив {negative}, нейтрал {neutral}, позитив {positive}"
    )

    semaphore = asyncio.Semaphore(PARALLEL_BATCHES)
    results: List[Dict[str, Any]] = []
    tokens_used = 0

    async def worker(number: int, batch: List[Dict[str, Any]]) -> None:
        nonlocal tokens_used
        lines = [_message_line(doc["msg_id"], doc) for doc in batch]
        async with semaphore:
            outcome = await _run_batch(ctx, number, len(batches), scope, focus_text, lines)
        tokens_used += int(outcome.get("tokens") or 0)
        results.append(outcome)

    await asyncio.gather(*(worker(n, batch) for n, batch in enumerate(batches, start=1)))
    parsed_batches = [item["parsed"] for item in results if item.get("ok")]
    failed = len([item for item in results if not item.get("ok")])
    if not parsed_batches:
        raise ToolError(
            "Локальная модель не разобрала ни одну пачку сообщений — повторите запуск позже "
            "или уменьшите limit"
        )

    total_read = selected
    merged = _merge(parsed_batches, docs_by_id, total_read)
    topics, categories, highlights = merged["topics"], merged["categories"], merged["highlights"]

    tone_note = f"негатив {negative}, нейтрал {neutral}, позитив {positive}"
    digest_topics = "\n".join(
        f"- «{row['topic']}» — {row['count']} сообщений, {row['share_pct']}%, тональность: {row['tone']}"
        + (f", суть: {row['essence']}" if row.get("essence") else "")
        for row in topics[:10]
    ) or "— темы не выделены"
    digest_quotes = "\n".join(
        f"- «{quote['text']}» ({row['topic']}; {quote['hub']}, {quote['date']}, {quote['author']})"
        for row in topics[:8]
        for quote in (row.get("quotes") or [])[:2]
    ) or "— цитат нет"
    digest_highlights = "\n".join(
        f"- ({item['date']}, {item['hub']}, {item['tone']}) {item['text'][:200]}" for item in highlights[:5]
    ) or "— нет"

    summary = ""
    try:
        summary, used = await _qwen(
            ctx,
            SUMMARY_INSTRUCTION.format(
                dataset=index_name,
                period=f"{_iso(lo)} — {_iso(hi)}",
                count=total_read,
                negative=negative,
                tone=tone_note,
                topics=digest_topics,
                quotes=digest_quotes,
                highlights=digest_highlights,
            ),
            system=SUMMARY_SYSTEM,
            max_tokens=SUMMARY_MAX_TOKENS,
            temperature=0.25,
        )
        tokens_used += used
        if summary.lstrip().startswith("{"):
            parsed_summary = _extract_json(summary) or {}
            for key in ("summary", "анализ", "analysis", "text", "разбор"):
                value = parsed_summary.get(key)
                if isinstance(value, str) and value.strip():
                    summary = value.strip()
                    break
    except Exception as exc:  # noqa: BLE001 — без итогового текста раздел всё равно собирается
        await ctx.log(f"Итоговый текст по темам собрать не удалось: {exc}", level="error")
    if not summary.strip():
        summary = _fallback_summary(topics, total_read, negative, highlights)

    # ---------------------------------------------------------- раздел для отчёта
    findings = [
        {
            "topic": row["topic"],
            "count": row["count"],
            "share": row["share"],
            "share_pct": row["share_pct"],
            "tone": row["tone"],
            "category": row["category"],
            "essence": row["essence"],
            "importance": row["importance"],
            "quotes": row["quotes"],
            "msg_ids": row["msg_ids"][:20],
        }
        for row in topics
    ]
    heading = f"Темы и цитаты из текстов ({total_read} сообщений)"
    citations = [
        {"title": f"{quote['hub']} · {quote['date']} · {quote['author']}".strip(" ·"), "url": quote["url"]}
        for row in topics
        for quote in (row.get("quotes") or [])
        if quote.get("url")
    ][:10]
    section = {
        "heading": heading,
        "text": summary,
        "findings": findings,
        "citations": citations,
        "chart_ids": [],
    }
    # build_report подставит раздел сам, если модель про него забудет
    ctx.text_analysis = section
    ctx.negative_in_slice = int(negative)

    scope_note = (
        f"Прочитаны тексты {total_read} сообщений датасета «{index_name}» за период {_iso(lo)} — {_iso(hi)} "
        f"локальной моделью {model_label} (пачек: {len(batches)}). "
        f"Тональность среза: {tone_note}. Отбор важных сообщений: "
        f"{docs[0].get('importance_basis') if docs else '—'}."
    )
    if failed:
        scope_note += f" Не разобрано пачек: {failed}."

    seconds = round(time.time() - started, 1)
    return {
        "index": idx,
        "index_name": index_name,
        "period": {"from": _iso(lo), "to": _iso(hi)},
        "tone_filter": tone,
        "messages_in_slice": int(found.get("messages_in_slice") or 0),
        "messages_analyzed": total_read,
        "engagement_available": bool(found.get("engagement_available")),
        "engagement_max": compact(found.get("engagement_max") or {}, max_items=10, max_str=40),
        "tone_counts": {"negative": negative, "neutral": neutral, "positive": positive},
        # topics/categories/highlights отдаём как есть (без compact): compact добавляет в список
        # служебный элемент {"_total_items": ...} и ломает перебор, а он нужен и модели, и отчёту.
        "topics": _public_topics(topics),
        "categories": _public_categories(categories),
        "highlights": _public_highlights(highlights),
        "summary": summary,
        "scope_note": scope_note,
        "report_section": section,
        "stats": {
            "messages_processed": total_read,
            "messages_selected": selected,
            "messages_in_slice": int(found.get("messages_in_slice") or 0),
            "batches": len(batches),
            "batches_failed": failed,
            "batch_size": batch_size,
            "topics": len(topics),
            "highlights": len(highlights),
            "seconds": seconds,
            "model": model_label,
            "local_model_tokens": tokens_used,
            "cost_usd": 0.0,
            "importance_basis": docs[0].get("importance_basis") if docs else "",
            "engagement_available": bool(found.get("engagement_available")),
        },
        "note": (
            "Раздел report_section обязательно передай в build_report (findings с темами, долями и цитатами "
            "попадут в DOCX/PDF отдельным блоком). Пояснения к каждому графику строй на этих темах и цитатах, "
            "а не только на статистике. Цитаты уже сверены с текстами сообщений."
        ),
    }
