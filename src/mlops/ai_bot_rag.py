"""RAG helpers for /ai-bot: metadata-aware retrieval, citations, corpus briefing."""
from __future__ import annotations

import hashlib
import json
import re
import threading
import uuid
from datetime import datetime, timezone
from html import unescape
from typing import Any, Callable, Iterable

try:
    from fastapi.responses import JSONResponse
except ImportError:  # pragma: no cover - local tests without FastAPI
    class JSONResponse:
        def __init__(self, content=None, status_code=200, **_kwargs):
            self.status_code = status_code
            self.content = content or {}
            self.body = content

ES_SOURCE_FIELDS = [
    "text",
    "title",
    "hub",
    "hubtype",
    "url",
    "hash",
    "authorObject",
    "timeCreate",
    "audienceCount",
    "duplicateCount",
    "toneMark",
    "citeIndex",
    "viewsCount",
    "likesCount",
    "commentsCount",
    "repostsCount",
    "type",
    "city",
    "region",
]

TONE_FILTER = {"neg": -1, "neu": 0, "pos": 1}

CHANNEL_FILTER = {
    "smi": ["Онлайн-СМИ"],
    "social": ["Соцсети", "Мессенджеры каналы", "Микроблоги", "Блоги"],
    "reviews": ["Отзывы", "Форумы"],
}

TOPIC_LENS = {
    "Обзор": "Дай сжатый брифинг по медиаполю: сюжеты, тональность, площадки, что проверить дальше.",
    "PR и репутация": "Сфокусируйся на репутационных рисках, спикерах, тиражировании и формулировках для ответа PR.",
    "Маркетинг": "Сфокусируйся на продукте, обещаниях бренда, претензиях клиентов, конкурентах и каналах продвижения.",
    "СМИ": "Сфокусируйся на онлайн-СМИ: кто задаёт повестку, какие заголовки и сюжеты, есть ли перепечатки.",
    "Соцсети": "Сфокусируйся на соцсетях, мессенджерах и отзывах: авторы, охват, повторяющиеся претензии, признаки копипаста.",
}

PROMPT_CITATIONS = 24
CARD_CITATIONS = 12
DOC_SNIPPET = 350
DEEP_SAMPLE = 96
DEEP_BATCH = 8
DEEP_PER_TYPE = 12
MEMO_TTL = 6 * 3600
MEMO_PREFIX = "tellscope:ai-bot-memo:"

STANDARD_AGGS = {
    "tone": {"terms": {"field": "toneMark", "size": 10}},
    "hubtype": {
        "terms": {"field": "hubtype.keyword", "size": 16},
        "aggs": {"tone": {"terms": {"field": "toneMark", "size": 10}}},
    },
    "period": {"stats": {"field": "timeCreate"}},
    "audience": {"sum": {"field": "audienceCount"}},
    "hubs": {"terms": {"field": "hub.keyword", "size": 8}},
}

FOLLOW_UPS = {
    "Обзор": [
        "Разложи найденное по тональности и типам площадок",
        "Выдели 5 главных сюжетов с цитатами",
        "Где больше риска — в СМИ, соцсетях или отзывах?",
    ],
    "PR и репутация": [
        "Какие формулировки чаще звучат в негативе?",
        "Кто из площадок и авторов задаёт тон обсуждения?",
        "Какие сообщения стоит разобрать для ответа PR в первую очередь?",
    ],
    "Маркетинг": [
        "Какие продуктовые претензии повторяются чаще всего?",
        "Есть ли сравнения с конкурентами и какие аргументы?",
        "Какие каналы дают наибольший охват в этой выборке?",
    ],
    "СМИ": [
        "Какие онлайн-СМИ задают повестку и с какой тональностью?",
        "Какие материалы похожи на перепечатки или тиражирование?",
        "Кратко: информационные поводы периода",
    ],
    "Соцсети": [
        "Какие каналы и авторы наиболее заметны?",
        "Что уходит в тираж в мессенджерах?",
        "Какие претензии в отзывах повторяются?",
    ],
}


_TAG_RE = re.compile(r"<[^>]+>")


def _clean_text(value: Any) -> str:
    text = unescape(str(value or ""))
    text = _TAG_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def tone_label(value: Any) -> str:
    try:
        mark = int(value)
    except (TypeError, ValueError):
        return "не указана"
    if mark < 0:
        return "негатив"
    if mark > 0:
        return "позитив"
    return "нейтраль"


def _num(value: Any, default: float | int = 0):
    try:
        if value in (None, ""):
            return default
        return type(default)(float(value)) if isinstance(default, float) else int(float(value))
    except (TypeError, ValueError):
        return default


def _author(source: dict | None) -> str:
    obj = (source or {}).get("authorObject")
    if isinstance(obj, dict):
        return str(obj.get("fullname") or obj.get("name") or obj.get("nick") or "").strip()
    if isinstance(obj, str):
        return obj.strip()
    return ""


def _format_ts(value: Any) -> str:
    try:
        raw = float(value)
    except (TypeError, ValueError):
        return ""
    if raw <= 0:
        return ""
    if raw > 1e12:
        raw /= 1000.0
    try:
        return datetime.fromtimestamp(raw, tz=timezone.utc).strftime("%d.%m.%Y %H:%M")
    except (OSError, OverflowError, ValueError):
        return ""


def rrf_merge(rank_lists: Iterable[Iterable[str]], k: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    for ranks in rank_lists:
        for i, key in enumerate(ranks):
            if not key:
                continue
            scores[str(key)] = scores.get(str(key), 0.0) + 1.0 / (k + i + 1)
    return [item for item, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]


def es_filter_clauses(filters: dict | None) -> list[dict]:
    filters = filters or {}
    clauses: list[dict] = []
    tone = filters.get("tone")
    if tone in TONE_FILTER:
        clauses.append({"term": {"toneMark": TONE_FILTER[tone]}})
    hubtypes = [
        str(name).strip()
        for name in (filters.get("hubtypes") or [])
        if str(name).strip()
    ]
    if hubtypes:
        clauses.append({"terms": {"hubtype.keyword": hubtypes}})
    else:
        channel = filters.get("channel")
        if channel in CHANNEL_FILTER:
            clauses.append({"terms": {"hubtype.keyword": CHANNEL_FILTER[channel]}})
    return clauses


def resolve_collection(indexes: dict, db_name: str) -> str | None:
    if not db_name:
        return None
    for name in indexes.values():
        if name == db_name or db_name in str(name):
            return name
    return None


def payload_hash(payload: dict | None) -> str | None:
    payload = payload or {}
    meta = payload.get("metadata")
    if isinstance(meta, dict) and meta.get("hash"):
        return meta.get("hash")
    if payload.get("hash"):
        return payload.get("hash")
    return None


def source_from_es(hit_source: dict, db_name: str, score: float) -> dict:
    src = hit_source or {}
    text = _clean_text(src.get("text"))
    return {
        "text": text,
        "title": _clean_text(src.get("title")),
        "hash": src.get("hash") or "",
        "score": float(score or 0),
        "source": {
            "hub": src.get("hub") or "",
            "hubtype": src.get("hubtype") or "",
            "url": src.get("url") or "",
            "database": db_name,
            "author": _author(src),
            "timeCreate": src.get("timeCreate") or "",
            "audienceCount": _num(src.get("audienceCount"), 0),
            "duplicateCount": _num(src.get("duplicateCount"), 0),
            "toneMark": src.get("toneMark"),
            "viewsCount": _num(src.get("viewsCount"), 0),
            "likesCount": _num(src.get("likesCount"), 0),
            "commentsCount": _num(src.get("commentsCount"), 0),
            "city": src.get("city") or "",
            "region": src.get("region") or "",
        },
    }


def source_from_qdrant(payload: dict, db_name: str, score: float, point_id: Any) -> dict:
    payload = payload or {}
    meta = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    text = str(payload.get("content") or payload.get("text") or "")
    return {
        "text": text,
        "title": str(meta.get("title") or ""),
        "hash": str(meta.get("hash") or point_id),
        "score": float(score or 0),
        "source": {
            "hub": meta.get("hub") or "",
            "hubtype": meta.get("hubtype") or "",
            "url": meta.get("url") or "",
            "database": db_name,
            "author": meta.get("author") or "",
            "timeCreate": meta.get("timeCreate") or "",
            "audienceCount": _num(meta.get("audienceCount"), 0),
            "duplicateCount": _num(meta.get("duplicateCount"), 0),
            "toneMark": meta.get("toneMark"),
            "viewsCount": 0,
            "likesCount": 0,
            "commentsCount": 0,
            "city": "",
            "region": "",
        },
    }


def public_sources(items: list[dict], limit: int = 12) -> list[dict]:
    cards = []
    for i, item in enumerate(items[:limit], 1):
        src = item.get("source") or {}
        text = str(item.get("text") or "")
        snippet = text[:280] + ("…" if len(text) > 280 else "")
        cards.append(
            {
                "id": i,
                "title": item.get("title") or src.get("hub") or f"Документ {i}",
                "hub": src.get("hub") or "",
                "hubtype": src.get("hubtype") or "",
                "url": src.get("url") or "",
                "author": src.get("author") or "",
                "timeCreate": src.get("timeCreate"),
                "timeLabel": _format_ts(src.get("timeCreate")),
                "audienceCount": src.get("audienceCount") or 0,
                "duplicateCount": src.get("duplicateCount") or 0,
                "toneMark": src.get("toneMark"),
                "tone": tone_label(src.get("toneMark")),
                "score": round(float(item.get("score") or 0), 3),
                "snippet": snippet,
                "viewsCount": src.get("viewsCount") or 0,
                "likesCount": src.get("likesCount") or 0,
                "commentsCount": src.get("commentsCount") or 0,
            }
        )
    return cards


def docs_markdown(top_texts: list[dict], question: str) -> str:
    lines = [
        f'## Цитаты по запросу «{question}» (выборка, не весь корпус)',
        "",
        "Нумерация [n] должна совпадать со ссылками в ответе. Сюжеты и формулировки — только отсюда.",
        "",
    ]
    for i, item in enumerate(top_texts, 1):
        src = item.get("source") or {}
        text = str(item.get("text") or "")
        if len(text) > DOC_SNIPPET:
            text = text[:DOC_SNIPPET] + "…"
        meta = (
            f"{src.get('hub') or 'площадка не указана'} ({src.get('hubtype') or 'тип не указан'}) · "
            f"{_format_ts(src.get('timeCreate')) or 'без даты'} · {tone_label(src.get('toneMark'))} · "
            f"охват {src.get('audienceCount') or 0} · дубли {src.get('duplicateCount') or 0}"
        )
        lines.extend(
            [
                f"### [{i}] {item.get('title') or 'без заголовка'}",
                meta,
                f"Автор: {src.get('author') or 'не указан'}. URL: {src.get('url') or ''}",
                f"Текст: {text}",
                "",
            ]
        )
    return "\n".join(lines)


def history_block(history: list | None, limit: int = 6) -> str:
    if not history:
        return ""
    lines = ["**Предыдущие реплики (для уточнения, не как источник фактов):**"]
    for turn in history[-limit:]:
        role = "Пользователь" if (turn or {}).get("role") == "user" else "Ассистент"
        content = str((turn or {}).get("content") or "").strip()
        if content:
            lines.append(f"- {role}: {content[:400]}")
    return "\n".join(lines) + "\n"


def normalize_instructions(raw, limit: int = 12000) -> str:
    if isinstance(raw, list):
        parts = []
        for item in raw:
            if isinstance(item, dict):
                name = str(item.get("name") or "файл").strip() or "файл"
                text = str(item.get("text") or "").strip()
                if text:
                    parts.append(f"### {name}\n{text}")
            else:
                text = str(item or "").strip()
                if text:
                    parts.append(text)
        text = "\n\n".join(parts)
    else:
        text = str(raw or "").strip()
    if len(text) > limit:
        return text[:limit] + "\n…"
    return text


def user_prompt(
    question: str,
    topic: str,
    selected_databases: list[str],
    docs_md: str,
    history: list | None = None,
    filtered: bool = False,
    instructions: str = "",
    evidence_md: str = "",
    deep_memo: str = "",
) -> str:
    lens = TOPIC_LENS.get(topic) or "Ответь по задаче аналитика медиаполя."
    filter_note = (
        "К поиску применён фильтр по метаданным: цифры «фильтр» и «по запросу» уже это учитывают. "
        "Не обобщай отфильтрованные цифры на весь корпус без блока «Корпус темы»."
        if filtered
        else "Цифры объёма и тональности бери из блока Elasticsearch. Цитаты [n] не равны всей теме."
    )
    hist = history_block(history)
    instr = ""
    extra = normalize_instructions(instructions)
    if extra:
        instr = f"""
**Инструкции пользователя (как отвечать / на что смотреть). Это не факты корпуса:**
{extra}

Не используй инструкции как источник цифр, сюжетов и цитат.
"""
    memo = ""
    if str(deep_memo or "").strip():
        memo = f"""
**Мемо глубокого разбора Qwen (стратифицированная выборка текстов, не все сообщения):**
{str(deep_memo).strip()[:8000]}
"""
    stats = evidence_md or ""
    return f"""
**Вопрос:** {question}
**Линза анализа:** {topic}
**Задача линзы:** {lens}
**Темы/коллекции:** {', '.join(selected_databases)}
{instr}{hist}{stats}{memo}
{docs_md}

**Как отвечать:**
- Объём, тональность, типы площадок, охват, период — только из блока статистики Elasticsearch. Их можно цитировать как расчёт по всем сообщениям.
- Сюжеты, формулировки, авторы и URL — только из цитат [n]. Ссылайся как [1], [2].
- {filter_note}
- Не называй набор цитат всей темой и не считай доли по [n] вместо ES.
- Структура: краткий вывод; цифры по корпусу; факты с цитатами [n]; смысл для PR/маркетинга; ограничения.
- Если части вопроса нет в ES и в цитатах — скажи прямо.
"""


def follow_ups(topic: str) -> list[str]:
    return list(FOLLOW_UPS.get(topic) or FOLLOW_UPS["Обзор"])


def bm25_hashes(es, index: str, question: str, size: int = 40) -> list[str]:
    try:
        res = es.search(
            index=index,
            size=size,
            _source=["hash"],
            query={
                "multi_match": {
                    "query": question,
                    "fields": ["text^2", "title^3", "hub"],
                    "type": "best_fields",
                }
            },
        )
        hashes = []
        for hit in res.get("hits", {}).get("hits", []):
            value = (hit.get("_source") or {}).get("hash")
            if value:
                hashes.append(value)
        return hashes
    except Exception:
        return []


def fetch_es_docs(es, index: str, hashes: list[str], filters: dict | None) -> list[dict]:
    if not hashes:
        return []
    query: dict[str, Any] = {"terms": {"hash": hashes}}
    clauses = es_filter_clauses(filters)
    if clauses:
        query = {"bool": {"must": [{"terms": {"hash": hashes}}], "filter": clauses}}
    res = es.search(
        index=index,
        size=min(len(hashes), 80),
        _source=ES_SOURCE_FIELDS,
        query=query,
    )
    return [hit.get("_source") or {} for hit in res.get("hits", {}).get("hits", [])]


def _tone_map(buckets: list | None) -> dict:
    out = {"negative": 0, "neutral": 0, "positive": 0}
    for bucket in buckets or []:
        try:
            mark = int(bucket.get("key"))
        except (TypeError, ValueError):
            continue
        count = int(bucket.get("doc_count") or 0)
        if mark < 0:
            out["negative"] += count
        elif mark > 0:
            out["positive"] += count
        else:
            out["neutral"] += count
    return out


def _terms_agg(es, index: str, field: str, size: int = 8) -> list[dict]:
    try:
        res = es.search(
            index=index,
            size=0,
            track_total_hits=True,
            query={"match_all": {}},
            aggs={"t": {"terms": {"field": field, "size": size}}},
        )
        return res.get("aggregations", {}).get("t", {}).get("buckets") or []
    except Exception:
        return []


def _fmt(value: Any) -> str:
    try:
        return f"{int(value):,}".replace(",", "\u00a0")
    except (TypeError, ValueError):
        return "0"


def empty_stats() -> dict:
    return {
        "count": 0,
        "tone": {"negative": 0, "neutral": 0, "positive": 0},
        "hubtypes": [],
        "hubs": [],
        "period": {"from": None, "to": None, "fromLabel": "", "toLabel": ""},
        "audienceSum": 0,
    }


def resolve_collections(indexes: dict, selected_databases: list[str]) -> list[str]:
    collections = []
    seen = set()
    for db_name in selected_databases:
        name = resolve_collection(indexes, db_name)
        if name and name not in seen:
            collections.append(name)
            seen.add(name)
    return collections


def es_bool_query(filters: dict | None = None, question: str = "") -> dict:
    clauses = es_filter_clauses(filters)
    must = []
    query_text = str(question or "").strip()
    if query_text:
        must.append(
            {
                "multi_match": {
                    "query": query_text,
                    "fields": ["text^2", "title^3", "hub"],
                    "type": "best_fields",
                }
            }
        )
    if not must and not clauses:
        return {"match_all": {}}
    body: dict[str, Any] = {"bool": {}}
    if must:
        body["bool"]["must"] = must
    if clauses:
        body["bool"]["filter"] = clauses
    return body


def _hit_total(res: dict | None) -> int:
    total = (res or {}).get("hits", {}).get("total", {})
    if isinstance(total, dict):
        return int(total.get("value") or 0)
    return int(total or 0)


def parse_search_aggs(res: dict | None) -> dict:
    aggs = (res or {}).get("aggregations") or {}
    hubtypes = []
    for bucket in (aggs.get("hubtype") or {}).get("buckets") or []:
        name = bucket.get("key")
        if not name:
            continue
        hubtypes.append(
            {
                "name": name,
                "count": int(bucket.get("doc_count") or 0),
                "tone": _tone_map((bucket.get("tone") or {}).get("buckets")),
            }
        )
    hubs = []
    for bucket in (aggs.get("hubs") or {}).get("buckets") or []:
        if bucket.get("key"):
            hubs.append({"name": bucket.get("key"), "count": int(bucket.get("doc_count") or 0)})
    period = aggs.get("period") or {}
    audience = (aggs.get("audience") or {}).get("value") or 0
    return {
        "count": _hit_total(res),
        "tone": _tone_map((aggs.get("tone") or {}).get("buckets")),
        "hubtypes": hubtypes,
        "hubs": hubs,
        "period": {
            "from": period.get("min"),
            "to": period.get("max"),
            "fromLabel": _format_ts(period.get("min")),
            "toLabel": _format_ts(period.get("max")),
        },
        "audienceSum": int(audience or 0),
    }


def run_aggregations(es, index: str, query: dict) -> dict:
    try:
        res = es.search(
            index=index,
            size=0,
            track_total_hits=True,
            query=query,
            aggs=STANDARD_AGGS,
        )
        return parse_search_aggs(res)
    except Exception:
        return empty_stats()


def corpus_summary(es, indexes: dict, selected_databases: list[str], filters: dict | None = None) -> dict:
    collections = resolve_collections(indexes, selected_databases)
    if not collections:
        return {"error": "Не найдена коллекция для выбранных тем"}
    stats = run_aggregations(es, ",".join(collections), es_bool_query(filters))
    stats["collections"] = collections
    return stats


def evidence_pack(
    es,
    indexes: dict,
    selected_databases: list[str],
    question: str = "",
    filters: dict | None = None,
) -> dict:
    collections = resolve_collections(indexes, selected_databases)
    if not collections:
        empty = empty_stats()
        return {
            "error": "Не найдена коллекция для выбранных тем",
            "collections": [],
            "corpus": empty,
            "filtered": empty,
            "query_hits": empty,
            "applied_filters": False,
        }
    index = ",".join(collections)
    corpus = run_aggregations(es, index, {"match_all": {}})
    filtered = run_aggregations(es, index, es_bool_query(filters))
    query_hits = run_aggregations(es, index, es_bool_query(filters, question))
    return {
        "collections": collections,
        "corpus": corpus,
        "filtered": filtered,
        "query_hits": query_hits,
        "applied_filters": bool(es_filter_clauses(filters)),
    }


def _tone_line(tone: dict | None) -> str:
    tone = tone or {}
    return (
        f"нег. {_fmt(tone.get('negative'))} · "
        f"нейтр. {_fmt(tone.get('neutral'))} · "
        f"поз. {_fmt(tone.get('positive'))}"
    )


def _hubtype_line(hubtypes: list | None, limit: int = 10) -> str:
    parts = [f"{item.get('name')} {_fmt(item.get('count'))}" for item in (hubtypes or [])[:limit] if item.get("name")]
    return "; ".join(parts) if parts else "нет данных"


def _stats_block(title: str, stats: dict | None) -> str:
    stats = stats or empty_stats()
    period = stats.get("period") or {}
    window = " — ".join(part for part in [period.get("fromLabel"), period.get("toLabel")] if part) or "не указан"
    return "\n".join(
        [
            f"### {title}",
            f"Сообщений: {_fmt(stats.get('count'))}",
            f"Тональность: {_tone_line(stats.get('tone'))}",
            f"Типы площадок: {_hubtype_line(stats.get('hubtypes'))}",
            f"Суммарный охват audienceCount: {_fmt(stats.get('audienceSum'))}",
            f"Период: {window}",
        ]
    )


def evidence_markdown(pack: dict | None) -> str:
    pack = pack or {}
    corpus = pack.get("corpus") or empty_stats()
    filtered = pack.get("filtered") or empty_stats()
    hits = pack.get("query_hits") or empty_stats()
    lines = [
        "## Статистика Elasticsearch по всем сообщениям",
        "Цифры ниже — агрегаты по индексу темы, не по цитатам [n]. Их можно цитировать как расчёт по всему корпусу.",
        "",
        _stats_block("Корпус темы", corpus),
    ]
    if pack.get("applied_filters") and int(filtered.get("count") or 0) != int(corpus.get("count") or 0):
        lines.extend(["", _stats_block("После фильтра по типам/тональности", filtered)])
    lines.extend(
        [
            "",
            _stats_block("По запросу во всём индексе (все попадания ES, не top-N)", hits),
        ]
    )
    return "\n".join(lines)


def coverage_payload(pack: dict | None, citations: int, cards: int, pool: int) -> dict:
    pack = pack or {}
    corpus_n = int((pack.get("corpus") or {}).get("count") or 0)
    hits_n = int((pack.get("query_hits") or {}).get("count") or 0)
    label = (
        f"Корпус: {_fmt(corpus_n)} · по запросу в ES: {_fmt(hits_n)} · "
        f"цитаты для разбора: {cards}/{citations}"
    )
    note = (
        "Цифры тональности и типов площадок — по всем сообщениям Elasticsearch. "
        "Цитаты — ближайшие тексты для сюжетов, это не вся тема."
    )
    return {
        "corpus_count": corpus_n,
        "query_hits": hits_n,
        "citations": citations,
        "cards": cards,
        "pool": pool,
        "label": label,
        "note": note,
    }


def doc_key(item: dict | None) -> str:
    item = item or {}
    return str(
        item.get("hash")
        or (item.get("source") or {}).get("url")
        or str(item.get("text") or "")[:80]
    )


def dedupe_docs(items: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for item in items:
        key = doc_key(item)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _reach_rank(item: dict) -> float:
    src = item.get("source") or {}
    audience = _num(src.get("audienceCount"), 0)
    dups = _num(src.get("duplicateCount"), 0)
    score = float(item.get("score") or 0)
    return score * 2.0 + (max(audience, 0) ** 0.5) * 0.04 + dups * 0.08


def stratify_citations(items: list[dict], limit: int = PROMPT_CITATIONS, per_hubtype: int = 3) -> list[dict]:
    by_type: dict[str, list[dict]] = {}
    for item in items:
        hubtype = str((item.get("source") or {}).get("hubtype") or "прочее")
        by_type.setdefault(hubtype, []).append(item)
    picked: list[dict] = []
    seen = set()
    for group in by_type.values():
        for item in sorted(group, key=_reach_rank, reverse=True)[:per_hubtype]:
            key = doc_key(item)
            if key in seen:
                continue
            seen.add(key)
            picked.append(item)
    for item in sorted(items, key=lambda row: float(row.get("score") or 0), reverse=True):
        if len(picked) >= limit:
            break
        key = doc_key(item)
        if key in seen:
            continue
        seen.add(key)
        picked.append(item)
    picked.sort(key=lambda row: float(row.get("score") or 0), reverse=True)
    return picked[:limit]


def fetch_reach_docs(
    es,
    index: str,
    db_name: str,
    filters: dict | None,
    hubtypes: list[dict] | None,
    per_type: int = 3,
) -> list[dict]:
    names = [str(item.get("name") or "").strip() for item in (hubtypes or []) if item.get("name")]
    if filters and (filters.get("hubtypes") or []):
        names = [str(name).strip() for name in filters.get("hubtypes") if str(name).strip()]
    if not names:
        names = [""]
    docs: list[dict] = []
    for name in names[:12]:
        extra = dict(filters or {})
        if name:
            extra["hubtypes"] = [name]
        clauses = es_filter_clauses(extra)
        query: dict[str, Any] = {"bool": {"filter": clauses}} if clauses else {"match_all": {}}
        try:
            res = es.search(
                index=index,
                size=per_type,
                _source=ES_SOURCE_FIELDS,
                query=query,
                sort=[{"audienceCount": {"order": "desc", "unmapped_type": "long"}}],
            )
            for hit in res.get("hits", {}).get("hits", []):
                docs.append(source_from_es(hit.get("_source") or {}, db_name, 0.05))
        except Exception:
            continue
    return docs


def memo_key(collections: list[str], filters: dict | None) -> str:
    raw = json.dumps({"c": collections, "f": filters or {}}, ensure_ascii=False, sort_keys=True)
    return MEMO_PREFIX + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def store_memo(collections: list[str], filters: dict | None, memo: str) -> None:
    text = str(memo or "").strip()
    if not text:
        return
    try:
        from mlops.jobs import _r

        _r().setex(memo_key(collections, filters), MEMO_TTL, text)
    except Exception:
        return


def load_stored_memo(collections: list[str], filters: dict | None) -> str:
    try:
        from mlops.jobs import _r

        return str(_r().get(memo_key(collections, filters)) or "").strip()
    except Exception:
        return ""


def gpu_ready() -> tuple[bool, list]:
    try:
        from mlops.runtime import running_batches

        holders = running_batches()
        return (not holders, holders)
    except Exception:
        return True, []


def sample_deep_docs(es, index: str, db_name: str, filters: dict | None, hubtypes: list[dict] | None) -> list[dict]:
    types = hubtypes or []
    per_type = max(8, DEEP_PER_TYPE if types else DEEP_SAMPLE)
    if types:
        per_type = max(8, min(DEEP_PER_TYPE, max(8, DEEP_SAMPLE // max(len(types), 1))))
    return dedupe_docs(fetch_reach_docs(es, index, db_name, filters, types or [{"name": ""}], per_type))[:DEEP_SAMPLE]


def _vllm_chat(messages: list[dict], max_tokens: int = 400, timeout: float = 120) -> str:
    from mlops.gateway import chat

    result = chat(
        provider="vllm",
        messages=messages,
        temperature=0.1,
        max_tokens=max_tokens,
        timeout=timeout,
        extra={"chat_template_kwargs": {"enable_thinking": False}},
        profile="dashboard_qa_bot",
    )
    return str(result.content or "").strip()


def _map_batch_prompt(batch: list[dict]) -> str:
    lines = []
    for i, item in enumerate(batch, 1):
        src = item.get("source") or {}
        text = str(item.get("text") or "")[:400]
        lines.append(
            f"{i}. [{src.get('hubtype') or 'тип?'}] {src.get('hub') or ''} · "
            f"{tone_label(src.get('toneMark'))} · охват {src.get('audienceCount') or 0} · "
            f"дубли {src.get('duplicateCount') or 0}\n{text}"
        )
    return (
        "Собери 5–8 пунктов: повторяющиеся сюжеты, претензии, площадки, тональность, заметный охват. "
        "Не выдумывай цифры сверх указанных охват/дубли.\n\n" + "\n\n".join(lines)
    )


def run_deep_brief(es, indexes: dict, selected_databases: list[str], filters: dict | None, job_id: str, logger) -> str:
    from mlops.jobs import register

    pack = evidence_pack(es, indexes, selected_databases, "", filters)
    collections = pack.get("collections") or []
    if not collections:
        raise RuntimeError("Не найдена коллекция для выбранных тем")
    index = ",".join(collections)
    db_name = selected_databases[0]
    hubtypes = (pack.get("filtered") or pack.get("corpus") or {}).get("hubtypes") or []
    register(job_id, product="ai-bot-deep", status="running", message="Собираем выборку по типам площадок…")
    docs = sample_deep_docs(es, index, db_name, filters, hubtypes)
    if len(docs) < 8:
        extra = fetch_reach_docs(es, index, db_name, filters, [{"name": ""}], 40)
        docs = dedupe_docs(docs + extra)[:DEEP_SAMPLE]
    if not docs:
        raise RuntimeError("Не удалось взять тексты для глубокого разбора")

    summaries = []
    batches = [docs[i : i + DEEP_BATCH] for i in range(0, len(docs), DEEP_BATCH)]
    for i, batch in enumerate(batches, 1):
        register(
            job_id,
            product="ai-bot-deep",
            status="running",
            message=f"Qwen читает пачку {i}/{len(batches)} ({len(docs)} текстов, не весь корпус)…",
        )
        summary = _vllm_chat(
            [
                {
                    "role": "system",
                    "content": "/no_think\nТы аналитик русскоязычных соцмедиа. Отвечай готовым текстом, без рассуждений.",
                },
                {"role": "user", "content": "/no_think\n" + _map_batch_prompt(batch)},
            ],
            max_tokens=420,
        )
        if summary:
            summaries.append(f"Пачка {i}: {summary}")
    if not summaries:
        raise RuntimeError("Qwen не вернул разбор пачек")
    register(job_id, product="ai-bot-deep", status="running", message="Собираем мемо темы…")
    memo = _vllm_chat(
        [
            {
                "role": "system",
                "content": "/no_think\nТы аналитик медиаполя. Цифры только из блока Elasticsearch.",
            },
            {
                "role": "user",
                "content": (
                    "/no_think\n"
                    f"{evidence_markdown(pack)}\n\n"
                    "Ниже — разборы стратифицированных пачек (не весь корпус).\n\n"
                    + "\n\n".join(summaries)
                    + "\n\nСобери мемо: о чём пишут; тональность и типы по ES; главные сюжеты; риски PR; чего нельзя утверждать. До 900 слов."
                ),
            },
        ],
        max_tokens=1400,
        timeout=180,
    )
    if not memo:
        raise RuntimeError("Qwen не собрал мемо")
    store_memo(collections, filters, memo)
    register(
        job_id,
        product="ai-bot-deep",
        status="done",
        message=f"Готово: {len(docs)} текстов, корпус {_fmt((pack.get('corpus') or {}).get('count'))}",
        memo=memo,
        sampled=len(docs),
        corpus_count=(pack.get("corpus") or {}).get("count") or 0,
    )
    if logger:
        logger.info("deep brief done job=%s docs=%s", job_id, len(docs))
    return memo


def _vector_hashes(search_result) -> list[str]:
    hashes = []
    for point in search_result or []:
        value = payload_hash(getattr(point, "payload", None) or {})
        if value:
            hashes.append(value)
    return hashes


def analyze_question(
    body: dict,
    *,
    es,
    qdrant_client,
    model_manager,
    models,
    client,
    ai_model: str,
    logger,
    load_indexes: Callable[[], dict],
    system_prompt: str,
) -> JSONResponse:
    question = str(body.get("question") or "").strip()
    topic = str(body.get("topic") or "Обзор").strip() or "Обзор"
    selected_databases = [name for name in (body.get("selected_databases") or []) if name]
    filters = body.get("filters") if isinstance(body.get("filters"), dict) else {}
    history = body.get("history") if isinstance(body.get("history"), list) else []
    instructions = normalize_instructions(body.get("instructions"))
    deep_memo = str(body.get("deep_memo") or "").strip()

    if not question:
        return JSONResponse(status_code=400, content={"error": "Вопрос не может быть пустым"})
    if not selected_databases:
        return JSONResponse(
            status_code=400,
            content={"error": "Необходимо выбрать хотя бы одну тему для анализа"},
        )

    try:
        indexes = load_indexes()
    except Exception as exc:
        logger.error("indexes load failed: %s", exc)
        return JSONResponse(status_code=500, content={"error": "Ошибка при загрузке индексов"})

    first_collection = None
    for db_name in selected_databases:
        first_collection = resolve_collection(indexes, db_name)
        if first_collection:
            break
    if not first_collection:
        return JSONResponse(status_code=404, content={"error": "Не найдена ни одна коллекция для выбранных баз данных"})

    pack = {"corpus": empty_stats(), "query_hits": empty_stats(), "filtered": empty_stats(), "collections": []}
    try:
        pack = evidence_pack(es, indexes, selected_databases, question, filters)
    except Exception as exc:
        logger.warning("evidence pack failed: %s", exc)
    if not deep_memo:
        deep_memo = load_stored_memo(pack.get("collections") or [], filters)

    collection_normalized = True
    avg_collection_norm = 1.0
    try:
        import numpy as np

        sample_points = qdrant_client.scroll(collection_name=first_collection, limit=10, with_vectors=True)[0]
        if sample_points:
            norms = [np.linalg.norm(p.vector) for p in sample_points if getattr(p, "vector", None) is not None]
            if norms:
                avg_collection_norm = float(np.mean(norms))
                collection_normalized = abs(avg_collection_norm - 1.0) < 0.05
    except Exception as exc:
        logger.warning("collection norm check failed: %s", exc)

    try:
        import numpy as np

        query_embedding = model_manager.encode_texts(
            [question],
            batch_size=1,
            normalize_embeddings=collection_normalized,
        )
        if hasattr(query_embedding, "ndim") and query_embedding.ndim == 2:
            query_embedding = query_embedding[0]
        query_embedding = query_embedding.astype("float32")
        query_norm = float(np.linalg.norm(query_embedding))
        if collection_normalized and abs(query_norm - 1.0) > 0.01 and query_norm:
            query_embedding = query_embedding / query_norm
        query_vector = query_embedding.tolist()
    except Exception as exc:
        logger.error("embedding failed: %s", exc, exc_info=True)
        return JSONResponse(status_code=500, content={"error": "Ошибка при создании эмбеддинга", "details": str(exc)})

    all_relevant_texts: list[dict] = []
    search_results_summary: list[dict] = []
    search_params = models.SearchParams(hnsw_ef=128, exact=False)

    for db_name in selected_databases:
        collection_name = resolve_collection(indexes, db_name)
        if not collection_name:
            search_results_summary.append({"database": db_name, "error": "Коллекция не найдена"})
            continue
        try:
            info = qdrant_client.get_collection(collection_name=collection_name)
            if getattr(info, "points_count", 0) == 0:
                search_results_summary.append({"database": db_name, "status": "empty_collection", "error": "Коллекция не содержит векторов"})
                continue
            search_result = qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=50,
                with_payload=True,
                score_threshold=0.3,
                search_params=search_params,
                with_vectors=False,
            )
            if len(search_result) < 5:
                search_result = qdrant_client.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=50,
                    with_payload=True,
                    score_threshold=0.1,
                    search_params=search_params,
                    with_vectors=False,
                )
            if not search_result:
                search_result = qdrant_client.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=10,
                    with_payload=True,
                    with_vectors=False,
                )

            vector_hashes = _vector_hashes(search_result)
            lexical_hashes = bm25_hashes(es, collection_name, question)
            merged_hashes = rrf_merge([vector_hashes, lexical_hashes])[:50] or vector_hashes
            hash_to_score = {}
            for point in search_result or []:
                value = payload_hash(getattr(point, "payload", None) or {})
                if value:
                    hash_to_score[value] = float(getattr(point, "score", 0) or 0)

            docs = []
            used_filter = bool(es_filter_clauses(filters))
            try:
                docs = fetch_es_docs(es, collection_name, merged_hashes, filters)
                if used_filter and len(docs) < 3:
                    docs = fetch_es_docs(es, collection_name, merged_hashes, {})
                    used_filter = False
            except Exception as elastic_error:
                logger.error("elasticsearch fetch failed for %s: %s", collection_name, elastic_error)

            texts = []
            if docs:
                for src in docs:
                    h = src.get("hash") or ""
                    texts.append(source_from_es(src, db_name, hash_to_score.get(h, 0.0)))
            else:
                for point in search_result or []:
                    texts.append(
                        source_from_qdrant(
                            getattr(point, "payload", None) or {},
                            db_name,
                            float(getattr(point, "score", 0) or 0),
                            getattr(point, "id", ""),
                        )
                    )
            try:
                hubtypes = (pack.get("filtered") or pack.get("corpus") or {}).get("hubtypes") or []
                texts.extend(fetch_reach_docs(es, collection_name, db_name, filters, hubtypes, per_type=3))
            except Exception as reach_error:
                logger.warning("reach docs failed for %s: %s", collection_name, reach_error)
            all_relevant_texts.extend(texts)
            search_results_summary.append(
                {
                    "database": db_name,
                    "found_documents": len(texts),
                    "collection_name": collection_name,
                    "source": "hybrid" if lexical_hashes else "qdrant",
                    "filtered": bool(es_filter_clauses(filters)) and used_filter,
                }
            )
        except Exception as db_error:
            logger.error("search failed for %s: %s", db_name, db_error, exc_info=True)
            search_results_summary.append({"database": db_name, "error": str(db_error)})

    unique = dedupe_docs(sorted(all_relevant_texts, key=lambda item: item.get("score") or 0, reverse=True))
    top_texts = stratify_citations(unique, limit=PROMPT_CITATIONS, per_hubtype=3)
    coverage = coverage_payload(pack, len(top_texts), min(CARD_CITATIONS, len(top_texts)), len(unique))
    corpus_count = int((pack.get("corpus") or {}).get("count") or 0)
    filtered = bool(pack.get("applied_filters")) or any(item.get("filtered") for item in search_results_summary)

    if not top_texts and not corpus_count:
        return JSONResponse(
            content={
                "answer": "По запросу не нашлось достаточно близких фрагментов в выбранной теме. Уточните формулировку или снимите фильтры по тональности и типу площадки.",
                "sources": [],
                "confidence": None,
                "coverage": coverage,
                "status": "no_results",
                "topic": topic,
                "search_summary": search_results_summary,
                "follow_ups": follow_ups(topic),
                "grounding": coverage["note"],
            }
        )

    docs_md = docs_markdown(top_texts, question) if top_texts else "Цитат по запросу нет — отвечай только цифрами Elasticsearch, без выдуманных сюжетов."
    prompt = user_prompt(
        question,
        topic,
        selected_databases,
        docs_md,
        history,
        filtered=filtered,
        instructions=instructions,
        evidence_md=evidence_markdown(pack),
        deep_memo=deep_memo,
    )

    try:
        chat_result = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            model=ai_model,
            max_tokens=2200,
            temperature=0.3,
        )
        answer = chat_result.choices[0].message.content
        avg_score = (
            sum(item.get("score") or 0 for item in top_texts) / len(top_texts) if top_texts else 0.0
        )
        return JSONResponse(
            content={
                "answer": answer,
                "sources": public_sources(top_texts, CARD_CITATIONS),
                "confidence": None,
                "coverage": coverage,
                "status": "success",
                "topic": topic,
                "search_summary": search_results_summary,
                "documents_analyzed": len(top_texts),
                "total_documents_found": len(unique),
                "average_relevance": round(float(avg_score), 3),
                "follow_ups": follow_ups(topic),
                "grounding": coverage["note"],
                "filters": filters,
                "deep_memo_used": bool(deep_memo),
            }
        )
    except Exception as llm_error:
        logger.error("LLM failed: %s", llm_error, exc_info=True)
        return JSONResponse(status_code=500, content={"error": "Ошибка при генерации ответа", "details": str(llm_error)})


async def handle_question_analysis(request, **deps) -> JSONResponse:
    try:
        body = await request.json()
    except Exception as exc:
        return JSONResponse(status_code=400, content={"error": "Неверный формат данных", "details": str(exc)})
    return analyze_question(body, **deps)


async def handle_corpus_summary(request, *, es, load_indexes, logger) -> JSONResponse:
    try:
        body = await request.json()
    except Exception as exc:
        return JSONResponse(status_code=400, content={"error": "Неверный формат данных", "details": str(exc)})
    selected = [name for name in (body.get("selected_databases") or []) if name]
    if not selected:
        return JSONResponse(status_code=400, content={"error": "Необходимо выбрать тему"})
    try:
        indexes = load_indexes()
        return JSONResponse(content=corpus_summary(es, indexes, selected))
    except Exception as exc:
        logger.error("corpus summary failed: %s", exc, exc_info=True)
        return JSONResponse(status_code=500, content={"error": "Не удалось собрать сводку по теме"})


async def handle_deep_brief(request, *, es, load_indexes, logger) -> JSONResponse:
    try:
        body = await request.json()
    except Exception as exc:
        return JSONResponse(status_code=400, content={"error": "Неверный формат данных", "details": str(exc)})
    selected = [name for name in (body.get("selected_databases") or []) if name]
    if not selected:
        return JSONResponse(status_code=400, content={"error": "Необходимо выбрать тему"})
    filters = body.get("filters") if isinstance(body.get("filters"), dict) else {}
    ready, holders = gpu_ready()
    if not ready:
        first = holders[0] if holders else {}
        return JSONResponse(
            status_code=503,
            content={
                "error": "GPU занят другой задачей. Запустите глубокий разбор, когда очередь пуста.",
                "jobs": holders,
                "busy": first.get("product") or "",
            },
        )
    try:
        indexes = load_indexes()
    except Exception as exc:
        logger.error("indexes load failed: %s", exc)
        return JSONResponse(status_code=500, content={"error": "Ошибка при загрузке индексов"})

    job_id = uuid.uuid4().hex[:16]
    try:
        from mlops.jobs import register

        register(
            job_id,
            product="ai-bot-deep",
            status="running",
            message="Готовим стратифицированную выборку…",
        )
    except Exception:
        pass

    def worker():
        try:
            run_deep_brief(es, indexes, selected, filters, job_id, logger)
        except Exception as exc:
            logger.error("deep brief failed job=%s: %s", job_id, exc, exc_info=True)
            try:
                from mlops.jobs import register as mark

                mark(job_id, product="ai-bot-deep", status="error", message=str(exc)[:400])
            except Exception:
                pass

    threading.Thread(target=worker, daemon=True).start()
    return JSONResponse(
        content={
            "job_id": job_id,
            "status": "running",
            "eta_min": "3–8",
            "warning": "Qwen прочитает около 80–120 текстов по типам площадок, не все сообщения темы. Цифры по-прежнему из Elasticsearch.",
        }
    )


async def handle_deep_brief_status(request) -> JSONResponse:
    job_id = ""
    try:
        job_id = str(request.query_params.get("job_id") or "").strip()
    except Exception:
        job_id = ""
    if not job_id:
        return JSONResponse(status_code=400, content={"error": "Нужен job_id"})
    try:
        from mlops.jobs import get

        job = get(job_id) or {}
    except Exception:
        job = {}
    if not job:
        return JSONResponse(status_code=404, content={"error": "Задача не найдена", "job_id": job_id})
    status = str(job.get("status") or "running")
    return JSONResponse(
        content={
            "job_id": job_id,
            "status": status,
            "message": job.get("message") or "",
            "memo": (job.get("memo") or "") if status == "done" else "",
            "sampled": job.get("sampled"),
            "corpus_count": job.get("corpus_count"),
            "error": job.get("message") if status == "error" else "",
        }
    )
