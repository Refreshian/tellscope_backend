"""RAG helpers for /ai-bot: metadata-aware retrieval, citations, corpus briefing."""
from __future__ import annotations

import re
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
        f'## Найденные материалы по запросу «{question}»',
        "",
        "Нумерация [n] должна совпадать со ссылками в ответе.",
        "",
    ]
    for i, item in enumerate(top_texts, 1):
        src = item.get("source") or {}
        text = str(item.get("text") or "")
        if len(text) > 700:
            text = text[:700] + "…"
        lines.extend(
            [
                f"### [{i}] {item.get('title') or 'без заголовка'}",
                f"- Площадка: {src.get('hub') or 'не указана'} ({src.get('hubtype') or 'тип не указан'})",
                f"- Дата: {_format_ts(src.get('timeCreate')) or 'не указана'}",
                f"- Тональность: {tone_label(src.get('toneMark'))}",
                f"- Охват аудитории: {src.get('audienceCount') or 0}",
                f"- Дубликаты: {src.get('duplicateCount') or 0}",
                f"- Автор: {src.get('author') or 'не указан'}",
                f"- Релевантность: {float(item.get('score') or 0):.3f}",
                f"- URL: {src.get('url') or ''}",
                f"- Текст: {text}",
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
) -> str:
    lens = TOPIC_LENS.get(topic) or "Ответь по задаче аналитика медиаполя."
    filter_note = (
        "К документам применён фильтр по метаданным; не обобщай на весь корпус."
        if filtered
        else "Это RAG-выборка, а не полная статистика темы. Не выдавай доли и суммы за расчёт по всему массиву, если не сказано иное."
    )
    hist = history_block(history)
    instr = ""
    extra = normalize_instructions(instructions)
    if extra:
        instr = f"""
**Инструкции пользователя (как отвечать / на что смотреть). Это не факты корпуса:**
{extra}

Не используй инструкции как источник цифр, сюжетов и цитат. Факты бери только из документов ниже.
"""
    return f"""
**Вопрос:** {question}
**Линза анализа:** {topic}
**Задача линзы:** {lens}
**Темы/коллекции:** {', '.join(selected_databases)}
{instr}{hist}
{docs_md}

**Как отвечать:**
- Используй только факты из документов выше. Не выдумывай цифры, охват, даты и тональность.
- Ссылайся на документы как [1], [2] по их номерам.
- {filter_note}
- Структура: краткий вывод; факты с цитатами и ссылками [n]; что это значит для PR/маркетинга/аналитики; что нельзя утверждать по этой выборке.
- Если часть вопроса не покрыта документами — скажи прямо.
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


def corpus_summary(es, indexes: dict, selected_databases: list[str]) -> dict:
    collections = []
    seen = set()
    for db_name in selected_databases:
        name = resolve_collection(indexes, db_name)
        if name and name not in seen:
            collections.append(name)
            seen.add(name)
    if not collections:
        return {"error": "Не найдена коллекция для выбранных тем"}

    index = ",".join(collections)
    res = es.search(
        index=index,
        size=0,
        track_total_hits=True,
        query={"match_all": {}},
        aggs={
            "tone": {"terms": {"field": "toneMark", "size": 10}},
            "hubtype": {"terms": {"field": "hubtype.keyword", "size": 12}},
            "period": {"stats": {"field": "timeCreate"}},
            "audience": {"sum": {"field": "audienceCount"}},
        },
    )
    total = res.get("hits", {}).get("total", {})
    count = int(total.get("value") or 0) if isinstance(total, dict) else int(total or 0)
    tone_buckets = {
        int(b["key"]): int(b["doc_count"])
        for b in (res.get("aggregations", {}).get("tone", {}).get("buckets") or [])
        if str(b.get("key")) not in ("", "None")
    }
    hubtypes = [
        {"name": b.get("key"), "count": int(b.get("doc_count") or 0)}
        for b in (res.get("aggregations", {}).get("hubtype", {}).get("buckets") or [])
        if b.get("key")
    ]
    hubs = _terms_agg(es, index, "hub.keyword", 8) or _terms_agg(es, index, "hub", 8)
    period = res.get("aggregations", {}).get("period") or {}
    audience = (res.get("aggregations", {}).get("audience") or {}).get("value") or 0
    return {
        "collections": collections,
        "count": count,
        "tone": {
            "negative": tone_buckets.get(-1, 0),
            "neutral": tone_buckets.get(0, 0),
            "positive": tone_buckets.get(1, 0),
        },
        "hubtypes": hubtypes,
        "hubs": [{"name": b.get("key"), "count": int(b.get("doc_count") or 0)} for b in hubs if b.get("key")],
        "period": {
            "from": period.get("min"),
            "to": period.get("max"),
            "fromLabel": _format_ts(period.get("min")),
            "toLabel": _format_ts(period.get("max")),
        },
        "audienceSum": int(audience or 0),
    }


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

    all_relevant_texts.sort(key=lambda item: item.get("score") or 0, reverse=True)
    # de-dupe by hash/url
    seen_keys = set()
    unique = []
    for item in all_relevant_texts:
        key = item.get("hash") or (item.get("source") or {}).get("url") or item.get("text", "")[:80]
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique.append(item)
    top_texts = unique[:12]

    if not top_texts:
        return JSONResponse(
            content={
                "answer": "По запросу не нашлось достаточно близких фрагментов в выбранной теме. Уточните формулировку или снимите фильтры по тональности и типу площадки.",
                "sources": [],
                "confidence": 0.0,
                "status": "no_results",
                "topic": topic,
                "search_summary": search_results_summary,
                "follow_ups": follow_ups(topic),
                "grounding": "Пустая RAG-выборка: модель не опиралась на документы темы.",
            }
        )

    docs_md = docs_markdown(top_texts, question)
    filtered = any(item.get("filtered") for item in search_results_summary)
    prompt = user_prompt(
        question,
        topic,
        selected_databases,
        docs_md,
        history,
        filtered=filtered,
        instructions=instructions,
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
        avg_score = sum(item.get("score") or 0 for item in top_texts) / len(top_texts)
        confidence = min(0.95, avg_score * 0.8 + (len(top_texts) / 50) * 0.2)
        return JSONResponse(
            content={
                "answer": answer,
                "sources": public_sources(top_texts),
                "confidence": round(float(confidence), 2),
                "status": "success",
                "topic": topic,
                "search_summary": search_results_summary,
                "documents_analyzed": len(top_texts),
                "total_documents_found": len(unique),
                "average_relevance": round(float(avg_score), 3),
                "follow_ups": follow_ups(topic),
                "grounding": f"Ответ опирается на {len(top_texts)} фрагментов RAG, это не расчёт по всему массиву темы.",
                "filters": filters,
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
