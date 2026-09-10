# -*- coding: utf-8 -*-
"""Инструменты анализа текстов локальной моделью Qwen3-32B (наши GPU, без оплаты внешних API).

Как устроен подробный анализ темы (map-reduce):
  1. берём сообщения по теме из Elasticsearch (сортировка по вовлечённости);
  2. режем на пачки и просим модель разобрать каждую: смысловые блоки, конкретные претензии
     с цитатами, детали (продукты, места, суммы), тональность;
  3. склеиваем результаты пачек и просим модель написать итоговый разбор;
  4. ссылки подставляем сами по идентификаторам сообщений — модель их не выдумывает.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from .context import compact
from .registry import ToolError, tool

BATCH_SIZE = 16
MAX_BATCHES = 10
MESSAGE_CHARS = 500
BATCH_MAX_TOKENS = 700
REDUCE_MAX_TOKENS = 1200
PARALLEL_BATCHES = 4

THINK_OFF = {"chat_template_kwargs": {"enable_thinking": False}}

MAP_SYSTEM = (
    "Ты аналитик соцмедиа и СМИ. Тебе дают сообщения по одной теме. Разбери их строго по фактам "
    "из текста, ничего не додумывай. Отвечай ТОЛЬКО JSON без пояснений."
)

MAP_INSTRUCTION = """Разбери сообщения по теме «{topic}».{focus}
Верни JSON строго такого вида:
{{
  "subtopics": [{{"name": "короткое название смыслового блока", "mentions": 3}}],
  "claims": [{{"claim": "суть претензии или наблюдения", "quote": "дословная цитата из сообщения", "msg_id": "m12", "tone": "негатив"}}],
  "details": {{"products": ["что упоминают"], "places": ["города или адреса"], "money": ["суммы, цены, скидки"], "organisations": ["компании и ведомства"]}},
  "sentiment": {{"negative": 0, "neutral": 0, "positive": 0}}
}}
Правила: не больше 6 claims; quote — дословная выдержка (до 200 символов); msg_id — идентификатор сообщения из квадратных скобок; sentiment — счёт сообщений по тональности.

Сообщения:
{messages}"""

REDUCE_SYSTEM = (
    "Ты аналитик соцмедиа и СМИ. Пиши разбор деловым русским языком: только текст, без JSON, "
    "без вводных фраз вида «график показывает». Опирайся только на переданные данные."
)

REDUCE_INSTRUCTION = """Ты аналитик соцмедиа и СМИ. Ниже результат разбора сообщений по теме «{topic}»{focus}.
Напиши подробный аналитический разбор на русском: 4–6 абзацев, деловым языком, без вводных фраз.
Структура: что происходит по теме; на что конкретно жалуются или что хвалят (с числами); детали и факты (продукты, места, суммы);
как меняется картина по площадкам; что из этого следует и что делать. Опирайся только на данные ниже, не выдумывай цифры.
Ответ — обычный текст, без JSON и без заголовков вида «аналитический_разбор».

Данные разбора:
{digest}"""


def _gateway():
    from mlops import gateway

    return gateway


def _strip_think(text: str) -> str:
    clean = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S | re.I)
    clean = re.sub(r"</?think>", "", clean, flags=re.I)
    return clean.strip()


def _extract_json(text: str) -> Optional[dict]:
    clean = _strip_think(text)
    decoder = json.JSONDecoder()
    start = clean.find("{")
    while start != -1:
        try:
            obj, _ = decoder.raw_decode(clean[start:])
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        start = clean.find("{", start + 1)
    return None


async def _qwen(ctx, prompt: str, *, system: str = "", max_tokens: int = BATCH_MAX_TOKENS, temperature: float = 0.15) -> Tuple[str, int]:
    """Один вызов локальной модели. Возвращает (текст, потраченные токены)."""
    gateway = _gateway()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    result = await gateway.achat(
        provider="vllm",
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=420,
        extra=dict(THINK_OFF),
        profile="agent",
    )
    raw = getattr(result, "raw", None) or {}
    usage = raw.get("usage") or {}
    try:
        tokens = int(usage.get("total_tokens") or 0)
    except Exception:
        tokens = 0
    return _strip_think(result.content or ""), tokens


def _message_line(msg_id: str, doc: Dict[str, Any]) -> str:
    text = " ".join(str(doc.get("text") or "").split())[:MESSAGE_CHARS]
    date = (doc.get("date") or "")[:10]
    return f"[{msg_id}] ({doc.get('hub') or 'источник'}, {date}, лайков {doc.get('likes') or 0}) {text}"


def _fetch_messages(ctx, index_name: str, phrase: str, lo, hi, tone: str, limit: int) -> List[Dict[str, Any]]:
    from .tools_data import _es, _iso, _query, _sample

    query = _query(phrase, lo, hi, tone)
    body = {
        "size": int(limit),
        "_source": [
            "text", "timeCreate", "hub", "url", "likesCount", "commentsCount", "toneMark", "city", "authorObject",
        ],
        "query": query,
        "sort": [{"likesCount": {"order": "desc"}}, {"timeCreate": {"order": "desc"}}],
    }
    try:
        res = _es().search(index=index_name, body=body)
    except Exception as exc:
        raise ToolError(f"Ошибка выборки сообщений: {exc}") from exc
    docs = []
    for hit in (res.get("hits") or {}).get("hits") or []:
        sample = _sample(hit)
        sample["time"] = _iso((hit.get("_source") or {}).get("timeCreate"))
        sample["city"] = (hit.get("_source") or {}).get("city") or ""
        docs.append(sample)
    return docs


def _merge_results(parsed_batches: List[dict], docs_by_id: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    subtopics: Dict[str, int] = {}
    claims: List[Dict[str, Any]] = []
    details: Dict[str, List[str]] = {"products": [], "places": [], "money": [], "organisations": []}
    sentiment = {"negative": 0, "neutral": 0, "positive": 0}

    for parsed in parsed_batches:
        for item in parsed.get("subtopics") or []:
            name = " ".join(str((item or {}).get("name") or "").split()).lower()
            if not name:
                continue
            try:
                mentions = int((item or {}).get("mentions") or 1)
            except Exception:
                mentions = 1
            subtopics[name] = subtopics.get(name, 0) + max(1, mentions)
        for item in parsed.get("claims") or []:
            claim = (item or {}).get("claim") or ""
            if not str(claim).strip():
                continue
            doc = docs_by_id.get(str((item or {}).get("msg_id") or "")) or {}
            claims.append(
                {
                    "claim": str(claim).strip()[:300],
                    "quote": str((item or {}).get("quote") or "")[:300],
                    "tone": (item or {}).get("tone") or "",
                    "url": doc.get("url") or "",
                    "date": doc.get("date") or "",
                    "platform": doc.get("hub") or "",
                    "author": doc.get("author") or "",
                }
            )
        got = parsed.get("details") or {}
        for key in details:
            for value in got.get(key) or []:
                value = " ".join(str(value or "").split())
                if value and value.lower() not in {v.lower() for v in details[key]}:
                    details[key].append(value)
        sent = parsed.get("sentiment") or {}
        for key in sentiment:
            try:
                sentiment[key] += int(sent.get(key) or 0)
            except Exception:
                pass

    unique_claims: List[Dict[str, Any]] = []
    seen = set()
    for item in sorted(claims, key=lambda c: -len(c.get("quote") or "")):
        key = item["claim"].lower()[:80]
        if key in seen:
            continue
        seen.add(key)
        unique_claims.append(item)

    top_subtopics = sorted(subtopics.items(), key=lambda kv: -kv[1])[:12]
    return {
        "subtopics": [{"name": name, "mentions": count} for name, count in top_subtopics],
        "key_claims": unique_claims[:12],
        "details": {key: value[:12] for key, value in details.items()},
        "sentiment": sentiment,
    }


@tool(
    "deep_text_analysis",
    title="Подробный LLM-анализ темы",
    description=(
        "Подробный разбор сообщений по теме локальной моделью Qwen (бесплатно, наши GPU): смысловые блоки, "
        "конкретные претензии и похвалы с цитатами и ссылками, детали (продукты, места, суммы, организации), "
        "тональность внутри темы и итоговый аналитический текст. Работает 1–3 минуты, поэтому вызывай его, "
        "когда нужен именно глубокий разбор темы, а не общие счётчики."
    ),
    parameters={
        "type": "object",
        "properties": {
            "phrase": {"type": "string", "description": "тема для разбора: фраза, по которой ищем сообщения"},
            "focus": {"type": "string", "description": "на что смотреть в первую очередь (например «претензии к качеству», «цены», «сервис»)"},
            "index": {"type": "integer", "description": "index датасета"},
            "min_date": {"type": "string", "description": "начало периода: YYYY-MM-DD или unix-секунды"},
            "max_date": {"type": "string", "description": "конец периода: YYYY-MM-DD или unix-секунды"},
            "tone": {"type": "string", "enum": ["any", "negative", "positive", "neutral"], "description": "фильтр тональности сообщений"},
            "max_messages": {"type": "integer", "description": "сколько сообщений разобрать (по умолчанию 120, максимум 160)"},
        },
        "required": ["phrase"],
    },
    group="analytics",
    timeout=900.0,
)
async def deep_text_analysis(
    ctx,
    phrase: str,
    focus: Optional[str] = None,
    index: Optional[int] = None,
    min_date: Any = None,
    max_date: Any = None,
    tone: str = "any",
    max_messages: int = 120,
):
    from .tools_data import _iso, dates, guard

    if not phrase or not str(phrase).strip():
        raise ToolError("Укажите тему для разбора")
    topic = " ".join(str(phrase).split())
    idx, index_name = guard(ctx, index)
    lo, hi = dates(ctx, min_date, max_date)
    limit = max(16, min(int(max_messages or 120), BATCH_SIZE * MAX_BATCHES))
    docs = _fetch_messages(ctx, index_name, topic, lo, hi, tone, limit)
    if not docs:
        return {
            "index": idx,
            "index_name": index_name,
            "topic": topic,
            "messages_analyzed": 0,
            "note": "По этой теме сообщений не найдено — анализировать нечего.",
        }

    docs_by_id: Dict[str, Dict[str, Any]] = {}
    lines: List[str] = []
    for pos, doc in enumerate(docs, start=1):
        msg_id = f"m{pos}"
        docs_by_id[msg_id] = doc
        lines.append(_message_line(msg_id, doc))

    batches = [lines[i : i + BATCH_SIZE] for i in range(0, len(lines), BATCH_SIZE)][:MAX_BATCHES]
    focus_text = f" Особое внимание: {focus}." if focus else ""
    await ctx.log(
        f"Подробный разбор темы «{topic}»: {len(docs)} сообщений, {len(batches)} пачек, локальная модель Qwen"
    )

    semaphore = asyncio.Semaphore(PARALLEL_BATCHES)
    parsed_batches: List[dict] = []
    failed = 0
    tokens_used = 0

    async def run_batch(number: int, batch: List[str]) -> None:
        nonlocal failed, tokens_used
        prompt = MAP_INSTRUCTION.format(topic=topic, focus=focus_text, messages="\n".join(batch))
        async with semaphore:
            for attempt in (1, 2):
                try:
                    text, tokens = await _qwen(ctx, prompt, system=MAP_SYSTEM)
                    tokens_used += tokens
                    parsed = _extract_json(text)
                    if parsed:
                        parsed_batches.append(parsed)
                        await ctx.log(f"Пачка {number}/{len(batches)} разобрана")
                        return
                    if attempt == 1:
                        prompt = prompt + "\n\nНапоминаю: ответ — только JSON, без текста вокруг."
                except Exception as exc:
                    if attempt == 2:
                        await ctx.log(f"Пачка {number} не разобрана: {exc}", level="error")
        failed += 1

    await asyncio.gather(*(run_batch(i, batch) for i, batch in enumerate(batches, start=1)))
    if not parsed_batches:
        raise ToolError("Локальная модель не вернула разбор ни по одной пачке — попробуйте позже или короче тему")

    merged = _merge_results(parsed_batches, docs_by_id)

    digest = {
        "тема": topic,
        "сообщений_разобрано": len(docs),
        "период": {"с": _iso(lo) if lo else None, "по": _iso(hi) if hi else None},
        "смысловые_блоки": merged["subtopics"],
        "претензии_и_наблюдения": merged["key_claims"],
        "детали": merged["details"],
        "тональность": merged["sentiment"],
    }
    summary = ""
    try:
        summary, tokens = await _qwen(
            ctx,
            REDUCE_INSTRUCTION.format(topic=topic, focus=focus_text, digest=json.dumps(digest, ensure_ascii=False)[:9000]),
            system=REDUCE_SYSTEM,
            max_tokens=REDUCE_MAX_TOKENS,
            temperature=0.25,
        )
        tokens_used += tokens
        if summary.lstrip().startswith("{"):
            # модель всё равно ответила JSON — достаём текст разбора из него
            parsed_summary = _extract_json(summary) or {}
            for key in ("аналитический_разбор", "разбор", "analysis", "summary", "text"):
                value = parsed_summary.get(key)
                if isinstance(value, dict):
                    summary = "\n\n".join(str(v) for v in value.values() if v)
                    break
                if isinstance(value, str) and value.strip():
                    summary = value.strip()
                    break
    except Exception as exc:
        await ctx.log(f"Не удалось собрать итоговый текст разбора: {exc}", level="error")

    artifact = None
    try:
        os.makedirs(ctx.artifacts_dir, exist_ok=True)
        safe_topic = re.sub(r'[\\/:*?"<>|\s]+', "_", topic)[:40]
        path = os.path.join(ctx.artifacts_dir, f"deep_analysis_{safe_topic}.md")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(f"# Подробный разбор темы «{topic}»\n\n")
            fh.write(f"Датасет: {index_name} · сообщений разобрано: {len(docs)} · модель: Qwen3-32B (локально)\n\n")
            fh.write(summary or "Итоговый текст модель не сформировала.")
            fh.write("\n\n## Смысловые блоки\n")
            for item in merged["subtopics"]:
                fh.write(f"- {item['name']} — {item['mentions']} упоминаний\n")
            fh.write("\n## Претензии и наблюдения с цитатами\n")
            for item in merged["key_claims"]:
                fh.write(f"- {item['claim']}: «{item['quote']}» — {item['platform']} {item['date']} {item['url']}\n")
        artifact = ctx.add_artifact("analysis", f"Подробный разбор: {topic}", path, meta={"url": f"/api/agent/artifact/{ctx.run_id}/{os.path.basename(path)}"})
        artifact["url"] = f"/api/agent/artifact/{ctx.run_id}/{os.path.basename(path)}"
    except Exception as exc:
        await ctx.log(f"Не удалось сохранить файл разбора: {exc}", level="error")

    return {
        "index": idx,
        "index_name": index_name,
        "topic": topic,
        "focus": focus or "",
        "messages_analyzed": len(docs),
        "batches": len(batches),
        "batches_failed": failed,
        "subtopics": compact(merged["subtopics"], max_items=12, max_str=120),
        "key_claims": compact(merged["key_claims"], max_items=12, max_str=260),
        "details": compact(merged["details"], max_items=12, max_str=120),
        "sentiment": merged["sentiment"],
        "summary": summary,
        "artifact": {"name": os.path.basename(artifact["path"])} if artifact else None,
        "usage": {"local_model_tokens": tokens_used, "cost_usd": 0.0},
        "note": "Разбор выполнен локальной моделью Qwen3-32B на сервере Tellscope: внешние API не оплачиваются.",
    }
