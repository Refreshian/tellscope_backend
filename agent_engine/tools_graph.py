# -*- coding: utf-8 -*-
"""Инструменты графов и инфоповодов: информационный граф, цепочки распространения, популярные поводы."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi.encoders import jsonable_encoder

from .context import compact
from .registry import ToolError, tool
from .tools_data import _m, _iso, dates, guard


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


def _full_period(ctx, idx: int, lo, hi):
    """Период для инструментов, которым он обязателен; понятная ошибка, если времени нет в индексе."""
    from .tools_data import NO_TIME_NOTE, _period, _period_or_none

    lo, hi = _period_or_none(lo, hi)
    if lo and hi:
        return lo, hi
    _, index_name = guard(ctx, idx)
    lo2, hi2 = _period(index_name)
    if lo2 and hi2:
        return lo2, hi2
    raise ToolError(NO_TIME_NOTE)


@tool(
    "popular_hooks",
    title="Популярные инфоповоды",
    description=(
        "Инфоповоды датасета: самые частые фразы, извлечённые из сообщений именно этого датасета, "
        "с точным числом упоминаний. Темы не переносятся из других проектов — используй то, что вернул инструмент."
    ),
    parameters={
        "type": "object",
        "properties": {
            "index": {"type": "string", "description": "тема: название датасета или его номер"},
            "limit": {"type": "integer", "description": "сколько поводов вернуть (по умолчанию 15, максимум 40)"},
            "min_date": {"type": "string", "description": "начало периода: YYYY-MM-DD или unix-секунды"},
            "max_date": {"type": "string", "description": "конец периода: YYYY-MM-DD или unix-секунды"},
        },
    },
    group="graph",
    timeout=300.0,
)
async def popular_hooks(ctx, index: Optional[int] = None, limit: int = 15, min_date: Any = None, max_date: Any = None):
    m = _m()
    idx, index_name = guard(ctx, index)
    limit = max(1, min(int(limit or 15), 40))
    lo, hi = dates(ctx, min_date, max_date)
    data = await _guard_call(m.popular_hooks, user=ctx.user, index=idx, limit=limit, min_date=lo, max_date=hi)
    payload = jsonable_encoder(data)
    return {
        "index": idx,
        "index_name": index_name,
        "hooks": compact(payload.get("values") or [], max_items=limit, max_str=120),
        "note": payload.get("note") or "фразы извлечены из сообщений датасета",
    }


@tool(
    "chain_graph",
    title="Цепочка распространения",
    description=(
        "Цепочка распространения инфоповода: первоисточник, распространители, площадки, таймлайн. "
        "Показывает, откуда пошла волна и кто её разогнал. Ключевой инструмент для расследований всплесков."
    ),
    parameters={
        "type": "object",
        "properties": {
            "phrase": {"type": "string", "description": "тема/фраза инфоповода, например «Курочка с душком»"},
            "index": {"type": "string", "description": "тема: название датасета или его номер"},
            "min_date": {"type": "string", "description": "YYYY-MM-DD или unix-секунды"},
            "max_date": {"type": "string", "description": "YYYY-MM-DD или unix-секунды"},
        },
        "required": ["phrase"],
    },
    group="graph",
    timeout=360.0,
)
async def chain_graph(ctx, phrase: str, index: Optional[int] = None, min_date: Any = None, max_date: Any = None):
    m = _m()
    if not phrase or not str(phrase).strip():
        raise ToolError("Укажите фразу инфоповода")
    idx, _ = guard(ctx, index)
    lo, hi = dates(ctx, min_date, max_date)
    data = await _guard_call(m.chain_graph, user=ctx.user, index=idx, phrase=str(phrase).strip(), min_date=lo, max_date=hi)
    payload = jsonable_encoder(data)
    graph = payload.get("graph") or {}
    stats = payload.get("stats") or {}
    nodes = graph.get("nodes") or []
    clusters = graph.get("clusters") or []
    timeline = payload.get("timeline") or []
    top_nodes = sorted(nodes, key=lambda n: -(n.get("posts_count") or 0))[:15]
    return {
        "index": idx,
        "phrase": phrase,
        "stats": compact(stats, max_items=15, max_str=200),
        "top_spreaders": [
            {"author": n.get("label"), "hub": n.get("hub"), "posts": n.get("posts_count"), "url": n.get("url"), "topic": ((n.get("topics") or [{}])[0] or {}).get("text")}
            for n in top_nodes
        ],
        "clusters": compact(clusters, max_items=12, max_str=200),
        "timeline_summary": {
            "hours_span": (timeline[-1]["hour"] if timeline else 0),
            "points": len(timeline),
            "first_hours": compact(timeline[:12], max_items=12),
            "peak": max(timeline, key=lambda p: p.get("count") or 0) if timeline else None,
            "total_messages": timeline[-1]["cumulative"] if timeline else 0,
        },
        "nodes_total": len(nodes),
    }


@tool(
    "information_graph",
    title="Информационный граф",
    description=(
        "Информационный граф по теме: уникальные авторы, их аудитория и динамика аудитории по дням, "
        "сообщения и репосты. Показывает, кто формирует информационное поле вокруг темы. "
        "На больших датасетах обязательно указывайте min_date и max_date — иначе обработка идёт по всему периоду."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query_str": {"type": "string", "description": "тема запроса, например «карта», «бургер», «Ростикс»"},
            "index": {"type": "string", "description": "тема: название датасета или его номер"},
            "min_date": {"type": "string", "description": "YYYY-MM-DD или unix-секунды"},
            "max_date": {"type": "string", "description": "YYYY-MM-DD или unix-секунды"},
            "post": {"type": "boolean", "description": "учитывать посты"},
            "repost": {"type": "boolean", "description": "учитывать репосты"},
            "SMI": {"type": "boolean", "description": "учитывать СМИ"},
        },
        "required": ["query_str"],
    },
    group="graph",
    timeout=420.0,
)
async def information_graph(
    ctx,
    query_str: str,
    index: Optional[int] = None,
    min_date: Any = None,
    max_date: Any = None,
    post: Optional[bool] = None,
    repost: Optional[bool] = None,
    SMI: Optional[bool] = None,
):
    m = _m()
    idx, _ = guard(ctx, index)
    lo, hi = _full_period(ctx, idx, *dates(ctx, min_date, max_date))
    data = await _guard_call(
        m.information_graph,
        user=ctx.user,
        index=idx,
        min_date=lo,
        max_date=hi,
        query_str=query_str,
        post=post,
        repost=repost,
        SMI=SMI,
    )
    payload = jsonable_encoder(data)
    values = payload.get("values") or []
    audience = payload.get("dynamicdata_audience") or {}
    return {
        "index": idx,
        "query_str": query_str,
        "num_messages": payload.get("num_messages"),
        "num_unique_authors": payload.get("num_unique_authors"),
        "authors_stream": compact(values, max_items=15, max_str=200),
        "audience_dynamics": compact(audience, max_items=40, max_str=80),
    }
