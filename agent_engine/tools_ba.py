# -*- coding: utf-8 -*-
"""Инструмент загрузки темы из Brand Analytics.

Если нужной темы нет среди локальных датасетов Tellscope (а она есть в аккаунте Brand Analytics),
агент может выгрузить её за нужный период: Tellscope запускает экспорт BA, сохраняет данные,
индексирует их в Elasticsearch и отдаёт номер нового датасета — дальше работают обычные инструменты
аналитики (поиск, тональность, авторы, отчёты).
"""
from __future__ import annotations

import asyncio
import threading
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from .registry import ToolError, tool
from .tools_data import _norm_text, _split_name

DEFAULT_WAIT = 480
MAX_WAIT = 540


def _period(date_from: str, date_to: str) -> "tuple[str, str]":
    """'2026-09-10' → unix-секунды начала и конца суток (как в интерфейсе Tellscope)."""
    def parse(value: str, end: bool) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        if text.isdigit():
            return text
        for fmt in ("%Y-%m-%d", "%d.%m.%Y"):
            try:
                day = datetime.strptime(text[: len(fmt) + 2 if fmt == "%d.%m.%Y" else 10], fmt)
                if end:
                    day = day + timedelta(days=1) - timedelta(seconds=1)
                return str(int(day.timestamp()))
            except Exception:
                continue
        raise ToolError(f"Не понял дату «{text}»: используйте формат ГГГГ-ММ-ДД")

    return parse(date_from, False), parse(date_to, True)


def _best_theme(query: str, themes: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Ищет тему BA по названию: точное совпадение, затем вхождение (с учётом транслита)."""
    text = _norm_text(query)
    if len(text) < 2:
        return None
    for item in themes:
        if _norm_text(item.get("title")) == text:
            return item
    for item in themes:
        if text in _norm_text(item.get("title")):
            return item
    for item in themes:
        title = _norm_text(item.get("title"))
        if len(title) > 3 and title in text:
            return item
    return None


@tool(
    "fetch_dataset",
    title="Загрузить тему из Brand Analytics",
    description=(
        "Выгружает тему из аккаунта Brand Analytics за период и добавляет её как датасет Tellscope. "
        "Используй, когда пользователь называет тему, которой нет в list_datasets (например «Озон отзывы»). "
        "После загрузки работай с полученным датасетом обычными инструментами (search_messages, tonality_summary и т.д.)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "theme": {"type": "string", "description": "название темы в Brand Analytics, например «Озон отзывы»"},
            "date_from": {"type": "string", "description": "начало периода: ГГГГ-ММ-ДД или unix-секунды"},
            "date_to": {"type": "string", "description": "конец периода: ГГГГ-ММ-ДД или unix-секунды"},
            "wait_seconds": {"type": "integer", "description": "сколько секунд ждать выгрузку (по умолчанию 8 минут)"},
        },
        "required": ["theme", "date_from", "date_to"],
    },
    group="analytics",
    scope="write",
    timeout=570.0,
    cost=3,
)
async def fetch_dataset(ctx, theme: str, date_from: str = "", date_to: str = "",
                        wait_seconds: Optional[int] = None) -> Dict[str, Any]:
    """Загружает тему из Brand Analytics и возвращает новый датасет."""
    import ba_api

    themes_payload = ba_api.themes(str(ctx.user_id), refresh=0)
    items = themes_payload.get("themes") or []
    match = _best_theme(theme, items)
    if match is None and items:
        # снапшот мог устареть — обновляем список тем из аккаунта
        try:
            themes_payload = ba_api.themes(str(ctx.user_id), refresh=1)
            items = themes_payload.get("themes") or []
            match = _best_theme(theme, items)
        except Exception:
            match = None
    if match is None:
        titles = ", ".join(f"«{item.get('title')}»" for item in items[:20]) or "список пуст"
        if not items:
            raise ToolError(
                "Аккаунт Brand Analytics не подключён: добавьте логин и пароль BA в разделе настроек Tellscope"
            )
        raise ToolError(f"В Brand Analytics нет темы «{theme}». Доступные темы: {titles}")

    tsf, tst = _period(date_from, date_to)
    job_id = uuid.uuid4().hex[:12]
    body = ba_api.ImportBody(
        theme_id=str(match["theme_id"]),
        user_id=str(ctx.user_id),
        folder="",
        date_from=tsf,
        date_to=tst,
        force=True,
    )
    threading.Thread(target=ba_api._run_import, args=(job_id, body), daemon=True).start()

    limit = min(int(wait_seconds or DEFAULT_WAIT), MAX_WAIT)
    deadline = time.time() + limit
    while time.time() < deadline:
        await asyncio.sleep(5)
        try:
            status = ba_api.job_status(job_id)
        except Exception:
            continue
        state = status.get("status")
        if state == "done":
            rec = status.get("summary") or {}
            index_key = rec.get("index_key")
            index_name = rec.get("index_name") or ""
            label, period = _split_name(index_name)
            return {
                "status": "готово",
                "theme": match.get("title"),
                "dataset": index_name,
                "index": index_key,
                "label": label,
                "period": period,
                "file": rec.get("file"),
                "bytes": rec.get("bytes"),
                "hint": (
                    f"Данные загружены: работайте с темой «{index_name}» "
                    f"(index {index_key}) обычными инструментами"
                ),
            }
        if state == "error":
            raise ToolError(f"Выгрузка из Brand Analytics не удалась: {status.get('message')}")

    return {
        "status": "идёт загрузка",
        "theme": match.get("title"),
        "job_id": job_id,
        "waited_seconds": limit,
        "hint": (
            "Brand Analytics ещё выгружает данные. Подождите пару минут и снова вызовите этот инструмент "
            "или просто повторите анализ — датасет появится в list_datasets"
        ),
    }
