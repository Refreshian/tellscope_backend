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
import re
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from .context import to_unix_end, to_unix_start
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
        stamp = to_unix_end(text) if end else to_unix_start(text)
        if stamp is None:
            raise ToolError(f"Не понял дату «{text}»: используйте формат ГГГГ-ММ-ДД")
        return str(stamp)

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


def _ru_period(text: str) -> Optional["tuple[str, str]"]:
    """«01.03.2025 — 31.03.2025» → («2025-03-01», «2025-03-31») для проверки покрытия периода."""
    parts = re.split(r"\s*[—–]\s*|\s+-\s+", str(text or "").strip())
    if len(parts) != 2:
        return None
    out = []
    for part in parts:
        match = re.match(r"^(\d{2})\.(\d{2})\.(\d{4})$", part.strip())
        if not match:
            return None
        day, month, year = match.groups()
        out.append("%s-%s-%s" % (year, month, day))
    return out[0], out[1]


def _covers_period(item: Dict[str, Any], want_from: Any, want_to: Any) -> bool:
    """Покрывает ли готовый датасет запрошенный период. Неизвестный период — не считаем покрытием."""
    if not want_from or not want_to:
        return True
    try:
        lo_want, hi_want = int(want_from), int(want_to)
    except (TypeError, ValueError):
        return True
    bounds = _ru_period(str(item.get("period") or ""))
    if bounds:
        lo, hi = _period(bounds[0], bounds[1])
    else:
        # В подписи периода нет (например «БА Риномарис 20260917») — спрашиваем Elasticsearch.
        # Имя индекса берём из справочника датасетов: у старых датасетов name и индекс не совпадают.
        es_name = ""
        try:
            from .tools_data import index_map

            es_name = (index_map() or {}).get(int(item.get("index"))) or ""
        except Exception:
            es_name = ""
        if not es_name:
            es_name = str(item.get("name") or "")
        try:
            from .tools_data import _period as _dataset_period

            lo, hi = _dataset_period(es_name)
        except Exception:
            lo, hi = (None, None)
        lo, hi = (str(lo) if lo else "", str(hi) if hi else "")
        if not lo or not hi:
            return False
    try:
        return int(lo) <= lo_want and int(hi) >= hi_want
    except (TypeError, ValueError):
        return False


def _existing_dataset(theme_title: str, want_from: Any = None,
                      want_to: Any = None) -> Optional[Dict[str, Any]]:
    """Уже загруженный в Tellscope датасет по названию темы (без выгрузки из Brand Analytics)."""
    try:
        from . import tools_data as _td
        items = _td.datasets_public()
    except Exception:
        return None
    norm = getattr(_td, "_norm_text", None)

    def _n(text: str) -> str:
        text = str(text or "").lower()
        if norm:
            try:
                return norm(text)
            except Exception:
                pass
        return text

    words = [w for w in re.split(r"[^0-9a-zA-Zа-яА-ЯёЁ]+", _n(theme_title)) if len(w) >= 3]
    if not words:
        return None
    threshold = max(1, len(words) - 1)
    scored = []
    for item in items or []:
        hay = _n(str(item.get("name") or "")) + " " + _n(str(item.get("label") or ""))
        score = sum(1 for word in words if word in hay)
        if score >= threshold:
            scored.append((score, item))
    if not scored:
        return None
    # Из подходящих по названию берём тот, что реально покрывает запрошенный период:
    # иначе на запрос «Мониторинг тем за 14–21.09.2026» подставлялся готовый датасет
    # за март 2025 — отчёт выходил не про тот период, о котором спросили.
    scored.sort(key=lambda pair: -pair[0])
    for _score, item in scored:
        if _covers_period(item, want_from, want_to):
            return item
    return None


def _remember_dataset(ctx: Any, index: Any, label: str = "",
                      period_from: Any = None, period_to: Any = None) -> None:
    """Делает выгруженный датасет текущим для остальных инструментов запуска.

    Иначе после успешной выгрузки инструменты без явного index продолжали работать с датасетом,
    выбранным в интерфейсе: отчёт выходил по одной теме, а данные в нём — по другой.
    """
    try:
        if index is not None:
            ctx.dataset_index = int(index)
        if label:
            ctx.dataset_name = label
            ctx.dataset_label = label
        if period_from:
            ctx.min_date = int(period_from)
        if period_to:
            ctx.max_date = int(period_to)
    except Exception:
        return


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
    # Тема могла выгружаться раньше — берём готовый датасет, только если он покрывает
    # запрошенный период; иначе идём в BA за нужными датами.
    existing = _existing_dataset(str(match.get("title") or theme), tsf, tst)
    if existing:
        bounds = _ru_period(str(existing.get("period") or ""))
        if bounds:
            use_from, use_to = _period(bounds[0], bounds[1])
            _remember_dataset(ctx, existing.get("index"),
                              str(existing.get("label") or existing.get("name") or ""),
                              use_from, use_to)
        else:
            _remember_dataset(ctx, existing.get("index"),
                              str(existing.get("label") or existing.get("name") or ""))
        return {
            "status": "используется ранее загруженный датасет",
            "theme": match.get("title"),
            "dataset": existing.get("name"),
            "index": existing.get("index"),
            "label": existing.get("label") or existing.get("name"),
            "period": existing.get("period") or "",
            "hint": (
                f"Тема «{match.get('title')}» уже выгружалась в Tellscope: работайте с датасетом "
                f"«{existing.get('label') or existing.get('name')}» (index {existing.get('index')}) — "
                "новая выгрузка из Brand Analytics не нужна"
            ),
        }

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
    check = 0
    while time.time() < deadline:
        await asyncio.sleep(5)
        check += 1
        try:
            status = ba_api.job_status(job_id)
        except Exception:
            status = {}
        state = status.get("status")
        if state == "done":
            rec = status.get("summary") or {}
            index_key = rec.get("index_key")
            index_name = rec.get("index_name") or ""
            label, period = _split_name(index_name)
            _remember_dataset(ctx, index_key, index_name or label, tsf, tst)
            return {
                "status": "готово",
                "theme": match.get("title"),
                "dataset": index_name,
                "index": index_key,
                "label": label,
                "period": period,
                "period_requested": {"from": date_from, "to": date_to},
                "file": rec.get("file"),
                "bytes": rec.get("bytes"),
                "hint": (
                    f"Данные загружены: работайте с темой «{index_name}» "
                    f"(index {index_key}) обычными инструментами"
                ),
            }
        if state == "error":
            raise ToolError(f"Выгрузка из Brand Analytics не удалась: {status.get('message')}")
        if check % 6 == 0:
            # Статус задачи в BA иногда не доходит до «готово», хотя датасет уже собран:
            # проверяем справочник датасетов, чтобы не отдавать «идёт загрузка» на готовые данные.
            ready = _existing_dataset(str(match.get("title") or theme), tsf, tst)
            if ready:
                _remember_dataset(ctx, ready.get("index"),
                                  str(ready.get("label") or ready.get("name") or ""), tsf, tst)
                return {
                    "status": "готово",
                    "theme": match.get("title"),
                    "dataset": ready.get("name"),
                    "index": ready.get("index"),
                    "label": ready.get("label") or ready.get("name"),
                    "period": ready.get("period") or "",
                    "period_requested": {"from": date_from, "to": date_to},
                    "hint": (
                        f"Данные загружены: работайте с темой «{ready.get('name')}» "
                        f"(index {ready.get('index')}) обычными инструментами"
                    ),
                }

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
