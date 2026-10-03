# -*- coding: utf-8 -*-
"""Разделы (вкладки) интерфейса Tellscope и доступ пользователей к ним.

Доступы к темам (датасетам) живут в Redis: hash ``<user_id>`` → ``json_files_directory``.
Разделы интерфейса хранятся там же, в поле ``allowed_sections``:

* поле отсутствует или ``["*"]`` — доступны все разделы (так ведут себя все текущие учётки);
* список слагов (``["tonality", "media"]``) — доступны только эти разделы.

Каталог разделов повторяет левое меню приложения: слаг, название, путь и префиксы API,
которые этому разделу принадлежат. Проверка идёт по самому длинному подходящему префиксу,
поэтому ``/agent/agents`` (Мои агенты) не попадает под ``/agent`` (Центр ИИ-задач).
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, Iterable, List, Optional, Set

ALL_SECTIONS = "*"

# Порядок совпадает с меню: сначала ИИ-разделы, затем аналитика и данные.
SECTIONS: List[Dict[str, Any]] = [
    {
        "slug": "harness",
        "title": "Центр ИИ-задач",
        "path": "/harness",
        "prefixes": ["/harness", "/agent/run", "/agent/runs", "/agent/task"],
    },
    {
        "slug": "dify",
        "title": "Dify",
        "path": "/dify-constructor",
        "prefixes": ["/dify"],
        "hint": "редактор Dify открыт только администратору: вкладка видна, но конструктор закрыт.",
    },
    {
        "slug": "agents",
        "title": "Мои агенты",
        "path": "/agents",
        "prefixes": ["/agents", "/agent/agents", "/agent/connectors", "/agent/tools"],
    },
    {
        "slug": "tonality",
        "title": "Тональный ландшафт",
        "path": "/user-tonality",
        "prefixes": ["/tonality_landscape", "/tonality", "/user-tonality"],
    },
    {
        "slug": "information-graph",
        "title": "Граф информации",
        "path": "/information-graf",
        "prefixes": ["/information-graf", "/information_graph", "/information-graph"],
    },
    {
        "slug": "media",
        "title": "СМИ",
        "path": "/media-rating",
        "prefixes": ["/media-rating", "/media_rating"],
    },
    {
        "slug": "voice",
        "title": "Голос клиента",
        "path": "/voice-of-customer",
        "prefixes": ["/voice-of-customer", "/voice_of_customer"],
    },
    {
        "slug": "ai-analytics",
        "title": "ИИ анализ",
        "path": "/ai-analytics",
        "prefixes": ["/ai-analytics", "/ai_analytics"],
    },
    {
        "slug": "graph-analysis",
        "title": "Связи авторов",
        "path": "/graph-analysis",
        "prefixes": ["/graph-analysis", "/graph_analysis"],
    },
    {
        "slug": "ai-bot",
        "title": "ИИ-Бот",
        "path": "/ai-bot",
        "prefixes": ["/ai-bot", "/ai_bot"],
    },
    {
        "slug": "datasets",
        "title": "Наборы данных",
        "path": "/data-set",
        "prefixes": ["/configs", "/datasets", "/dataset", "/folders", "/upload",
                     "/delete-theme-files", "/add-folder", "/create-folder"],
    },
    {
        "slug": "mosinform",
        "title": "ОИВ рейтинг",
        "path": "/mosinform-rating",
        "prefixes": ["/mosinform"],
    },
    {
        "slug": "pr-campaigns",
        "title": "PR-кампании",
        "path": "/pr-campaigns",
        "prefixes": ["/pr/"],
    },
    {
        "slug": "docs",
        "title": "Документация",
        "path": "https://wiki.tellscope40.headsmade.com",
        "prefixes": [],
    },
    {
        "slug": "admin",
        "title": "Администрирование",
        "path": "/admin",
        "prefixes": ["/admin"],
        # Чувствительный раздел: подсвечиваем в списке выдачи и объясняем последствия.
        "danger": True,
        "hint": ("открывает раздел администрирования. Полные возможности над темами и "
                 "пользователями даёт только флаг «админ» у учётной записи (кнопка «сделать "
                 "админом»): сама вкладка прав не добавляет."),
    },
]

_BY_SLUG = {item["slug"]: item for item in SECTIONS}
# Префиксы от длинных к коротким: побеждает самое точное совпадение.
_PREFIXES: List[tuple] = sorted(
    ((prefix, item["slug"]) for item in SECTIONS for prefix in item["prefixes"]),
    key=lambda pair: -len(pair[0]),
)

_CACHE: Dict[str, tuple] = {}
_CACHE_TTL = 20.0


def catalog() -> List[Dict[str, Any]]:
    """Каталог разделов для интерфейса: слаг, название, путь, пометка и пояснение.

    У чувствительных разделов (``danger``) интерфейс показывает пояснение: без него
    «Администрирование» выглядит как обычная вкладка в списке выдачи.
    """
    rows = []
    for item in SECTIONS:
        row = {"slug": item["slug"], "title": item["title"], "path": item["path"]}
        if item.get("danger"):
            row["danger"] = True
        if item.get("hint"):
            row["hint"] = item["hint"]
        rows.append(row)
    return rows


def known_slugs() -> Set[str]:
    return set(_BY_SLUG)


def section_for_path(path: str) -> Optional[str]:
    """Раздел, которому принадлежит путь API. None — путь не относится к вкладкам."""
    text = str(path or "")
    if not text:
        return None
    for prefix, slug in _PREFIXES:
        if text == prefix or text.startswith(prefix + "/") or text.startswith(prefix + "?"):
            return slug
        # префиксы вида "/pr/" уже содержат слэш
        if prefix.endswith("/") and text.startswith(prefix):
            return slug
    return None


def _redis():
    import redis as _rds

    return _rds.Redis(host="localhost", port=6379, db=0, decode_responses=True)


def allowed_sections(user_id: Any) -> Set[str]:
    """Разрешённые разделы пользователя. Возвращает {ALL_SECTIONS} для полного доступа."""
    key = str(user_id)
    now = time.time()
    cached = _CACHE.get(key)
    if cached and now - cached[0] < _CACHE_TTL:
        return cached[1]
    value: Set[str] = {ALL_SECTIONS}
    try:
        raw = _redis().hget(key, "allowed_sections")
        if raw:
            parsed = json.loads(raw)
            if isinstance(parsed, str):
                parsed = [parsed]
            if isinstance(parsed, list):
                items = {str(item).strip() for item in parsed if str(item).strip()}
                if items:
                    value = items
    except Exception:
        value = {ALL_SECTIONS}
    _CACHE[key] = (now, value)
    return value


def set_allowed_sections(user_id: Any, slugs: Optional[Iterable[str]]) -> Set[str]:
    """Записывает доступы: пустой список или «*» — все разделы."""
    items = {str(item).strip() for item in (slugs or []) if str(item).strip()}
    unknown = items - known_slugs() - {ALL_SECTIONS}
    if unknown:
        raise ValueError("неизвестные разделы: %s" % ", ".join(sorted(unknown)))
    if not items or ALL_SECTIONS in items:
        items = {ALL_SECTIONS}
    _redis().hset(str(user_id), "allowed_sections", json.dumps(sorted(items), ensure_ascii=False))
    _CACHE.pop(str(user_id), None)
    return items


def is_section_allowed(user_id: Any, slug: Optional[str]) -> bool:
    if not slug:
        return True
    allowed = allowed_sections(user_id)
    return ALL_SECTIONS in allowed or slug in allowed


async def deny_reason(user_id: Any, path: str) -> Optional[str]:
    """Причина отказа для пути или None, если доступ есть.

    Суперпользователи не ограничиваются: они настраивают доступы других.
    """
    slug = section_for_path(path)
    if not slug:
        return None
    if is_section_allowed(user_id, slug):
        return None
    try:
        from access_guard import user_by_id

        user = await user_by_id(user_id)
        if user is not None and getattr(user, "is_superuser", False):
            return None
    except Exception:
        pass
    title = (_BY_SLUG.get(slug) or {}).get("title") or slug
    return "Нет доступа к разделу «%s»" % title
