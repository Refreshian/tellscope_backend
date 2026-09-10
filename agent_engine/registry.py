# -*- coding: utf-8 -*-
"""Реестр инструментов агентного режима Tellscope.

Инструмент описывается декларативно: имя, JSON-схема параметров, обработчик,
группа, политика (scope) и лимиты. Обработчики выполняются в том же процессе,
что и FastAPI, и вызывают те же функции, что и эндпоинты интерфейса, поэтому
проверки доступа к датасетам остаются едиными (main._guard_index_access).
"""
from __future__ import annotations

import asyncio
import inspect
import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

TOOL_GROUPS: Dict[str, Dict[str, str]] = {
    "analytics": {
        "title": "Данные и метрики",
        "hint": "Поиск сообщений, тональность, рейтинг СМИ, голос клиента, ИИ-аналитика",
    },
    "graph": {
        "title": "Графы и инфоповоды",
        "hint": "Информационный граф, цепочки распространения, популярные поводы",
    },
    "reports": {
        "title": "Отчёты и артефакты",
        "hint": "Графики, сборка DOCX/PDF, сохранение во вкладке «Отчёты»",
    },
    "connectors": {
        "title": "Внешние коннекторы",
        "hint": "HTTP/MCP-инструменты и пользовательские секреты",
    },
}

MAX_RESULT_CHARS = 3500
MAX_TOOL_TIMEOUT = 600.0

# Ограничения длины описаний в схеме, которую отправляем модели каждый шаг.
# Схемы 17 инструментов пересылаются в каждом запросе, поэтому их размер = прямые деньги.
SCHEMA_DESC_LIMIT = 200
SCHEMA_PARAM_DESC_LIMIT = 110


class ToolError(RuntimeError):
    """Ошибка выполнения инструмента, безопасная для показа в журнале агента."""


def slim_schema(parameters: Dict[str, Any], desc_limit: int = SCHEMA_PARAM_DESC_LIMIT) -> Dict[str, Any]:
    """Урезает описания в JSON-схеме параметров, сохраняя структуру и типы."""
    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            out = {}
            for key, value in node.items():
                if key == "description" and isinstance(value, str):
                    out[key] = value if len(value) <= desc_limit else value[:desc_limit].rstrip() + "…"
                else:
                    out[key] = walk(value)
            return out
        if isinstance(node, list):
            return [walk(item) for item in node]
        return node

    return walk(parameters or {"type": "object", "properties": {}})


@dataclass
class ToolSpec:
    name: str
    title: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable[..., Any]
    group: str = "analytics"
    scope: str = "read"
    default_enabled: bool = True
    timeout: float = 240.0
    cost: int = 1

    def openai_schema(self, compact: bool = True) -> Dict[str, Any]:
        description = self.description or ""
        if compact and len(description) > SCHEMA_DESC_LIMIT:
            description = description[:SCHEMA_DESC_LIMIT].rstrip() + "…"
        parameters = slim_schema(self.parameters) if compact else (self.parameters or {"type": "object", "properties": {}})
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": description,
                "parameters": parameters,
            },
        }

    def public(self) -> Dict[str, Any]:
        meta = TOOL_GROUPS.get(self.group) or {}
        return {
            "name": self.name,
            "title": self.title,
            "group": self.group,
            "group_title": meta.get("title", self.group),
            "description": self.description,
            "parameters": self.parameters,
            "scope": self.scope,
            "default_enabled": self.default_enabled,
        }


_REGISTRY: Dict[str, ToolSpec] = {}


def register(spec: ToolSpec) -> ToolSpec:
    if spec.name in _REGISTRY:
        raise ValueError(f"инструмент уже зарегистрирован: {spec.name}")
    _REGISTRY[spec.name] = spec
    return spec


def tool(
    name: str,
    *,
    title: str,
    description: str,
    parameters: Dict[str, Any],
    group: str = "analytics",
    scope: str = "read",
    default_enabled: bool = True,
    timeout: float = 240.0,
    cost: int = 1,
):
    """Декоратор регистрации инструмента."""

    def deco(func: Callable[..., Any]) -> Callable[..., Any]:
        register(
            ToolSpec(
                name=name,
                title=title,
                description=description,
                parameters=parameters,
                handler=func,
                group=group,
                scope=scope,
                default_enabled=default_enabled,
                timeout=timeout,
                cost=cost,
            )
        )
        return func

    return deco


def all_tools() -> Dict[str, ToolSpec]:
    return dict(_REGISTRY)


def get_tool(name: str) -> Optional[ToolSpec]:
    return _REGISTRY.get(name)


def default_tool_names() -> List[str]:
    return [name for name, spec in sorted(_REGISTRY.items()) if spec.default_enabled]


def catalog() -> Dict[str, Any]:
    """Каталог инструментов для интерфейса: группы + описания + схемы."""
    groups: List[Dict[str, Any]] = []
    for gid, meta in TOOL_GROUPS.items():
        items = [spec.public() for spec in sorted(_REGISTRY.values(), key=lambda s: s.name) if spec.group == gid]
        groups.append({"id": gid, "title": meta.get("title", gid), "hint": meta.get("hint", ""), "tools": items})
    return {
        "groups": groups,
        "total": len(_REGISTRY),
        "default_enabled": default_tool_names(),
    }


def resolve_tools(requested: Optional[List[str]]) -> List[str]:
    """Проверяет список включённых инструментов; неизвестные имена отбрасывает."""
    if not requested:
        return default_tool_names()
    known = [name for name in requested if name in _REGISTRY]
    return known


def openai_tools(names: List[str], compact: bool = True) -> List[Dict[str, Any]]:
    out = []
    for name in names:
        spec = _REGISTRY.get(name)
        if spec:
            out.append(spec.openai_schema(compact=compact))
    return out


def json_text(value: Any, max_chars: int = MAX_RESULT_CHARS) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        text = json.dumps({"repr": repr(value)[:max_chars]}, ensure_ascii=False)
    if len(text) > max_chars:
        text = text[:max_chars] + '... [результат усечён]'
    return text


async def execute(spec: ToolSpec, ctx: Any, args: Dict[str, Any]) -> Dict[str, Any]:
    """Выполняет инструмент с лимитом времени и безопасной обработкой ошибок."""
    started = time.time()
    timeout = min(float(spec.timeout or 240.0), MAX_TOOL_TIMEOUT)
    try:
        result = spec.handler(ctx, **(args or {}))
        if inspect.isawaitable(result):
            result = await asyncio.wait_for(result, timeout=timeout)
    except asyncio.TimeoutError:
        return {
            "ok": False,
            "error": f"инструмент {spec.name} превысил лимит {int(timeout)} с",
            "ms": int((time.time() - started) * 1000),
        }
    except Exception as exc:  # noqa: BLE001 — ошибка уходит агенту текстом
        import traceback

        traceback.print_exc()
        detail = str(exc)
        try:
            from fastapi import HTTPException

            if isinstance(exc, HTTPException):
                detail = f"HTTP {exc.status_code}: {exc.detail}"
        except Exception:
            pass
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {detail}"[:600],
            "ms": int((time.time() - started) * 1000),
        }
    return {
        "ok": True,
        "result": result,
        "ms": int((time.time() - started) * 1000),
    }
