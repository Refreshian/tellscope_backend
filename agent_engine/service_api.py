# -*- coding: utf-8 -*-
"""Сервисный (машинный) доступ к инструментам Tellscope.

Тот же реестр инструментов, что у агентного режима (/agent/run) и MCP-сервера (/mcp),
публикуется наружу двумя способами:

  * ``POST /api/agent/tool/<имя>`` — прямой вызов инструмента по HTTP с сервисным токеном;
  * ``GET /api/agent/openapi.json`` — OpenAPI 3.0 описание всех инструментов, которое
    визуальный конструктор Dify импортирует как «Custom Tool» (и n8n / GPT Actions тоже).

Сервисные токены лежат в JSON-файле рядом с каталогом data, поэтому их можно выдавать и
отзывать без перезапуска API. Дополнительно поддерживается токен из переменной окружения
``AGENT_SERVICE_TOKEN`` (для быстрого подключения из скриптов).
"""
from __future__ import annotations

import json
import os
import secrets
import time
from typing import Any, Dict, List, Optional

DEFAULT_TOKENS_PATH = "/home/dev/tellscope_app/tellscope_backend/data/agent_service_tokens.json"
TOKENS_PATH = (os.getenv("AGENT_SERVICE_TOKENS_PATH") or "").strip() or DEFAULT_TOKENS_PATH

# Группы инструментов, которые НЕ отдаём внешним конструкторам: управление коннекторами
# и секретами должно оставаться внутри Tellscope.
SKIP_GROUPS = {"connectors"}

# Ограничения ответа инструмента: Dify и LLM-узлы не переваривают мегабайты JSON.
MAX_PAYLOAD_CHARS = 24000
MAX_LIST_ITEMS = 30
MAX_STRING_CHARS = 4000

_ENV_TOKEN = (os.getenv("AGENT_SERVICE_TOKEN") or "").strip()
_ENV_USER_ID = int((os.getenv("AGENT_SERVICE_USER_ID") or "1").strip() or 1)

_cache: Dict[str, Any] = {"mtime": 0.0, "tokens": []}


# ----------------------------- хранилище токенов -----------------------------

def _read_store(force: bool = False) -> List[Dict[str, Any]]:
    """Читает файл токенов (с кэшем по времени изменения)."""
    try:
        mtime = os.path.getmtime(TOKENS_PATH)
    except OSError:
        return []
    if not force and mtime == _cache.get("mtime"):
        return list(_cache.get("tokens") or [])
    try:
        with open(TOKENS_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return []
    tokens = data.get("tokens") if isinstance(data, dict) else data
    if not isinstance(tokens, list):
        tokens = []
    _cache["mtime"] = mtime
    _cache["tokens"] = tokens
    return list(tokens)


def _write_store(tokens: List[Dict[str, Any]]) -> None:
    payload = {"tokens": tokens, "updated": int(time.time())}
    directory = os.path.dirname(TOKENS_PATH)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)
    tmp = TOKENS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(tmp, TOKENS_PATH)
    try:
        os.chmod(TOKENS_PATH, 0o600)
    except OSError:
        pass
    _cache["mtime"] = 0.0


def _mask(token: str) -> str:
    token = token or ""
    if len(token) <= 12:
        return token[:4] + "…"
    return token[:10] + "…" + token[-4:]


def list_tokens() -> List[Dict[str, Any]]:
    """Список выданных токенов без самих значений (для интерфейса)."""
    out = []
    for item in _read_store(force=True):
        out.append(
            {
                "id": item.get("id"),
                "name": item.get("name") or "без имени",
                "user_id": item.get("user_id"),
                "created": item.get("created"),
                "last_used": item.get("last_used"),
                "masked": _mask(str(item.get("token") or "")),
            }
        )
    if _ENV_TOKEN:
        out.append(
            {
                "id": "env",
                "name": "AGENT_SERVICE_TOKEN (переменная окружения)",
                "user_id": _ENV_USER_ID,
                "created": None,
                "last_used": None,
                "masked": _mask(_ENV_TOKEN),
            }
        )
    return out


def create_service_token(name: str = "dify", user_id: int = 1) -> Dict[str, Any]:
    """Выдаёт новый сервисный токен и сохраняет его в файл."""
    token = "ts_" + secrets.token_urlsafe(33)
    item = {
        "id": "tok_" + secrets.token_hex(4),
        "name": (name or "dify").strip()[:60],
        "user_id": int(user_id or 1),
        "token": token,
        "created": int(time.time()),
        "last_used": None,
    }
    tokens = _read_store(force=True)
    tokens.append(item)
    _write_store(tokens)
    return {"id": item["id"], "name": item["name"], "user_id": item["user_id"], "token": token}


def revoke_service_token(token_id: str) -> bool:
    tokens = _read_store(force=True)
    left = [item for item in tokens if str(item.get("id")) != str(token_id)]
    if len(left) == len(tokens):
        return False
    _write_store(left)
    return True


def match_service_token(token: str) -> Optional[Dict[str, Any]]:
    """Проверяет сервисный токен и возвращает владельца (id/имя), либо None."""
    token = (token or "").strip()
    if not token:
        return None
    if _ENV_TOKEN and secrets.compare_digest(token, _ENV_TOKEN):
        return {"id": "env", "name": "env", "user_id": _ENV_USER_ID}
    tokens = _read_store()
    for item in tokens:
        stored = str(item.get("token") or "")
        if stored and secrets.compare_digest(token, stored):
            try:
                if not item.get("last_used") or time.time() - float(item["last_used"]) > 300:
                    item["last_used"] = int(time.time())
                    _write_store(tokens)
                    _cache["mtime"] = 0.0
            except Exception:
                pass
            return {"id": item.get("id"), "name": item.get("name"), "user_id": int(item.get("user_id") or 1)}
    return None


# ----------------------------- описание инструментов -----------------------------

def _tool_specs() -> List[Any]:
    """Все инструменты реестра, кроме служебных групп."""
    import agent_engine  # noqa: F401 — импорт пакета регистрирует все инструменты
    from . import registry

    specs = [spec for spec in registry.all_tools().values() if spec.group not in SKIP_GROUPS]
    return sorted(specs, key=lambda spec: spec.name)


def public_tool_names() -> List[str]:
    return [spec.name for spec in _tool_specs()]


def _sanitize_schema(node: Any, depth: int = 0) -> Any:
    """Приводит JSON-схему к виду, который понимает парсер OpenAPI в Dify."""
    if isinstance(node, list):
        return [_sanitize_schema(item, depth + 1) for item in node]
    if not isinstance(node, dict):
        return node

    out: Dict[str, Any] = {}
    node_type = node.get("type")
    nullable = False
    if isinstance(node_type, list):
        types = [item for item in node_type if item != "null"]
        nullable = len(types) != len(node_type)
        node_type = types[0] if types else "string"
    for key, value in node.items():
        if key in ("$schema", "additionalProperties", "title", "examples", "example"):
            continue
        if key == "type":
            out["type"] = node_type if node_type in ("string", "integer", "number", "boolean", "array", "object") else "string"
        elif key == "required" and isinstance(value, list):
            if value:
                out["required"] = value
        elif key == "properties" and isinstance(value, dict):
            out["properties"] = {k: _sanitize_schema(v, depth + 1) for k, v in value.items()}
        elif key == "items":
            out["items"] = _sanitize_schema(value, depth + 1)
        elif key == "anyOf":
            variants = [_sanitize_schema(v, depth + 1) for v in (value or []) if isinstance(v, dict)]
            simple = [v for v in variants if v.get("type") in ("string", "integer", "number", "boolean")]
            if simple:
                out.update(simple[0])
                nullable = nullable or any(v.get("type") == "null" for v in variants)
            elif variants:
                out.update(variants[0])
        elif key in ("description", "default", "enum", "minimum", "maximum", "minLength", "maxLength", "minimum_length"):
            out[key] = value
    if nullable:
        out["nullable"] = True
    if out.get("type") == "object" and "properties" not in out:
        out.pop("type", None)
    return out


def openapi_spec(base_url: str) -> Dict[str, Any]:
    """OpenAPI 3.0 описание инструментов Tellscope (по одному маршруту на инструмент)."""
    specs = _tool_specs()
    paths: Dict[str, Any] = {}
    for spec in specs:
        schema = spec.parameters if isinstance(spec.parameters, dict) else {"type": "object", "properties": {}}
        schema = _sanitize_schema(schema)
        if schema.get("type") != "object":
            schema = {"type": "object", "properties": {}}
        schema.setdefault("properties", {})
        paths["/api/agent/tool/" + spec.name] = {
            "post": {
                "operationId": spec.name,
                "summary": spec.title or spec.name,
                "description": (spec.description or spec.title or spec.name).strip(),
                "tags": [spec.group],
                "requestBody": {
                    "required": True,
                    "content": {"application/json": {"schema": schema}},
                },
                "responses": {
                    "200": {
                        "description": "Результат инструмента в JSON",
                        "content": {"application/json": {"schema": {"type": "object"}}},
                    }
                },
            }
        }
    return {
        "openapi": "3.0.1",
        "info": {
            "title": "Tellscope — аналитика соцмедиа и СМИ",
            "version": "1.1.0",
            "description": (
                "Инструменты Tellscope для внешних конструкторов (Dify, n8n, GPT Actions): "
                "поиск сообщений, тональность, рейтинг СМИ, инфоповоды, подробный ИИ-разбор текстов "
                "и сборка отчётов DOCX/PDF. Авторизация — сервисный токен Tellscope в заголовке "
                "X-Service-Token."
            ),
        },
        "servers": [{"url": base_url.rstrip("/")}],
        "components": {
            "securitySchemes": {
                "TellscopeServiceToken": {"type": "apiKey", "in": "header", "name": "X-Service-Token"}
            }
        },
        "security": [{"TellscopeServiceToken": []}],
        "paths": paths,
    }


# ----------------------------- ужимание результата -----------------------------

def shrink(value: Any, depth: int = 0) -> Any:
    """Аккуратно ужимает большой результат: длинные списки и строки обрезаются с пометкой."""
    if depth > 6:
        return "…"
    if isinstance(value, str):
        if len(value) > MAX_STRING_CHARS:
            return value[:MAX_STRING_CHARS] + f"… [строка укорочена, было {len(value)} символов]"
        return value
    if isinstance(value, list):
        items = [shrink(item, depth + 1) for item in value[:MAX_LIST_ITEMS]]
        if len(value) > MAX_LIST_ITEMS:
            items.append(f"… [ещё {len(value) - MAX_LIST_ITEMS} элементов, список укорочен]")
        return items
    if isinstance(value, dict):
        return {str(key): shrink(item, depth + 1) for key, item in value.items()}
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return str(value)[:MAX_STRING_CHARS]


def dump_limited(payload: Any, limit: int = MAX_PAYLOAD_CHARS) -> Dict[str, Any]:
    """Возвращает JSON-совместимый ответ: при превышении лимита ужимает и предупреждает."""
    try:
        text = json.dumps(payload, ensure_ascii=False, default=str)
    except Exception:
        return {"ok": False, "error": "результат не удалось сериализовать"}
    if len(text) <= limit:
        return payload if isinstance(payload, dict) else {"result": payload}
    payload = shrink(payload, depth=0) if not isinstance(payload, dict) else {k: shrink(v, 1) for k, v in payload.items()}
    try:
        text = json.dumps(payload, ensure_ascii=False, default=str)
    except Exception:
        text = ""
    if len(text) > limit:
        while len(text) > limit and isinstance(payload, dict):
            biggest = None
            for key, value in payload.items():
                size = len(json.dumps(value, ensure_ascii=False, default=str))
                if biggest is None or size > biggest[1]:
                    biggest = (key, size)
            if not biggest or biggest[1] < 1200:
                break
            payload[biggest[0]] = "… [значение укорочено из-за размера ответа]"
            text = json.dumps(payload, ensure_ascii=False, default=str)
    out = payload if isinstance(payload, dict) else {"result": payload}
    out.setdefault("note", "ответ был укорочен из-за размера")
    return out
