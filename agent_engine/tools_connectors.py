# -*- coding: utf-8 -*-
"""Внешние коннекторы: пользовательские HTTP/MCP-инструменты.

Коннектор описывается JSON-файлом data/<user_id>/agent_connectors.json:
{
  "connectors": [
    {"name": "crm", "type": "http", "base_url": "https://crm.example.com/api",
     "headers": {"Authorization": "Bearer ..."}, "allow_paths": ["/leads"], "enabled": true},
    {"name": "weather", "type": "mcp", "url": "https://mcp.example.com/rpc", "token": "..."}
  ]
}

Правила безопасности: только https, запрет приватных адресов и localhost, лимит размера
ответа, список разрешённых путей у HTTP-коннектора, токены не возвращаются в интерфейс.
"""
from __future__ import annotations

import ipaddress
import json
import os
import socket
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

from .context import compact
from .registry import ToolError, tool

BACKEND_ROOT = "/home/dev/tellscope_app/tellscope_backend"
MAX_RESPONSE_BYTES = 200_000


def connectors_path(user_id: Any) -> str:
    return os.path.join(BACKEND_ROOT, "data", str(user_id), "agent_connectors.json")


def load_connectors(user_id: Any) -> List[Dict[str, Any]]:
    path = connectors_path(user_id)
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh) or {}
    except Exception:
        return []
    items = data.get("connectors") if isinstance(data, dict) else data
    return [it for it in (items or []) if isinstance(it, dict) and it.get("name")]


def save_connectors(user_id: Any, items: List[Dict[str, Any]]) -> None:
    path = connectors_path(user_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"connectors": items}, fh, ensure_ascii=False, indent=2)
    try:
        os.chmod(path, 0o600)
    except Exception:
        pass


def find_connector(user_id: Any, name: str) -> Dict[str, Any]:
    for item in load_connectors(user_id):
        if str(item.get("name")).lower() == str(name or "").lower():
            return item
    raise ToolError(f"Коннектор «{name}» не найден")


def _check_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ToolError("Разрешены только https-адреса коннекторов")
    host = parsed.hostname or ""
    if not host:
        raise ToolError("Некорректный адрес коннектора")
    try:
        infos = socket.getaddrinfo(host, parsed.port or 443)
    except Exception as exc:
        raise ToolError(f"Не удалось разрешить адрес {host}: {exc}") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise ToolError("Запрещены адреса во внутренней сети")


@tool(
    "list_connectors",
    title="Внешние коннекторы",
    description="Список подключённых пользователем внешних инструментов (HTTP/MCP) с их назначением.",
    parameters={"type": "object", "properties": {}},
    group="connectors",
)
async def list_connectors(ctx):
    items = load_connectors(ctx.user_id)
    safe = []
    for item in items:
        safe.append(
            {
                "name": item.get("name"),
                "type": item.get("type") or "http",
                "base_url": item.get("base_url") or item.get("url"),
                "description": item.get("description") or "",
                "enabled": item.get("enabled", True),
            }
        )
    return {"connectors": compact(safe, max_items=30), "total": len(safe)}


@tool(
    "connector_request",
    title="Запрос во внешний сервис",
    description=(
        "Выполняет HTTP-запрос через подключённый пользователем коннектор (например, CRM, склад, BI). "
        "Секреты коннектора подставляются автоматически. Доступны только коннекторы типа http."
    ),
    parameters={
        "type": "object",
        "properties": {
            "connector": {"type": "string", "description": "имя коннектора"},
            "path": {"type": "string", "description": "путь относительно base_url, например /leads?limit=10"},
            "method": {"type": "string", "enum": ["GET", "POST"], "description": "HTTP-метод"},
            "body": {"type": "object", "description": "JSON-тело запроса для POST"},
        },
        "required": ["connector", "path"],
    },
    group="connectors",
    scope="write",
    default_enabled=False,
    timeout=120.0,
)
async def connector_request(ctx, connector: str, path: str, method: str = "GET", body: Optional[Dict[str, Any]] = None):
    import httpx

    item = find_connector(ctx.user_id, connector)
    if (item.get("type") or "http") != "http":
        raise ToolError(f"Коннектор «{connector}» не является HTTP-коннектором")
    if item.get("enabled") is False:
        raise ToolError(f"Коннектор «{connector}» отключён")
    base = str(item.get("base_url") or "").rstrip("/") + "/"
    url = urljoin(base, str(path or "").lstrip("/"))
    _check_url(url)
    allow_paths = [str(p) for p in (item.get("allow_paths") or [])]
    if allow_paths:
        parsed_path = urlparse(url).path
        if not any(parsed_path.startswith(p) for p in allow_paths):
            raise ToolError(f"Путь {parsed_path} не разрешён для коннектора «{connector}»")
    headers = {str(k): str(v) for k, v in (item.get("headers") or {}).items()}
    token = item.get("token")
    if token and "Authorization" not in headers:
        headers["Authorization"] = f"Bearer {token}"
    try:
        async with httpx.AsyncClient(timeout=60, follow_redirects=False) as client:
            resp = await client.request(str(method or "GET").upper(), url, headers=headers, json=body)
    except Exception as exc:
        raise ToolError(f"Ошибка запроса к коннектору: {exc}") from exc
    text = resp.text[:MAX_RESPONSE_BYTES]
    payload: Any = text
    try:
        payload = json.loads(text)
    except Exception:
        payload = text[:4000]
    return {"connector": connector, "status": resp.status_code, "url": url, "response": compact(payload, max_items=25, max_str=400)}


@tool(
    "mcp_call",
    title="Вызов MCP-инструмента",
    description=(
        "Вызывает инструмент внешнего MCP-сервера, подключённого пользователем (JSON-RPC tools/call). "
        "Используй, когда нужны данные из внешних систем, подключённых через MCP."
    ),
    parameters={
        "type": "object",
        "properties": {
            "connector": {"type": "string", "description": "имя MCP-коннектора"},
            "tool": {"type": "string", "description": "имя инструмента на MCP-сервере"},
            "arguments": {"type": "object", "description": "аргументы инструмента"},
        },
        "required": ["connector", "tool"],
    },
    group="connectors",
    scope="write",
    default_enabled=False,
    timeout=180.0,
)
async def mcp_call(ctx, connector: str, tool: str, arguments: Optional[Dict[str, Any]] = None):
    import httpx

    item = find_connector(ctx.user_id, connector)
    if (item.get("type") or "").lower() != "mcp":
        raise ToolError(f"Коннектор «{connector}» не является MCP-сервером")
    url = str(item.get("url") or "")
    _check_url(url)
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if item.get("token"):
        headers["Authorization"] = f"Bearer {item['token']}"
    payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": tool, "arguments": arguments or {}}}
    try:
        async with httpx.AsyncClient(timeout=120, follow_redirects=False) as client:
            resp = await client.post(url, headers=headers, json=payload)
    except Exception as exc:
        raise ToolError(f"Ошибка вызова MCP-инструмента: {exc}") from exc
    try:
        data = resp.json()
    except Exception:
        data = {"raw": resp.text[:4000]}
    if isinstance(data, dict) and data.get("error"):
        raise ToolError(f"MCP вернул ошибку: {data['error']}")
    return {"connector": connector, "tool": tool, "status": resp.status_code, "result": compact(data.get("result") if isinstance(data, dict) else data, max_items=25, max_str=400)}


@tool(
    "mcp_list_tools",
    title="Инструменты MCP-сервера",
    description="Возвращает список инструментов, которые предоставляет подключённый MCP-сервер (JSON-RPC tools/list).",
    parameters={
        "type": "object",
        "properties": {"connector": {"type": "string", "description": "имя MCP-коннектора"}},
        "required": ["connector"],
    },
    group="connectors",
    default_enabled=False,
    timeout=120.0,
)
async def mcp_list_tools(ctx, connector: str):
    import httpx

    item = find_connector(ctx.user_id, connector)
    url = str(item.get("url") or "")
    _check_url(url)
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if item.get("token"):
        headers["Authorization"] = f"Bearer {item['token']}"
    payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    try:
        async with httpx.AsyncClient(timeout=60, follow_redirects=False) as client:
            resp = await client.post(url, headers=headers, json=payload)
        data = resp.json()
    except Exception as exc:
        raise ToolError(f"Ошибка запроса списка инструментов: {exc}") from exc
    tools_list = ((data or {}).get("result") or {}).get("tools") or []
    return {"connector": connector, "tools": compact(tools_list, max_items=40, max_str=200)}
