# -*- coding: utf-8 -*-
"""Контекст одного агентного запуска: пользователь, датасет, лимиты, артефакты."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set

BACKEND_ROOT = "/home/dev/tellscope_app/tellscope_backend"
AGENT_RUNS_ROOT = os.path.join(BACKEND_ROOT, "data", "agent_runs")


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def to_unix(value: Any) -> Optional[int]:
    """Приводит 'YYYY-MM-DD', ISO-строку или число к unix-секундам."""
    if value in (None, "", "null"):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%d.%m.%Y"):
        try:
            return int(datetime.strptime(text[: len(fmt) + 3 if fmt.endswith("%S") else len(fmt)], fmt).replace(tzinfo=timezone.utc).timestamp())
        except Exception:
            continue
    try:
        cleaned = text.replace("Z", "+00:00")
        return int(datetime.fromisoformat(cleaned).timestamp())
    except Exception:
        return None


def compact(value: Any, max_items: int = 15, max_str: int = 260, depth: int = 0, max_depth: int = 5) -> Any:
    """Сжимает структуру ответа до размера, пригодного для передачи в LLM."""
    if depth > max_depth:
        return "..."
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for key, item in value.items():
            if item in (None, "", [], {}):
                continue
            out[str(key)] = compact(item, max_items, max_str, depth + 1, max_depth)
        return out
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        trimmed = [compact(item, max_items, max_str, depth + 1, max_depth) for item in items[:max_items]]
        if len(items) > max_items:
            trimmed.append({"_total_items": len(items), "_shown": max_items})
        return trimmed
    if isinstance(value, str):
        text = value.strip()
        return text if len(text) <= max_str else text[:max_str] + "…"
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    try:
        return compact(dict(value), max_items, max_str, depth + 1, max_depth)
    except Exception:
        return str(value)[:max_str]


@dataclass
class AgentContext:
    """Всё, что нужно инструментам и циклу агента во время одного запуска."""

    run_id: str
    user: Any
    user_id: str
    task: str
    dataset_index: Optional[int] = None
    dataset_name: Optional[str] = None
    dataset_label: str = ""
    min_date: Optional[int] = None
    max_date: Optional[int] = None
    allowed_tools: Set[str] = field(default_factory=set)
    model_choice: str = "claude"
    folder: str = "Агент"
    emit: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None
    artifacts_dir: str = ""
    deadline: float = 0.0

    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    charts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Раздел подробного разбора (deep_text_analysis) — обязан попасть в итоговый отчёт
    deep_analysis: Optional[Dict[str, Any]] = None
    # Чтение текстов (analyze_texts): темы и цитаты, обязательные для отчёта
    text_analysis: Optional[Dict[str, Any]] = None
    # Сколько негативных сообщений в срезе — нужно для правила качества отчёта
    negative_in_slice: int = 0
    # Заполняется build_report: в срезе есть негатив, но в отчёте нет темы с цитатой
    text_gap: str = ""
    llm_calls: int = 0
    tokens: int = 0
    cost_usd: float = 0.0
    token_budget: int = 0
    notes: List[str] = field(default_factory=list)

    def time_left(self) -> float:
        if not self.deadline:
            return 10 ** 6
        return self.deadline - time.time()

    def out_of_time(self) -> bool:
        return self.time_left() <= 5

    async def event(self, payload: Dict[str, Any]) -> None:
        if self.emit is None:
            return
        data = {"ts": now_iso(), **payload}
        try:
            await self.emit(data)
        except Exception:
            pass

    async def log(self, message: str, level: str = "info") -> None:
        await self.event({"type": "log", "level": level, "message": message})

    def add_artifact(self, kind: str, title: str, path: str, url: Optional[str] = None, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        item = {
            "kind": kind,
            "title": title,
            "name": os.path.basename(path),
            "path": path,
            "url": url,
            "meta": meta or {},
        }
        self.artifacts.append(item)
        return item

    def registry_tool(self, tool_id: str) -> Dict[str, Any]:
        return {
            "id": tool_id,
            "run_id": self.run_id,
            "kind": "chart",
        }
