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


def local_tz():
    """Часовой пояс пользователя: системная зона процесса (на проде — Europe/Moscow, UTC+3)."""
    return datetime.now(timezone.utc).astimezone().tzinfo or timezone.utc


LOCAL_TZ = local_tz()

DATE_FORMATS = ("%Y-%m-%d", "%d.%m.%Y")
DATETIME_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M", "%d.%m.%Y %H:%M:%S")


def _local_day(day: datetime, end: bool) -> int:
    """Сутки в локальной зоне: 00:00:00 для начала периода, 23:59:59 для конца."""
    if end:
        day = day.replace(hour=23, minute=59, second=59)
    return int(day.replace(tzinfo=LOCAL_TZ).timestamp())


def _parse_datetime(text: str) -> Optional[datetime]:
    """Дата со временем: смещение ('+03:00', 'Z') учитывается, без смещения — локальная зона."""
    candidate = text[:-1] + "+00:00" if text.endswith("Z") else text
    parsed: Optional[datetime] = None
    try:
        parsed = datetime.fromisoformat(candidate)
    except Exception:
        parsed = None
    if parsed is None:
        for fmt in DATETIME_FORMATS:
            try:
                parsed = datetime.strptime(text, fmt)
                break
            except Exception:
                continue
    if parsed is None:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=LOCAL_TZ)


def to_unix(value: Any, end: bool = False) -> Optional[int]:
    """Приводит 'YYYY-MM-DD', ISO-строку или число к unix-секундам.

    Дата без времени — это полные сутки в локальном времени пользователя (MSK, UTC+3):
    начало суток (00:00:00) для min_date и конец суток (23:59:59) для max_date (end=True).
    Иначе терялись бы сообщения, попавшие в первые часы суток.
    """
    if value in (None, "", "null"):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip()
    if not text or text in ("null", "None"):
        return None
    if text.isdigit():
        return int(text)
    if len(text) == 10:  # только дата — границы суток в локальной зоне
        for fmt in DATE_FORMATS:
            try:
                return _local_day(datetime.strptime(text, fmt), end)
            except Exception:
                continue
    parsed = _parse_datetime(text)
    if parsed is not None:
        return int(parsed.timestamp())
    for fmt in DATE_FORMATS:  # нестандартная запись даты, например '2026-9-10'
        try:
            return _local_day(datetime.strptime(text, fmt), end)
        except Exception:
            continue
    try:
        return int(float(text))
    except Exception:
        return None


def to_unix_start(value: Any) -> Optional[int]:
    """Начало периода: дата без времени — 00:00:00 локального времени."""
    return to_unix(value, end=False)


def to_unix_end(value: Any) -> Optional[int]:
    """Конец периода: дата без времени — 23:59:59 локального времени (сутки включительно)."""
    return to_unix(value, end=True)


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
    # Модель-оркестратор (планирование шагов и вызовы инструментов) берётся из единой настройки
    # mlops.lock.agent_cfg: orchestrator + orchestrator_fallbacks. Может смениться на ходу,
    # поэтому храним и активный ключ, и цепочку отказов, и причину переключения.
    orchestrator_choice: str = ""
    orchestrator_chain: List[str] = field(default_factory=list)
    orchestrator_info: Dict[str, Any] = field(default_factory=dict)
    orchestrator_probe_mode: str = "tools"
    # Ключ модели, ответившей последней: нужен для честного учёта цены прогона
    # (оркестратор и модель анализа могут быть разными).
    last_choice_key: str = ""
    # Сколько вызовов и токенов ушло на каждую модель в этом запуске.
    models_used: Dict[str, Dict[str, Any]] = field(default_factory=dict)
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
