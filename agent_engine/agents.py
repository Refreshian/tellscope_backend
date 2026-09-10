# -*- coding: utf-8 -*-
"""Конструктор агентов Tellscope: сохранённые агенты, готовые шаблоны, расписания.

Агент — это сохранённая конфигурация: инструкция, набор инструментов, модель, бюджет,
датасет, папка отчётов и расписание. Запуск агента создаёт обычный прогон агентного режима
(agent_engine.runs), поэтому лимиты, стоимость и журнал шагов остаются общими.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional

BACKEND_ROOT = "/home/dev/tellscope_app/tellscope_backend"

WEEKDAYS = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]

DEFAULT_TOOLS = [
    "dataset_overview",
    "search_messages",
    "tonality_summary",
    "popular_hooks",
    "media_rating",
    "make_chart",
    "build_report",
]

# Готовые шаблоны агентов: пользователь добавляет их одним кликом и правит под себя.
PRESETS: List[Dict[str, Any]] = [
    {
        "id": "negative_watch",
        "name": "Мониторинг негатива и жалоб",
        "description": "Ежедневно проверяет негатив и жалобы: где, сколько, по каким темам, кто пишет, что делать.",
        "instruction": (
            "Проверь негативные сообщения и жалобы за последние сутки доступного периода. Если датасет с фиксированным "
            "периодом, бери последние сутки этого периода, а если данных за сутки мало — последнюю неделю периода. "
            "Нужно: 1) сколько негативных сообщений и как это отличается от среднего уровня по датасету; 2) площадки и города, "
            "где негатив сконцентрирован; 3) повторяющиеся темы жалоб (качество еды, отравления, обслуживание, очереди, цены, "
            "доставка) с количеством по каждой; 4) пять конкретных примеров сообщений со ссылками и датами; "
            "5) если есть всплеск — определи первоисточник через цепочку распространения и назови авторов, которые разогнали волну. "
            "По самой частой теме жалоб сделай подробный разбор инструментом deep_text_analysis: смысловые блоки, конкретные претензии "
            "с цитатами и ссылками, детали (продукты, точки, суммы). "
            "Собери короткий отчёт с графиком динамики негатива и выводами: что случилось, где остро, что делать в первую очередь."
        ),
        "tools": [
            "dataset_overview",
            "search_messages",
            "tonality_summary",
            "deep_text_analysis",
            "chain_graph",
            "popular_hooks",
            "media_rating",
            "make_chart",
            "build_report",
        ],
        "folder": "Мониторинг негатива",
        "token_budget": 120000,
        "schedule": {"enabled": True, "mode": "daily", "hour": 9, "minute": 0, "weekdays": [1, 2, 3, 4, 5, 6, 7]},
    },
    {
        "id": "brand_report",
        "name": "Сводный отчёт по бренду",
        "description": "Регулярный отчёт: динамика, тональность, площадки, инфоповоды, примеры и выводы.",
        "instruction": (
            "Собери регулярный сводный отчёт по бренду за период. Разделы: 1) динамика упоминаний по месяцам и её изменение; "
            "2) тональность: доли и как меняются; 3) площадки: топ-5 и их вклад; 4) ключевые инфоповоды с примерами сообщений "
            "и ссылками; 5) активные авторы и каналы; 6) выводы и рекомендации. Построй графики (динамика, тональность, площадки) "
            "и сохрани отчёт DOCX/PDF в папку «Отчёт по бренду». В чат верни краткое резюме с ключевыми цифрами."
        ),
        "tools": [
            "dataset_overview",
            "tonality_summary",
            "popular_hooks",
            "search_messages",
            "media_rating",
            "make_chart",
            "build_report",
        ],
        "folder": "Отчёт по бренду",
        "token_budget": 150000,
        "schedule": {"enabled": True, "mode": "weekly", "hour": 9, "minute": 0, "weekdays": [1]},
    },
    {
        "id": "competitor_watch",
        "name": "Сравнение с конкурентами",
        "description": "Сравнивает бренд и конкурентов по объёму, тональности, площадкам и темам.",
        "instruction": (
            "Сравни наш бренд и конкурентов в датасете. Для каждого: объём упоминаний, тональность (доли негатива и позитива), "
            "площадки, ключевые темы и инфоповоды. Покажи сравнительную таблицу, отметь, где мы выигрываем и где проигрываем, "
            "приведи примеры сообщений со ссылками по каждому бренду, построй сравнительные графики и собери отчёт в папку "
            "«Конкуренты». Если по какому-то бренду данных мало — скажи об этом прямо."
        ),
        "tools": [
            "list_datasets",
            "dataset_overview",
            "search_messages",
            "tonality_summary",
            "popular_hooks",
            "media_rating",
            "make_chart",
            "build_report",
        ],
        "folder": "Конкуренты",
        "token_budget": 150000,
        "schedule": {"enabled": True, "mode": "weekly", "hour": 10, "minute": 0, "weekdays": [1]},
    },
    {
        "id": "topic_watch",
        "name": "Слежение за инфоповодом",
        "description": "Отслеживает конкретную тему: всплески, первоисточник, кто разогнал, динамика.",
        "instruction": (
            "Проверь активность по теме: подставьте свою тему в инструкции агента. Нужно: сколько сообщений, динамика по дням, "
            "кто написал первым, кто разогнал волну (цепочка распространения: первоисточник и распространители), площадки и "
            "авторы, тональность. Обязательно сделай подробный разбор текстов инструментом deep_text_analysis: смысловые блоки, "
            "о чём именно пишут, конкретные претензии или аргументы с цитатами и ссылками, детали (места, суммы, организации). "
            "Если активности нет — ответь одной строкой, что тема не обсуждается, и не собирай отчёт."
        ),
        "tools": [
            "search_messages",
            "deep_text_analysis",
            "chain_graph",
            "popular_hooks",
            "information_graph",
            "make_chart",
            "build_report",
        ],
        "folder": "Инфоповоды",
        "token_budget": 120000,
        "schedule": {"enabled": True, "mode": "daily", "hour": 12, "minute": 0, "weekdays": [1, 2, 3, 4, 5, 6, 7]},
    },
]

_PRESET_BY_ID = {preset["id"]: preset for preset in PRESETS}


def store_path(user_id: Any) -> str:
    return os.path.join(BACKEND_ROOT, "data", str(user_id), "agent_agents.json")


def list_agents(user_id: Any) -> List[Dict[str, Any]]:
    path = store_path(user_id)
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh) or {}
    except Exception:
        return []
    items = data.get("agents") if isinstance(data, dict) else data
    return [item for item in (items or []) if isinstance(item, dict) and item.get("id")]


def save_agents(user_id: Any, items: List[Dict[str, Any]]) -> None:
    path = store_path(user_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"agents": items}, fh, ensure_ascii=False, indent=2)


def get_agent(user_id: Any, agent_id: str) -> Optional[Dict[str, Any]]:
    for item in list_agents(user_id):
        if str(item.get("id")) == str(agent_id):
            return item
    return None


def normalize_schedule(raw: Any, base: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    base = base or {"enabled": False, "mode": "manual", "hour": 9, "minute": 0, "weekdays": list(range(1, 8))}
    if not isinstance(raw, dict):
        return base
    mode = str(raw.get("mode") or base.get("mode") or "manual")
    if mode not in ("manual", "daily", "weekly"):
        mode = "manual"
    try:
        hour = max(0, min(int(raw.get("hour", base.get("hour", 9))), 23))
    except Exception:
        hour = 9
    try:
        minute = max(0, min(int(raw.get("minute", base.get("minute", 0))), 59))
    except Exception:
        minute = 0
    weekdays = raw.get("weekdays") or base.get("weekdays") or list(range(1, 8))
    try:
        weekdays = sorted({max(1, min(int(day), 7)) for day in weekdays})
    except Exception:
        weekdays = list(range(1, 8))
    if not weekdays:
        weekdays = list(range(1, 8))
    return {
        "enabled": bool(raw.get("enabled", base.get("enabled", False))),
        "mode": mode,
        "hour": hour,
        "minute": minute,
        "weekdays": weekdays,
    }


def upsert_agent(user_id: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    items = list_agents(user_id)
    agent_id = str(payload.get("id") or "").strip()
    existing = next((item for item in items if str(item.get("id")) == agent_id), None) if agent_id else None
    now = time.strftime("%Y-%m-%d %H:%M:%S")

    if existing is None:
        preset = _PRESET_BY_ID.get(str(payload.get("preset") or ""))
        base = dict(preset or {})
        agent_id = agent_id or ("ag_" + uuid.uuid4().hex[:10])
        record = {
            "id": agent_id,
            "user_id": str(user_id),
            "created_at": now,
            "last_run_at": None,
            "last_run_id": None,
            "preset": payload.get("preset") or base.get("id"),
            "name": payload.get("name") or base.get("name") or "Новый агент",
            "description": payload.get("description") or base.get("description") or "",
            "instruction": payload.get("instruction") or base.get("instruction") or "",
            "tools": payload.get("tools") or base.get("tools") or DEFAULT_TOOLS,
            "model": payload.get("model") or base.get("model") or "gpt",
            "token_budget": payload.get("token_budget") or base.get("token_budget") or 120000,
            "dataset_index": payload.get("dataset_index"),
            "dataset_name": payload.get("dataset_name") or "",
            "folder": payload.get("folder") or base.get("folder") or "Агент",
            "schedule": normalize_schedule(payload.get("schedule") if payload.get("schedule") is not None else base.get("schedule")),
            "enabled": bool(payload.get("enabled", True)),
        }
        items.append(record)
    else:
        record = existing
        for key in ("name", "description", "instruction", "folder", "model", "dataset_name"):
            if payload.get(key) is not None:
                record[key] = payload[key]
        if payload.get("tools") is not None:
            record["tools"] = payload["tools"]
        if payload.get("token_budget") is not None:
            try:
                record["token_budget"] = int(payload["token_budget"])
            except Exception:
                pass
        if payload.get("dataset_index") is not None:
            record["dataset_index"] = payload["dataset_index"]
        if payload.get("schedule") is not None:
            record["schedule"] = normalize_schedule(payload["schedule"], record.get("schedule"))
        if payload.get("enabled") is not None:
            record["enabled"] = bool(payload["enabled"])
    record["updated_at"] = now
    save_agents(user_id, items)
    return record


def delete_agent(user_id: Any, agent_id: str) -> bool:
    items = list_agents(user_id)
    left = [item for item in items if str(item.get("id")) != str(agent_id)]
    if len(left) == len(items):
        return False
    save_agents(user_id, left)
    return True


def mark_run(user_id: Any, agent_id: str, run_id: str) -> None:
    items = list_agents(user_id)
    for item in items:
        if str(item.get("id")) == str(agent_id):
            item["last_run_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            item["last_run_id"] = run_id
            break
    save_agents(user_id, items)


def _slot_key(now: time.struct_time, schedule: Dict[str, Any]) -> str:
    if schedule.get("mode") == "weekly":
        return time.strftime("%Y-%m-%d", now)
    return time.strftime("%Y-%m-%d", now)


def is_due(agent: Dict[str, Any], now: Optional[time.struct_time] = None) -> bool:
    """Пора ли запускать агента по расписанию."""
    if not agent.get("enabled"):
        return False
    schedule = normalize_schedule(agent.get("schedule"))
    if not schedule.get("enabled") or schedule.get("mode") == "manual":
        return False
    now = now or time.localtime()
    if time.mktime(now) < time.mktime((now.tm_year, now.tm_mon, now.tm_mday, schedule["hour"], schedule["minute"], 0, 0, 0, -1)):
        return False
    if schedule["mode"] == "weekly":
        iso_weekday = now.tm_wday + 1
        if iso_weekday not in schedule["weekdays"]:
            return False
    last = str(agent.get("last_run_at") or "")
    if last.startswith(time.strftime("%Y-%m-%d", now)):
        return False
    return True


def due_agents(user_id: Any, now: Optional[time.struct_time] = None) -> List[Dict[str, Any]]:
    return [agent for agent in list_agents(user_id) if is_due(agent, now)]


async def _load_user(user_id: Any):
    """Пользователь из БД для планового запуска (нужен для проверки доступа к датасету)."""
    import main

    from auth.database import User as AuthUser

    async with main.async_session_maker() as session:
        return await session.get(AuthUser, int(user_id))


def start_agent_run(user_id: Any, agent: Dict[str, Any], user: Any, main_loop: Any = None) -> Dict[str, Any]:
    """Создаёт и запускает прогон по конфигурации агента."""
    from . import runs as agent_runs

    run = agent_runs.create_run(
        user=user,
        user_id=str(user_id),
        task=str(agent.get("instruction") or agent.get("name") or ""),
        dataset_index=agent.get("dataset_index"),
        dataset_name=agent.get("dataset_name") or "",
        dataset_label=agent.get("dataset_name") or "",
        tools=agent.get("tools"),
        model_choice=str(agent.get("model") or "gpt"),
        folder=str(agent.get("folder") or "Агент"),
        token_budget=agent.get("token_budget"),
    )
    run["agent_id"] = agent.get("id")
    run["agent_name"] = agent.get("name")
    # Пользователь обязателен: инструменты проверяют по нему доступ к датасету
    run["_user"] = user
    if main_loop is not None:
        future = asyncio.run_coroutine_threadsafe(agent_runs.execute_run(run["run_id"], main_loop), agent_runs.worker_loop())
        agent_runs._ACTIVE_TASKS[run["run_id"]] = future
    else:
        agent_runs.start_run(run, user)
    mark_run(user_id, str(agent.get("id")), run["run_id"])
    return run


def user_ids_with_agents() -> List[str]:
    """Все пользователи, у которых есть сохранённые агенты (для планировщика)."""
    root = os.path.join(BACKEND_ROOT, "data")
    out = []
    if not os.path.isdir(root):
        return out
    for name in os.listdir(root):
        if os.path.isfile(os.path.join(root, name, "agent_agents.json")):
            out.append(name)
    return out


async def _tick(main_loop: Any) -> None:
    from . import runs as agent_runs

    now = time.localtime()
    for user_id in user_ids_with_agents():
        try:
            due = due_agents(user_id, now)
            if not due:
                continue
            if agent_runs.active_runs_for_user(user_id):
                continue  # у пользователя уже идёт прогон
            if agent_runs.tokens_today_for_user(user_id) >= agent_runs.MAX_TOKENS_PER_DAY:
                continue
            user = await _load_user(user_id)
            if user is None:
                continue
            for agent in due:
                start_agent_run(user_id, agent, user, main_loop)
                print(f"[agent-scheduler] запущен агент «{agent.get('name')}» для пользователя {user_id}")
        except Exception as exc:
            print(f"[agent-scheduler] ошибка для пользователя {user_id}: {exc}")


async def _scheduler(main_loop: Any) -> None:
    await asyncio.sleep(20)
    while True:
        try:
            await _tick(main_loop)
        except Exception as exc:
            print(f"[agent-scheduler] tick error: {exc}")
        await asyncio.sleep(60)


def start_scheduler(main_loop: Any) -> None:
    """Поднимает планировщик в рабочем цикле агентного режима (один раз за процесс)."""
    from .runs import worker_loop

    loop = worker_loop()
    asyncio.run_coroutine_threadsafe(_scheduler(main_loop), loop)
    print("[agent-scheduler] планировщик запущен")


def presets_public() -> List[Dict[str, Any]]:
    out = []
    for preset in PRESETS:
        item = dict(preset)
        item["schedule_text"] = describe_schedule(item.get("schedule") or {})
        out.append(item)
    return out


def describe_schedule(schedule: Dict[str, Any]) -> str:
    schedule = normalize_schedule(schedule)
    if not schedule.get("enabled") or schedule.get("mode") == "manual":
        return "только вручную"
    at = f"{schedule['hour']:02d}:{schedule['minute']:02d}"
    if schedule["mode"] == "daily":
        return f"ежедневно в {at}"
    days = ", ".join(WEEKDAYS[day - 1] for day in schedule["weekdays"])
    return f"еженедельно ({days}) в {at}"
