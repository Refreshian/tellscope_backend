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
        "steps": [
            {"kind": "tool", "title": "Обзор датасета", "tool": "dataset_overview",
             "args": {"top_n": 10}, "save_as": "overview"},
            {"kind": "tool", "title": "Тональность и площадки", "tool": "tonality_summary",
             "save_as": "tonality"},
            {"kind": "tool", "title": "Ключевые инфоповоды", "tool": "popular_hooks",
             "args": {"limit": 12}, "save_as": "hooks"},
            {"kind": "chart", "title": "Динамика упоминаний по месяцам", "from": "{{overview.monthly_dynamics}}",
             "label_field": "month", "value_field": "count", "chart_type": "line",
             "series_name": "Сообщений", "save_as": "chart_dynamics"},
            {"kind": "chart", "title": "Тональность: доли", "from": "{{overview.tonality}}",
             "label_field": "tone", "value_field": "count", "chart_type": "pie",
             "series_name": "Сообщений", "save_as": "chart_tone"},
            {"kind": "chart", "title": "Топ площадок", "from": "{{overview.hubs}}",
             "label_field": "key", "value_field": "count", "chart_type": "hbar",
             "series_name": "Сообщений", "save_as": "chart_hubs"},
            {"kind": "llm", "title": "Выводы аналитика", "save_as": "synthesis", "max_tokens": 1600,
             "prompt": "Данные по бренду за период.\n\nОбъём и период: {{overview.messages_total}} сообщений, {{overview.period}}.\n\nДинамика по месяцам: {{overview.monthly_dynamics}}\n\nТональность: {{overview.tonality}}\n\nПлощадки: {{overview.hubs}}\n\nИнфоповоды: {{hooks.hooks}}\n\nНапиши аналитический разбор: 1) динамика упоминаний и что изменилось; 2) тональность; 3) площадки; 4) ключевые инфоповоды; 5) выводы и рекомендации. Только по этим данным, без вводных фраз."},
            {"kind": "report", "title": "Сборка отчёта", "report_title": "Сводный отчёт по бренду",
             "subtitle": "Динамика, тональность, площадки, инфоповоды", "save_as": "report",
             "sections": [{"heading": "Аналитика и выводы", "text": "{{synthesis.text}}",
                           "chart_ids": ["chart1", "chart2", "chart3"]}]},
        ],
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
            "Итог оформи документом: построй график динамики упоминаний и вызови build_report (DOCX и PDF), включив разбор отдельным разделом. "
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
    {
        "id": "topic_pipeline",
        "name": "Отчёт по теме (цепочка шагов)",
        "description": "Готовая цепочка: поиск по теме → подробный разбор текстов → график → выводы ИИ → отчёт DOCX/PDF. Тему укажите в первых двух шагах.",
        "instruction": (
            "Сделай подробный анализ указанной темы: о чём пишут, на что жалуются, какие детали, примеры со ссылками, "
            "и собери отчёт."
        ),
        "tools": [
            "search_messages",
            "deep_text_analysis",
            "popular_hooks",
            "make_chart",
            "build_report",
        ],
        "folder": "Отчёт по теме",
        "token_budget": 120000,
        "schedule": {"enabled": False, "mode": "manual", "hour": 9, "minute": 0, "weekdays": [1, 2, 3, 4, 5]},
        "steps": [
            {"kind": "tool", "title": "Поиск сообщений по теме", "tool": "search_messages",
             "args": {"phrase": "отравление", "limit": 30, "sort": "relevance"}, "save_as": "search"},
            {"kind": "tool", "title": "Подробный разбор текстов", "tool": "deep_text_analysis",
             "args": {"phrase": "отравление", "focus": "претензии, детали, продукты", "max_messages": 64},
             "save_as": "deep"},
            {"kind": "chart", "title": "Динамика упоминаний по месяцам", "from": "{{search.monthly_dynamics}}",
             "label_field": "month", "value_field": "count", "chart_type": "line",
             "series_name": "Сообщений", "save_as": "chart_dynamics"},
            {"kind": "llm", "title": "Выводы по теме", "save_as": "synthesis", "max_tokens": 1600,
             "prompt": "Тема разбора.\n\nНайдено сообщений: {{search.messages_found}}, формулировки поиска: {{search.search_terms}}, динамика: {{search.monthly_dynamics}}\n\nПодробный разбор текстов: {{deep.summary}}\n\nСмысловые блоки: {{deep.subtopics}}\n\nПретензии с цитатами: {{deep.key_claims}}\n\nНапиши разбор темы: о чём пишут, на что жалуются, ключевые детали и факты, 2–3 примера со ссылками, выводы. Только по этим данным."},
            {"kind": "report", "title": "Сборка отчёта", "report_title": "Отчёт по теме",
             "subtitle": "Подробный разбор сообщений по теме", "save_as": "report",
             "sections": [{"heading": "Разбор темы", "text": "{{synthesis.text}}", "chart_ids": ["chart1"]}]},
        ],
    },
    {
        "id": "subtopic_report",
        "name": "Отчёт по подтеме: тональность, авторы, цепочки, негатив и позитив",
        "description": (
            "Готовая цепочка из 14 шагов: поисковые слова подтемы → ИИ-аналитика → тональность, авторы и рейтинг СМИ, "
            "цепочки авторов, примеры негатива и позитива → три графика → выводы ИИ → отчёт DOCX/PDF. "
            "Свою подтему (поисковые слова) укажите в шагах 1, 4, 5, 6 и 7 вместо слова «просрочка»."
        ),
        "instruction": (
            "Собери отчёт по подтеме: тональность, авторы, цепочки авторов, негатив, позитив и выводы."
        ),
        "tools": [
            "search_messages",
            "tonality_summary",
            "media_rating",
            "chain_graph",
            "ai_analytics",
            "make_chart",
            "build_report",
        ],
        "folder": "Отчёт по подтеме",
        "token_budget": 150000,
        "schedule": {"enabled": False, "mode": "manual", "hour": 9, "minute": 0, "weekdays": [1, 2, 3, 4, 5]},
        "steps": [
            {"kind": "tool", "title": "1. Поиск сообщений по подтеме", "tool": "search_messages",
             "args": {"phrase": "просрочка", "limit": 30, "sort": "relevance"}, "save_as": "search"},
            {"kind": "tool", "title": "2. Тональность: всего и по площадкам", "tool": "tonality_summary",
             "save_as": "tonality"},
            {"kind": "tool", "title": "3. Рейтинг СМИ и авторов", "tool": "media_rating",
             "args": {"limit": 12}, "save_as": "media"},
            {"kind": "tool", "title": "4. Цепочки авторов по подтеме", "tool": "chain_graph",
             "args": {"phrase": "просрочка"}, "save_as": "chain"},
            {"kind": "tool", "title": "5. Негатив: примеры сообщений", "tool": "search_messages",
             "args": {"phrase": "просрочка", "tone": "негатив", "limit": 15, "sort": "relevance"},
             "save_as": "negative"},
            {"kind": "tool", "title": "6. Позитив: примеры сообщений", "tool": "search_messages",
             "args": {"phrase": "просрочка", "tone": "позитив", "limit": 15, "sort": "relevance"},
             "save_as": "positive"},
            {"kind": "tool", "title": "7. ИИ-аналитика по подтеме", "tool": "ai_analytics",
             "args": {"query_str": "просрочка"}, "save_as": "ai"},
            {"kind": "chart", "title": "Тональность по подтеме", "from": "{{tonality.tonality_total}}",
             "label_field": "tone", "value_field": "count", "chart_type": "pie",
             "series_name": "Сообщений", "save_as": "chart_tone"},
            {"kind": "chart", "title": "Динамика упоминаний по месяцам", "from": "{{search.monthly_dynamics}}",
             "label_field": "month", "value_field": "count", "chart_type": "line",
             "series_name": "Сообщений", "save_as": "chart_dynamics"},
            {"kind": "chart", "title": "Топ авторов негатива", "from": "{{tonality.top_negative_authors}}",
             "label_field": "author", "value_field": "posts_in_top1000", "chart_type": "hbar",
             "series_name": "Постов в топ-1000", "save_as": "chart_authors"},
            {"kind": "llm", "title": "8. Раздел «Тональность»", "save_as": "sum_tone", "max_tokens": 900,
             "prompt": (
                 "Подтема: {{search.phrase}}. Датасет: {{search.index_name}}, период {{search.period.from}} — {{search.period.to}}.\n\n"
                 "Найдено по подтеме: {{search.messages_found}} сообщений; формулировки поиска: {{search.search_terms}}.\n"
                 "Тональность по подтеме: {{search.tonality}}; тональность по всему датасету: {{tonality.tonality_total}}.\n"
                 "Тональность по площадкам: {{tonality.tonality_by_hub}}.\n"
                 "Динамика по месяцам: {{search.monthly_dynamics}}.\n"
                 "Примеры негатива: {{negative.examples}}\n"
                 "Примеры позитива: {{positive.examples}}\n\n"
                 "Напиши раздел отчёта «Тональность и динамика»: 1) сколько сообщений по подтеме и как это выглядит на фоне "
                 "датасета; 2) доли негатива, нейтрала и позитива, чего больше и как меняется по месяцам; 3) какие площадки "
                 "дают негатив, а какие позитив; 4) конкретные претензии из негатива (3–4 с формулировками и ссылками); "
                 "5) что хвалят в позитиве (3 пункта). Только факты из данных, без вводных фраз, объём 5–8 абзацев."
             )},
            {"kind": "llm", "title": "9. Раздел «Авторы и цепочки»", "save_as": "sum_authors", "max_tokens": 900,
             "prompt": (
                 "Подтема: {{search.phrase}}. Датасет: {{search.index_name}}, период {{search.period.from}} — {{search.period.to}}.\n\n"
                 "Авторы негатива: {{tonality.top_negative_authors}}\n"
                 "Авторы позитива: {{tonality.top_positive_authors}}\n"
                 "Рейтинг СМИ (всего публикаций: {{media.smi_messages_total}}): негативные {{media.negative_smi}}; "
                 "позитивные {{media.positive_smi}}; лента публикаций {{media.media_feed}}\n"
                 "Цепочки распространения: статистика {{chain.stats}}; топ распространителей {{chain.top_spreaders}}; "
                 "кластеры {{chain.clusters}}; хронология {{chain.timeline_summary}}\n"
                 "Что показывает встроенная ИИ-аналитика ({{ai.total_rows}} сообщений): {{ai.examples}}\n\n"
                 "Напиши раздел отчёта «Авторы и цепочки распространения»: 1) кто задаёт повестку по подтеме — топ-5 авторов "
                 "с цифрами; 2) кто разгоняет негатив и кто поддерживает позитив; 3) как инфоповод расходится: первоисточник, "
                 "кто подхватил, через какие площадки, сколько времени заняла волна; 4) какие кластеры сообщений появились; "
                 "5) какие СМИ пишут по теме. Только факты из данных, 4–6 абзацев."
             )},
            {"kind": "llm", "title": "10. Выводы и рекомендации", "save_as": "sum_conclusions", "max_tokens": 800,
             "prompt": (
                 "Подтема: {{search.phrase}}. Найдено {{search.messages_found}} сообщений, тональность: {{search.tonality}}.\n\n"
                 "Раздел «Тональность»:\n{{sum_tone.text}}\n\n"
                 "Раздел «Авторы и цепочки»:\n{{sum_authors.text}}\n\n"
                 "Сформулируй 5–7 нумерованных выводов и рекомендаций: что происходит с подтемой, где риск, что делать "
                 "в первую очередь, за чем следить дальше. Каждый пункт — одно-два предложения, с цифрами из данных."
             )},
            {"kind": "report", "title": "11. Сборка отчёта DOCX/PDF", "save_as": "report",
             "report_title": "Отчёт по подтеме: {{search.phrase}}",
             "subtitle": "Датасет {{search.index_name}}, период {{search.period.from}} — {{search.period.to}}",
             "folder": "Отчёт по подтеме",
             "sections": [
                 {"heading": "Тональность и динамика", "text": "{{sum_tone.text}}", "chart_ids": ["chart1", "chart2"]},
                 {"heading": "Авторы и цепочки распространения", "text": "{{sum_authors.text}}", "chart_ids": ["chart3"]},
                 {"heading": "Выводы и рекомендации", "text": "{{sum_conclusions.text}}"},
             ]},
        ],
    },
]

_PRESET_BY_ID = {preset["id"]: preset for preset in PRESETS}


def normalize_steps(raw: Any) -> List[Dict[str, Any]]:
    """Приводит цепочку шагов к безопасному виду (шаги конструктора)."""
    from .pipeline import STEP_KINDS

    out: List[Dict[str, Any]] = []
    for number, step in enumerate(raw or [], start=1):
        if not isinstance(step, dict):
            continue
        kind = str(step.get("kind") or "tool")
        if kind not in STEP_KINDS:
            continue
        item = {
            "id": str(step.get("id") or f"step{number}"),
            "kind": kind,
            "title": str(step.get("title") or STEP_KINDS.get(kind, kind))[:200],
            "save_as": str(step.get("save_as") or f"step{number}")[:60],
        }
        for key in (
            "tool",
            "prompt",
            "system",
            "report_title",
            "subtitle",
            "folder",
            "from",
            "label_field",
            "value_field",
            "chart_type",
            "series_name",
            "x_label",
            "y_label",
        ):
            if step.get(key) not in (None, ""):
                item[key] = step[key]
        for key in ("args", "sections", "series_fields"):
            if isinstance(step.get(key), (dict, list)):
                item[key] = step[key]
        if step.get("limit") is not None:
            try:
                item["limit"] = int(step["limit"])
            except Exception:
                pass
        if step.get("max_tokens") is not None:
            try:
                item["max_tokens"] = int(step["max_tokens"])
            except Exception:
                pass
        out.append(item)
    return out[:14]


def step_kinds_public() -> List[Dict[str, str]]:
    """Виды шагов для интерфейса конструктора."""
    from .pipeline import STEP_KINDS

    return [{"id": key, "title": value} for key, value in STEP_KINDS.items()]


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
            "steps": normalize_steps(payload.get("steps") if payload.get("steps") is not None else base.get("steps")),
        }
        items.append(record)
    else:
        record = existing
        for key in ("name", "description", "instruction", "folder", "model", "dataset_name"):
            if payload.get(key) is not None:
                record[key] = payload[key]
        if payload.get("tools") is not None:
            record["tools"] = payload["tools"]
        if payload.get("steps") is not None:
            record["steps"] = normalize_steps(payload["steps"])
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
        steps=agent.get("steps"),
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
