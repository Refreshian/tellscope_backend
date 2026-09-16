# -*- coding: utf-8 -*-
"""Конструктор шагов: детерминированная цепочка (собрать данные → разбор → графики → отчёт).

Агент может быть двух видов:
  * с инструкцией (LLM сам выбирает инструменты — agent_engine.loop);
  * со шагами (этот модуль): шаги выполняются по порядку, результат каждого доступен
    следующим шагам через подстановки вида {{step_id.field}}.

Виды шагов:
  tool   — вызов инструмента из реестра (search_messages, deep_text_analysis, ...);
  chart  — график из данных предыдущего шага (from + label_field + value_field);
  llm    — текстовые выводы моделью (собранные данные подставляются в промпт);
  report — сборка документа DOCX/PDF из разделов.

Расследования остаются в режиме с инструкцией: там, где нужен свободный поиск, LLM-цикл
гибче цепочки. Шаги — для повторяемых регламентных отчётов.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional

from .loop import MODEL_CHOICES, DEFAULT_CHOICE
from .progress import tracker as progress_tracker
from .registry import execute as run_tool
from .registry import get_tool

STEP_KINDS = {
    "tool": "Собрать данные или разбор",
    "chart": "Построить график",
    "llm": "Выводы ИИ (текст)",
    "report": "Собрать отчёт DOCX/PDF",
}

MAX_STEPS = 16

_PLACEHOLDER = re.compile(r"\{\{\s*([^}]+?)\s*\}\}")


def _unwrap(value: Any) -> Any:
    """Результат шага хранится в обёртке {'ok', 'result'} — для подстановок нужен сам результат."""
    if isinstance(value, dict) and "ok" in value and "result" in value:
        return value.get("result")
    return value


def _resolve_path(path: str, ctx: Any, results: Dict[str, Any]) -> Any:
    """{{overview.monthly_dynamics}} → данные из результатов шага; {{ctx.dataset_index}} → из контекста."""
    parts = [p.strip() for p in str(path).split(".") if p.strip()]
    if not parts:
        return None
    if parts[0] == "ctx":
        value = getattr(ctx, parts[1], None) if len(parts) > 1 else None
        parts = parts[2:]
    else:
        value = _unwrap(results.get(parts[0]))
        parts = parts[1:]
    for part in parts:
        value = _unwrap(value)
        if value is None:
            return None
        if isinstance(value, dict):
            value = value.get(part)
        elif isinstance(value, list):
            try:
                value = value[int(part)]
            except Exception:
                return None
        else:
            value = getattr(value, part, None)
    return _unwrap(value)


def render(value: Any, ctx: Any, results: Dict[str, Any]) -> Any:
    """Подставляет значения шагов и данные контекста в аргументы шага."""
    if isinstance(value, str):
        matches = list(_PLACEHOLDER.finditer(value))
        if len(matches) == 1 and matches[0].group(0) == value.strip():
            return _resolve_path(matches[0].group(1), ctx, results)
        def repl(match):
            resolved = _resolve_path(match.group(1), ctx, results)
            if isinstance(resolved, (dict, list)):
                return json.dumps(resolved, ensure_ascii=False)
            return "" if resolved is None else str(resolved)

        return _PLACEHOLDER.sub(repl, value)
    if isinstance(value, dict):
        return {key: render(item, ctx, results) for key, item in value.items()}
    if isinstance(value, list):
        return [render(item, ctx, results) for item in value]
    return value


def _series_from(items: List[dict], label_field: str, value_field: str, limit: int = 24) -> (List[str], List[float]):
    labels: List[str] = []
    values: List[float] = []
    for item in (items or [])[:limit]:
        if not isinstance(item, dict):
            continue
        label = item.get(label_field)
        raw = item.get(value_field)
        try:
            number = float(raw)
        except (TypeError, ValueError):
            continue
        labels.append(str(label) if label is not None else "")
        values.append(number)
    return labels, values


def _account_llm(ctx: Any, result: Any) -> None:
    """Учёт токенов и стоимости шага LLM (как в цикле агента)."""
    raw = getattr(result, "raw", None) or {}
    usage = raw.get("usage") or {}
    try:
        prompt = int(usage.get("prompt_tokens") or 0)
        completion = int(usage.get("completion_tokens") or 0)
        total = int(usage.get("total_tokens") or (prompt + completion))
    except Exception:
        prompt = completion = total = 0
    choice = MODEL_CHOICES.get(getattr(ctx, "model_choice", DEFAULT_CHOICE)) or MODEL_CHOICES[DEFAULT_CHOICE]
    ctx.tokens += total
    ctx.cost_usd += (prompt * float(choice.get("price_in") or 0) + completion * float(choice.get("price_out") or 0)) / 1_000_000.0


# --- санитайзер аргументов шага -------------------------------------------------------
# Модель, собирая цепочку, регулярно кладёт в аргументы литерал вместо значения
# ("index": "search.index", "phrase": "search.phrase") или пустое значение (phrase: null,
# tone: "", limit: 0). Отдельные инструменты это молча проглатывают и отдают пустой срез,
# поэтому месяц «закрывался» за секунды без анализа. Ниже — единая чистка аргументов.

# Значения тональности, которые понимает _tone_filter / _tone_label.
_TONE_OK = {
    "any", "all", "любая", "любой", "все", "всё",
    "negative", "негатив", "негативный", "негативная",
    "positive", "позитив", "позитивный", "позитивная",
    "neutral", "нейтрал", "нейтральный", "нейтральная",
}
# Разумные пределы выборки для инструментов, у которых есть аргумент limit.
_SANE_LIMITS = {"search_messages": 30, "media_rating": 30, "popular_hooks": 30, "analyze_texts": 400}
_MAX_LIMIT = 20000

# Строка-заглушка: подстановка, имя поля или служебное слово вместо значения.
_LITERAL_ARGS = re.compile(
    r"^\s*(\{\{.*?\}\}|<[^>]{1,40}>|none|null|nil|nan|undefined|n/?a|—|-\s*|\.{2,}"
    r"|(phrase|topic|query|tone|index|limit|keyword|focus|value|count)\s*"
    r"|[a-z_][\w]*(\.[\w]+)+\s*)\s*$",
    re.I,
)

_MONTHS_NOM = ["январь", "февраль", "март", "апрель", "май", "июнь", "июль",
               "август", "сентябрь", "октябрь", "ноябрь", "декабрь"]
_MONTHS_GEN = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля",
               "августа", "сентября", "октября", "ноября", "декабря"]
_MONTH_YEAR = re.compile(
    r"(?<![\w])(январ[ья]|феврал[ья]|март[а]?|апрел[ья]|ма[йя]|июн[ья]|июл[ья]"
    r"|август[а]?|сентябр[ья]|октябр[ья]|ноябр[ья]|декабр[ья])(?![\w])\s+(\d{4})",
    re.I,
)
_ISO_RANGE = re.compile(r"\d{4}-\d{2}-\d{2}\s*[–—-]\s*\d{4}-\d{2}-\d{2}")


def _is_blank_or_literal(value: Any) -> bool:
    """Пустое значение или подстановка вместо значения («search.phrase», «phrase», null)."""
    if value is None:
        return True
    if isinstance(value, (list, dict, tuple, set)):
        return not value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return False
    text = str(value).strip()
    if not text:
        return True
    return bool(_LITERAL_ARGS.match(text))


def _date_only(value: Any) -> str:
    """Дата ГГГГ-ММ-ДД из ISO-строки или unix-времени (иначе пустая строка)."""
    if value is None or value == "" or isinstance(value, bool):
        return ""
    if isinstance(value, (int, float)):
        try:
            return time.strftime("%Y-%m-%d", time.localtime(float(value)))
        except Exception:
            return ""
    match = re.match(r"\s*(\d{4})-(\d{2})-(\d{2})", str(value))
    return "%s-%s-%s" % match.groups() if match else ""


def _task_topic(ctx: Any) -> str:
    """Тема из постановки задачи: «Проанализируй тему KFC за декабрь…» → «KFC»."""
    text = str(getattr(ctx, "task", "") or "")
    for pattern in (
        r"(?:тему|тема|темы|по теме|по подтеме)\s*[«\"'„]?\s*"
        r"([^»\"'“\n,.;:()]{2,60}?)\s*[»\"'“]?\s+(?:за|в|с|по)\s",
        r"(?:про|о|об)\s+теме\s*[«\"'„]?\s*([^»\"'“\n,.;:()]{2,60}?)\s*[»\"'“]?\s+(?:за|в|с)\s",
    ):
        match = re.search(pattern, text, re.I)
        if match:
            value = match.group(1).strip(" «»\"'")
            if value and not _is_blank_or_literal(value):
                return value[:80]
    # Запасной вариант — имя датасета без дат: «kfc_13.05.2024-22.09.2026» → «kfc».
    name = str(getattr(ctx, "dataset_name", "") or "")
    head = re.split(r"[_\d]", name.strip())[0].strip(" .-_")
    return head[:40]


def _fix_period_literals(text: Any, ctx: Any) -> Any:
    """Чужой месяц/период в заголовке меняем на период задачи.

    Шаги цепочки собираются под конкретный месяц и хранятся как есть: если задачу потом
    переиспользуют для другого периода (или планировщик ошибся с месяцем), в заголовке
    оставался прежний месяц, и отчёт за декабрь назывался январским.
    """
    if not isinstance(text, str) or not text:
        return text
    iso = _date_only(getattr(ctx, "min_date", None))
    if not iso:
        return text
    year, month = iso[:4], int(iso[5:7])
    nom, gen = _MONTHS_NOM[month - 1], _MONTHS_GEN[month - 1]

    def month_repl(match):
        word, found_year = match.group(1), match.group(2)
        low = word.lower()
        form = "gen" if low in _MONTHS_GEN else ("nom" if low in _MONTHS_NOM else
                                                ("nom" if low.endswith(("ь", "й")) else "gen"))
        target = nom if form == "nom" else gen
        if found_year == year and low == target:
            return match.group(0)
        return "%s %s" % (target, year)

    fixed = _MONTH_YEAR.sub(month_repl, text)
    lo, hi = _date_only(getattr(ctx, "min_date", None)), _date_only(getattr(ctx, "max_date", None))
    if lo and hi:
        fixed = _ISO_RANGE.sub("%s–%s" % (lo, hi), fixed)
    return fixed


def _align_period(ctx: Any, args: Dict[str, Any]) -> List[str]:
    """Период задачи главнее литералов в шаге: без этого шаг читал чужой месяц."""
    notes: List[str] = []
    t_lo, t_hi = _date_only(getattr(ctx, "min_date", None)), _date_only(getattr(ctx, "max_date", None))
    if not (t_lo or t_hi):
        return notes
    a_lo, a_hi = _date_only(args.get("min_date")), _date_only(args.get("max_date"))
    if a_lo and a_hi and t_lo and t_hi and (a_hi < t_lo or a_lo > t_hi):
        args["min_date"], args["max_date"] = ctx.min_date, ctx.max_date
        notes.append("период шага %s–%s не пересекается с периодом задачи — беру %s–%s"
                     % (a_lo, a_hi, t_lo, t_hi))
        return notes
    if t_lo and ctx.min_date and (not a_lo or a_lo < t_lo):
        args["min_date"] = ctx.min_date
        if a_lo:
            notes.append("начало периода %s вне задачи — беру %s" % (a_lo, t_lo))
    if t_hi and ctx.max_date and (not a_hi or a_hi > t_hi):
        args["max_date"] = ctx.max_date
        if a_hi:
            notes.append("конец периода %s вне задачи — беру %s" % (a_hi, t_hi))
    return notes


async def _sanitize_tool_args(ctx, step: Dict[str, Any], args: Any) -> Dict[str, Any]:
    """Чистит аргументы шага: литералы и пустые значения → значения из задачи."""
    name = str(step.get("tool") or "").strip()
    if not isinstance(args, dict):
        args = {}
    notes: List[str] = []

    # index: не номер и не известное имя темы — литерал или мусор.
    if args.get("index") not in (None, ""):
        raw_index = str(args.get("index")).strip()
        keep = raw_index.lstrip("-").isdigit()
        if not keep:
            try:
                from .tools_data import _match_dataset

                keep = _match_dataset(raw_index) is not None
            except Exception:
                keep = False
        if not keep:
            notes.append("index=%r не является темой — беру датасет из задачи" % raw_index)
            args.pop("index", None)

    # phrase/topic/query/focus: пусто или подстановка → тема из постановки задачи.
    topic = _task_topic(ctx)
    for key in ("phrase", "topic", "query", "keyword", "focus"):
        if key not in args or not _is_blank_or_literal(args.get(key)):
            continue
        raw = args.pop(key)
        if key in ("phrase", "topic", "query") and topic:
            args[key] = topic
            notes.append("%s=%r пусто или литерал — подставляю тему задачи %r" % (key, raw, topic))
        else:
            notes.append("убрал пустое значение %s=%r" % (key, raw))

    # tone: неизвестная тональность молча превращалась в «без фильтра» — говорим об этом явно.
    if "tone" in args and str(args.get("tone") or "").strip().lower() not in _TONE_OK:
        raw = args.get("tone")
        args["tone"] = "all"
        notes.append("тональность %r неизвестна — читаю все сообщения" % raw)

    # limit: 0/пусто/мусор вместо числа.
    if "limit" in args:
        raw_limit = args.get("limit")
        try:
            number = int(float(str(raw_limit).strip()))
        except (TypeError, ValueError):
            number = 0
        if number < 1:
            sane = _SANE_LIMITS.get(name)
            if sane:
                args["limit"] = sane
                notes.append("limit=%r некорректен — беру %d" % (raw_limit, sane))
            else:
                args.pop("limit", None)
                notes.append("убрал некорректный limit=%r" % (raw_limit,))
        elif number > _MAX_LIMIT:
            args["limit"] = _MAX_LIMIT
            notes.append("limit=%d слишком велик — срезал до %d" % (number, _MAX_LIMIT))

    notes.extend(_align_period(ctx, args))

    if ctx.dataset_index is not None and "index" not in args:
        args["index"] = ctx.dataset_index
    if ctx.min_date and "min_date" not in args:
        args["min_date"] = ctx.min_date
    if ctx.max_date and "max_date" not in args:
        args["max_date"] = ctx.max_date

    if notes:
        await ctx.log("Шаг «%s»: %s" % (name, "; ".join(notes)))
    return args


async def _run_tool_step(ctx, step: Dict[str, Any], results: Dict[str, Any]) -> Dict[str, Any]:
    name = str(step.get("tool") or "").strip()
    spec = get_tool(name)
    if spec is None:
        return {"ok": False, "error": f"инструмент {name} не найден"}
    args = render(step.get("args") or {}, ctx, results)
    args = await _sanitize_tool_args(ctx, step, args)
    for key in ("title", "subtitle", "heading", "name", "caption", "report_title"):
        # Заголовки инструментов тоже подчиняем периоду задачи.
        if isinstance(args.get(key), str):
            args[key] = _fix_period_literals(args[key], ctx)
    outcome = await run_tool(spec, ctx, args)
    return {"ok": outcome.get("ok"), "result": outcome.get("result"), "error": outcome.get("error"), "args": args, "tool": name}


async def _run_chart_step(ctx, step: Dict[str, Any], results: Dict[str, Any]) -> Dict[str, Any]:
    spec = get_tool("make_chart")
    if spec is None:
        return {"ok": False, "error": "инструмент make_chart недоступен"}
    source = render(step.get("from"), ctx, results)
    label_field = str(step.get("label_field") or "key")
    value_field = str(step.get("value_field") or "count")
    labels: List[str] = []
    values: List[float] = []
    if isinstance(source, dict):
        # например {'месяц': число} или {'key': ..., 'count': ...} в одном объекте
        if label_field in source and value_field in source:
            labels, values = [str(source.get(label_field))], [float(source.get(value_field))]
        else:
            for key, val in list(source.items())[:24]:
                try:
                    values.append(float(val))
                    labels.append(str(key))
                except (TypeError, ValueError):
                    continue
    elif isinstance(source, list):
        labels, values = _series_from(source, label_field, value_field, int(step.get("limit") or 24))

    if not labels:
        return {"ok": False, "error": "нет данных для графика: проверьте поле «откуда брать данные» и поля подписей"}

    args = {
        "title": render(_fix_period_literals(step.get("title"), ctx), ctx, results) or "График",
        "chart_type": step.get("chart_type") or "bar",
        "categories": labels,
        "series": [{"name": render(step.get("series_name"), ctx, results) or "Значение", "values": values}],
        "x_label": render(step.get("x_label"), ctx, results) or "",
        "y_label": render(step.get("y_label"), ctx, results) or "",
    }
    outcome = await run_tool(spec, ctx, args)

    # мультирядный вариант: несколько колонок одного набора данных (например тональность)
    series_fields = step.get("series_fields") or []
    if outcome.get("ok") and series_fields and isinstance(source, list):
        series = []
        for field in series_fields:
            field_labels, field_values = _series_from(source, label_field, field, int(step.get("limit") or 24))
            if field_values:
                series.append({"name": str(field), "values": field_values})
        if series:
            args["series"] = series
            outcome = await run_tool(spec, ctx, args)
    return {"ok": outcome.get("ok"), "result": outcome.get("result"), "error": outcome.get("error"), "args": args}


async def _run_llm_step(ctx, step: Dict[str, Any], results: Dict[str, Any]) -> Dict[str, Any]:
    """Шаг «выводы ИИ» — это генерация текста, а не планирование.

    Шаги цепочки заданы заранее и выполняются детерминированно, инструменты не выбираются
    моделью, поэтому здесь используется модель анализа (выбор пользователя), а не
    модель-оркестратор из настройки agent.orchestrator.
    """
    from mlops import gateway
    from .loop import _load_prompt  # переиспользуем системный промпт аналитика

    choice = MODEL_CHOICES.get(getattr(ctx, "model_choice", DEFAULT_CHOICE)) or MODEL_CHOICES[DEFAULT_CHOICE]
    ctx.last_choice_key = getattr(ctx, "model_choice", DEFAULT_CHOICE)
    prompt = render(step.get("prompt") or "", ctx, results)
    system = render(step.get("system") or "", ctx, results) or _load_prompt(
        "agent_system_v1",
        "Ты аналитик соцмедиа и СМИ. Пиши деловым русским языком, без вводных фраз, только по переданным данным.",
    )
    extra = {"chat_template_kwargs": {"enable_thinking": False}} if choice.get("provider") == "vllm" else None
    tracker = getattr(ctx, "progress", None)
    try:
        if tracker is not None:
            # Шаг «выводы ИИ» длится десятки секунд — heartbeat виден в интерфейсе.
            async with tracker.heartbeat():
                result = await gateway.achat(
                    provider=choice["provider"],
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                    temperature=0.25,
                    max_tokens=int(step.get("max_tokens") or 1600),
                    timeout=420,
                    extra=extra,
                    profile=choice["profile"],
                    usage_ctx={"user_id": ctx.user_id, "case": "agent-mode"},
                )
        else:
            result = await gateway.achat(
                provider=choice["provider"],
                messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                temperature=0.25,
                max_tokens=int(step.get("max_tokens") or 1600),
                timeout=420,
                extra=extra,
                profile=choice["profile"],
                usage_ctx={"user_id": ctx.user_id, "case": "agent-mode"},
            )
        text = re.sub(r"<think>.*?</think>", "", result.content or "", flags=re.S | re.I).strip()
        _account_llm(ctx, result)
        return {"ok": bool(text), "text": text, "error": None if text else "модель вернула пустой ответ"}
    except Exception as exc:
        return {"ok": False, "text": "", "error": f"{type(exc).__name__}: {exc}"}


async def _run_report_step(ctx, step: Dict[str, Any], results: Dict[str, Any]) -> Dict[str, Any]:
    spec = get_tool("build_report")
    if spec is None:
        return {"ok": False, "error": "инструмент build_report недоступен"}
    raw_sections = step.get("sections") or []
    if not raw_sections and step.get("text"):
        raw_sections = [{"heading": step.get("heading") or "Аналитика", "text": step.get("text")}]
    sections = []
    for section in raw_sections:
        item = {
            "heading": render(_fix_period_literals(section.get("heading") or "Раздел", ctx), ctx, results),
            "text": render(section.get("text") or "", ctx, results),
            "bullets": [render(b, ctx, results) for b in (section.get("bullets") or [])],
            "chart_ids": section.get("chart_ids") or [],
        }
        for key in ("findings", "highlights"):
            # Темы с цитатами и ключевые сообщения из analyze_texts попадают в DOCX/PDF отдельным блоком.
            value = render(section.get(key), ctx, results)
            if value:
                item[key] = value
        if section.get("citations"):
            item["citations"] = render(section.get("citations"), ctx, results)
        sections.append(item)
    args = {
        "title": render(_fix_period_literals(step.get("report_title") or step.get("title"), ctx),
                        ctx, results) or "Аналитический отчёт",
        "subtitle": render(_fix_period_literals(step.get("subtitle") or "", ctx), ctx, results),
        "folder": render(step.get("folder") or ctx.folder, ctx, results),
        "sections": sections,
    }
    outcome = await run_tool(spec, ctx, args)
    return {"ok": outcome.get("ok"), "result": outcome.get("result"), "error": outcome.get("error"), "args": args}


async def run_pipeline(ctx, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Выполняет цепочку шагов и возвращает результат последнего смыслового шага."""
    results: Dict[str, Any] = {}
    answer = ""
    failed = 0
    executed = 0
    llm_steps = 0
    plan = list((steps or [])[:MAX_STEPS])
    # План известен заранее — интерфейс получает «шаг 4 из 12» и оценку остатка.
    tracker = progress_tracker(ctx, total=len(plan))
    await tracker.start_run()
    await ctx.event({"type": "start", "model": (MODEL_CHOICES.get(ctx.model_choice) or {}).get("label"), "tools": ["шаги"], "token_budget": ctx.token_budget})

    for number, step in enumerate(plan, start=1):
        # Остановка и мягкая пауза: проверяем перед каждым шагом цепочки.
        ctx.check_cancelled()
        await ctx.wait_if_paused()
        if ctx.out_of_time():
            ctx.notes.append("истёк лимит времени запуска")
            break
        if ctx.token_budget and ctx.tokens >= max(1000, int(ctx.token_budget) - 8000):
            ctx.notes.append(f"достигнут бюджет прогона: {ctx.tokens} токенов из {ctx.token_budget}")
            break

        kind = str(step.get("kind") or "tool")
        title = render(_fix_period_literals(step.get("title"), ctx), ctx, results) or STEP_KINDS.get(kind, kind)
        save_as = str(step.get("save_as") or f"step{number}")
        await tracker.begin_stage(f"Шаг {number}. {title}", detail=STEP_KINDS.get(kind, kind))
        await ctx.event({"type": "tool_start", "name": kind, "title": f"Шаг {number}. {title}", "args": step.get("args") or {}})

        if kind == "tool":
            outcome = await _run_tool_step(ctx, step, results)
        elif kind == "chart":
            outcome = await _run_chart_step(ctx, step, results)
        elif kind == "llm":
            outcome = await _run_llm_step(ctx, step, results)
            llm_steps += 1
        elif kind == "report":
            outcome = await _run_report_step(ctx, step, results)
        else:
            outcome = {"ok": False, "error": f"неизвестный тип шага: {kind}"}

        executed += 1
        if not outcome.get("ok"):
            failed += 1
        results[save_as] = outcome
        payload = outcome.get("result") if outcome.get("ok") else None
        if kind == "llm":
            payload = {"text": outcome.get("text")}
            if outcome.get("text"):
                answer = outcome["text"]
        if kind == "report" and outcome.get("ok"):
            files = (outcome.get("result") or {}).get("files") or []
            answer = answer or f"Отчёт собран: {', '.join(f.get('name', '') for f in files)}"
        ctx.tool_calls.append(
            {
                "name": step.get("tool") or kind,
                "args": outcome.get("args") or step.get("args") or {},
                "ok": outcome.get("ok"),
                "ms": 0,
                "summary": outcome.get("error") or f"шаг {number}: {title}",
            }
        )
        await ctx.event(
            {
                "type": "tool_end",
                "name": step.get("tool") or kind,
                "title": f"Шаг {number}. {title}",
                "ok": bool(outcome.get("ok")),
                "summary": outcome.get("error") or f"готово ({title})",
                "error": outcome.get("error"),
            }
        )
        # Шаг завершён: инкремент счётчика, уточнение средней длительности шага и ETA.
        await tracker.end_stage(
            detail=outcome.get("error") or f"шаг {number} готов: {title}",
            ok=bool(outcome.get("ok")),
        )
        # Остановка или пауза могли прийти во время шага — дальше не идём.
        ctx.check_cancelled()
        await ctx.wait_if_paused()
        if payload is not None:
            results[save_as + "_payload"] = payload

    stats = {
        "llm_calls": llm_steps,
        "tool_calls": len(ctx.tool_calls),
        "tokens": ctx.tokens,
        "cost_usd": round(ctx.cost_usd, 4),
        "token_budget": ctx.token_budget,
        "model": (MODEL_CHOICES.get(ctx.model_choice) or {}).get("label"),
        "artifacts": len(ctx.artifacts),
        "tools_used": sorted({c["name"] for c in ctx.tool_calls}),
        "steps": executed,
        "steps_failed": failed,
        "notes": ctx.notes,
    }
    if answer:
        await ctx.event({"type": "answer", "text": answer})
    await ctx.event({"type": "final", "answer": answer or "Шаги выполнены, смотрите артефакты запуска.", "artifacts": ctx.artifacts, "stats": stats})
    return {
        "answer": answer or "Шаги выполнены, смотрите артефакты запуска.",
        "stats": stats,
        "tool_calls": ctx.tool_calls,
        "artifacts": ctx.artifacts,
        "no_data": bool(getattr(ctx, "no_data", False)),
        "text_gap": str(getattr(ctx, "text_gap", "") or ""),
    }
