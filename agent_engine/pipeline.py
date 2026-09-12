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


async def _run_tool_step(ctx, step: Dict[str, Any], results: Dict[str, Any]) -> Dict[str, Any]:
    name = str(step.get("tool") or "").strip()
    spec = get_tool(name)
    if spec is None:
        return {"ok": False, "error": f"инструмент {name} не найден"}
    args = render(step.get("args") or {}, ctx, results)
    if ctx.dataset_index is not None and "index" not in (args or {}):
        args["index"] = ctx.dataset_index
    if ctx.min_date and "min_date" not in (args or {}):
        args["min_date"] = ctx.min_date
    if ctx.max_date and "max_date" not in (args or {}):
        args["max_date"] = ctx.max_date
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
        "title": render(step.get("title"), ctx, results) or "График",
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
            "heading": render(section.get("heading") or "Раздел", ctx, results),
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
        "title": render(step.get("report_title") or step.get("title"), ctx, results) or "Аналитический отчёт",
        "subtitle": render(step.get("subtitle") or "", ctx, results),
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
        if ctx.out_of_time():
            ctx.notes.append("истёк лимит времени запуска")
            break
        if ctx.token_budget and ctx.tokens >= max(1000, int(ctx.token_budget) - 8000):
            ctx.notes.append(f"достигнут бюджет прогона: {ctx.tokens} токенов из {ctx.token_budget}")
            break

        kind = str(step.get("kind") or "tool")
        title = render(step.get("title"), ctx, results) or STEP_KINDS.get(kind, kind)
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
