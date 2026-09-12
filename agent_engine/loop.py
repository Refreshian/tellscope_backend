# -*- coding: utf-8 -*-
"""Цикл агента: LLM выбирает инструменты, Tellscope выполняет их и стримит шаги в интерфейс.

Работает с OpenAI-совместимыми провайдерами через mlops.gateway: локальный vLLM (Qwen3)
и внешний aitunnel (Claude/GPT). Если модель не поддерживает tools, автоматически
включается резервный JSON-протокол: модель отвечает {"tool": {...}} или {"final": "..."}.
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Tuple

from .registry import execute, get_tool, json_text, openai_tools

MAX_STEPS = 14
MAX_TOOL_CALLS = 28
DEFAULT_TOKEN_BUDGET = 120_000

# Цены за 1M токенов (USD) — только для оценки расхода в интерфейсе.
# Локальный Qwen считается бесплатным: это наши GPU, внешних платежей нет.
MODEL_CHOICES: Dict[str, Dict[str, Any]] = {
    "gpt": {
        "provider": "aitunnel",
        "profile": "dashboard_qa",
        "label": "GPT-4.1 mini — дёшево",
        "price_in": 0.4,
        "price_out": 1.6,
        "tier": "cheap",
    },
    "qwen": {
        "provider": "vllm",
        "profile": "agent",
        "label": "Qwen3-32B — локально, без оплаты (экспериментально)",
        "price_in": 0.0,
        "price_out": 0.0,
        "tier": "free",
    },
    "deepseek": {
        "provider": "aitunnel",
        "profile": "harness",
        "label": "Ассистент — быстро и дёшево (DeepSeek)",
        "price_in": 0.28,
        "price_out": 0.42,
        "tier": "cheap",
    },
    "claude": {
        "provider": "aitunnel",
        "profile": "smart_agent_planner",
        "label": "Claude Sonnet 4.5 — максимум качества, дорого",
        "price_in": 3.0,
        "price_out": 15.0,
        "tier": "premium",
    },
}
DEFAULT_CHOICE = "gpt"
# Куда переключаться, если выбранная модель не умеет вызывать инструменты.
FALLBACK_CHOICE = "gpt"
# Ставить ли в ответ предупреждение, если агент не получил данные инструментами.
NO_DATA_WARNING = (
    "⚠️ Данные не подтверждены инструментами: агент не смог получить факты из датасета, "
    "поэтому цифрам ниже доверять нельзя. Запустите задачу ещё раз или выберите модель GPT-4.1 mini."
)

# Сколько символов результата инструмента остаётся в истории для старых шагов.
# Без этого промпт растёт квадратично и каждый следующий шаг стоит дороже предыдущего.
TOOL_RESULT_KEEP_CHARS = 2200
TOOL_RESULT_RECENT_FULL = 3

REPORT_KEYWORDS = (
    "отчёт", "отчет", "report", "презентац", "документ", "docx", "pdf", "выгрузк", "слайд",
    "разбор", "анализ", "аналитик", "исследован", "обзор", "доклад", "записк", "справк",
)
# Если пользователь явно просит короткий ответ, отчёт не навязываем
REPORT_SKIP_PHRASES = (
    "без отчёта", "без отчета", "не нужен отчёт", "не нужен отчет", "не надо отчёт", "не надо отчет",
    "только текстом", "только текст", "в чат", "одной строкой", "кратко", "коротко", "не собирай отчёт",
)


def wants_report(task: str) -> bool:
    """Нужен ли пользователю файл отчёта, а не только текст в чате."""
    low = str(task or "").lower()
    if any(phrase in low for phrase in REPORT_SKIP_PHRASES):
        return False
    return any(keyword in low for keyword in REPORT_KEYWORDS)


def _gateway():
    from mlops import gateway

    return gateway


def _load_prompt(prompt_id: str, fallback_text: str) -> str:
    try:
        from mlops.prompts import load_prompt

        return load_prompt(prompt_id)
    except Exception:
        return fallback_text


def _extract_json(text: str) -> Optional[dict]:
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            obj, _ = decoder.raw_decode(text[start:])
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        start = text.find("{", start + 1)
    return None


def _tool_catalog_text(ctx) -> str:
    lines = []
    for name in sorted(ctx.allowed_tools):
        spec = get_tool(name)
        if spec:
            lines.append(f"- {name} ({spec.group}): {spec.description}")
    return "\n".join(lines)


def system_prompt(ctx) -> str:
    base = _load_prompt(
        "agent_system_v1",
        (
            "Ты — аналитик соцмедиа и СМИ в системе Tellscope. Ты работаешь только с данными датасета "
            "пользователя и вызываешь инструменты Tellscope, чтобы получить факты.\n"
            "Правила: 1) не выдумывай числа — любые цифры бери из результатов инструментов; "
            "2) если данных нет, скажи об этом прямо; 3) приводи конкретные примеры сообщений со ссылками; "
            "4) пиши по-русски, деловым языком, без вводных фраз вида «график показывает»; "
            "5) в конце собери отчёт инструментом build_report, если пользователь просит отчёт."
        ),
    )
    context = [
        "",
        "Контекст запуска:",
        f"- датасет: {ctx.dataset_label or ctx.dataset_name or 'не выбран'} (index={ctx.dataset_index})",
        f"- период: {'весь доступный' if not (ctx.min_date or ctx.max_date) else 'ограничен выбранными датами'}",
        f"- папка отчётов: {ctx.folder}",
        "",
        "Доступные инструменты:",
        _tool_catalog_text(ctx),
    ]
    return base + "\n".join(context)


def user_prompt(ctx) -> str:
    lines = [f"Задача пользователя: {ctx.task}", ""]
    lines.append(
        "Работай по шагам: сначала пойми, какие данные нужны, затем вызывай инструменты и используй их результаты. "
        "Все цифры в ответе должны опираться на данные инструментов."
    )
    if wants_report(ctx.task) and "build_report" in ctx.allowed_tools:
        lines.append(
            "Пользователь просит отчёт. Обязательный порядок: собрать данные инструментами, построить 2–4 графика "
            "через make_chart (динамика, тональность, площадки, сравнение периодов), затем вызвать build_report "
            "со всеми разделами, выводами и ссылками на источники. Ответ в чате — короткое резюме, сам отчёт только файлом."
        )
    return "\n".join(lines)


def _summarize(name: str, result: Any) -> str:
    """Короткое человекочитаемое описание результата инструмента для журнала."""
    try:
        if not isinstance(result, dict):
            return "готово"
        if name == "search_messages":
            base = f"найдено {result.get('messages_found')} сообщений" + (f" по «{result.get('phrase')}»" if result.get("phrase") else "")
            terms = result.get("search_terms") or []
            if len(terms) > 1:
                base += f" (смысловой поиск, формулировок: {len(terms)})"
            return base
        if name == "deep_text_analysis":
            terms = result.get("search_terms") or []
            vector = " + векторный" if result.get("semantic_vectors") else ""
            return (
                f"разобрано {result.get('messages_analyzed')} сообщений, формулировок поиска: {len(terms)}{vector}, "
                f"претензий: {len(result.get('key_claims') or [])}"
            )
        if name == "dataset_overview":
            period = result.get("period") or {}
            return f"{result.get('messages_total')} сообщений, период {period.get('from', '')} — {period.get('to', '')}"
        if name == "popular_hooks":
            return f"поводов: {len(result.get('hooks') or [])}"
        if name == "chain_graph":
            stats = result.get("stats") or {}
            return f"найдено {stats.get('found')} сообщений, первоисточников {stats.get('unique_texts')}"
        if name == "information_graph":
            return f"авторов {result.get('num_unique_authors')}, сообщений {result.get('num_messages')}"
        if name == "media_rating":
            return f"СМИ: {len(result.get('negative_smi') or [])} негативных, {len(result.get('positive_smi') or [])} позитивных"
        if name == "tonality_summary":
            return "тональный ландшафт получен"
        if name == "voice_of_customer":
            return f"тем: {result.get('topics_total')}"
        if name == "ai_analytics":
            return f"сообщений: {result.get('total_rows')}"
        if name == "make_chart":
            return f"график {result.get('chart_id')} построен"
        if name == "build_report":
            files = result.get("files") or []
            return "отчёт сохранён: " + ", ".join(f.get("name", "") for f in files)
        if name == "list_datasets":
            return f"датасетов доступно: {result.get('total_datasets')}"
        if name == "list_reports":
            return f"папок с отчётами: {len(result.get('reports') or [])}"
        if name in ("connector_request", "mcp_call", "mcp_list_tools"):
            return f"статус {result.get('status', 'ok')}"
        keys = ", ".join(list(result.keys())[:6])
        return f"получены поля: {keys}"
    except Exception:
        return "готово"


async def _call_llm(ctx, messages: List[dict], tools: Optional[List[dict]], max_tokens: int = 3000, force_tool: Optional[str] = None) -> Tuple[Any, bool]:
    """Возвращает (ChatResult, tools_supported). force_tool принудительно выбирает инструмент."""
    gateway = _gateway()
    choice = MODEL_CHOICES.get(ctx.model_choice or DEFAULT_CHOICE) or MODEL_CHOICES[DEFAULT_CHOICE]
    extra: Dict[str, Any] = {}
    if tools:
        extra = {"tools": tools, "tool_choice": "auto", "parallel_tool_calls": False}
        if force_tool:
            extra["tool_choice"] = {"type": "function", "function": {"name": force_tool}}
    usage_ctx = {"user_id": ctx.user_id, "case": "agent-mode"}
    try:
        result = await gateway.achat(
            provider=choice["provider"],
            messages=messages,
            temperature=0.2,
            max_tokens=max_tokens,
            timeout=300,
            extra=extra or None,
            profile=choice["profile"],
            usage_ctx=usage_ctx,
        )
        return result, True
    except gateway.GatewayError as exc:
        status = getattr(exc, "status_code", 0)
        if tools and status in (400, 404, 422, 500, 501):
            if choice.get("provider") != "aitunnel":
                # Локальная модель без поддержки function calling: продолжаем на внешней дешёвой,
                # чтобы агент не начал выдумывать данные вместо вызова инструментов.
                fallback = MODEL_CHOICES[FALLBACK_CHOICE]
                ctx.notes.append(
                    f"{choice['label']} не поддерживает инструменты — прогон продолжен на {fallback['label']}"
                )
                ctx.model_choice = FALLBACK_CHOICE
                await ctx.event({"type": "log", "level": "error", "message": f"Модель переключена на {fallback['label']}"})
                result = await gateway.achat(
                    provider=fallback["provider"],
                    messages=messages,
                    temperature=0.2,
                    max_tokens=max_tokens,
                    timeout=300,
                    extra=extra,
                    profile=fallback["profile"],
                    usage_ctx=usage_ctx,
                )
                return result, True
            ctx.notes.append(f"модель {choice['label']} не приняла tools (HTTP {status}), включён JSON-протокол")
            result = await gateway.achat(
                provider=choice["provider"],
                messages=messages,
                temperature=0.2,
                max_tokens=max_tokens,
                timeout=300,
                profile=choice["profile"],
                usage_ctx=usage_ctx,
            )
            return result, False
        raise


def _parse_message(result: Any) -> Tuple[str, List[dict]]:
    raw = getattr(result, "raw", None) or {}
    choices = raw.get("choices") or [{}]
    message = (choices[0] or {}).get("message") or {}
    content = message.get("content") or getattr(result, "content", "") or ""
    calls = message.get("tool_calls") or []
    return content, calls


def _usage_tokens(result: Any) -> int:
    raw = getattr(result, "raw", None) or {}
    usage = raw.get("usage") or {}
    try:
        return int(usage.get("total_tokens") or 0)
    except Exception:
        return 0


def _usage_parts(result: Any) -> Tuple[int, int, int]:
    raw = getattr(result, "raw", None) or {}
    usage = raw.get("usage") or {}
    try:
        prompt = int(usage.get("prompt_tokens") or 0)
        completion = int(usage.get("completion_tokens") or 0)
        total = int(usage.get("total_tokens") or (prompt + completion))
    except Exception:
        return 0, 0, 0
    return prompt, completion, total


def _account(ctx, result: Any) -> None:
    """Пишет расход токенов и денег в контекст прогона."""
    prompt, completion, total = _usage_parts(result)
    choice = MODEL_CHOICES.get(ctx.model_choice or DEFAULT_CHOICE) or MODEL_CHOICES[DEFAULT_CHOICE]
    ctx.tokens += total
    ctx.cost_usd += (prompt * float(choice.get("price_in") or 0) + completion * float(choice.get("price_out") or 0)) / 1_000_000.0


def _compact_history(messages: List[dict]) -> None:
    """Сжимает старые результаты инструментов: они уже использованы, но весят больше всего."""
    indexes = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    # в резервном JSON-протоколе результаты лежат в user-сообщениях
    indexes += [i for i, m in enumerate(messages) if m.get("role") == "user" and str(m.get("content") or "").startswith("Результат инструмента")]
    for pos, idx in enumerate(indexes):
        if pos >= len(indexes) - TOOL_RESULT_RECENT_FULL:
            continue
        content = str(messages[idx].get("content") or "")
        if len(content) > TOOL_RESULT_KEEP_CHARS:
            messages[idx]["content"] = content[:TOOL_RESULT_KEEP_CHARS] + " … [результат сжат для экономии токенов]"


def _budget_exceeded(ctx) -> bool:
    """Бюджет считаем с запасом на один шаг, чтобы не выходить за заявленный лимит."""
    if not ctx.token_budget:
        return False
    return ctx.tokens >= max(1000, int(ctx.token_budget) - 8000)


def _tool_payload(outcome: Dict[str, Any]) -> Dict[str, Any]:
    """Компактный результат инструмента для передачи модели."""
    if outcome.get("ok"):
        return {"ok": True, "result": outcome.get("result")}
    return {"ok": False, "error": outcome.get("error")}


async def _run_tool(ctx, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Выполняет один инструмент, стримит шаги в журнал и запоминает вызов."""
    spec = get_tool(name)
    if spec is None or name not in ctx.allowed_tools:
        return {"ok": False, "error": f"инструмент {name} недоступен"}
    await ctx.event({"type": "tool_start", "name": name, "title": spec.title, "args": args})
    outcome = await execute(spec, ctx, args)
    summary = _summarize(name, outcome.get("result")) if outcome.get("ok") else str(outcome.get("error"))[:200]
    ctx.tool_calls.append({"name": name, "args": args, "ok": outcome.get("ok"), "ms": outcome.get("ms"), "summary": summary})
    await ctx.event(
        {
            "type": "tool_end",
            "name": name,
            "title": spec.title,
            "ok": bool(outcome.get("ok")),
            "ms": outcome.get("ms"),
            "summary": summary,
            "error": None if outcome.get("ok") else outcome.get("error"),
        }
    )
    return outcome


async def _ensure_report(ctx, messages: List[dict], tools_schema: List[dict]) -> bool:
    """Гарантирует, что при запросе отчёта файл действительно собран."""
    if "build_report" not in ctx.allowed_tools:
        return False
    if any(call.get("name") == "build_report" for call in ctx.tool_calls):
        return True
    if ctx.out_of_time():
        return False
    instruction = (
        "Собери итоговый отчёт прямо сейчас: вызови инструмент build_report с заголовком, разделами "
        "(динамика, тональность, площадки, инфоповоды, выводы), графиками по их chart_id и ссылками на источники. "
        "Если графиков ещё нет, сначала вызови make_chart 2–4 раза, затем build_report. "
        "Если ранее делал подробный разбор темы (deep_text_analysis) — включи его итоговый текст отдельным разделом "
        "с цитатами и ссылками, а не пересказывай своими словами."
    )
    messages.append({"role": "user", "content": instruction})
    for force in ("build_report", None):
        if ctx.out_of_time():
            break
        try:
            result, _ = await _call_llm(ctx, messages, tools_schema, max_tokens=3000, force_tool=force)
        except Exception as exc:
            await ctx.log(f"Не удалось принудительно собрать отчёт: {exc}", level="error")
            continue
        ctx.llm_calls += 1
        _account(ctx, result)
        content, calls = _parse_message(result)
        if not calls:
            parsed = _extract_json(content or "")
            if parsed and parsed.get("tool"):
                calls = [{"id": "forced", "function": {"name": parsed.get("tool"), "arguments": json.dumps(parsed.get("arguments") or {}, ensure_ascii=False)}}]
        if not calls:
            continue
        for call in calls:
            fn = call.get("function") or {}
            name = str(fn.get("name") or "")
            if name not in ("build_report", "make_chart"):
                continue
            raw_args = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
            except Exception:
                args = {}
            outcome = await _run_tool(ctx, name, args)
            if name == "build_report" and outcome.get("ok"):
                return True
            messages.append(
                {
                    "role": "user",
                    "content": "Результат инструмента " + name + ": " + json_text(_tool_payload(outcome)),
                }
            )
        if any(call.get("name") == "build_report" for call in ctx.tool_calls):
            return True
    return any(call.get("name") == "build_report" for call in ctx.tool_calls)


async def _ensure_data(ctx, messages: List[dict], tools_schema: List[dict]) -> bool:
    """Не даёт агенту отвечать цифрами без данных: принудительно вызывает инструмент."""
    if ctx.tool_calls:
        return True
    candidates = [name for name in ("dataset_overview", "search_messages", "list_datasets") if name in ctx.allowed_tools]
    if not candidates or ctx.out_of_time() or not tools_schema:
        return False
    tool_name = candidates[0]
    messages.append(
        {
            "role": "user",
            "content": f"Ты ещё не обращался к данным. Вызови инструмент {tool_name} прямо сейчас и работай только с его результатами.",
        }
    )
    try:
        result, _ = await _call_llm(ctx, messages, tools_schema, max_tokens=1200, force_tool=tool_name)
    except Exception as exc:
        await ctx.log(f"Не удалось принудительно вызвать инструмент: {exc}", level="error")
        return False
    ctx.llm_calls += 1
    _account(ctx, result)
    content, calls = _parse_message(result)
    if not calls:
        parsed = _extract_json(content or "")
        if parsed and parsed.get("tool"):
            calls = [
                {
                    "id": "forced",
                    "function": {"name": parsed.get("tool"), "arguments": json.dumps(parsed.get("arguments") or {}, ensure_ascii=False)},
                }
            ]
    if not calls:
        return False
    for call in calls:
        fn = call.get("function") or {}
        name = str(fn.get("name") or "")
        if name not in ctx.allowed_tools:
            continue
        raw_args = fn.get("arguments") or "{}"
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
        except Exception:
            args = {}
        outcome = await _run_tool(ctx, name, args)
        messages.append(
            {"role": "user", "content": "Результат инструмента " + name + ": " + json_text(_tool_payload(outcome))}
        )
    return bool(ctx.tool_calls)


async def run_agent(ctx) -> Dict[str, Any]:
    """Основной цикл агента. Возвращает итоговый ответ, шаги и артефакты."""
    allowed = sorted(ctx.allowed_tools)
    tools_schema = openai_tools(allowed, compact=True)
    use_tools = bool(tools_schema)
    messages: List[dict] = [
        {"role": "system", "content": system_prompt(ctx)},
        {"role": "user", "content": user_prompt(ctx)},
    ]
    if not use_tools:
        messages[0]["content"] += (
            "\n\nФормат ответа (инструменты недоступны как функции): отвечай ТОЛЬКО JSON без пояснений: "
            '{"tool": "<имя инструмента>", "arguments": {...}} для вызова инструмента или {"final": "<итоговый ответ>"} для завершения.'
        )
    tool_call_count = 0
    answer = ""
    await ctx.event(
        {
            "type": "start",
            "model": (MODEL_CHOICES.get(ctx.model_choice) or {}).get("label"),
            "tools": allowed,
            "token_budget": ctx.token_budget,
        }
    )

    for step in range(1, MAX_STEPS + 1):
        if ctx.out_of_time():
            ctx.notes.append("истёк лимит времени запуска")
            break
        if tool_call_count >= MAX_TOOL_CALLS:
            ctx.notes.append("достигнут лимит вызовов инструментов")
            break
        if _budget_exceeded(ctx):
            ctx.notes.append(f"достигнут бюджет прогона: {ctx.tokens} токенов из {ctx.token_budget}")
            break
        _compact_history(messages)
        try:
            result, tools_ok = await _call_llm(ctx, messages, tools_schema if use_tools else None)
        except Exception as exc:
            import traceback

            traceback.print_exc()
            await ctx.log(f"Ошибка обращения к модели: {exc}", level="error")
            ctx.notes.append(f"ошибка модели: {exc}")
            break
        if use_tools and not tools_ok:
            use_tools = False
        ctx.llm_calls += 1
        _account(ctx, result)
        content, calls = _parse_message(result)

        if use_tools and calls:
            messages.append({"role": "assistant", "content": content or "", "tool_calls": calls})
            await ctx.event({"type": "llm", "step": step, "planned": [((c.get("function") or {}).get("name")) for c in calls]})
            for call in calls:
                fn = call.get("function") or {}
                name = str(fn.get("name") or "")
                args_raw = fn.get("arguments") or "{}"
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
                except Exception:
                    args = {}
                spec = get_tool(name)
                if spec is None or name not in ctx.allowed_tools:
                    payload = {"ok": False, "error": f"инструмент {name} недоступен"}
                    messages.append({"role": "tool", "tool_call_id": call.get("id"), "name": name, "content": json_text(payload)})
                    await ctx.event({"type": "tool_end", "name": name, "ok": False, "summary": "инструмент недоступен"})
                    continue
                tool_call_count += 1
                outcome = await _run_tool(ctx, name, args)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id"),
                        "name": name,
                        "content": json_text(_tool_payload(outcome)),
                    }
                )
            continue

        # Нет вызовов инструментов: либо финальный ответ, либо JSON-протокол
        if use_tools:
            answer = (content or "").strip()
            await ctx.event({"type": "llm", "step": step, "final": True})
            break

        parsed = _extract_json(content or "")
        if not parsed:
            answer = (content or "").strip()
            break
        if parsed.get("final"):
            answer = str(parsed.get("final")).strip()
            break
        name = str(parsed.get("tool") or "")
        args = parsed.get("arguments") or parsed.get("args") or {}
        spec = get_tool(name)
        if spec is None or name not in ctx.allowed_tools:
            messages.append({"role": "user", "content": f"Инструмент {name} недоступен. Доступны: {', '.join(allowed)}"})
            continue
        tool_call_count += 1
        outcome = await _run_tool(ctx, name, args)
        messages.append(
            {
                "role": "user",
                "content": "Результат инструмента " + name + ": " + json_text(_tool_payload(outcome)),
            }
        )

    # Агент не должен отвечать цифрами, не обратившись к данным
    if not ctx.tool_calls:
        await _ensure_data(ctx, messages, tools_schema)
        if not ctx.tool_calls:
            ctx.notes.append("агент не вызвал ни одного инструмента — ответ не подтверждён данными")

    # Если просили отчёт (или делали подробный разбор текстов) — файл должен быть собран
    deep_analysis_done = any(call.get("name") == "deep_text_analysis" for call in ctx.tool_calls)
    if (wants_report(ctx.task) or deep_analysis_done) and not _budget_exceeded(ctx):
        reported = await _ensure_report(ctx, messages, tools_schema)
        if reported:
            messages.append(
                {
                    "role": "user",
                    "content": "Отчёт собран. Дай короткое резюме для чата: 5–8 строк с ключевыми цифрами и перечнем того, что вошло в отчёт.",
                }
            )
            try:
                brief_result, _ = await _call_llm(ctx, messages, None, max_tokens=900)
                brief = (_parse_message(brief_result)[0] or "").strip()
                ctx.llm_calls += 1
                _account(ctx, brief_result)
                if brief:
                    answer = brief
            except Exception:
                pass

    if not answer:
        if ctx.out_of_time():
            messages.append({"role": "user", "content": "Время вышло. Сформулируй итог по уже собранным данным, без новых вызовов инструментов."})
        else:
            messages.append({"role": "user", "content": "Заверши работу: сформулируй итоговый ответ по собранным данным, без новых вызовов инструментов."})
        try:
            final_result, _ = await _call_llm(ctx, messages, None, max_tokens=2500)
            answer = (_parse_message(final_result)[0] or "").strip()
            ctx.llm_calls += 1
            _account(ctx, final_result)
        except Exception as exc:
            answer = "Не удалось получить итоговый ответ: " + str(exc)

    if answer and not ctx.tool_calls:
        answer = NO_DATA_WARNING + "\n\n" + answer

    if answer:
        await ctx.event({"type": "answer", "text": answer})
    stats = {
        "llm_calls": ctx.llm_calls,
        "tool_calls": len(ctx.tool_calls),
        "tokens": ctx.tokens,
        "cost_usd": round(ctx.cost_usd, 4),
        "token_budget": ctx.token_budget,
        "model": (MODEL_CHOICES.get(ctx.model_choice) or {}).get("label"),
        "artifacts": len(ctx.artifacts),
        "tools_used": sorted({c["name"] for c in ctx.tool_calls}),
        "notes": ctx.notes,
    }
    await ctx.event({"type": "final", "answer": answer, "artifacts": ctx.artifacts, "stats": stats})
    return {"answer": answer, "stats": stats, "tool_calls": ctx.tool_calls, "artifacts": ctx.artifacts, "no_data": bool(getattr(ctx, "no_data", False))}
