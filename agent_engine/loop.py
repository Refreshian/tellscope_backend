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

MODEL_CHOICES: Dict[str, Dict[str, str]] = {
    "claude": {"provider": "aitunnel", "profile": "smart_agent_planner", "label": "Claude Sonnet 4.5"},
    "gpt": {"provider": "aitunnel", "profile": "dashboard_qa", "label": "GPT-4.1 mini"},
    "qwen": {"provider": "vllm", "profile": "agent", "label": "Qwen3-32B (локальная GPU)"},
}
DEFAULT_CHOICE = "claude"


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
    return "\n".join(lines)


def _summarize(name: str, result: Any) -> str:
    """Короткое человекочитаемое описание результата инструмента для журнала."""
    try:
        if not isinstance(result, dict):
            return "готово"
        if name == "search_messages":
            return f"найдено {result.get('messages_found')} сообщений" + (f" по «{result.get('phrase')}»" if result.get("phrase") else "")
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


async def _call_llm(ctx, messages: List[dict], tools: Optional[List[dict]], max_tokens: int = 3000) -> Tuple[Any, bool]:
    """Возвращает (ChatResult, tools_supported)."""
    gateway = _gateway()
    choice = MODEL_CHOICES.get(ctx.model_choice or DEFAULT_CHOICE) or MODEL_CHOICES[DEFAULT_CHOICE]
    extra: Dict[str, Any] = {}
    if tools:
        extra = {"tools": tools, "tool_choice": "auto", "parallel_tool_calls": False}
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


def _tool_payload(outcome: Dict[str, Any]) -> Dict[str, Any]:
    """Компактный результат инструмента для передачи модели."""
    if outcome.get("ok"):
        return {"ok": True, "result": outcome.get("result")}
    return {"ok": False, "error": outcome.get("error")}


async def run_agent(ctx) -> Dict[str, Any]:
    """Основной цикл агента. Возвращает итоговый ответ, шаги и артефакты."""
    allowed = sorted(ctx.allowed_tools)
    tools_schema = openai_tools(allowed)
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
    await ctx.event({"type": "start", "model": (MODEL_CHOICES.get(ctx.model_choice) or {}).get("label"), "tools": allowed})

    for step in range(1, MAX_STEPS + 1):
        if ctx.out_of_time():
            ctx.notes.append("истёк лимит времени запуска")
            break
        if tool_call_count >= MAX_TOOL_CALLS:
            ctx.notes.append("достигнут лимит вызовов инструментов")
            break
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
        ctx.tokens += _usage_tokens(result)
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
                await ctx.event({"type": "tool_start", "name": name, "title": spec.title, "args": args})
                outcome = await execute(spec, ctx, args)
                summary = _summarize(name, outcome.get("result")) if outcome.get("ok") else str(outcome.get("error"))[:200]
                ctx.tool_calls.append({"name": name, "args": args, "ok": outcome.get("ok"), "ms": outcome.get("ms"), "summary": summary})
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id"),
                        "name": name,
                        "content": json_text(_tool_payload(outcome)),
                    }
                )
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
        await ctx.event({"type": "tool_start", "name": name, "title": spec.title, "args": args})
        outcome = await execute(spec, ctx, args)
        summary = _summarize(name, outcome.get("result")) if outcome.get("ok") else str(outcome.get("error"))[:200]
        ctx.tool_calls.append({"name": name, "args": args, "ok": outcome.get("ok"), "ms": outcome.get("ms"), "summary": summary})
        messages.append(
            {
                "role": "user",
                "content": "Результат инструмента " + name + ": " + json_text(_tool_payload(outcome)),
            }
        )
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

    if not answer:
        if ctx.out_of_time():
            messages.append({"role": "user", "content": "Время вышло. Сформулируй итог по уже собранным данным, без новых вызовов инструментов."})
        else:
            messages.append({"role": "user", "content": "Заверши работу: сформулируй итоговый ответ по собранным данным, без новых вызовов инструментов."})
        try:
            final_result, _ = await _call_llm(ctx, messages, None, max_tokens=2500)
            answer = (_parse_message(final_result)[0] or "").strip()
            ctx.llm_calls += 1
            ctx.tokens += _usage_tokens(final_result)
        except Exception as exc:
            answer = "Не удалось получить итоговый ответ: " + str(exc)

    if answer:
        await ctx.event({"type": "answer", "text": answer})
    stats = {
        "llm_calls": ctx.llm_calls,
        "tool_calls": len(ctx.tool_calls),
        "tokens": ctx.tokens,
        "artifacts": len(ctx.artifacts),
        "tools_used": sorted({c["name"] for c in ctx.tool_calls}),
        "notes": ctx.notes,
    }
    await ctx.event({"type": "final", "answer": answer, "artifacts": ctx.artifacts, "stats": stats})
    return {"answer": answer, "stats": stats, "tool_calls": ctx.tool_calls, "artifacts": ctx.artifacts}
