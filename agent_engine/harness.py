# -*- coding: utf-8 -*-
"""Центр задач Tellscope — ассистент ИИ-аналитики соцмедиа и СМИ.

Пользователь описывает задачу обычным текстом, а ассистент:

  * ``explain`` — объясняет, как задачу решить (план, инструменты, что получится);
  * ``run``     — сразу выполняет её агентным циклом Tellscope (инструменты + отчёт);
  * ``chain``   — собирает цепочку шагов и сохраняет её как агента во вкладке «Мои агенты»;
  * ``flow``    — генерирует Dify-workflow (DSL-файл), который импортируется в конструктор Dify.

Задачи хранятся по пользователю: ``data/<user_id>/harness_tasks.json`` — каждый видит только свои.
Сгенерированные DSL лежат рядом: ``data/<user_id>/harness_flows/``.
"""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from . import agents as agents_store
from . import registry as tool_registry

BACKEND_ROOT = "/home/dev/tellscope_app/tellscope_backend"
HARNESS_DIR_NAME = "harness_flows"

MODES: List[Dict[str, str]] = [
    {"id": "explain", "title": "Объясни, как это сделать", "hint": "план шагов и рекомендации без запуска"},
    {"id": "run", "title": "Выполнить сейчас", "hint": "агент сам вызовет инструменты и соберёт отчёт"},
    {"id": "chain", "title": "Собрать цепочку", "hint": "готовый агент с шагами в «Мои агенты»"},
    {"id": "flow", "title": "Собрать Dify-flow", "hint": "DSL-файл для визуального конструктора Dify"},
]

DEFAULT_MODEL = "deepseek"
MAX_CHAIN_STEPS = 12

# UUID провайдера-инструмента tellscope в Dify (для генерации DSL)
DIFY_PROVIDER_ID = os.environ.get("DIFY_TOOL_PROVIDER_ID", "b2eff1d5-6432-4631-aa0a-a2b0a36e3f3b")
DIFY_PUBLIC_URL = os.environ.get("DIFY_PUBLIC_URL", "https://tellscope40.headsmade.com:8443")


# ------------------------------------------------------------------ хранилище

def _store_path(user_id: Any) -> str:
    return os.path.join(BACKEND_ROOT, "data", str(user_id), "harness_tasks.json")


def list_tasks(user_id: Any, limit: int = 60) -> List[Dict[str, Any]]:
    path = _store_path(user_id)
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh) or {}
    except Exception:
        return []
    items = data.get("tasks") if isinstance(data, dict) else data
    items = [item for item in (items or []) if isinstance(item, dict)]
    items.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    return items[:limit]


def save_tasks(user_id: Any, items: List[Dict[str, Any]]) -> None:
    path = _store_path(user_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"tasks": items[:200]}, fh, ensure_ascii=False, indent=1)


def get_task(user_id: Any, task_id: str) -> Optional[Dict[str, Any]]:
    for item in list_tasks(user_id, limit=200):
        if str(item.get("id")) == str(task_id):
            return item
    return None


def update_task(user_id: Any, task_id: str, patch: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    items = list_tasks(user_id, limit=200)
    found = None
    for item in items:
        if str(item.get("id")) == str(task_id):
            item.update(patch or {})
            found = item
            break
    if found is not None:
        save_tasks(user_id, items)
    return found


def delete_task(user_id: Any, task_id: str) -> bool:
    items = list_tasks(user_id, limit=200)
    left = [item for item in items if str(item.get("id")) != str(task_id)]
    if len(left) == len(items):
        return False
    save_tasks(user_id, left)
    return True


def create_task(user_id: Any, text: str, mode: str, index: Optional[int], dataset_name: str = "") -> Dict[str, Any]:
    task = {
        "id": "ht_" + uuid.uuid4().hex[:10],
        "user_id": str(user_id),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "text": (text or "").strip()[:4000],
        "mode": mode,
        "status": "new",
        "dataset_index": index,
        "dataset_name": dataset_name,
        "run_id": None,
        "result": None,
        "answer": "",
        "error": None,
    }
    items = list_tasks(user_id, limit=200)
    items.insert(0, task)
    save_tasks(user_id, items)
    return task


# ------------------------------------------------------------------- модель

def _model_choice(name: Optional[str]) -> str:
    """Выбор модели: deepseek по умолчанию, с откатом на gpt, если профиль недоступен."""
    from .loop import MODEL_CHOICES

    if name and name in MODEL_CHOICES:
        return name
    return DEFAULT_MODEL if DEFAULT_MODEL in MODEL_CHOICES else "gpt"


async def _llm(user: Any, messages: List[Dict[str, str]], *, model_choice: Optional[str] = None,
               temperature: float = 0.2, max_tokens: int = 1400, timeout: float = 300.0) -> Tuple[str, Dict[str, Any]]:
    """Один вызов модели через общий шлюз Tellscope. Возвращает (текст, учёт)."""
    from mlops import gateway
    from .loop import MODEL_CHOICES, DEFAULT_CHOICE

    choice_key = _model_choice(model_choice)
    choice = MODEL_CHOICES.get(choice_key) or MODEL_CHOICES[DEFAULT_CHOICE]
    extra = {"chat_template_kwargs": {"enable_thinking": False}} if choice.get("provider") == "vllm" else None
    try:
        result = await gateway.achat(
            provider=choice["provider"],
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            extra=extra,
            profile=choice["profile"],
            usage_ctx={"user_id": str(getattr(user, "id", "")), "case": "harness"},
        )
    except Exception as exc:  # noqa: BLE001
        # DeepSeek может быть недоступен в профиле — падаем на дефолтную модель, чтобы задача не терялась
        if choice_key != DEFAULT_CHOICE:
            fallback = MODEL_CHOICES[DEFAULT_CHOICE]
            result = await gateway.achat(
                provider=fallback["provider"],
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                extra=None,
                profile=fallback["profile"],
                usage_ctx={"user_id": str(getattr(user, "id", "")), "case": "harness"},
            )
            choice = fallback
        else:
            raise
    usage = (getattr(result, "raw", None) or {}).get("usage") or {}
    text = re.sub(r"<think>.*?</think>", "", getattr(result, "content", "") or "", flags=re.S | re.I).strip()
    account = {
        "model": choice.get("label"),
        "provider": choice.get("provider"),
        "tokens": int(usage.get("total_tokens") or 0),
        "cost_usd": round(
            (
                int(usage.get("prompt_tokens") or 0) * float(choice.get("price_in") or 0)
                + int(usage.get("completion_tokens") or 0) * float(choice.get("price_out") or 0)
            )
            / 1_000_000.0,
            4,
        ),
    }
    return text, account


def _json_block(text: str) -> Optional[Any]:
    """Достаёт JSON из ответа модели (в том числе из ```json ... ```)."""
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, flags=re.S)
    candidate = fence.group(1) if fence else text
    for opener, closer in (("{", "}"), ("[", "]")):
        start = candidate.find(opener)
        end = candidate.rfind(closer)
        if start >= 0 and end > start:
            chunk = candidate[start:end + 1]
            try:
                return json.loads(chunk)
            except Exception:
                continue
    return None


# -------------------------------------------------------- описание возможностей

def _tools_brief() -> str:
    """Краткое описание инструментов для промпта планировщика."""
    lines = []
    for spec in sorted(tool_registry.all_tools().values(), key=lambda s: s.name):
        if spec.group == "connectors":
            continue
        params = (spec.parameters or {}).get("properties") or {}
        keys = ", ".join(list(params)[:9])
        lines.append(f"- {spec.name} ({spec.title}): {spec.description} Параметры: {keys}")
    return "\n".join(lines)


PLANNER_SYSTEM = (
    "Ты ИИ-ассистент платформы Tellscope — аналитика соцмедиа и СМИ. "
    "Если нужной темы нет среди датасетов, её можно выгрузить из Brand Analytics инструментом fetch_dataset. "
    "Ты знаешь инструменты платформы, её конструктор цепочек и визуальный конструктор Dify. "
    "ВАЖНО: любой план или цепочка, которая заканчивается отчётом, обязана включать шаг чтения текстов "
    "инструментом analyze_texts ПЕРЕД сборкой отчёта: темы, доли и цитаты из текстов — основа выводов, "
    "а статистика и графики только дополняют их. "
    "Отвечай деловым русским языком, без воды, только по делу. Если данных не хватает — прямо скажи, "
    "какой датасет или период нужен."
)


def _plan_prompt(text: str, index: Optional[int], dataset_name: str) -> str:
    return (
        f"Задача пользователя: {text}\n\n"
        f"Датасет: {index if index is not None else 'не выбран'} ({dataset_name or 'название неизвестно'}).\n\n"
        "Инструменты Tellscope:\n" + _tools_brief() + "\n\n"
        "Обязательное правило: если в плане есть отчёт (build_report), перед ним должен быть шаг analyze_texts — "
        "чтение текстов сообщений локальной моделью (темы, доли, цитаты). Пояснения к графикам опираются на "
        "темы и цитаты из текстов, а не только на статистику.\n\n"
        "Ответь СТРОГО одним JSON-объектом без пояснений вокруг:\n"
        "{\n"
        '  "summary": "в двух-трёх предложениях, что будет сделано и какой результат получит пользователь",\n'
        '  "steps": [{"title": "шаг", "tool": "имя_инструмента", "why": "зачем", "params": {"ключ": "значение"}}],\n'
        '  "outputs": ["что будет на выходе: отчёт DOCX/PDF, графики, таблица"],\n'
        '  "cautions": ["на что обратить внимание: мало данных, нужен период"],\n'
        '  "can_run_now": true,\n'
        '  "mode_hint": "run | chain | flow"\n'
        "}\n"
    )


CHAINS_SYSTEM = (
    "Ты ИИ-ассистент Tellscope и собираешь цепочки шагов. Ты собираешь цепочки шагов, "
    "которые выполняются детерминированно: шаг → шаг → отчёт. Пиши по-русски и только валидный JSON."
)


def _chain_prompt(text: str, index: Optional[int], dataset_name: str) -> str:
    return (
        f"Задача пользователя: {text}\n\n"
        f"Датасет: {index if index is not None else 'не выбран'} ({dataset_name or ''}).\n\n"
        "Инструменты Tellscope:\n" + _tools_brief() + "\n\n"
        "Собери цепочку шагов. Формат шага (только эти поля и виды):\n"
        '{"kind":"tool","title":"Поиск по теме","tool":"search_messages","args":{"phrase":"..."},"save_as":"search"}\n'
        '{"kind":"tool","title":"Чтение текстов","tool":"analyze_texts","args":{"tone":"all","limit":90},"save_as":"texts"}\n'
        '{"kind":"chart","title":"Тональность","from":"{{search.tonality}}","label_field":"tone","value_field":"count",'
        '"chart_type":"pie","save_as":"chart_tone"}\n'
        '{"kind":"llm","title":"Выводы","prompt":"... {{search.messages_found}} ... темы и цитаты: {{texts.topics}} ...","save_as":"synthesis"}\n'
        '{"kind":"report","title":"Отчёт","report_title":"...","subtitle":"...","save_as":"report",'
        '"sections":[{"heading":"Темы и цитаты","text":"{{texts.summary}}","findings":"{{texts.report_section.findings}}",'
        '"highlights":"{{texts.report_section.highlights}}"},{"heading":"Раздел","text":"{{synthesis.text}}","chart_ids":["chart1"]}]}\n\n'
        "Правила: index/min_date/max_date подставляются автоматически, их указывать не нужно; "
        "ссылки вида {{save_as.поле}} работают для шагов, выполненных раньше; "
        "если в цепочке есть отчёт, шаг analyze_texts перед ним ОБЯЗАТЕЛЕН, а раздел отчёта с текстовыми "
        "находками передаёт findings и highlights из {{texts.report_section}} — так в DOCX/PDF попадают темы, "
        "доли и цитаты; пояснения к графикам опирай на темы и цитаты из текстов; "
        f"не больше {MAX_CHAIN_STEPS} шагов; в конце обязательно шаг report с разделами.\n\n"
        "Ответь СТРОГО одним JSON-объектом:\n"
        '{"name": "название агента", "description": "что делает", "instruction": "короткая инструкция", '
        '"folder": "папка отчётов", "steps": [ ... ]}\n'
    )


FLOWS_SYSTEM = (
    "Ты ИИ-ассистент Tellscope и собираешь workflow для визуального конструктора Dify. Ты описываешь цепочку узлов Dify для аналитики "
    "соцмедиа и СМИ и отвечаешь только валидным JSON."
)


def _flow_prompt(text: str, index: Optional[int], dataset_name: str) -> str:
    return (
        f"Задача пользователя: {text}\n\n"
        f"Датасет: {index if index is not None else 'не выбран'} ({dataset_name or ''}).\n\n"
        "Инструменты Tellscope:\n" + _tools_brief() + "\n\n"
        "Опиши линейный workflow для Dify: старт → вызовы инструментов → разбор моделью → отчёт.\n"
        "Ответь СТРОГО одним JSON-объектом:\n"
        "{\n"
        '  "title": "название приложения",\n'
        '  "subtitle": "подзаголовок отчёта",\n'
        '  "folder": "папка отчёта в Tellscope",\n'
        '  "tools": [{"tool": "search_messages", "title": "Поиск по теме", "args": {"phrase": "{{subtopic}}"}}],\n'
        '  "report_sections": [{"heading": "Тональность и динамика", "ask": "что написать в этом разделе"}],\n'
        '  "conclusions": "что должно быть в выводах"\n'
        "}\n"
        "В args можно ссылаться на поля ввода: {{subtopic}}, {{focus}} — они станут полями формы в Dify.\n"
    )


# ------------------------------------------------------------- режим explain

async def explain(user: Any, text: str, index: Optional[int] = None, dataset_name: str = "",
                  model_choice: Optional[str] = None) -> Dict[str, Any]:
    """План решения задачи без выполнения."""
    raw, account = await _llm(
        user,
        [
            {"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user", "content": _plan_prompt(text, index, dataset_name)},
        ],
        model_choice=model_choice,
        temperature=0.2,
        max_tokens=1400,
    )
    data = _json_block(raw)
    if not isinstance(data, dict):
        data = {"summary": raw or "Не удалось разобрать ответ модели", "steps": [], "outputs": [], "cautions": []}
    steps = []
    known = tool_registry.all_tools()
    for step in (data.get("steps") or [])[:MAX_CHAIN_STEPS]:
        if not isinstance(step, dict):
            continue
        name = str(step.get("tool") or "").strip()
        steps.append(
            {
                "title": str(step.get("title") or step.get("tool") or "шаг")[:200],
                "tool": name if name in known else "",
                "tool_title": (known.get(name).title if name in known else ""),
                "why": str(step.get("why") or "")[:400],
                "params": step.get("params") if isinstance(step.get("params"), dict) else {},
            }
        )
    return {
        "summary": str(data.get("summary") or "")[:2000],
        "steps": steps,
        "outputs": [str(item)[:200] for item in (data.get("outputs") or [])][:8],
        "cautions": [str(item)[:300] for item in (data.get("cautions") or [])][:6],
        "can_run_now": bool(data.get("can_run_now", True)),
        "mode_hint": str(data.get("mode_hint") or "run")[:20],
        "model": account,
    }


# --------------------------------------------------------------- режим chain

async def make_chain(user: Any, text: str, index: Optional[int] = None, dataset_name: str = "",
                     model_choice: Optional[str] = None) -> Dict[str, Any]:
    """Собирает цепочку шагов и сохраняет её как агента пользователя."""
    raw, account = await _llm(
        user,
        [
            {"role": "system", "content": CHAINS_SYSTEM},
            {"role": "user", "content": _chain_prompt(text, index, dataset_name)},
        ],
        model_choice=model_choice,
        temperature=0.15,
        max_tokens=2400,
    )
    data = _json_block(raw)
    if not isinstance(data, dict):
        raise ValueError("модель не вернула корректную цепочку шагов, попробуйте переформулировать задачу")
    steps = agents_store.normalize_steps(data.get("steps"))[:MAX_CHAIN_STEPS]
    if not steps:
        raise ValueError("в цепочке не оказалось ни одного валидного шага")

    agent = agents_store.upsert_agent(
        user.id,
        {
            "name": str(data.get("name") or f"Задача: {text[:60]}")[:200],
            "description": str(data.get("description") or f"Цепочка собрана ассистентом по задаче: {text[:200]}"),
            "instruction": str(data.get("instruction") or text)[:2000],
            "folder": str(data.get("folder") or "Центр задач")[:80],
            "dataset_index": index,
            "dataset_name": dataset_name,
            "model": _model_choice(model_choice),
            "steps": steps,
            "tools": sorted({str(step.get("tool")) for step in steps if step.get("tool")}),
            "schedule": {"enabled": False, "mode": "manual", "hour": 9, "minute": 0, "weekdays": [1, 2, 3, 4, 5]},
        },
    )
    return {"agent": agent, "steps": steps, "model": account}


# ---------------------------------------------------------------- режим flow

def dataset_options(limit: int = 15) -> List[str]:
    """Понятные варианты темы для выпадающего списка в Dify (подпись · период)."""
    from .tools_data import datasets_public

    bad = ("zz", "tiny", "probe", "test", "demo", "tmp", "sample", "converted", "аукцион", ".doc", ".txt", "serv cros")
    options: List[str] = []
    for item in datasets_public():
        name = str(item.get("name") or "").lower()
        if any(marker in name for marker in bad):
            continue
        label = str(item.get("label") or item.get("name"))
        period = str(item.get("period") or "")
        options.append(f"{label} · {period}" if period else label)
        if len(options) >= limit:
            break
    return options


def _yaml_value(value: Any) -> str:
    return "'" + str(value).replace("'", "''") + "'"


# Поля формы в сгенерированном Dify-workflow
START_FIELDS = ("index", "subtopic", "date_from", "date_to", "focus")


def _normalize_ref(value: str) -> str:
    """{{subtopic}} → {{#start.subtopic#}} — модель пишет короткие ссылки, Dify ждёт полные."""
    def repl(match: "re.Match[str]") -> str:
        name = match.group(1).strip()
        return "{{#start.%s#}}" % name if name in START_FIELDS else match.group(0)

    return re.sub(r"\{\{\s*#?\s*([A-Za-z0-9_]+)\s*\}?\s*\}\}", repl, value)


def build_dify_dsl(spec: Dict[str, Any], index: Optional[int]) -> str:
    """Собирает DSL-файл Dify: старт → инструменты → разбор моделью → отчёт → End."""
    title = str(spec.get("title") or "Tellscope — задача")[:120]
    subtitle = str(spec.get("subtitle") or "Аналитика соцмедиа и СМИ")[:200]
    folder = str(spec.get("folder") or "Центр задач")[:80]
    tools = [item for item in (spec.get("tools") or []) if isinstance(item, dict) and item.get("tool")][:10]
    sections = [item for item in (spec.get("report_sections") or []) if isinstance(item, dict)][:6] or [
        {"heading": "Аналитика и выводы", "ask": "ключевые факты, тональность и выводы"}
    ]
    conclusions = str(spec.get("conclusions") or "5 нумерованных выводов и рекомендаций")[:600]

    nodes: List[str] = []
    edges: List[str] = []
    order: List[str] = ["start"]

    def node(node_id: str, data_lines: str, x: int, height: int = 120) -> str:
        body = "\n".join("        " + line if line else "" for line in data_lines.split("\n"))
        return (
            "    - data:\n" + body + "\n"
            f"      height: {height}\n"
            f"      id: {_yaml_value(node_id)}\n"
            "      position:\n"
            f"        x: {x}\n"
            "        y: 300\n"
            "      positionAbsolute:\n"
            f"        x: {x}\n"
            "        y: 300\n"
            "      selected: false\n"
            "      sourcePosition: right\n"
            "      targetPosition: left\n"
            "      type: custom\n"
            "      width: 244\n"
        )

    theme_lines = ["        - label: 'Тема (набор данных)'", "          max_length: null", "          options:"]
    for option in dataset_options():
        theme_lines.append("          - " + _yaml_value(option))
    theme_lines += ["          required: true", "          type: select", "          variable: dataset"]
    start_vars = (
        "\n".join(theme_lines) + "\n"
        "        - label: 'Подтема: поисковые слова'\n          max_length: null\n          options: []\n"
        "          required: true\n          type: text-input\n          variable: subtopic\n"
        "        - label: 'Период с (YYYY-MM-DD)'\n          max_length: null\n          options: []\n"
        "          required: false\n          type: text-input\n          variable: date_from\n"
        "        - label: 'Период по (YYYY-MM-DD)'\n          max_length: null\n          options: []\n"
        "          required: false\n          type: text-input\n          variable: date_to\n"
        "        - label: 'На что смотреть в первую очередь'\n          max_length: null\n          options: []\n"
        "          required: false\n          type: text-input\n          variable: focus\n"
    )
    nodes.append(
        node(
            "start",
            "desc: ''\ntitle: Начало\ntype: start\nvariables:\n" + start_vars + "selected: false",
            30,
            height=200,
        )
    )

    x = 300
    tool_ids: List[Tuple[str, str]] = []
    for position, item in enumerate(tools, start=1):
        node_id = f"tool{position}"
        tool_name = str(item.get("tool"))
        spec_tool = tool_registry.get_tool(tool_name)
        if spec_tool is None:
            continue
        args = item.get("args") if isinstance(item.get("args"), dict) else {}
        # index/период всегда берём из формы Dify, а не из придуманных моделью значений
        normalized: Dict[str, str] = {}
        for key, value in args.items():
            key = str(key)
            if key == "index":
                normalized[key] = "{{#start.dataset#}}"
            elif key == "min_date":
                normalized[key] = "{{#start.date_from#}}"
            elif key == "max_date":
                normalized[key] = "{{#start.date_to#}}"
            else:
                normalized[key] = _normalize_ref(str(value))
        normalized.setdefault("index", "{{#start.dataset#}}")
        normalized.setdefault("min_date", "{{#start.date_from#}}")
        normalized.setdefault("max_date", "{{#start.date_to#}}")

        params: List[Tuple[str, str, str]] = []
        for key, text_value in normalized.items():
            if "{{" in text_value:
                params.append((key, text_value, "variable"))
            else:
                params.append((key, text_value, "constant"))

        lines = [
            "desc: ''",
            f"title: {_yaml_value(str(item.get('title') or spec_tool.title))}",
            "type: tool",
            f"provider_id: {_yaml_value(DIFY_PROVIDER_ID)}",
            "provider_type: api",
            f"provider_name: 'tellscope'",
            f"tool_name: {_yaml_value(tool_name)}",
            f"tool_label: {_yaml_value(spec_tool.title)}",
            "tool_configurations: {}",
            "is_team_authorization: true",
            "tool_node_version: '2'",
            "tool_parameters:",
        ]
        for key, value, kind in params:
            lines.append(f"  {key}:")
            if kind == "constant":
                lines.append("    type: constant")
                lines.append(f"    value: {_yaml_value(value)}")
            else:
                ref = re.match(r"^\{\{#([A-Za-z0-9_]+)\.([A-Za-z0-9_]+)#\}\}$", value)
                if ref:
                    lines.append("    type: variable")
                    lines.append("    value:")
                    lines.append(f"    - {_yaml_value(ref.group(1))}")
                    lines.append(f"    - {_yaml_value(ref.group(2))}")
                else:
                    lines.append("    type: mixed")
                    lines.append(f"    value: {_yaml_value(value)}")
        lines.append("selected: false")
        nodes.append(node(node_id, "\n".join(lines), x, height=190))
        order.append(node_id)
        tool_ids.append((node_id, str(item.get("title") or spec_tool.title)))
        x += 270

    # узел модели: пишет разделы отчёта
    prompt_lines = [f"Задача: {title}. Подтема: {{{{#start.subtopic#}}}}."]
    for pos, section in enumerate(sections, start=1):
        heading = str(section.get("heading") or f"Раздел {pos}")
        ask = str(section.get("ask") or "")
        prompt_lines.append(f"\nРаздел «{heading}»: {ask}")
        if tool_ids:
            refs = ", ".join(f"{{{{#{node_id}.text#}}}}" for node_id, _ in tool_ids[:7])
            prompt_lines.append(f"Данные: {refs}." if pos == 1 else "")
    prompt_lines.append(f"\nВ конце: {conclusions}. Только факты из данных, без вводных фраз.")
    prompt = "\n".join(prompt_lines)
    nodes.append(
        node(
            "summarize",
            "context:\n  enabled: false\n  variable_selector: []\ndesc: ''\nmemory:\n  query_prompt_template: ''\n"
            "  window:\n    enabled: false\n    size: 10\nmodel:\n  completion_params:\n    temperature: 0.3\n"
            "    max_tokens: 1500\n  mode: chat\n  name: 'gpt-4.1-mini'\n"
            "  provider: 'langgenius/openai_api_compatible/openai_api_compatible'\nprompt_template:\n"
            "  - role: system\n    text: 'Ты аналитик соцмедиа и СМИ. Пиши деловым русским языком, только по данным.'\n"
            "  - role: user\n    text: " + _yaml_value(prompt) + "\nselected: false\n"
            f"title: {_yaml_value('Разбор моделью')}\ntype: llm\nvariables: []\nvision:\n  enabled: false",
            x,
            height=170,
        )
    )
    order.append("summarize")
    x += 270

    # отчёт
    report_params = [
        ("title", title, "constant"),
        ("subtitle", subtitle, "constant"),
        ("folder", folder, "constant"),
        ("author", "ИИ-ассистент Tellscope", "constant"),
    ]
    lines = [
        "desc: ''",
        f"title: {_yaml_value('Сборка отчёта')}",
        "type: tool",
        f"provider_id: {_yaml_value(DIFY_PROVIDER_ID)}",
        "provider_type: api",
        "provider_name: 'tellscope'",
        "tool_name: 'build_report'",
        "tool_label: 'Собрать отчёт (DOCX/PDF)'",
        "tool_configurations: {}",
        "is_team_authorization: true",
        "tool_node_version: '2'",
        "tool_parameters:",
    ]
    for key, value, kind in report_params:
        lines.append(f"  {key}:")
        lines.append("    type: constant")
        lines.append(f"    value: {_yaml_value(value)}")
    lines.append("  sections:")
    lines.append("    type: mixed")
    lines.append("    value: " + _yaml_value("[{\"heading\": \"" + str(sections[0].get('heading') or 'Аналитика') +
                                           "\", \"text\": \"{{#summarize.text#}}\"}]"))
    lines.append("selected: false")
    nodes.append(node("report", "\n".join(lines), x, height=180))
    order.append("report")
    x += 270

    nodes.append(
        node(
            "end",
            "desc: ''\noutputs:\n  - value_selector:\n    - summarize\n    - text\n    value_type: string\n"
            "    variable: analysis\n  - value_selector:\n    - report\n    - text\n    value_type: string\n"
            "    variable: report_files\nselected: false\ntitle: Конец\ntype: end",
            x,
            height=130,
        )
    )
    order.append("end")

    types = {"start": "start", "end": "end", "summarize": "llm", "report": "tool"}
    for source, target in zip(order, order[1:]):
        edges.append(
            "    - data:\n        isInIteration: false\n        isInLoop: false\n"
            f"        sourceType: {types.get(source, 'tool')}\n        targetType: {types.get(target, 'tool')}\n"
            f"      id: {source}-{target}\n      source: {_yaml_value(source)}\n      sourceHandle: source\n"
            f"      target: {_yaml_value(target)}\n      targetHandle: target\n      type: custom\n      zIndex: 0\n"
        )

    return (
        "app:\n"
        f"  description: {_yaml_value(title)}\n"
        "  icon: 📊\n"
        "  icon_background: '#E4FBCC'\n"
        "  mode: workflow\n"
        f"  name: {_yaml_value(title)}\n"
        "  use_icon_as_answer_icon: false\n"
        "dependencies: []\n"
        "kind: app\n"
        "version: 0.3.1\n"
        "workflow:\n"
        "  conversation_variables: []\n"
        "  environment_variables: []\n"
        "  features:\n"
        "    file_upload:\n      enabled: false\n"
        "    opening_statement: ''\n"
        "    retriever_resource:\n      enabled: false\n"
        "    sensitive_word_avoidance:\n      enabled: false\n"
        "    speech_to_text:\n      enabled: false\n"
        "    suggested_questions: []\n"
        "    suggested_questions_after_answer:\n      enabled: false\n"
        "    text_to_speech:\n      enabled: false\n"
        "  graph:\n"
        "    edges:\n" + "".join(edges) + "    nodes:\n" + "".join(nodes) +
        "    viewport:\n      x: 0\n      y: 0\n      zoom: 0.5\n"
    )


def _slug(text: str) -> str:
    value = re.sub(r"[^0-9A-Za-zА-Яа-яЁё]+", "_", (text or "").strip())[:60].strip("_")
    return value or "flow"


def flow_path(user_id: Any, task_id: str, name: str = "") -> str:
    folder = os.path.join(BACKEND_ROOT, "data", str(user_id), HARNESS_DIR_NAME)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f"{_slug(name)}_{task_id}.yml")


async def make_flow(user: Any, text: str, index: Optional[int] = None, dataset_name: str = "",
                    model_choice: Optional[str] = None) -> Dict[str, Any]:
    """Генерирует DSL-файл workflow для Dify."""
    raw, account = await _llm(
        user,
        [
            {"role": "system", "content": FLOWS_SYSTEM},
            {"role": "user", "content": _flow_prompt(text, index, dataset_name)},
        ],
        model_choice=model_choice,
        temperature=0.2,
        max_tokens=2000,
    )
    spec = _json_block(raw)
    if not isinstance(spec, dict):
        raise ValueError("модель не вернула описание workflow, попробуйте переформулировать задачу")
    yaml_text = build_dify_dsl(spec, index)
    node_count = yaml_text.count("\n    - data:")
    return {
        "spec": spec,
        "yaml": yaml_text,
        "nodes": node_count,
        "dify_url": DIFY_PUBLIC_URL,
        "instructions": (
            "В Dify нажмите «Создать приложение» → «Импорт DSL-файла» и выберите скачанный файл, "
            "либо вкладка «Импорт DSL» в студии. После импорта задайте поля формы (index, подтема, период) и запустите."
        ),
        "model": account,
    }
