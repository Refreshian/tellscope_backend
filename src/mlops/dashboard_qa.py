"""Dashboard «Проанализировать данные через ИИ»: local vLLM, filtered context only."""
from __future__ import annotations

import re
from typing import Any

from mlops.prompts import load_prompt

_THINK_BLOCK = re.compile(r"<think>[\s\S]*?</think>", re.I)
_THINK_TAG = re.compile(r"</?think>", re.I)
_FILTER_RULE = (
    "Отвечай только по переданной выборке. Если данные уже отфильтрованы — опирайся "
    "на эти цифры и объекты, не восстанавливай то, что скрыто фильтром. "
    "Не выдумывай ссылки, авторов и числа. Если выборки мало — скажи об этом прямо. "
    "Пиши по-русски готовым текстом. Всегда доводи ответ до конца: последнее "
    "предложение должно быть законченным, markdown — закрытым."
)
_CONTINUE = (
    "/no_think\nПродолжи ответ с того места, где он оборвался. "
    "Допиши до конца, не повторяя уже сказанное. Закрой markdown, если он остался открытым."
)


def _clean_llm(text: str) -> str:
    cleaned = _THINK_BLOCK.sub("", text or "")
    cleaned = _THINK_TAG.sub("", cleaned)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _trim(value: Any, limit: int = 160) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _rng(value: Any, index: int, default: int = 0) -> int:
    if isinstance(value, (list, tuple)) and len(value) > index:
        return _as_int(value[index], default)
    return default


def _hub_line(hub: dict, limit: int = 12) -> str:
    name = _trim(hub.get("name") or hub.get("Название источника") or "без имени", 80)
    messages = _as_int(hub.get("values") if hub.get("values") is not None else hub.get("Количество сообщений"))
    audience = _as_int(hub.get("audience_sum") if hub.get("audience_sum") is not None else hub.get("Суммарная аудитория"))
    return f"- {name}: {messages} сообщ., охват {audience}"


def _top(items: list, key, limit: int = 12) -> list:
    ranked = sorted(items or [], key=key, reverse=True)
    return ranked[:limit]


def compact_tonality(body: dict) -> str:
    data = body.get("data") if isinstance(body.get("data"), dict) else {}
    tab = str(body.get("current_tab") or "Негативные упоминания")
    filters = body.get("filters") if isinstance(body.get("filters"), dict) else {}
    values = data.get("tonality_values") or {}
    hubs = data.get("tonality_hubs_values") or {}
    pos_hubs = hubs.get("positive_hubs") or []
    neg_hubs = hubs.get("negative_hubs") or []
    pos_n = _as_int(values.get("positive_count"))
    neg_n = _as_int(values.get("negative_count"))
    original = _as_int(filters.get("original_mentions"))
    filtered = pos_n + neg_n
    lines = [
        f"Вкладка: {tab}",
        f"Выборка после фильтров: {filtered} упоминаний"
        + (f" из {original}" if original else "")
        + f" (негатив {neg_n}, позитив {pos_n}).",
    ]
    if filters:
        lines.append(
            "Фильтры: комментарии {c0}–{c1}, лайки {l0}–{l1}, просмотры {v0}–{v1}, аудитория {a0}–{a1}.".format(
                c0=_rng(filters.get("commentsRange"), 0),
                c1=_rng(filters.get("commentsRange"), 1),
                l0=_rng(filters.get("likesRange"), 0),
                l1=_rng(filters.get("likesRange"), 1),
                v0=_rng(filters.get("viewsRange"), 0),
                v1=_rng(filters.get("viewsRange"), 1),
                a0=_rng(filters.get("audienceRange"), 0),
                a1=_rng(filters.get("audienceRange"), 1),
            )
        )
    if tab.startswith("Позитив"):
        chosen = _top(pos_hubs, lambda h: _as_int(h.get("audience_sum") or h.get("values")), 14)
        lines.append("Топ позитивных источников:")
        lines.extend(_hub_line(h) for h in chosen)
    elif "автор" in tab.lower():
        authors = []
        for group_name, bucket in (
            ("негатив", data.get("negative_authors_values") or []),
            ("позитив", data.get("positive_authors_values") or []),
        ):
            for group in bucket:
                for author in group.get("author_data") or []:
                    texts = author.get("texts") or []
                    authors.append(
                        {
                            "side": group_name,
                            "name": author.get("fullname") or "Без имени",
                            "count": _as_int(author.get("count_texts") or len(texts)),
                            "texts": texts,
                        }
                    )
        authors = _top(authors, lambda a: a["count"], 16)
        lines.append("Авторы в выборке:")
        snippets = []
        for author in authors:
            lines.append(f"- {author['name']} ({author['side']}): {author['count']} сообщ.")
            for text in author["texts"][:2]:
                if not isinstance(text, dict):
                    continue
                snippets.append(
                    f"- {author['name']}: {_trim(text.get('title') or text.get('text') or text.get('hub'), 140)}"
                    + (f" · {text.get('url')}" if text.get("url") else "")
                )
        if snippets:
            lines.append("Примеры сообщений:")
            lines.extend(snippets[:10])
    else:
        chosen = _top(neg_hubs, lambda h: _as_int(h.get("audience_sum") or h.get("values")), 14)
        lines.append("Топ негативных источников:")
        lines.extend(_hub_line(h) for h in chosen)
    return "\n".join(line for line in lines if line)


def compact_information(body: dict) -> str:
    data = body.get("data") if isinstance(body.get("data"), dict) else {}
    filters = body.get("filters") if isinstance(body.get("filters"), dict) else {}
    values = data.get("values") or []
    platforms: dict[str, int] = {}
    types: dict[str, int] = {}
    ranked = []
    for item in values:
        author = item.get("author") if isinstance(item.get("author"), dict) else {}
        hub = str(author.get("hub") or "неизвестно")
        atype = str(author.get("author_type") or "неизвестно")
        platforms[hub] = platforms.get(hub, 0) + 1
        types[atype] = types.get(atype, 0) + 1
        ranked.append(
            {
                "name": author.get("fullname") or author.get("url") or "без имени",
                "hub": hub,
                "type": atype,
                "audience": _as_int(author.get("audienceCount")),
                "views": _as_int(author.get("viewsCount")),
                "er": author.get("er") or 0,
                "url": author.get("url") or "",
                "reposts": len(item.get("reposts") or []),
            }
        )
    ranked = _top(ranked, lambda row: row["audience"], 18)
    original = _as_int(filters.get("original_messages") or data.get("num_messages"))
    lines = [
        f"Выборка после фильтров: {len(values)} сообщений"
        + (f" из {original}" if original else "")
        + ".",
        "Фильтры: аудитория {a0}–{a1}, репосты {r0}–{r1}, ER {e0}–{e1}, просмотры {v0}–{v1}.".format(
            a0=_rng(filters.get("audienceRange"), 0),
            a1=_rng(filters.get("audienceRange"), 1),
            r0=_rng(filters.get("repostsRange"), 0),
            r1=_rng(filters.get("repostsRange"), 1),
            e0=_rng(filters.get("erRange"), 0),
            e1=_rng(filters.get("erRange"), 1),
            v0=_rng(filters.get("viewsCountRange"), 0),
            v1=_rng(filters.get("viewsCountRange"), 1),
        ),
        "Площадки: " + ", ".join(f"{name} {count}" for name, count in sorted(platforms.items(), key=lambda x: -x[1])[:8]),
        "Типы авторов: " + ", ".join(f"{name} {count}" for name, count in sorted(types.items(), key=lambda x: -x[1])[:8]),
        "Топ авторов по охвату:",
    ]
    for row in ranked:
        lines.append(
            f"- {row['name']} · {row['hub']} · {row['type']} · охват {row['audience']} · "
            f"просмотры {row['views']} · репосты {row['reposts']}"
            + (f" · {row['url']}" if row["url"] else "")
        )
    return "\n".join(lines)


def _media_graph(data: dict) -> tuple[dict, list]:
    first = data.get("filtered_first_graph") or data.get("first_graph") or {}
    second = data.get("filtered_second_graph")
    if second is None:
        second = data.get("second_graph") or []
    if not isinstance(first, dict):
        first = {}
    if not isinstance(second, list):
        second = []
    return first, second


def compact_media(body: dict) -> str:
    data = body.get("data") if isinstance(body.get("data"), dict) else {}
    filters = body.get("filters") if isinstance(body.get("filters"), dict) else {}
    first, second = _media_graph(data)
    pos = first.get("positive_smi") or []
    neg = first.get("negative_smi") or []
    index_range = filters.get("indexRange") or filters.get("sliderRange") or []
    lines = [
        f"Выборка СМИ после фильтров: {len(pos)} позитивных и {len(neg)} негативных ресурсов, "
        f"{len(second)} ссылок.",
    ]
    if len(index_range) == 2:
        lines.append(f"Фильтр индекса: {index_range[0]}–{index_range[1]}.")
    lines.append("Негатив:")
    for item in _top(neg, lambda x: _as_int(x.get("index") or x.get("message_count")), 12):
        lines.append(
            f"- {_trim(item.get('name'), 80)} · индекс {item.get('index')} · сообщений {item.get('message_count')}"
        )
    lines.append("Позитив:")
    for item in _top(pos, lambda x: _as_int(x.get("index") or x.get("message_count")), 12):
        lines.append(
            f"- {_trim(item.get('name'), 80)} · индекс {item.get('index')} · сообщений {item.get('message_count')}"
        )
    if second:
        lines.append("Ссылки:")
        for item in second[:16]:
            url = item.get("url") or ""
            lines.append(f"- {_trim(item.get('name'), 80)} · индекс {item.get('index')}" + (f" · {url}" if url else ""))
    return "\n".join(lines)


def compact_graph(graph_data: dict) -> str:
    payload = graph_data if isinstance(graph_data, dict) else {}
    graph = payload.get("graph") if isinstance(payload.get("graph"), dict) else payload
    stats = payload.get("statistics") if isinstance(payload.get("statistics"), dict) else {}
    meta = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    nodes = graph.get("nodes") or []
    links = graph.get("links") or []
    clusters = graph.get("clusters") or []
    edge_types = stats.get("edge_types") or {}
    if not edge_types:
        for link in links:
            key = str((link or {}).get("type") or "similar")
            edge_types[key] = _as_int(edge_types.get(key)) + 1
    filtered_n = _as_int(stats.get("nodes_count") or len(nodes))
    filtered_e = _as_int(stats.get("edges_count") or len(links))
    original_n = _as_int(stats.get("original_nodes_count") or meta.get("original_nodes_count"))
    original_e = _as_int(stats.get("original_edges_count") or meta.get("original_edges_count"))
    lines = [
        f"Выборка графа после фильтров: {filtered_n} узлов, {filtered_e} связей"
        + (f" (исходно {original_n} узлов, {original_e} связей)" if original_n else "")
        + ".",
    ]
    if meta.get("focused_cluster_id"):
        lines.append(f"Сейчас выбран кластер {meta.get('focused_cluster_id')}.")
    if meta.get("enabled_link_types"):
        lines.append("Типы связей в выборке: " + ", ".join(str(x) for x in meta.get("enabled_link_types")))
    if edge_types:
        lines.append(
            "Счётчик связей: "
            + ", ".join(f"{name} {count}" for name, count in sorted(edge_types.items(), key=lambda x: -_as_int(x[1])))
        )
    if meta.get("selected_topics"):
        lines.append("Темы: " + ", ".join(_trim(t, 60) for t in (meta.get("selected_topics") or [])[:8]))
    if meta.get("author_search"):
        lines.append(f"Поиск автора: {meta.get('author_search')}")
    if meta.get("search_query"):
        lines.append(f"Поиск по темам ({meta.get('search_mode') or 'include'}): {meta.get('search_query')}")
    if clusters:
        lines.append("Кластеры в выборке:")
        for cluster in clusters[:12]:
            about = _trim(cluster.get("about") or "; ".join(cluster.get("topics") or []), 180)
            lines.append(
                f"- кластер {cluster.get('id')}: {cluster.get('size') or cluster.get('visible_size')} авт."
                + (f" · {about}" if about else "")
            )
    ranked = _top(nodes, lambda n: _as_int(n.get("audience") or n.get("posts_count")), 18)
    lines.append("Топ узлов:")
    for node in ranked:
        lines.append(
            f"- {node.get('label') or node.get('id')} · {node.get('type') or ''} · "
            f"{node.get('hubtype') or ''} · охват {_as_int(node.get('audience'))} · "
            f"постов {_as_int(node.get('posts_count'))} · кластер {node.get('cluster_id') or '—'}"
        )
    return "\n".join(lines)


def _system(prompt_id: str, fallback: str) -> str:
    try:
        text = load_prompt(prompt_id)
    except Exception:
        text = fallback
    return "/no_think\n" + text + "\n\n" + _FILTER_RULE


def _finish_reason(result) -> str:
    reason = str(getattr(result, "finish_reason", "") or "")
    if reason:
        return reason
    choice = ((getattr(result, "raw", None) or {}).get("choices") or [{}])[0] or {}
    return str(choice.get("finish_reason") or "")


def ask_vllm(prompt_id: str, question: str, context: str, fallback: str, max_tokens: int = 2200) -> str:
    from mlops.gateway import chat

    user = (
        "/no_think\n"
        f"Вопрос: {question.strip()}\n\n"
        f"Данные выборки:\n{context}"
    )
    messages = [
        {"role": "system", "content": _system(prompt_id, fallback)},
        {"role": "user", "content": user},
    ]
    parts: list[str] = []
    for _step in range(2):
        result = chat(
            provider="vllm",
            messages=messages,
            temperature=0.1,
            max_tokens=max_tokens,
            timeout=180,
            extra={"chat_template_kwargs": {"enable_thinking": False}},
            profile="dashboard_qa",
        )
        chunk = str(result.content or "")
        if not chunk.strip() and not parts:
            raise RuntimeError("Модель вернула пустой ответ")
        parts.append(chunk)
        if _finish_reason(result) != "length":
            break
        messages = messages + [
            {"role": "assistant", "content": chunk},
            {"role": "user", "content": _CONTINUE},
        ]
    memo = _clean_llm("".join(parts))
    if not memo:
        raise RuntimeError("Модель вернула пустой ответ")
    return memo


def _busy():
    from fastapi.responses import JSONResponse
    try:
        from mlops.ai_bot_rag import gpu_ready

        ready, holders = gpu_ready()
    except Exception:
        return None
    if ready:
        return None
    first = holders[0] if holders else {}
    return JSONResponse(
        status_code=503,
        content={
            "error": "Сейчас выполняется другая задача. Попробуйте через несколько минут.",
            "jobs": holders,
            "busy": first.get("product") or "",
        },
    )


def _ok(text: str, extra: dict | None = None):
    from fastapi.responses import JSONResponse

    payload = {"content": text, "answer": text, "role": "assistant", "provider": "local"}
    if extra:
        payload.update(extra)
    return JSONResponse(content=payload)


def _err(message: str, status: int = 400):
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=status, content={"error": message})


async def handle_question_raw(request):
    busy = _busy()
    if busy:
        return busy
    try:
        body = await request.json()
    except Exception:
        return _err("Неверный формат данных")
    question = str(body.get("question") or "").strip()
    if not question:
        return _err("Нужен вопрос")
    if not isinstance(body.get("data"), dict):
        return _err("Нет отфильтрованных данных для анализа")
    context = compact_tonality(body)
    try:
        prompt_id = (
            "dashboard_qa_tonality_v1"
            if "автор" in str(body.get("current_tab") or "").lower()
            else "dashboard_qa_raw_v1"
        )
        text = ask_vllm(
            prompt_id,
            question,
            context,
            "Ты аналитик тональности соцмедиа. Ответь по выборке.",
        )
    except Exception as exc:
        return _err(str(exc)[:400], 500)
    return _ok(text, {"filtered": True, "tab": body.get("current_tab")})


async def handle_question_information(request):
    busy = _busy()
    if busy:
        return busy
    try:
        body = await request.json()
    except Exception:
        return _err("Неверный формат данных")
    question = str(body.get("question") or "").strip()
    if not question:
        return _err("Нужен вопрос")
    data = body.get("data") if isinstance(body.get("data"), dict) else {}
    if not (data.get("values") or []):
        return _err("Нет отфильтрованных данных для анализа")
    context = compact_information(body)
    try:
        text = ask_vllm(
            "dashboard_qa_graph_v1",
            question,
            context,
            "Ты аналитик распространения информации в соцмедиа. Ответь по выборке.",
        )
    except Exception as exc:
        return _err(str(exc)[:400], 500)
    return _ok(text, {"filtered": True})


async def handle_question_media(request):
    busy = _busy()
    if busy:
        return busy
    try:
        body = await request.json()
    except Exception:
        return _err("Неверный формат данных")
    question = str(body.get("question") or "").strip()
    if not question:
        return _err("Нужен вопрос")
    data = body.get("data") if isinstance(body.get("data"), dict) else {}
    first, second = _media_graph(data)
    if not (first.get("positive_smi") or first.get("negative_smi") or second):
        return _err("Нет отфильтрованных данных для анализа")
    context = compact_media(body)
    try:
        text = ask_vllm(
            "dashboard_qa_media_v1",
            question,
            context,
            "Ты аналитик рейтинга СМИ. Ответь по выборке.",
        )
    except Exception as exc:
        return _err(str(exc)[:400], 500)
    return _ok(text, {"filtered": True})


async def handle_analyze_graph(question: str, graph_data: Any):
    busy = _busy()
    if busy:
        return busy
    question = str(question or "").strip()
    if not question:
        return _err("Нужен вопрос")
    if not isinstance(graph_data, dict):
        return _err("graph_data must be a dictionary")
    graph = graph_data.get("graph", graph_data)
    nodes = (graph or {}).get("nodes") or []
    if not nodes:
        return _err("Graph has no nodes")
    context = compact_graph(graph_data)
    try:
        text = ask_vllm(
            "dashboard_qa_graph_v1",
            question,
            context,
            "Ты аналитик графа авторов в соцмедиа. Ответь по выборке.",
        )
    except Exception as exc:
        return _err(str(exc)[:400], 500)
    stats = graph_data.get("statistics") if isinstance(graph_data.get("statistics"), dict) else {}
    return _ok(
        text,
        {
            "filtered": True,
            "context_used": {
                "nodes_analyzed": min(18, len(nodes)),
                "nodes_in_view": stats.get("nodes_count") or len(nodes),
                "links_in_view": stats.get("edges_count") or len(graph.get("links") or []),
            },
        },
    )
