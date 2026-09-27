#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Граф авторов и связей по негативу KFC: кто в каких сюжетах участвует.

Идея: авторы связаны, если пишут в один и тот же инфоповод (собственное название сюжета)
или в один и тот же материал (одна ветка комментариев). Сообщества в таком графе — это
устойчивые группы аудитории вокруг одних и тех же тем: их видно, можно назвать и понять,
что именно они разносят.

На выходе:
  * /tmp/kfc_topics/author_graph.json — сообщества: размер, охват, площадки, города, сюжеты,
    лидеры, примеры сообщений со ссылками;
  * /tmp/kfc_topics/author_network.png — картинка графа для отчёта;
  * /tmp/kfc_topics/author_communities.png — диаграмма сообществ по объёму.

Запуск: setsid --fork venv_py312_clean/bin/python -u kfc_author_graph.py > /tmp/kfc_graph.log 2>&1
"""
from __future__ import annotations

import collections
import datetime
import io
import json
import os
import re
import time

import networkx as nx
import numpy as np
from elasticsearch import Elasticsearch

INDEX = "kfc_13.05.2024-22.09.2026"
OUT = "/tmp/kfc_topics"
MSK = datetime.timezone(datetime.timedelta(hours=3))
LO_TS = int(datetime.datetime(2024, 5, 13, tzinfo=MSK).timestamp())
HI_TS = int(datetime.datetime(2026, 8, 31, 23, 59, 59, tzinfo=MSK).timestamp())
MIN_AUTHOR_MESSAGES = 4
LLM_URL = "http://127.0.0.1:8000/v1/chat/completions"
LLM_MODEL = "Qwen/Qwen3-32B-FP8"


def log(message: str) -> None:
    print("[%s] %s" % (datetime.datetime.now().strftime("%H:%M:%S"), message), flush=True)


def base_url(url: str) -> str:
    """Материал без параметров: comments одного поста собираются в один узел."""
    if not url:
        return ""
    cut = re.split(r"[?#]", url)[0]
    return cut.rstrip("/")[:160]


def collect(es) -> dict:
    path = os.path.join(OUT, "author_scan.json")
    if os.path.isfile(path):
        with io.open(path, encoding="utf-8") as fh:
            return json.load(fh)
    query = {"bool": {"must": [{"term": {"toneMark": -1}}],
                      "filter": [{"range": {"timeCreate": {"gte": LO_TS, "lte": HI_TS}}}]}}
    source = ["authorObject", "hub", "story", "url", "city", "timeCreate", "audienceCount",
              "text", "review_rating"]
    authors = {}
    story_authors = collections.defaultdict(collections.Counter)
    material_authors = collections.defaultdict(collections.Counter)
    started = time.time()
    scanned = 0
    resp = es.search(index=INDEX, body={"size": 3000, "query": query, "_source": source, "sort": ["_doc"]},
                     scroll="20m")
    sid = resp["_scroll_id"]
    try:
        while True:
            hits = resp["hits"]["hits"]
            if not hits:
                break
            for hit in hits:
                src = hit["_source"]
                obj = src.get("authorObject") or {}
                if isinstance(obj, str):
                    try:
                        obj = json.loads(obj.replace("'", '"'))
                    except Exception:  # noqa: BLE001
                        obj = {}
                author = (obj.get("hash") or "").strip()
                if not author:
                    continue
                scanned += 1
                row = authors.get(author)
                if row is None:
                    row = authors[author] = {
                        "name": (obj.get("fullname") or "").strip()[:80],
                        "url": obj.get("url") or "", "type": obj.get("author_type") or "",
                        "messages": 0, "reach": 0, "hubs": collections.Counter(),
                        "cities": collections.Counter(), "stories": collections.Counter(),
                        "months": collections.Counter(), "texts": [],
                    }
                row["messages"] += 1
                row["reach"] += int(src.get("audienceCount") or 0)
                row["hubs"][src.get("hub") or ""] += 1
                city = (src.get("city") or "").strip()
                if city:
                    row["cities"][city] += 1
                story = (src.get("story") or "").strip()
                if story:
                    row["stories"][story] += 1
                    story_authors[story][author] += 1
                material = base_url(src.get("url") or "")
                if material:
                    material_authors[material][author] += 1
                try:
                    month = datetime.datetime.fromtimestamp(
                        int(float(src.get("timeCreate") or 0)), MSK).strftime("%Y-%m")
                    row["months"][month] += 1
                except Exception:  # noqa: BLE001
                    pass
                if len(row["texts"]) < 3:
                    text = (src.get("text") or "").strip().replace("\n", " ")
                    if len(text) > 40:
                        row["texts"].append({"text": text[:300], "hub": src.get("hub") or "",
                                             "url": src.get("url") or ""})
            if scanned and scanned % 40000 < 3000:
                log("разобрано авторов-сообщений %d (%.0f с)" % (scanned, time.time() - started))
            resp = es.scroll(scroll_id=sid, scroll="20m")
            sid = resp["_scroll_id"]
    finally:
        try:
            es.clear_scroll(scroll_id=sid)
        except Exception:  # noqa: BLE001
            pass
    data = {
        "authors": {key: {**{k: v for k, v in row.items()
                             if k not in ("hubs", "cities", "stories", "months")},
                          "hubs": dict(row["hubs"]), "cities": dict(row["cities"]),
                          "stories": dict(row["stories"]), "months": dict(row["months"])}
                    for key, row in authors.items()},
        "story_authors": {story: dict(counter) for story, counter in story_authors.items()},
        "material_authors": {material: dict(counter) for material, counter in material_authors.items()
                             if len(counter) > 1},
        "scanned": scanned,
    }
    with io.open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False)
    log("собрано авторов: %d, сюжетов: %d, материалов: %d"
        % (len(authors), len(story_authors), len(data["material_authors"])))
    return data


def build_graph(data: dict) -> tuple:
    """Двудольный граф автор ↔ сюжет, затем сообщества."""
    graph = nx.Graph()
    authors = {key: row for key, row in data["authors"].items()
               if row["messages"] >= MIN_AUTHOR_MESSAGES}
    for key, row in authors.items():
        graph.add_node("a:" + key, kind="author", messages=row["messages"], reach=row["reach"],
                       name=row["name"])
    for story, counter in data["story_authors"].items():
        members = [(key, count) for key, count in counter.items() if key in authors]
        if len(members) < 3:
            continue
        graph.add_node("s:" + story, kind="story", messages=sum(count for _, count in members))
        for key, count in members:
            graph.add_edge("a:" + key, "s:" + story, weight=count)
    # связи по одной ветке комментариев: авторы одного материала
    for material, counter in list(data["material_authors"].items())[:400000]:
        members = [(key, count) for key, count in counter.items() if key in authors]
        if len(members) < 2:
            continue
        for index in range(len(members) - 1):
            left, left_weight = members[index]
            right, right_weight = members[index + 1]
            edge = graph.get_edge_data("a:" + left, "a:" + right)
            weight = min(left_weight, right_weight)
            if edge:
                edge["weight"] += weight
            else:
                graph.add_edge("a:" + left, "a:" + right, weight=weight, kind="material")
    log("граф: узлов %d, связей %d" % (graph.number_of_nodes(), graph.number_of_edges()))
    communities = nx.community.louvain_communities(graph, weight="weight", seed=42)
    log("сообществ: %d" % len(communities))
    return graph, communities, authors


def describe(graph, communities, authors, data) -> list:
    rows = []
    for index, community in enumerate(communities, 1):
        members = [node for node in community if node.startswith("a:")]
        if len(members) < 5:
            continue
        hubs, cities, stories, months = collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
        messages = reach = 0
        leaders = []
        for node in members:
            key = node[2:]
            row = authors.get(key) or {}
            messages += row.get("messages", 0)
            reach += row.get("reach", 0)
            hubs.update(row.get("hubs") or {})
            cities.update(row.get("cities") or {})
            stories.update(row.get("stories") or {})
            months.update(row.get("months") or {})
            leaders.append((row.get("messages", 0), row.get("name") or key, row.get("url") or "",
                            (row.get("texts") or [{}])[0]))
        leaders.sort(reverse=True)
        rows.append({
            "id": index, "authors": len(members), "messages": messages, "reach": reach,
            "hubs": hubs.most_common(6), "cities": cities.most_common(6),
            "stories": stories.most_common(5), "months": dict(months),
            "leaders": [{"messages": m, "name": n, "url": u,
                         "example": ex.get("text", ""), "example_url": ex.get("url", ""),
                         "example_hub": ex.get("hub", "")} for m, n, u, ex in leaders[:5]],
        })
    rows.sort(key=lambda row: -row["messages"])
    for place, row in enumerate(rows, 1):
        row["place"] = place
    return rows


def name_communities(rows: list) -> None:
    """Название сообщества — по тому, что в нём обсуждают (внешняя модель, сильный русский)."""
    sys.path.insert(0, "/home/dev/tellscope_app/tellscope_backend")
    import kfc_ai

    system = ("Ты аналитик медиаполя сети фастфуда Rostic's (бывший KFC). Пишешь по-русски, "
              "деловым языком, без выдуманных фактов: только то, что видно в данных ниже.")
    for row in rows[:14]:
        themes = ", ".join(story for story, _ in row["stories"][:6])
        hubs = ", ".join("%s (%d)" % (hub, count) for hub, count in row["hubs"][:4])
        cities = ", ".join("%s (%d)" % (city, count) for city, count in row["cities"][:4])
        examples = " | ".join(item["example"][:150] for item in row["leaders"][:3] if item.get("example"))
        prompt = (
            "Данные о группе авторов, которые пишут в одни и те же сюжеты о бренде.\n"
            "Сюжеты: %s\nПлощадки: %s\nГорода: %s\nСообщений: %d, авторов: %d\n"
            "Примеры их сообщений: %s\n\n"
            "Ответь строго JSON: {\"название\": \"до 7 слов, о чём эта группа\", "
            "\"суть\": \"1-2 предложения: что эта группа делает и чем полезна или опасна для бренда\"}"
            % (themes or "нет", hubs or "нет", cities or "нет", row["messages"], row["authors"],
               examples or "нет"))
        parsed = kfc_ai.ask_json(prompt, system=system, model="deepseek-chat", max_tokens=500)
        if parsed:
            row["title"] = str(parsed.get("название") or "").strip()
            row["essence"] = str(parsed.get("суть") or "").strip()
            log("сообщество %d: %s" % (row["place"], row["title"][:70]))
        else:
            log("сообщество %d: название не получено" % row["place"])


def render(graph, communities, authors, rows) -> str:
    """Картинка графа: крупнейшие сообщества авторов вокруг сюжетов."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sizes = {}
    for index, community in enumerate(communities, 1):
        for node in community:
            sizes[node] = index
    top_authors = sorted(((row["messages"], "a:" + key, row["name"]) for key, row in authors.items()),
                         reverse=True)[:150]
    keep = {node for _, node, _ in top_authors}
    for story_node in [node for node in graph.nodes if node.startswith("s:")]:
        if graph.degree(story_node) >= 6:
            keep.add(story_node)
    sub = graph.subgraph(keep)
    palette = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(11, 8.5), dpi=170)
    positions = nx.spring_layout(sub, k=0.28, iterations=140, seed=7, weight="weight")
    node_colors = [palette((sizes.get(node, 0) - 1) % 20) if node.startswith("a:") else (0.75, 0.75, 0.78, 1)
                   for node in sub.nodes]
    node_sizes = [40 + 6 * np.sqrt(sub.nodes[node].get("messages", 1)) if node.startswith("a:") else 130
                  for node in sub.nodes]
    nx.draw_networkx_edges(sub, positions, ax=ax, alpha=0.25, width=0.6, edge_color="#98a2b3")
    nx.draw_networkx_nodes(sub, positions, ax=ax, node_color=node_colors, node_size=node_sizes,
                           linewidths=0.4, edgecolors="#ffffff")
    labels = {}
    for _, node, name in top_authors[:22]:
        if node in positions:
            labels[node] = (name or node)[:18]
    for node in sub.nodes:
        if node.startswith("s:") and sub.degree(node) >= 10:
            labels[node] = node[2:][:26]
    nx.draw_networkx_labels(sub, positions, labels=labels, ax=ax, font_size=7, font_color="#101828")
    ax.set_title("Сообщества авторов вокруг сюжетов о бренде (негатив, 2024–2026)", fontsize=13,
                 fontweight="bold")
    ax.axis("off")
    path = os.path.join(OUT, "author_network.png")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    log("картинка графа: %s" % path)

    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=170)
    top = rows[:12][::-1]
    ax.barh([("%s (%d авт.)" % ((row.get("title") or "сообщество %d" % row["place"])[:40], row["authors"]))
             for row in top], [row["messages"] for row in top], color="#1760e8")
    ax.set_xlabel("негативных сообщений")
    ax.set_title("Крупнейшие сообщества авторов", fontsize=12, fontweight="bold")
    fig.tight_layout()
    path2 = os.path.join(OUT, "author_communities.png")
    fig.savefig(path2, dpi=170)
    plt.close(fig)
    log("диаграмма сообществ: %s" % path2)
    return path


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    es = Elasticsearch(hosts=["http://localhost:9200"], basic_auth=("elastic", "biz8z5i1w0nLPmEweKgP"),
                       verify_certs=False, request_timeout=600)
    data = collect(es)
    graph, communities, authors = build_graph(data)
    rows = describe(graph, communities, authors, data)
    log("сообществ с данными: %d" % len(rows))
    name_communities(rows)
    render(graph, communities, authors, rows)
    with io.open(os.path.join(OUT, "author_graph.json"), "w", encoding="utf-8") as fh:
        json.dump({"communities": rows, "authors_total": len(authors),
                   "built_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}, fh,
                  ensure_ascii=False)
    log("готово")


if __name__ == "__main__":
    main()
