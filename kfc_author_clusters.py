#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Сообщества авторов вокруг смысловых кластеров: кто в каких болях участвует.

Берём крупнейшие кластеры жалоб и выгружаем из хранилища авторов их сообщений. Дальше
строим двудольный граф «автор ↔ боль» и ищем сообщества: устойчивые группы людей,
которые пишут об одних и тех же проблемах. Для каждой группы считаем объём, охват,
города, площадки и лидеров, название и суть даёт внешняя модель.

Результат: /tmp/kfc_topics/author_clusters.json + author_clusters.png
"""
from __future__ import annotations

import collections
import datetime
import io
import json
import os
import sys
import time

import networkx as nx
import numpy as np
from elasticsearch import Elasticsearch

OUT = "/tmp/kfc_topics"
BACKEND = "/home/dev/tellscope_app/tellscope_backend"
sys.path.insert(0, BACKEND)
MSK = datetime.timezone(datetime.timedelta(hours=3))
LO_TS = int(datetime.datetime(2024, 5, 13, tzinfo=MSK).timestamp())
HI_TS = int(datetime.datetime(2026, 8, 31, 23, 59, 59, tzinfo=MSK).timestamp())
TOP_CLUSTERS = 14
PER_CLUSTER = 2500


def log(message: str) -> None:
    print("[%s] %s" % (datetime.datetime.now().strftime("%H:%M:%S"), message), flush=True)


def rows():
    with io.open(os.path.join(OUT, "negatives.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> None:
    path = os.path.join(OUT, "author_clusters.json")
    if os.path.isfile(path):
        log("уже посчитано")
        return
    labels = np.load(os.path.join(OUT, "labels_final.npy"))
    clusters = json.load(io.open(os.path.join(OUT, "clusters_final.json"), encoding="utf-8"))
    titles = {int(item["cluster"]): item["title"] or ("группа %d" % item["cluster"])
              for item in clusters["clusters"]}
    sizes = collections.Counter(int(x) for x in labels)
    top = [cluster for cluster, _ in sizes.most_common(TOP_CLUSTERS)]

    # какие сообщения относятся к этим кластерам: индексы и идентификаторы
    wanted = collections.defaultdict(list)
    for index, row in enumerate(rows()):
        if index >= len(labels):
            break
        label = int(labels[index])
        if label in top and len(wanted[label]) < PER_CLUSTER:
            wanted[label].append(row["id"])
    log("отобрано сообщений: %s" % {titles[key][:24]: len(value) for key, value in wanted.items()})

    es = Elasticsearch(hosts=["http://localhost:9200"], basic_auth=("elastic", "biz8z5i1w0nLPmEweKgP"),
                       verify_certs=False, request_timeout=600)
    author_cluster = collections.Counter()
    author_meta = {}
    author_examples = collections.defaultdict(dict)
    for cluster, ids in wanted.items():
        if not ids:
            continue
        for start in range(0, len(ids), 500):
            chunk = ids[start:start + 500]
            res = es.msearch(index="kfc_13.05.2024-22.09.2026", searches=[]) if False else None
            body = {"size": len(chunk), "query": {"ids": {"values": chunk}},
                    "_source": ["authorObject", "hub", "city", "text", "url", "audienceCount"]}
            try:
                data = es.search(index="kfc_13.05.2024-22.09.2026", body=body)
            except Exception as exc:  # noqa: BLE001
                log("кластер %d: ошибка выборки %s" % (cluster, str(exc)[:80]))
                continue
            for hit in data["hits"]["hits"]:
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
                author_cluster[(author, cluster)] += 1
                row = author_meta.setdefault(author, {"name": (obj.get("fullname") or "").strip()[:80],
                                                      "url": obj.get("url") or "",
                                                      "type": obj.get("author_type") or "",
                                                      "hubs": collections.Counter(),
                                                      "cities": collections.Counter(), "messages": 0})
                row["messages"] += 1
                if src.get("hub"):
                    row["hubs"][src["hub"]] += 1
                if src.get("city"):
                    row["cities"][src["city"]] += 1
                example = author_examples[author].get(cluster)
                if not example:
                    text = (src.get("text") or "").strip().replace("\n", " ")
                    if len(text) > 40:
                        author_examples[author][cluster] = {"text": text[:300],
                                                            "hub": src.get("hub") or "",
                                                            "url": src.get("url") or ""}
        log("кластер %d обработан" % cluster)

    graph = nx.Graph()
    for (author, cluster), weight in author_cluster.items():
        graph.add_node("a:" + author, kind="author", messages=author_meta[author]["messages"])
        graph.add_node("c:%d" % cluster, kind="cluster", title=titles.get(cluster, ""))
        graph.add_edge("a:" + author, "c:%d" % cluster, weight=weight)
    log("граф: узлов %d, связей %d" % (graph.number_of_nodes(), graph.number_of_edges()))
    communities = nx.community.louvain_communities(graph, weight="weight", seed=7)

    result = []
    for community in communities:
        authors = [node for node in community if node.startswith("a:")]
        clusters_in = [node for node in community if node.startswith("c:")]
        if len(authors) < 4 or not clusters_in:
            continue
        messages = sum(author_meta[node[2:]]["messages"] for node in authors)
        hubs, cities = collections.Counter(), collections.Counter()
        leaders = []
        for node in authors:
            key = node[2:]
            row = author_meta[key]
            hubs.update(row["hubs"])
            cities.update(row["cities"])
            example = author_examples[key].get(int(clusters_in[0][2:])) or {}
            leaders.append((row["messages"], row["name"] or key, example.get("text", ""),
                            example.get("url", ""), example.get("hub", "")))
        leaders.sort(reverse=True)
        result.append({
            "authors": len(authors), "messages": messages,
            "clusters": [{"id": int(node[2:]), "title": titles.get(int(node[2:]), "")} for node in clusters_in],
            "hubs": hubs.most_common(5), "cities": cities.most_common(5),
            "leaders": [{"messages": m, "name": n, "example": ex, "url": url, "hub": hub}
                        for m, n, ex, url, hub in leaders[:5]],
        })
    result.sort(key=lambda row: -row["messages"])
    log("сообществ: %d" % len(result))

    import kfc_ai

    system = ("Ты аналитик медиаполя сети фастфуда Rostic's (бывший KFC). Называешь группу "
              "авторов по тому, о чём они пишут. По-русски, без выдумок.")
    for place, row in enumerate(result[:12], 1):
        row["place"] = place
        themes = ", ".join("%s (%d)" % (item["title"], item["id"]) for item in row["clusters"][:5])
        prompt = (
            "Группа авторов пишет о проблемах сети фастфуда. Боли: %s\n"
            "Сообщений: %d, авторов: %d, площадки: %s, города: %s\nПримеры: %s\n\n"
            "Ответь строго JSON: {\"название\": \"до 6 слов, кто эти авторы\", "
            "\"суть\": \"1-2 предложения: чем эта группа важна для бренда\"}"
            % (themes, row["messages"], row["authors"],
               ", ".join(hub for hub, _ in row["hubs"][:3]) or "нет",
               ", ".join(city for city, _ in row["cities"][:3]) or "нет",
               " | ".join(item["example"][:120] for item in row["leaders"][:3] if item.get("example"))))
        named = kfc_ai.ask_json(prompt, system=system, model="deepseek-chat", max_tokens=500)
        if named:
            row["title"] = str(named.get("название") or "").strip()
            row["essence"] = str(named.get("суть") or "").strip()
            log("сообщество %d: %s" % (place, row["title"][:60]))

    with io.open(path, "w", encoding="utf-8") as fh:
        json.dump({"communities": result, "built_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")},
                  fh, ensure_ascii=False)

    # картинка: авторы вокруг болей
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 8), dpi=170)
    positions = nx.spring_layout(graph, k=0.12, iterations=160, seed=11, weight="weight")
    colors = []
    for node in graph.nodes:
        colors.append("#1760e8" if node.startswith("c:") else "#f79009")
    sizes_nodes = [520 if node.startswith("c:") else 30 + 4 * np.sqrt(graph.nodes[node].get("messages", 1))
                   for node in graph.nodes]
    nx.draw_networkx_edges(graph, positions, ax=ax, alpha=0.18, width=0.5, edge_color="#98a2b3")
    nx.draw_networkx_nodes(graph, positions, ax=ax, node_color=colors, node_size=sizes_nodes,
                           linewidths=0.3, edgecolors="#ffffff")
    labels = {node: titles.get(int(node[2:]), "")[:26] for node in graph.nodes if node.startswith("c:")}
    nx.draw_networkx_labels(graph, positions, labels=labels, ax=ax, font_size=7)
    ax.set_title("Авторы вокруг болей бренда: каждая точка — автор, синие — смысловые группы жалоб",
                 fontsize=12, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "author_clusters.png"), dpi=170)
    plt.close(fig)
    log("картинка: author_clusters.png")
    log("готово")


if __name__ == "__main__":
    main()
