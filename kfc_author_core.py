#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Кто именно создаёт негатив: ядро авторов по каждой боли и концентрация.

Считаем:
  * сколько авторов стоит за каждой смысловой группой жалоб и какая доля объёма — от ядра
    (авторы с тремя и более сообщениями в группе);
  * топ авторов по всему негативу: сколько сообщений, площадки, города, охват, профиль;
  * концентрацию: какая доля всего негатива приходится на верхушку авторов.

Результат: /tmp/kfc_topics/author_core.json + author_core.png
"""
from __future__ import annotations

import collections
import datetime
import io
import json
import os
import sys
import time

import numpy as np
from elasticsearch import Elasticsearch

OUT = "/tmp/kfc_topics"
BACKEND = "/home/dev/tellscope_app/tellscope_backend"
sys.path.insert(0, BACKEND)
MSK = datetime.timezone(datetime.timedelta(hours=3))
LO_TS = int(datetime.datetime(2024, 5, 13, tzinfo=MSK).timestamp())
HI_TS = int(datetime.datetime(2026, 8, 31, 23, 59, 59, tzinfo=MSK).timestamp())
TOP_CLUSTERS = 12
PER_CLUSTER = 2500


def log(message: str) -> None:
    print("[%s] %s" % (datetime.datetime.now().strftime("%H:%M:%S"), message), flush=True)


def rows():
    with io.open(os.path.join(OUT, "negatives.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def author_of(obj) -> dict:
    if isinstance(obj, str):
        try:
            obj = json.loads(obj.replace("'", '"'))
        except Exception:  # noqa: BLE001
            return {}
    return obj or {}


def global_top(scan: dict) -> dict:
    """Топ авторов по всему негативу и концентрация."""
    authors = scan.get("authors") or {}
    total = sum(row["messages"] for row in authors.values())
    ranked = sorted(authors.items(), key=lambda kv: -kv[1]["messages"])
    top = []
    for key, row in ranked[:25]:
        top.append({
            "name": row["name"] or key, "url": row["url"], "type": row["type"],
            "messages": row["messages"], "reach": row["reach"],
            "hubs": sorted(row["hubs"].items(), key=lambda kv: -kv[1])[:3],
            "cities": sorted(row["cities"].items(), key=lambda kv: -kv[1])[:3],
            "months": sorted(row["months"].items())[:3],
            "example": (row.get("texts") or [{}])[0].get("text", ""),
            "example_url": (row.get("texts") or [{}])[0].get("url", ""),
        })
    cumulative = 0
    thresholds = {}
    for index, (_, row) in enumerate(ranked, 1):
        cumulative += row["messages"]
        for mark in (10, 100, 1000, 10000):
            if mark not in thresholds and index <= mark:
                continue
        if len(thresholds) < 4:
            for mark in (1, 10, 100, 1000, 10000):
                if mark not in thresholds and index == mark:
                    thresholds[mark] = round(cumulative / float(total) * 100, 1)
    return {
        "authors_total": len(authors), "messages_total": total,
        "top": top,
        "share_top10": round(sum(row["messages"] for _, row in ranked[:10]) / float(total) * 100, 1),
        "share_top100": round(sum(row["messages"] for _, row in ranked[:100]) / float(total) * 100, 1),
        "share_top1000": round(sum(row["messages"] for _, row in ranked[:1000]) / float(total) * 100, 1),
        "authors_with_one": sum(1 for _, row in ranked if row["messages"] == 1),
        "authors_with_many": sum(1 for _, row in ranked if row["messages"] >= 20),
    }


def main() -> None:
    path = os.path.join(OUT, "author_core.json")
    if os.path.isfile(path):
        log("уже посчитано")
        return
    scan = json.load(io.open(os.path.join(OUT, "author_scan.json"), encoding="utf-8"))
    aggregate = global_top(scan)
    log("авторов всего %d, сообщений %d; верхушка 100 авторов даёт %.1f%% негатива"
        % (aggregate["authors_total"], aggregate["messages_total"], aggregate["share_top100"]))

    labels = np.load(os.path.join(OUT, "labels_final.npy"))
    clusters = json.load(io.open(os.path.join(OUT, "clusters_final.json"), encoding="utf-8"))
    titles = {int(item["cluster"]): (item["title"] or "группа %d" % item["cluster"])
              for item in clusters["clusters"]}
    sizes = collections.Counter(int(x) for x in labels)
    top = [cluster for cluster, _ in sizes.most_common(TOP_CLUSTERS)]

    wanted = collections.defaultdict(list)
    for index, row in enumerate(rows()):
        if index >= len(labels):
            break
        label = int(labels[index])
        if label in top and len(wanted[label]) < PER_CLUSTER:
            wanted[label].append(row["id"])

    es = Elasticsearch(hosts=["http://localhost:9200"], basic_auth=("elastic", "biz8z5i1w0nLPmEweKgP"),
                       verify_certs=False, request_timeout=600)
    per_cluster = {}
    for cluster, ids in wanted.items():
        counter = collections.Counter()
        meta = {}
        examples = {}
        for start in range(0, len(ids), 500):
            chunk = ids[start:start + 500]
            body = {"size": len(chunk), "query": {"ids": {"values": chunk}},
                    "_source": ["authorObject", "hub", "city", "text", "url"]}
            try:
                data = es.search(index="kfc_13.05.2024-22.09.2026", body=body)
            except Exception as exc:  # noqa: BLE001
                log("группа %d: ошибка %s" % (cluster, str(exc)[:70]))
                continue
            for hit in data["hits"]["hits"]:
                src = hit["_source"]
                obj = author_of(src.get("authorObject"))
                key = (obj.get("hash") or "").strip()
                if not key:
                    continue
                counter[key] += 1
                row = meta.setdefault(key, {"name": (obj.get("fullname") or "").strip()[:70],
                                            "url": obj.get("url") or "",
                                            "hubs": collections.Counter(),
                                            "cities": collections.Counter()})
                if src.get("hub"):
                    row["hubs"][src["hub"]] += 1
                if src.get("city"):
                    row["cities"][src["city"]] += 1
                if key not in examples:
                    text = (src.get("text") or "").strip().replace("\n", " ")
                    if len(text) > 40:
                        examples[key] = {"text": text[:300], "hub": src.get("hub") or "",
                                         "url": src.get("url") or ""}
        authors_count = len(counter)
        core = [(key, count) for key, count in counter.items() if count >= 3]
        core_messages = sum(count for _, count in core)
        total_messages = sum(counter.values())
        top_authors = []
        for key, count in counter.most_common(6):
            row = meta.get(key) or {}
            example = examples.get(key) or {}
            top_authors.append({
                "name": row.get("name") or key, "messages": count,
                "url": row.get("url") or "",
                "hubs": (row.get("hubs") or collections.Counter()).most_common(2),
                "cities": (row.get("cities") or collections.Counter()).most_common(2),
                "example": example.get("text", ""), "example_url": example.get("url", ""),
            })
        per_cluster[cluster] = {
            "title": titles.get(cluster, ""), "sampled": total_messages,
            "authors": authors_count, "core_authors": len(core),
            "core_share": round(core_messages / float(total_messages or 1) * 100, 1),
            "messages_per_author": round(total_messages / float(authors_count or 1), 1),
            "top_authors": top_authors,
        }
        log("группа %s: авторов %d, ядро %d даёт %.1f%%" % (titles.get(cluster, "")[:40], authors_count,
                                                            len(core), per_cluster[cluster]["core_share"]))

    result = {"global": aggregate, "clusters": per_cluster,
              "built_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}
    with io.open(path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10.5, 5.4), dpi=170)
    ordered = sorted(per_cluster.values(), key=lambda row: -row["sampled"])[:12][::-1]
    labels_x = ["%s — %d авт." % (row["title"][:34], row["authors"]) for row in ordered]
    ax.barh(labels_x, [row["authors"] for row in ordered], color="#1760e8")
    ax.set_xlabel("авторов в группе (в выборке)")
    ax.set_title("Сколько людей стоит за каждой болью", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "author_core.png"), dpi=170)
    plt.close(fig)
    log("готово")


if __name__ == "__main__":
    main()
