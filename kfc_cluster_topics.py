#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Разбор смысловых кластеров негатива KFC: что это, где, когда, что делать.

Кластеры получены на эмбеддингах (шаг kfc_cluster_step.py). Здесь каждому кластеру
подбирается имя и суть (внешняя модель DeepSeek), считаются объём, динамика по месяцам,
города, площадки, охват и примеры сообщений со ссылками.

Результат: /tmp/kfc_topics/clusters.json
"""
from __future__ import annotations

import collections
import datetime
import io
import json
import os
import re
import sys
import time

import numpy as np

OUT = "/tmp/kfc_topics"
BACKEND = "/home/dev/tellscope_app/tellscope_backend"
sys.path.insert(0, BACKEND)

TOP_CLUSTERS = 60          # столько кластеров разбираем подробно
REPRESENTATIVES = 14       # примеров на кластер


def log(message: str) -> None:
    print("[%s] %s" % (datetime.datetime.now().strftime("%H:%M:%S"), message), flush=True)


def rows():
    with io.open(os.path.join(OUT, "negatives.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def centroids(labels: np.ndarray, keep: set) -> dict:
    """Центр каждого кластера: средний вектор его сообщений."""
    vectors = np.load(os.path.join(OUT, "vectors.npy"), mmap_mode="r")
    sums = {key: np.zeros(vectors.shape[1], dtype=np.float64) for key in keep}
    counts = collections.Counter()
    step = 20000
    for start in range(0, vectors.shape[0], step):
        stop = min(start + step, vectors.shape[0])
        block = np.asarray(vectors[start:stop])
        for offset in range(stop - start):
            label = int(labels[start + offset])
            if label in sums:
                sums[label] += block[offset]
                counts[label] += 1
        if start % 100000 == 0:
            log("центры: обработано %d" % stop)
    for key in sums:
        if counts[key]:
            sums[key] /= counts[key]
            norm = np.linalg.norm(sums[key])
            if norm:
                sums[key] /= norm
    return sums


def pick_representatives(labels: np.ndarray, sums: dict) -> dict:
    """Самые типичные сообщения кластера: ближайшие к его центру."""
    vectors = np.load(os.path.join(OUT, "vectors.npy"), mmap_mode="r")
    best = collections.defaultdict(list)
    step = 20000
    for start in range(0, vectors.shape[0], step):
        stop = min(start + step, vectors.shape[0])
        block = np.asarray(vectors[start:stop])
        for offset in range(stop - start):
            index = start + offset
            label = int(labels[index])
            center = sums.get(label)
            if center is None:
                continue
            score = float(np.dot(block[offset], center))
            bucket = best[label]
            if len(bucket) < REPRESENTATIVES * 6:
                bucket.append((score, index))
            else:
                worst = min(bucket)
                if score > worst[0]:
                    bucket.remove(worst)
                    bucket.append((score, index))
    return {label: [index for _, index in sorted(items, reverse=True)] for label, items in best.items()}


def main() -> None:
    labels = np.load(os.path.join(OUT, "labels.npy"))
    sizes = collections.Counter(int(x) for x in labels if x >= 0)
    keep = {cluster for cluster, _ in sizes.most_common(TOP_CLUSTERS)}
    log("кластеров всего %d, разбираем %d" % (len(sizes), len(keep)))

    sums = centroids(labels, keep)
    best = pick_representatives(labels, sums)
    log("представители собраны")

    rows_list = list(rows())
    log("сообщений в выгрузке: %d" % len(rows_list))
    profiles = {}
    for index, row in enumerate(rows_list):
        label = int(labels[index]) if index < len(labels) else -1
        if label not in keep:
            continue
        profile = profiles.get(label)
        if profile is None:
            profile = profiles[label] = {
                "size": 0, "reach": 0, "likes": 0, "months": collections.Counter(),
                "cities": collections.Counter(), "hubs": collections.Counter(),
                "hubtypes": collections.Counter(), "ratings": collections.Counter(),
                "stories": collections.Counter(),
            }
        profile["size"] += 1
        profile["reach"] += row.get("reach") or 0
        profile["likes"] += row.get("likes") or 0
        if row.get("month"):
            profile["months"][row["month"]] += 1
        if row.get("city"):
            profile["cities"][row["city"]] += 1
        if row.get("hub"):
            profile["hubs"][row["hub"]] += 1
        if row.get("hubtype"):
            profile["hubtypes"][row["hubtype"]] += 1
        if row.get("rating"):
            profile["ratings"][row["rating"]] += 1
        if row.get("story"):
            profile["stories"][row["story"]] += 1

    import kfc_ai

    result = []
    system = ("Ты аналитик медиаполя сети фастфуда Rostic's (бывший KFC). По выборке сообщений "
              "называешь смысловой кластер жалоб. Пишешь по-русски, деловым языком, без выдумок: "
              "только то, что видно в приведённых сообщениях.")
    for label, size in sizes.most_common(TOP_CLUSTERS):
        profile = profiles.get(label)
        if not profile or profile["size"] < 300:
            continue
        examples = [rows_list[index] for index in best.get(label, [])[:REPRESENTATIVES]
                    if index < len(rows_list)]
        texts = "\n".join("- " + (item["text"] or "")[:220].replace("\n", " ") for item in examples[:14])
        months = profile["months"]
        peak = max(months.items(), key=lambda kv: kv[1])[0] if months else ""
        recent = sum(count for month, count in months.items() if month >= "2026-01")
        prompt = (
            "Кластер жалоб клиентов сети фастфуда. Сообщений: %d. Пик: %s. С 2026 года: %d.\n"
            "Города: %s\nПлощадки: %s\nПримеры сообщений:\n%s\n\n"
            "Ответь строго JSON:\n"
            "{\"название\": \"до 6 слов, что это за боль\", "
            "\"суть\": \"2 предложения: на что именно жалуются и почему это важно бренду\", "
            "\"что_делать\": \"1 предложение: конкретное действие для сети\"}"
            % (profile["size"], peak, recent,
               ", ".join("%s" % city for city, _ in profile["cities"].most_common(5)) or "нет",
               ", ".join("%s" % hub for hub, _ in profile["hubs"].most_common(4)) or "нет", texts))
        named = kfc_ai.ask_json(prompt, system=system, model="deepseek-chat", max_tokens=700)
        entry = {
            "cluster": label, "size": profile["size"],
            "share": round(profile["size"] / float(len(rows_list)), 4),
            "reach": profile["reach"], "likes": profile["likes"],
            "months": dict(months), "peak_month": peak, "from_2026": recent,
            "cities": profile["cities"].most_common(8), "hubs": profile["hubs"].most_common(8),
            "hubtypes": profile["hubtypes"].most_common(4), "ratings": dict(profile["ratings"]),
            "stories": profile["stories"].most_common(4),
            "title": (named or {}).get("название", ""), "essence": (named or {}).get("суть", ""),
            "action": (named or {}).get("что_делать", ""),
            "examples": [{"text": (item.get("text") or "")[:400], "hub": item.get("hub", ""),
                          "city": item.get("city", ""), "month": item.get("month", ""),
                          "rating": item.get("rating", ""), "url": item.get("url", ""),
                          "reach": item.get("reach", 0)} for item in examples[:8]],
        }
        result.append(entry)
        log("кластер %d (%d): %s" % (label, size, entry["title"][:70]))
        with io.open(os.path.join(OUT, "clusters.json"), "w", encoding="utf-8") as fh:
            json.dump({"clusters": result, "total": len(rows_list),
                       "built_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}, fh,
                      ensure_ascii=False)
    log("готово: кластеров разобрано %d" % len(result))


if __name__ == "__main__":
    main()
