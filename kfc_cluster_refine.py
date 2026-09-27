#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Уточнение кластеров: распределение «шума» и разбиение самой крупной группы на подтемы.

После HDBSCAN часть сообщений (42%) остаётся без кластера — они относятся к темам, но не
попали в плотные группы. Здесь они относятся к ближайшему кластеру по косинусной близости,
а самая крупная группа разбивается на подтемы: внутри неё сидят разные боли, которые
на верхнем уровне сливаются в одну.

Результат: /tmp/kfc_topics/clusters_final.json, карта смыслов и графики.
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

OUT = "/tmp/kfc_topics"
BACKEND = "/home/dev/tellscope_app/tellscope_backend"
sys.path.insert(0, BACKEND)
BIG_CLUSTER_SUBTOPICS = 8
REPRESENTATIVES = 12


def log(message: str) -> None:
    print("[%s] %s" % (datetime.datetime.now().strftime("%H:%M:%S"), message), flush=True)


def rows():
    with io.open(os.path.join(OUT, "negatives.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def assign_noise(labels: np.ndarray) -> np.ndarray:
    """Каждое сообщение без кластера относим к ближайшему центру."""
    vectors = np.load(os.path.join(OUT, "vectors.npy"), mmap_mode="r")
    known = sorted(set(int(x) for x in labels if x >= 0))
    sums = {key: np.zeros(vectors.shape[1], dtype=np.float64) for key in known}
    counts = collections.Counter()
    step = 20000
    for start in range(0, vectors.shape[0], step):
        stop = min(start + step, vectors.shape[0])
        block = np.asarray(vectors[start:stop])
        for offset in range(stop - start):
            label = int(labels[start + offset])
            if label >= 0:
                sums[label] += block[offset]
                counts[label] += 1
    centers = np.zeros((len(known), vectors.shape[1]), dtype=np.float32)
    for index, key in enumerate(known):
        vector = sums[key] / max(1, counts[key])
        norm = np.linalg.norm(vector)
        centers[index] = vector / norm if norm else vector

    fixed = labels.copy()
    assigned = 0
    for start in range(0, vectors.shape[0], step):
        stop = min(start + step, vectors.shape[0])
        block = np.asarray(vectors[start:stop])
        noise_positions = [offset for offset in range(stop - start) if labels[start + offset] < 0]
        if not noise_positions:
            continue
        similarity = block[noise_positions] @ centers.T
        best = similarity.argmax(axis=1)
        for offset, index in zip(noise_positions, best):
            fixed[start + offset] = known[int(index)]
            assigned += 1
    log("шум распределён: %d сообщений" % assigned)
    return fixed


def split_biggest(labels: np.ndarray) -> tuple:
    """Самая крупная группа делится на подтемы методом k-средних."""
    from sklearn.cluster import MiniBatchKMeans

    sizes = collections.Counter(int(x) for x in labels)
    big = sizes.most_common(1)[0][0]
    compressed = np.load(os.path.join(OUT, "pca100.npy"), mmap_mode="r")
    members = np.where(labels == big)[0]
    log("разбиваем кластер %d (%d сообщений) на %d подтем" % (big, len(members), BIG_CLUSTER_SUBTOPICS))
    subset = np.asarray(compressed[members])
    model = MiniBatchKMeans(n_clusters=BIG_CLUSTER_SUBTOPICS, random_state=42, batch_size=4096,
                            n_init=5, max_iter=200)
    sub = model.fit_predict(subset)
    new_labels = labels.copy()
    for base in range(BIG_CLUSTER_SUBTOPICS):
        new_id = 100 + base
        new_labels[members[sub == base]] = new_id
    log("подтемы созданы: %s" % sorted(set(int(x) for x in new_labels if x >= 100)))
    return new_labels, big


def representatives(labels: np.ndarray, keep: set) -> dict:
    """Примеры, ближайшие к центру своего кластера."""
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
    centers = {}
    for key in sums:
        vector = sums[key] / max(1, counts[key])
        norm = np.linalg.norm(vector)
        centers[key] = vector / norm if norm else vector
    best = collections.defaultdict(list)
    for start in range(0, vectors.shape[0], step):
        stop = min(start + step, vectors.shape[0])
        block = np.asarray(vectors[start:stop])
        for offset in range(stop - start):
            index = start + offset
            center = centers.get(int(labels[index]))
            if center is None:
                continue
            score = float(np.dot(block[offset], center))
            bucket = best[int(labels[index])]
            if len(bucket) < REPRESENTATIVES * 5:
                bucket.append((score, index))
            elif score > min(bucket)[0]:
                bucket.remove(min(bucket))
                bucket.append((score, index))
    return {key: [index for _, index in sorted(items, reverse=True)] for key, items in best.items()}


def main() -> None:
    labels = np.load(os.path.join(OUT, "labels.npy"))
    labels = assign_noise(labels)
    labels, big = split_biggest(labels)
    np.save(os.path.join(OUT, "labels_final.npy"), labels)

    sizes = collections.Counter(int(x) for x in labels)
    keep = set(sizes)
    reps = representatives(labels, keep)
    log("примеры собраны")

    rows_list = list(rows())
    profiles = {}
    for index, row in enumerate(rows_list):
        label = int(labels[index]) if index < len(labels) else -1
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

    old = json.load(io.open(os.path.join(OUT, "clusters.json"), encoding="utf-8"))
    known = {int(item["cluster"]): item for item in old.get("clusters") or []}

    import kfc_ai

    system = ("Ты аналитик медиаполя сети фастфуда Rostic's (бывший KFC). По выборке сообщений "
              "называешь смысловую группу жалоб. По-русски, деловым языком, без выдумок.")
    result = []
    for label, size in sizes.most_common():
        profile = profiles.get(label)
        if not profile or profile["size"] < 400:
            continue
        examples = [rows_list[index] for index in reps.get(label, [])[:REPRESENTATIVES]
                    if index < len(rows_list)]
        entry = known.get(label)
        months = profile["months"]
        entry = {
            "cluster": label,
            "title": (entry or {}).get("title", ""),
            "essence": (entry or {}).get("essence", ""),
            "action": (entry or {}).get("action", ""),
            "size": profile["size"],
            "share": round(profile["size"] / float(len(rows_list)), 4),
            "reach": profile["reach"], "likes": profile["likes"],
            "months": dict(months),
            "peak_month": max(months.items(), key=lambda kv: kv[1])[0] if months else "",
            "from_2026": sum(count for month, count in months.items() if month >= "2026-01"),
            "cities": profile["cities"].most_common(8),
            "hubs": profile["hubs"].most_common(8),
            "hubtypes": profile["hubtypes"].most_common(4),
            "ratings": dict(profile["ratings"]),
            "stories": profile["stories"].most_common(4),
            "examples": [{"text": (item.get("text") or "")[:400], "hub": item.get("hub", ""),
                          "city": item.get("city", ""), "month": item.get("month", ""),
                          "rating": item.get("rating", ""), "url": item.get("url", ""),
                          "reach": item.get("reach", 0)} for item in examples[:8]],
        }
        if not entry["title"]:
            texts = "\n".join("- " + (item.get("text") or "")[:220].replace("\n", " ") for item in examples[:12])
            prompt = (
                "Группа жалоб клиентов сети фастфуда. Сообщений: %d.\nГорода: %s\nПлощадки: %s\n"
                "Примеры:\n%s\n\n"
                "Ответь строго JSON: {\"название\": \"до 6 слов, что это за боль\", "
                "\"суть\": \"2 предложения, на что жалуются и почему это важно бренду\", "
                "\"что_делать\": \"1 предложение, конкретное действие\"}"
                % (profile["size"],
                   ", ".join(city for city, _ in profile["cities"].most_common(5)) or "нет",
                   ", ".join(hub for hub, _ in profile["hubs"].most_common(4)) or "нет", texts))
            named = kfc_ai.ask_json(prompt, system=system, model="deepseek-chat", max_tokens=700)
            if named:
                entry["title"] = str(named.get("название") or "").strip()
                entry["essence"] = str(named.get("суть") or "").strip()
                entry["action"] = str(named.get("что_делать") or "").strip()
        result.append(entry)
        log("группа %d (%d): %s" % (label, size, entry["title"][:70]))

    with io.open(os.path.join(OUT, "clusters_final.json"), "w", encoding="utf-8") as fh:
        json.dump({"clusters": result, "total": len(rows_list), "split_cluster": big,
                   "built_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}, fh,
                  ensure_ascii=False)
    log("готово: групп %d" % len(result))


if __name__ == "__main__":
    main()
