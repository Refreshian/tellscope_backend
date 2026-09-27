#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Векторизация всего негативного корпуса KFC и кластеризация смыслов (BERTopic-подход).

Зачем: группы жалоб по словам видят только то, что мы заранее придумали. Эмбеддинги
позволяют найти смысловые кластеры, о которых мы не догадывались: что именно люди
называют проблемой, какие сюжеты повторяются, как они меняются по годам и городам.

Шаги:
  1. выгрузка всех негативных сообщений (текст + метаданные) в /tmp/kfc_topics/negatives.jsonl;
  2. векторы через сервис эмбеддингов (bge-m3, 1024) с чекпоинтами в vectors.npy;
  3. UMAP + HDBSCAN → кластеры;
  4. название и суть каждого кластера — локальная Qwen3-32B;
  5. разрез кластеров по месяцам, городам, площадкам + примеры со ссылками.

Результат: /tmp/kfc_topics/{negatives.jsonl, vectors.npy, clusters.json, labels.npy}
Запуск:  setsid --fork venv_py312_clean/bin/python -u kfc_embed_negatives.py > /tmp/kfc_embed.log 2>&1
"""
from __future__ import annotations

import datetime
import io
import json
import os
import re
import time
import urllib.request

import numpy as np
from elasticsearch import Elasticsearch

INDEX = "kfc_13.05.2024-22.09.2026"
OUT = "/tmp/kfc_topics"
EMBED_MODEL = "deepvk/USER-bge-m3"
EMBED_DEVICE = "cuda:0"
LLM_URL = "http://127.0.0.1:8000/v1/chat/completions"
LLM_MODEL = "Qwen/Qwen3-32B-FP8"
MSK = datetime.timezone(datetime.timedelta(hours=3))
LO_TS = int(datetime.datetime(2024, 5, 13, tzinfo=MSK).timestamp())
HI_TS = int(datetime.datetime(2026, 8, 31, 23, 59, 59, tzinfo=MSK).timestamp())

BATCH = 256
CHECKPOINT = 40          # каждые 40 батчей пишем векторы на диск

_MODEL = None


def model():
    """Модель эмбеддингов держим в памяти: через неё считаем все тексты."""
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        started = time.time()
        _MODEL = SentenceTransformer(EMBED_MODEL, device=EMBED_DEVICE)
        _MODEL.max_seq_length = 256
        log("модель эмбеддингов загружена за %.1f с" % (time.time() - started))
    return _MODEL


def log(message: str) -> None:
    print("[%s] %s" % (datetime.datetime.now().strftime("%H:%M:%S"), message), flush=True)


def dump_negatives() -> int:
    """Все негативные сообщения: текст и то, что нужно для разрезов."""
    path = os.path.join(OUT, "negatives.jsonl")
    if os.path.isfile(path) and os.path.getsize(path) > 1000:
        with io.open(path, encoding="utf-8") as fh:
            rows = sum(1 for _ in fh)
        log("выгрузка уже готова: %d сообщений" % rows)
        return rows
    os.makedirs(OUT, exist_ok=True)
    es = Elasticsearch(hosts=["http://localhost:9200"], basic_auth=("elastic", "biz8z5i1w0nLPmEweKgP"),
                       verify_certs=False, request_timeout=600)
    query = {"bool": {"must": [{"term": {"toneMark": -1}}],
                      "filter": [{"range": {"timeCreate": {"gte": LO_TS, "lte": HI_TS}}}]}}
    source = ["text", "title", "city", "region", "hub", "hubtype", "url", "timeCreate",
              "review_rating", "audienceCount", "likesCount", "story", "tag_1", "tag_2", "tag_3"]
    written = 0
    started = time.time()
    resp = es.search(index=INDEX, body={"size": 3000, "query": query, "_source": source, "sort": ["_doc"]},
                     scroll="20m")
    sid = resp["_scroll_id"]
    with io.open(path, "w", encoding="utf-8") as out:
        try:
            while True:
                hits = resp["hits"]["hits"]
                if not hits:
                    break
                for hit in hits:
                    src = hit["_source"]
                    text = (src.get("text") or src.get("title") or "").strip()
                    if len(text) < 25:
                        continue
                    month = ""
                    try:
                        month = datetime.datetime.fromtimestamp(
                            int(float(src.get("timeCreate") or 0)), MSK).strftime("%Y-%m")
                    except Exception:  # noqa: BLE001
                        pass
                    out.write(json.dumps({
                        "id": hit["_id"], "text": text[:700], "month": month,
                        "city": (src.get("city") or "").strip(), "hub": src.get("hub") or "",
                        "hubtype": (src.get("hubtype") or "").strip(),
                        "rating": str(src.get("review_rating") or "").strip(),
                        "reach": int(src.get("audienceCount") or 0),
                        "likes": int(src.get("likesCount") or 0),
                        "story": (src.get("story") or "").strip(),
                        "url": src.get("url") or "",
                    }, ensure_ascii=False) + "\n")
                    written += 1
                if written % 30000 < 3000:
                    log("выгружено %d (%.0f с)" % (written, time.time() - started))
                resp = es.scroll(scroll_id=sid, scroll="20m")
                sid = resp["_scroll_id"]
        finally:
            try:
                es.clear_scroll(scroll_id=sid)
            except Exception:  # noqa: BLE001
                pass
    log("выгрузка готова: %d сообщений за %.0f с" % (written, time.time() - started))
    return written


def read_rows():
    with io.open(os.path.join(OUT, "negatives.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def embed(texts: list) -> list:
    """Векторы пачкой: префикс passage — так обучена модель bge-m3."""
    vectors = model().encode(["passage: " + text for text in texts], batch_size=BATCH,
                             normalize_embeddings=True, show_progress_bar=False)
    return vectors


def build_vectors(total: int) -> str:
    """Векторы всех сообщений с чекпоинтами: можно прервать и продолжить."""
    path = os.path.join(OUT, "vectors.npy")
    done = 0
    if os.path.isfile(path):
        try:
            done = int(np.load(path, mmap_mode="r").shape[0])
        except Exception:  # noqa: BLE001
            done = 0
        log("векторов уже посчитано: %d" % done)
    if done >= total:
        return path
    matrix = np.lib.format.open_memmap(path, mode="r+" if done else "w+", dtype=np.float32,
                                       shape=(total, 1024))
    started = time.time()
    index = 0
    buffer = []

    def flush(position: int) -> int:
        if not buffer:
            return position
        vectors = embed(list(buffer))
        if len(vectors) != len(buffer):
            raise RuntimeError("модель вернула %d векторов на %d текстов" % (len(vectors), len(buffer)))
        matrix[position:position + len(buffer)] = np.asarray(vectors, dtype=np.float32)
        position += len(buffer)
        buffer.clear()
        if (position // BATCH) % CHECKPOINT == 0:
            matrix.flush()
            speed = (position - done) / max(1.0, time.time() - started)
            left = (total - position) / max(0.1, speed)
            log("векторов %d из %d, %.0f текстов/с, осталось ~%.0f мин"
                % (position, total, speed, left / 60.0))
        return position

    for row in read_rows():
        if index < done:
            index += 1
            continue
        buffer.append(row["text"])
        index += 1
        if len(buffer) >= BATCH:
            done = flush(done)
    done = flush(done)
    matrix.flush()
    log("векторы готовы: %d" % done)
    return path


def clusterize(matrix_path: str) -> None:
    """PCA → UMAP → HDBSCAN: смысловые кластеры жалоб.

    Сначала сжимаем 1024 признака до 100 главных компонент (это быстро и параллельно),
    потом строим соседей и кластеры: на полном наборе из 276 тысяч сообщений иначе
    расчёт растягивается на часы.
    """
    import umap
    from sklearn.cluster import HDBSCAN
    from sklearn.decomposition import PCA

    matrix = np.load(matrix_path, mmap_mode="r")
    log("кластеризация: %d векторов" % matrix.shape[0])
    pca = PCA(n_components=100, random_state=42, svd_solver="randomized")
    compressed = pca.fit_transform(np.asarray(matrix))
    np.save(os.path.join(OUT, "pca100.npy"), compressed.astype(np.float32))
    log("PCA готова: 100 компонент, объяснено %.1f%% дисперсии"
        % (pca.explained_variance_ratio_.sum() * 100))
    reducer = umap.UMAP(n_neighbors=30, n_components=10, min_dist=0.0, metric="euclidean",
                        n_jobs=8, low_memory=True, verbose=True)
    reduced = reducer.fit_transform(compressed)
    np.save(os.path.join(OUT, "umap10.npy"), reduced)
    log("соседи и проекция готовы")
    clusterer = HDBSCAN(min_cluster_size=500, min_samples=25, metric="euclidean",
                        cluster_selection_method="eom", core_dist_n_jobs=8)
    labels = clusterer.fit_predict(reduced)
    np.save(os.path.join(OUT, "labels.npy"), labels)
    unique = sorted(set(int(x) for x in labels if x >= 0))
    log("кластеров: %d, вне кластеров: %d" % (len(unique), int((labels < 0).sum())))

    map2d = umap.UMAP(n_neighbors=30, n_components=2, min_dist=0.1, metric="euclidean",
                      n_jobs=8, low_memory=True).fit_transform(compressed)
    np.save(os.path.join(OUT, "umap2.npy"), map2d)
    log("карта кластеров готова")


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    total = dump_negatives()
    path = build_vectors(total)
    if not os.path.isfile(os.path.join(OUT, "labels.npy")):
        clusterize(path)
    log("готово")


if __name__ == "__main__":
    main()
