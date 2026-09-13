# -*- coding: utf-8 -*-
"""Кластеризация всего корпуса среза: темы с частотами по ВСЕМУ корпусу.

Зачем отдельный модуль. Срез в десятки тысяч сообщений локальной моделью целиком не прочитать
(замер: ~840–2 250 сообщений за 5 минут), поэтому темы считаются по всему корпусу, а модель
читает только представителей кластеров и самые значимые сообщения — на них строятся цитаты и
пояснения, а частоты тем берутся из размеров кластеров по всему корпусу.

Что переиспользуется, а не пишется заново:
  * эмбеддинги — модель проекта (``embedding_model_manager.model_manager``), та же, что в
    семантическом поиске и в кластеризации датасетов в интерфейсе;
  * кластеризация — тот же стек, что в ``main.create_clustering_models_async``:
    UMAP (cosine, min_dist=0) + HDBSCAN (eom, prediction_data) + BERTopic;
  * источник документов — Elasticsearch через ``agent_engine.tools_data``.

Готовые темы датасета (Brand Analytics, поля ``tag_1..tag_33``) используются только как
ПОДСКАЗКА для названий кластеров. Они не подменяют кластеризацию и не подменяют частоты:
частоты всегда считаются по размеру кластера во всём корпусе.
"""
from __future__ import annotations

import re
import time
from typing import Any, Callable, Dict, List, Optional

CORPUS_SOURCE_FIELDS = [
    "text", "title", "timeCreate", "toneMark", "hub", "hubtype", "url", "city",
    "likesCount", "commentsCount", "repostsCount", "viewsCount", "audienceCount",
    "er", "massMediaAudience", "duplicateCount", "citeIndex", "rating", "review_rating",
    "authorObject",
] + ["tag_%d" % i for i in range(1, 34)]
TAG_FIELD_RE = re.compile(r"^tag_\d+$")
TONE_LABELS = {-1: "негатив", 0: "нейтрал", 1: "позитив"}



def _tag_labels(source: Dict[str, Any], prefix: str = "tag_") -> List[str]:
    """Подписи готовых тем сообщения (Brand Analytics): {tag_1: {"8450264": "HR KFC"}}."""
    labels: List[str] = []
    for name, value in (source or {}).items():
        if not TAG_FIELD_RE.match(str(name)) or not isinstance(value, dict):
            continue
        if prefix and not str(name).startswith(prefix):
            continue
        for label in value.values():
            text = " ".join(str(label or "").split()).strip()
            if text and text not in labels:
                labels.append(text)
    return labels


def fetch_corpus(index_name: str, query: Dict[str, Any], limit: int,
                 on_page: Optional[Callable[[int, int], None]] = None,
                 page_size: int = 2000) -> Dict[str, Any]:
    """Читает документы всего среза из Elasticsearch страницами (search_after).

    Возвращает {'docs', 'total', 'truncated'}: docs — по одному словарю на сообщение с текстом,
    метаданными, счётчиками вовлечённости и подписями готовых тем (подсказка для названий).
    """
    from .tools_data import _es

    es = _es()
    docs: List[Dict[str, Any]] = []
    total = 0
    truncated = False
    search_after: Optional[List[Any]] = None
    size = max(500, int(page_size))
    while True:
        body: Dict[str, Any] = {
            "size": size,
            "_source": CORPUS_SOURCE_FIELDS,
            "query": query,
            "sort": [{"timeCreate": {"order": "asc"}}, {"_shard_doc": {"order": "asc"}}],
            "track_total_hits": True if total == 0 else False,
        }
        if search_after:
            body["search_after"] = search_after
        try:
            res = es.search(index=index_name, body=body)
        except Exception:
            # очень старые индексы не умеют _shard_doc — откатываемся на сортировку по времени
            body["sort"] = [{"timeCreate": {"order": "asc"}}]
            body.pop("search_after", None)
            if search_after:
                body["search_after"] = search_after
            res = es.search(index=index_name, body=body)
        hits = (res.get("hits") or {}).get("hits") or []
        if total == 0:
            total = int(((res.get("hits") or {}).get("total") or {}).get("value") or 0)
        if not hits:
            break
        for hit in hits:
            source = hit.get("_source") or {}
            text = " ".join(str(source.get("text") or source.get("title") or "").split())
            if not text:
                continue
            docs.append({
                "es_id": hit.get("_id"),
                "text": text,
                # Исходные поля ES отдаём как есть: инструмент нормализует их тем же
                # _doc_from_hit, что и выборку среза, поэтому важность и метрики совпадают.
                "source": source,
                "tags": _tag_labels(source),
            })
        search_after = hits[-1].get("sort")
        if on_page is not None:
            try:
                on_page(len(docs), int(limit))
            except Exception:
                pass
        if search_after is None or len(docs) >= int(limit) or len(hits) < size:
            break
    if len(docs) > int(limit):
        docs = docs[: int(limit)]
        truncated = True
    # Сообщения без текста корпус не «обрезают»: это не предел, а отсутствие текста в выгрузке,
    # поэтому считаем их отдельно и не помечаем корпус усечённым.
    return {
        "docs": docs,
        "total": total,
        "truncated": truncated,
        "skipped_no_text": max(0, total - len(docs)),
    }


def _cluster_text(doc: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    """Текст сообщения для эмбеддингов: обрезаем по cluster_text_chars.

    Длина важнее всего: пачки выравниваются по самому длинному тексту, и на 600 символах
    замер дал ~180 сообщений/с, а на 300 — ~500 сообщений/с при той же структуре кластеров.
    """
    limit = max(80, int(cfg.get("cluster_text_chars") or 300))
    return str(doc.get("text") or "")[:limit]


def embed_matrix(texts: List[str], cfg: Dict[str, Any],
                 on_chunk: Optional[Callable[[int, int], None]] = None) -> Any:
    """Эмбеддинги корпуса моделью проекта. Возвращает float32-матрицу (n, dims)."""
    import numpy as np

    from embedding_model_manager import model_manager

    batch = max(16, int(cfg.get("cluster_embed_batch") or 64))
    chunks: List[Any] = []
    done = 0
    for start in range(0, len(texts), batch):
        part = texts[start:start + batch]
        if not part:
            continue
        vectors = model_manager.encode_texts(part, batch_size=len(part), normalize_embeddings=True)
        chunks.append(np.asarray(vectors, dtype="float32"))
        done += len(part)
        if on_chunk is not None:
            try:
                on_chunk(done, len(texts))
            except Exception:
                pass
    if not chunks:
        return np.zeros((0, 0), dtype="float32")
    return np.vstack(chunks)


def cluster_matrix(texts: List[str], embeddings: Any, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Кластеры корпуса тем же стеком, что в кластеризации датасетов: SVD → UMAP → HDBSCAN + BERTopic.

    Возвращает {'labels', 'sizes', 'keywords', 'count', 'noise'} — метка на каждое сообщение
    (по индексу), размеры кластеров по всему корпусу и ключевые слова кластера.
    """
    import numpy as np
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    n = len(texts)
    if n == 0 or getattr(embeddings, "shape", (0, 0))[0] == 0:
        return {"labels": [], "sizes": {}, "keywords": {}, "count": 0, "noise": 0}

    dims = int(cfg.get("cluster_svd_dims") or 50)
    dims = max(2, min(dims, embeddings.shape[1] - 1, n - 1))
    reduced = np.asarray(embeddings, dtype="float32")
    if dims and embeddings.shape[1] > dims:
        # SVD перед UMAP: на 50 измерениях UMAP в разы быстрее и не теряет структуру.
        reduced = TruncatedSVD(n_components=dims, random_state=42).fit_transform(reduced)

    neighbors = int(cfg.get("cluster_umap_neighbors") or 30)
    neighbors = max(2, min(neighbors, n - 2 if n > 2 else 2))
    min_size = max(5, int(cfg.get("cluster_min_size") or 30))
    umap_model = UMAP(
        n_neighbors=neighbors,
        n_components=int(cfg.get("cluster_umap_dims") or 5),
        min_dist=0.0,
        metric="cosine",
        random_state=42,
        n_jobs=1,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_size,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
        core_dist_n_jobs=1,
    )
    vectorizer = CountVectorizer(
        analyzer="word",
        token_pattern=r"(?u)\b\w+\b",
        lowercase=True,
        min_df=1,
        max_df=1.0,
        ngram_range=(1, 2) if n < 80 else (1, 3),
        stop_words=None,
    )
    model = BERTopic(
        embedding_model=None,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        min_topic_size=min_size,
        calculate_probabilities=False,
        verbose=False,
    )
    labels, _ = model.fit_transform(texts, embeddings=reduced)
    labels = [int(value) for value in labels]

    sizes: Dict[int, int] = {}
    for value in labels:
        if value < 0:
            continue
        sizes[value] = sizes.get(value, 0) + 1
    keywords: Dict[int, List[str]] = {}
    for topic_id in sizes:
        try:
            words = model.get_topic(int(topic_id)) or []
        except Exception:
            words = []
        keywords[int(topic_id)] = [str(word) for word, _score in words[:8]]
    return {
        "labels": labels,
        "sizes": sizes,
        "keywords": keywords,
        "count": len(sizes),
        "noise": sum(1 for value in labels if value < 0),
    }


def tag_hint_for_members(docs: List[Dict[str, Any]], members: List[int], limit: int = 6) -> str:
    """Подсказка для названия кластера: самые частые готовые темы его сообщений."""
    counts: Dict[str, int] = {}
    for pos in members:
        if pos < 0 or pos >= len(docs):
            continue
        for label in docs[pos].get("tags") or []:
            counts[label] = counts.get(label, 0) + 1
    if not counts:
        return ""
    best = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
    return best[0][0]


def cluster_corpus(index_name: str, query: Dict[str, Any], cfg: Dict[str, Any],
                   progress: Optional[Callable[[str, Dict[str, Any]], None]] = None,
                   on_page: Optional[Callable[[int, int], None]] = None,
                   on_chunk: Optional[Callable[[int, int], None]] = None) -> Optional[Dict[str, Any]]:
    """Полный проход: корпус → эмбеддинги → кластеры. None, если кластеризовать не удалось.

    cfg — настройка из mlops/lock.yaml (секция texts): cluster_embed_batch, cluster_min_size,
    cluster_max, cluster_umap_neighbors, cluster_svd_dims, cluster_corpus_limit.
    progress(stage, info) вызывается на каждом этапе (его пишет инструмент в прогресс запуска).
    """
    limit = int(cfg.get("cluster_corpus_limit") or 200000)
    started = time.time()
    stages: Dict[str, float] = {}

    def _stage(name: str, info: Optional[Dict[str, Any]] = None) -> None:
        if progress is not None:
            try:
                progress(name, info or {})
            except Exception:
                pass

    _stage("сбор корпуса", {"limit": limit})
    mark = time.time()
    corpus = fetch_corpus(index_name, query, limit, on_page=on_page)
    docs = corpus["docs"]
    stages["fetch"] = round(time.time() - mark, 1)
    if len(docs) < 2:
        return None
    _stage("сбор корпуса готов", {"docs": len(docs), "seconds": stages["fetch"],
                                  "truncated": bool(corpus.get("truncated"))})

    texts = [_cluster_text(doc, cfg) for doc in docs]
    mark = time.time()
    embeddings = embed_matrix(texts, cfg, on_chunk=on_chunk)
    stages["embed"] = round(time.time() - mark, 1)
    _stage("эмбеддинги готовы", {"docs": len(docs), "seconds": stages["embed"]})

    mark = time.time()
    result = cluster_matrix(texts, embeddings, cfg)
    stages["cluster"] = round(time.time() - mark, 1)
    labels = result["labels"]

    max_clusters = max(1, int(cfg.get("cluster_max") or 40))
    members: Dict[int, List[int]] = {}
    for pos, value in enumerate(labels):
        if value < 0:
            continue
        members.setdefault(int(value), []).append(pos)
    ordered = sorted(members.items(), key=lambda item: -len(item[1]))[:max_clusters]

    clusters: List[Dict[str, Any]] = []
    total = len(docs)
    for topic_id, positions in ordered:
        count = len(positions)
        # Тональность кластера — по ВСЕМ его сообщениям в корпусе, а не по прочитанным.
        tone_counts = {"negative": 0, "neutral": 0, "positive": 0}
        marks: List[int] = []
        for pos in positions:
            mark = (docs[pos].get("source") or {}).get("toneMark")
            try:
                mark = int(mark)
            except (TypeError, ValueError):
                continue
            if mark < 0:
                tone_counts["negative"] += 1
            elif mark > 0:
                tone_counts["positive"] += 1
            else:
                tone_counts["neutral"] += 1
            marks.append(1 if mark > 0 else (-1 if mark < 0 else 0))
        tone_avg = round(sum(marks) / float(len(marks)), 2) if marks else None
        if marks and min(marks) < 0 < max(marks):
            tone_label = "смешанная"
        elif tone_avg is not None:
            tone_label = TONE_LABELS.get(int(round(tone_avg)), "смешанная")
        else:
            tone_label = "—"
        clusters.append({
            "id": int(topic_id),
            "size": count,
            "share": round(count / float(total or 1), 4),
            "share_pct": round(100.0 * count / float(total or 1), 1),
            "keywords": result["keywords"].get(int(topic_id)) or [],
            "tag_hint": tag_hint_for_members(docs, positions),
            "tone_avg": tone_avg,
            "tone_label": tone_label,
            "tone_counts": tone_counts,
            "members": positions,
        })
    stages["total"] = round(time.time() - started, 1)
    _stage("кластеры готовы", {"clusters": len(clusters), "seconds": stages["cluster"],
                               "noise": result["noise"]})
    return {
        "docs": docs,
        "labels": labels,
        "clusters": clusters,
        "total": total,
        "skipped_no_text": int(corpus.get("skipped_no_text") or 0),
        "tagged": sum(1 for doc in docs if doc.get("tags")),
        "noise": int(result["noise"]),
        "clusters_total": int(result["count"]),
        "truncated": bool(corpus.get("truncated")),
        "stages": stages,
    }
