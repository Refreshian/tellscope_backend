#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Кластеризация негативных сообщений KFC и карта смыслов.

Берёт готовые эмбеддинги (bge-m3), сжатие PCA и проекцию UMAP и делит корпус на
смысловые кластеры. Дальше — названия кластеров, их динамика и примеры.

Результат: /tmp/kfc_topics/{labels.npy, umap2.npy, cluster_report.json}
"""
from __future__ import annotations

import collections
import datetime
import io
import json
import os
import time

import numpy as np

OUT = "/tmp/kfc_topics"


def log(message: str) -> None:
    print("[%s] %s" % (datetime.datetime.now().strftime("%H:%M:%S"), message), flush=True)


def cluster(reduced: np.ndarray) -> np.ndarray:
    """HDBSCAN с подбором доступных параметров: версии sklearn различаются."""
    from sklearn.cluster import HDBSCAN

    variants = [
        {"min_cluster_size": 500, "min_samples": 25, "metric": "euclidean",
         "cluster_selection_method": "eom", "n_jobs": 8},
        {"min_cluster_size": 500, "min_samples": 25, "metric": "euclidean",
         "cluster_selection_method": "eom"},
        {"min_cluster_size": 500, "min_samples": 25, "metric": "euclidean"},
    ]
    last = None
    for params in variants:
        try:
            started = time.time()
            labels = HDBSCAN(**params).fit_predict(reduced)
            log("HDBSCAN сработал с %s за %.0f с" % (list(params.keys()), time.time() - started))
            return labels
        except TypeError as exc:
            last = exc
            log("параметры не подошли: %s" % str(exc)[:120])
    raise RuntimeError("HDBSCAN не запустился: %s" % last)


def main() -> None:
    reduced_path = os.path.join(OUT, "umap10.npy")
    if not os.path.isfile(reduced_path):
        log("нет проекции umap10.npy — сначала шаг с эмбеддингами")
        return
    reduced = np.load(reduced_path)
    log("проекция: %s" % (reduced.shape,))
    labels = cluster(reduced)

    # карта смыслов: 2D-проекция для картинки
    map2d_path = os.path.join(OUT, "umap2.npy")
    if not os.path.isfile(map2d_path):
        import umap
        compressed = np.load(os.path.join(OUT, "pca100.npy"))
        started = time.time()
        map2d = umap.UMAP(n_neighbors=30, n_components=2, min_dist=0.15, metric="euclidean",
                          n_jobs=8, low_memory=True).fit_transform(compressed)
        np.save(map2d_path, map2d.astype(np.float32))
        log("карта смыслов готова за %.0f с" % (time.time() - started))
    np.save(os.path.join(OUT, "labels.npy"), labels)

    counts = collections.Counter(int(x) for x in labels if x >= 0)
    noise = int((labels < 0).sum())
    log("кластеров: %d | вне кластеров: %d (%.1f%%)" % (len(counts), noise,
                                                        100.0 * noise / len(labels)))
    for cluster_id, count in counts.most_common(15):
        log("   кластер %d: %d сообщений" % (cluster_id, count))
    report = {"clusters": len(counts), "noise": noise, "total": int(len(labels)),
              "sizes": {str(key): value for key, value in counts.most_common()}}
    with io.open(os.path.join(OUT, "cluster_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False)
    log("готово")


if __name__ == "__main__":
    main()
