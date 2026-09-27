#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Карта кластеризации: те же сообщения, но цвет — тема, в которую сведены группы.

Слева — проекция негативных сообщений (UMAP по эмбеддингам bge-m3): близкие точки
означают близкий смысл. Справа — сколько смысловых групп слилось в каждую тему.

Результат: /tmp/kfc_topics/cluster_map_themes.png
"""
from __future__ import annotations

import collections
import datetime
import io
import json
import math
import os
import sys

import numpy as np

OUT = "/tmp/kfc_topics"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kfc_clusters_themes import THEMES, theme_of  # noqa: E402

SHORT = {
    "food": "Еда и вкус",
    "wait": "Ожидание",
    "service": "Обслуживание",
    "rebrand": "Ребрендинг",
    "clean": "Чистота",
    "order": "Комплектация",
    "app": "Приложение",
    "payment": "Оплата",
    "drinks": "Напитки",
    "safety": "Отравления",
    "positive": "Без претензий",
}


def log(message: str) -> None:
    print("[%s] %s" % (datetime.datetime.now().strftime("%H:%M:%S"), message), flush=True)


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = np.load(os.path.join(OUT, "labels_final.npy"))
    coords = np.load(os.path.join(OUT, "umap2.npy"))
    groups = (json.load(io.open(os.path.join(OUT, "clusters_final.json"), encoding="utf-8"))
              .get("clusters") or [])
    titles = {int(item["cluster"]): (item.get("title") or "") for item in groups}

    themes = np.empty(len(labels), dtype=object)
    filled = 0
    with io.open(os.path.join(OUT, "negatives.jsonl"), encoding="utf-8") as fh:
        for index, line in enumerate(fh):
            if index >= len(labels):
                break
            line = line.strip()
            if not line:
                continue
            cluster = int(labels[index])
            themes[index] = theme_of(titles.get(cluster, ""))
            filled = index + 1
    if filled < len(labels):
        themes[filled:] = ""
    log("тем проставлено: %d из %d" % (filled, len(labels)))

    order = ["food", "wait", "service", "rebrand", "clean", "order", "app", "payment",
             "drinks", "safety", "positive"]
    counts = collections.Counter(themes.tolist())
    grouped = collections.Counter()
    for item in groups:
        theme = theme_of(item.get("title") or "")
        if theme:
            grouped[theme] += 1

    palette = {
        "food": "#1f6feb", "wait": "#f97316", "service": "#16a34a", "rebrand": "#dc2626",
        "clean": "#7c3aed", "order": "#a16207", "app": "#db2777", "payment": "#0d9488",
        "drinks": "#ca8a04", "safety": "#0891b2", "positive": "#cbd5e1",
    }

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.8), dpi=170,
                             gridspec_kw={"width_ratios": [2.35, 1.0]})
    ax = axes[0]
    rng = np.random.default_rng(11)
    limit = 60000
    if len(labels) > limit:
        sample = rng.choice(len(labels), size=limit, replace=False)
    else:
        sample = np.arange(len(labels))
    other = np.array([key not in palette or not key for key in themes[sample]])
    ax.scatter(coords[sample][other, 0], coords[sample][other, 1], s=0.7, c="#e4e7ec",
               alpha=0.4, linewidths=0)
    centers = {}
    for key in order:
        mask = themes[sample] == key
        if not mask.any():
            continue
        ax.scatter(coords[sample][mask, 0], coords[sample][mask, 1], s=1.6, color=palette[key],
                   alpha=0.5, linewidths=0,
                   label="%s — %s" % (SHORT[key], "{:,}".format(counts[key]).replace(",", " ")))
        centers[key] = (float(coords[sample][mask, 0].mean()), float(coords[sample][mask, 1].mean()))

    # показываем плотную часть карты: выбросы растягивают оси и прячут основную массу
    xs, ys = coords[sample][:, 0], coords[sample][:, 1]
    x0, x1 = np.quantile(xs, 0.012), np.quantile(xs, 0.988)
    y0, y1 = np.quantile(ys, 0.012), np.quantile(ys, 0.988)
    pad_x, pad_y = (x1 - x0) * 0.10, (y1 - y0) * 0.12
    ax.set_xlim(x0 - pad_x, x1 + pad_x)
    ax.set_ylim(y0 - pad_y, y1 + pad_y)

    # подписываем только крупные темы: их области крупные и не наезжают друг на друга
    for key, (cx, cy) in centers.items():
        if counts[key] < 13000:
            continue
        ax.annotate(SHORT[key], (cx, cy), fontsize=8.5, fontweight="bold", ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#98a2b3", alpha=0.92))
    ax.set_title("Каждая точка — негативное сообщение, цвет — тема жалоб\n"
                 "(близкие точки — близкие по смыслу жалобы)",
                 fontsize=11.5, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=8, ncol=2, loc="lower left", framealpha=0.85, markerscale=6)

    ax2 = axes[1]
    shown = [key for key in order if key != "positive" and grouped.get(key)]
    shown.sort(key=lambda key: grouped[key])
    ax2.barh([SHORT[key] for key in shown], [grouped[key] for key in shown],
             color=[palette[key] for key in shown])
    for index, key in enumerate(shown):
        ax2.text(grouped[key] + 0.08, index, "%d" % grouped[key], va="center", fontsize=9)
    ax2.set_xlabel("смысловых групп внутри темы")
    ax2.set_xlim(0, max(grouped.values()) + 1.2)
    ax2.set_title("Сколько групп жалоб\nслилось в одну тему", fontsize=11.5, fontweight="bold")
    ax2.tick_params(axis="y", labelsize=9)
    for spine in ("top", "right"):
        ax2.spines[spine].set_visible(False)

    fig.suptitle("Кластеризация жалоб: %s сообщений → %d смысловых групп → %d понятных тем"
                 % ("{:,}".format(len(labels)).replace(",", " "), len(groups),
                    len([key for key in order if key != "positive" and counts[key]])),
                 fontsize=12.5, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(os.path.join(OUT, "cluster_map_themes.png"), dpi=170)
    plt.close(fig)
    log("карта кластеризации по темам: cluster_map_themes.png")
    log("групп по темам: %s" % dict(grouped.most_common()))


if __name__ == "__main__":
    main()
