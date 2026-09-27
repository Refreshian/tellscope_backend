#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Графики смысловых кластеров жалоб: карта, объёмы, динамика по кварталам.

Результат: /tmp/kfc_topics/{cluster_map.png, cluster_bars.png, cluster_dynamics.png}
"""
from __future__ import annotations

import collections
import datetime
import io
import json
import os

import numpy as np

OUT = "/tmp/kfc_topics"


def log(message: str) -> None:
    print("[%s] %s" % (datetime.datetime.now().strftime("%H:%M:%S"), message), flush=True)


def quarter(month: str) -> str:
    if not month or len(month) < 7:
        return ""
    year, mon = month[:4], int(month[5:7])
    return "%s-%dкв" % (year, (mon - 1) // 3 + 1)


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = np.load(os.path.join(OUT, "labels_final.npy"))
    coords = np.load(os.path.join(OUT, "umap2.npy"))
    clusters = json.load(io.open(os.path.join(OUT, "clusters_final.json"), encoding="utf-8"))
    rows = clusters["clusters"]
    titles = {int(item["cluster"]): (item["title"] or ("группа %d" % item["cluster"])) for item in rows}

    # --- карта смыслов
    fig, ax = plt.subplots(figsize=(11, 8), dpi=170)
    top = [int(item["cluster"]) for item in rows[:16]]
    palette = plt.get_cmap("tab20")
    mask_other = ~np.isin(labels, top)
    ax.scatter(coords[mask_other, 0], coords[mask_other, 1], s=1.2, c="#d0d5dd", alpha=0.35,
               linewidths=0)
    for index, cluster in enumerate(top):
        mask = labels == cluster
        ax.scatter(coords[mask, 0], coords[mask, 1], s=2.2, color=palette(index % 20), alpha=0.75,
                   linewidths=0, label="%s (%d)" % (titles[cluster][:38], int(mask.sum())))
        center_x, center_y = coords[mask, 0].mean(), coords[mask, 1].mean()
        ax.annotate(titles[cluster][:34], (center_x, center_y), fontsize=7.5, fontweight="bold",
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#98a2b3", alpha=0.75))
    ax.set_title("Карта смыслов: каждая точка — негативное сообщение, цвет — смысловая группа",
                 fontsize=12, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=7, ncol=2, loc="lower left", framealpha=0.85)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "cluster_map.png"), dpi=170)
    plt.close(fig)
    log("карта смыслов готова")

    # --- объёмы
    fig, ax = plt.subplots(figsize=(10.5, 6.4), dpi=170)
    shown = list(reversed(rows[:18]))
    ax.barh([((item["title"] or "группа")[:44] + " — %d%%" % round(item["share"] * 100, 1))
             for item in shown], [item["size"] for item in shown], color="#1760e8")
    ax.set_xlabel("негативных сообщений")
    ax.set_title("Смысловые группы жалоб: объём и доля", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "cluster_bars.png"), dpi=170)
    plt.close(fig)
    log("диаграмма объёмов готова")

    # --- динамика по кварталам
    fig, ax = plt.subplots(figsize=(11, 6), dpi=170)
    quarters = sorted({quarter(month) for item in rows for month in (item["months"] or {})})
    quarters = [key for key in quarters if key]
    for item in rows[:10]:
        by_quarter = collections.Counter()
        for month, count in (item["months"] or {}).items():
            by_quarter[quarter(month)] += count
        series = [by_quarter.get(key, 0) for key in quarters]
        if sum(series) < 1000:
            continue
        ax.plot(range(len(quarters)), series, marker="o", markersize=3,
                label=(item["title"] or "группа")[:40])
    ax.set_xticks(range(len(quarters)))
    ax.set_xticklabels(quarters, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("негативных сообщений за квартал")
    ax.set_title("Как менялись боли по кварталам", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "cluster_dynamics.png"), dpi=170)
    plt.close(fig)
    log("динамика по кварталам готова")


if __name__ == "__main__":
    main()
