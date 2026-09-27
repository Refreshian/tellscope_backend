#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Понятная картинка про авторов: боли в центре, вокруг — люди, которые о них пишут.

Прежний вариант с десятками тысяч точек читать невозможно. Здесь два понятных слоя:
  1. радиальная схема: каждая крупная боль — узел на окружности, вокруг неё — самые
     активные авторы этой боли; подписано, сколько всего авторов в группе и сколько
     сообщений на автора;
  2. столбики: сколько авторов стоит за каждой болью и какая доля объёма приходится
     на ядро (авторы с тремя и более сообщениями).

Результат: /tmp/kfc_topics/author_map.png
"""
from __future__ import annotations

import io
import json
import math
import os

import numpy as np

OUT = "/tmp/kfc_topics"


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.load(io.open(os.path.join(OUT, "author_core.json"), encoding="utf-8"))
    per_cluster = data.get("clusters") or {}
    rows = sorted(per_cluster.values(), key=lambda row: -row["sampled"])[:10]
    if not rows:
        print("нет данных")
        return

    fig = plt.figure(figsize=(13, 7.2), dpi=170)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.05)

    # --- радиальная схема
    ax = fig.add_subplot(grid[0, 0])
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.25, 1.25)
    ax.axis("off")
    count = len(rows)
    palette = plt.get_cmap("tab10")
    for index, row in enumerate(rows):
        angle = 2 * math.pi * index / count - math.pi / 2
        cx, cy = math.cos(angle), math.sin(angle)
        radius = 0.10 + 0.12 * math.sqrt(row["authors"] / max(1, rows[0]["authors"]))
        ax.scatter([cx], [cy], s=320 * radius * 10, color=palette(index % 10), alpha=0.85,
                   edgecolors="white", linewidths=1.2, zorder=3)
        label = (row.get("title") or "боль")[:34]
        ax.text(cx * 1.16, cy * 1.16, "%s\n%d авт." % (label, row["authors"]), fontsize=8.5,
                ha="center", va="center", zorder=4,
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#d0d5dd", alpha=0.9))
        # авторы вокруг боли
        authors = row.get("top_authors") or []
        for position, author in enumerate(authors[:6]):
            spread = 0.34
            offset = (position - (min(len(authors), 6) - 1) / 2.0) * spread / max(1, min(len(authors), 6) - 1)
            base_angle = angle + math.pi / 2
            ax_x = cx + (radius + 0.30) * math.cos(angle) + offset * math.cos(base_angle)
            ax_y = cy + (radius + 0.30) * math.sin(angle) + offset * math.sin(base_angle)
            ax.plot([cx, ax_x], [cy, ax_y], color="#98a2b3", linewidth=0.6, alpha=0.55, zorder=1)
            ax.scatter([ax_x], [ax_y], s=18, color="#f79009", zorder=2)
            if position < 3 and author.get("name"):
                ax.text(ax_x, ax_y - 0.045, (author["name"] or "")[:16], fontsize=6.2, ha="center",
                        va="top", color="#475467")
    ax.set_title("Кто пишет о каждой боли: узел — боль, оранжевые точки — самые активные авторы",
                 fontsize=11.5, fontweight="bold")

    # --- столбики: авторы и ядро
    ax2 = fig.add_subplot(grid[0, 1])
    labels = [(row.get("title") or "боль")[:30] for row in rows][::-1]
    authors = [row["authors"] for row in rows][::-1]
    core = [row["core_authors"] for row in rows][::-1]
    positions = np.arange(len(rows))
    ax2.barh(positions, authors, color="#d0d5dd", label="все авторы в выборке")
    ax2.barh(positions, core, color="#1760e8", label="ядро: 3+ сообщения")
    ax2.set_yticks(positions)
    ax2.set_yticklabels(labels, fontsize=8)
    ax2.set_xlabel("авторов")
    ax2.set_title("Массовость: за каждой болью стоят тысячи людей,\nядро — единицы",
                  fontsize=11.5, fontweight="bold")
    ax2.legend(fontsize=8, loc="lower right")
    for position, row in enumerate(rows[::-1]):
        ax2.text(row["authors"] + max(authors) * 0.01, position,
                 "%s%% объёма от ядра" % ("%.1f" % row["core_share"]).replace(".", ","),
                 va="center", fontsize=7, color="#475467")
    ax2.margins(x=0.22)

    fig.suptitle("Авторы негатива: %s человек, %s сообщений"
                 % ("{:,}".format((data.get("global") or {}).get("authors_total") or 0).replace(",", " "),
                    "{:,}".format((data.get("global") or {}).get("messages_total") or 0).replace(",", " ")),
                 fontsize=12.5, fontweight="bold")
    path = os.path.join(OUT, "author_map.png")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print("готово:", path)


if __name__ == "__main__":
    main()
