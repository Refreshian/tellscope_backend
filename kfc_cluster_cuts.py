#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Три разреза, которых не было в отчёте:

  1. смысловые группы × города — где какая боль живёт;
  2. смысловые группы × площадки и их динамика по кварталам;
  3. кампании против болей — что усиливается, а что успокаивается во время кампаний.

Считается по всем негативным сообщениям (метки кластеров уже посчитаны), без обращения
к хранилищу: быстро и воспроизводимо.

Результат: /tmp/kfc_topics/cluster_cuts.json + графики heat_*.png, platform_dynamics.png,
campaign_pains.png
"""
from __future__ import annotations

import collections
import datetime
import glob
import io
import json
import os
import re

import numpy as np

OUT = "/tmp/kfc_topics"
CAMP_CACHE = "/tmp/kfc_cx_cache"
TOP_GROUPS = 12
TOP_CITIES = 12


def log(message: str) -> None:
    print("[%s] %s" % (datetime.datetime.now().strftime("%H:%M:%S"), message), flush=True)


def rows():
    with io.open(os.path.join(OUT, "negatives.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def quarter(month: str) -> str:
    if not month or len(month) < 7:
        return ""
    return "%s-%dкв" % (month[:4], (int(month[5:7]) - 1) // 3 + 1)


def month_span(start: str, end: str) -> list:
    """Месяцы между датами: кампании длятся один-три месяца."""
    try:
        first = datetime.date(int(start[:4]), int(start[4:6]), 1)
        last = datetime.date(int(end[:4]), int(end[4:6]), 1)
    except Exception:  # noqa: BLE001
        return []
    months = []
    current = first
    while current <= last:
        months.append("%04d-%02d" % (current.year, current.month))
        current = datetime.date(current.year + (1 if current.month == 12 else 0),
                                1 if current.month == 12 else current.month + 1, 1)
    return months


def previous_months(months: list, count: int) -> list:
    if not months:
        return []
    first = months[0]
    year, mon = int(first[:4]), int(first[5:7])
    out = []
    for _ in range(count):
        mon -= 1
        if mon == 0:
            mon = 12
            year -= 1
        out.append("%04d-%02d" % (year, mon))
    return list(reversed(out))


def campaigns() -> list:
    found = []
    for path in glob.glob(os.path.join(CAMP_CACHE, "camp_*_during_*.json")):
        name = os.path.basename(path)
        match = re.match(r"camp_(.+)_(\d{8})_during_(\d{8})\.json$", name)
        if not match:
            continue
        key, start, end = match.group(1), match.group(2), match.group(3)
        months = month_span(start, end)
        if not months:
            continue
        found.append({"key": key, "from": "%s-%s-%s" % (start[:4], start[4:6], start[6:]),
                      "to": "%s-%s-%s" % (end[:4], end[4:6], end[6:]), "months": months,
                      "human": re.sub(r"(?<=[а-яё])(?=[А-ЯЁ])", " ", key)})
    found.sort(key=lambda item: item["from"])
    return found


def main() -> None:
    labels = np.load(os.path.join(OUT, "labels_final.npy"))
    clusters = json.load(io.open(os.path.join(OUT, "clusters_final.json"), encoding="utf-8"))
    titles = {int(item["cluster"]): (item["title"] or "группа %d" % item["cluster"])
              for item in clusters["clusters"]}
    sizes = collections.Counter(int(x) for x in labels)
    top_groups = [cluster for cluster, _ in sizes.most_common(TOP_GROUPS)]

    group_city = collections.Counter()
    group_hub = collections.Counter()
    group_quarter = collections.Counter()
    group_month = collections.Counter()
    city_total = collections.Counter()
    platform_total = collections.Counter()
    group_city_quarter = collections.Counter()
    group_bucket_quarter = collections.Counter()

    def bucket(hubtype: str) -> str:
        """Две крупные корзины площадок: отзывы и всё остальное."""
        if "Отзыв" in hubtype:
            return "Отзывы"
        if any(word in hubtype for word in ("Мессенджер", "Соцсет", "Микроблог", "Блог", "Форум")):
            return "Соцсети и мессенджеры"
        return "СМИ и прочее"

    scanned = 0
    for index, row in enumerate(rows()):
        if index >= len(labels):
            break
        label = int(labels[index])
        if label not in top_groups:
            continue
        scanned += 1
        city = row.get("city") or ""
        hub = (row.get("hubtype") or row.get("hub") or "").strip()
        month = row.get("month") or ""
        if city:
            group_city[(label, city)] += 1
            city_total[city] += 1
            if month:
                group_city_quarter[(label, city, quarter(month))] += 1
        if hub:
            group_hub[(label, hub)] += 1
            platform_total[hub] += 1
            if month:
                group_bucket_quarter[(label, bucket(hub), quarter(month))] += 1
        if month:
            group_quarter[(label, quarter(month))] += 1
            group_month[(label, month)] += 1
    log("обработано сообщений: %d" % scanned)

    cities = [city for city, _ in city_total.most_common(TOP_CITIES)]
    platforms = [platform for platform, _ in platform_total.most_common(8)]
    quarters = sorted({key for _, key in group_quarter})

    # --- кампании против болей
    merged = collections.OrderedDict()
    for item in campaigns():
        key = (item["from"], item["to"])
        if key in merged:
            merged[key]["names"].append(item["human"])
            continue
        merged[key] = {"names": [item["human"]], "from": item["from"], "to": item["to"],
                       "months": item["months"]}
    log("окон кампаний: %d (из %d записей)" % (len(merged), len(campaigns())))

    campaign_rows = []
    for window in merged.values():
        during_months = window["months"]
        before_months = previous_months(during_months, len(during_months))
        growth = []
        for cluster in top_groups:
            during = sum(group_month.get((cluster, month), 0) for month in during_months)
            before = sum(group_month.get((cluster, month), 0) for month in before_months)
            if during < 300 or before < 150:
                continue          # малая база даёт обманчивые кратности
            growth.append({"cluster": cluster, "title": titles[cluster], "during": during,
                           "before": before, "growth": round((during + 1) / float(before + 1), 2)})
        if len(growth) < 3:
            continue
        grew = [row for row in sorted(growth, key=lambda item: -item["growth"]) if row["growth"] >= 1.15]
        fell = [row for row in sorted(growth, key=lambda item: item["growth"]) if row["growth"] <= 0.85]
        campaign_rows.append({
            "campaign": " и ".join(window["names"])[:70], "from": window["from"], "to": window["to"],
            "months": during_months, "before_months": before_months,
            "grew": grew[:3], "fell": fell[:2],
            "total_during": sum(row["during"] for row in growth),
            "checked_groups": len(growth),
        })
    log("кампаний разобрано: %d" % len(campaign_rows))

    result = {
        "groups": [{"cluster": cluster, "title": titles[cluster], "size": sizes[cluster]}
                   for cluster in top_groups],
        "group_city": {"%d|%s" % key: value for key, value in group_city.items()},
        "group_hub": {"%d|%s" % key: value for key, value in group_hub.items()},
        "group_quarter": {"%d|%s" % key: value for key, value in group_quarter.items()},
        "group_month": {"%d|%s" % key: value for key, value in group_month.items()},
        "group_bucket_quarter": {"%d|%s|%s" % key: value for key, value in group_bucket_quarter.items()},
        "city_total": dict(city_total),
        "platform_total": dict(platform_total),
        "cities": cities, "platforms": platforms, "quarters": quarters,
        "campaigns": campaign_rows,
        "built_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    with io.open(os.path.join(OUT, "cluster_cuts.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False)
    log("сохранено cluster_cuts.json")

    # ------------------------------------------------------------------ графики
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. группы × города
    matrix = np.zeros((len(top_groups), len(cities)))
    for row_index, cluster in enumerate(top_groups):
        total = sum(group_city.get((cluster, city), 0) for city in cities) or 1
        for col_index, city in enumerate(cities):
            matrix[row_index, col_index] = group_city.get((cluster, city), 0) / float(total) * 100
    fig, ax = plt.subplots(figsize=(11, 6.4), dpi=170)
    image = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(cities)))
    ax.set_xticklabels(cities, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(top_groups)))
    ax.set_yticklabels([titles[cluster][:38] for cluster in top_groups], fontsize=8)
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            if value >= 3:
                ax.text(col_index, row_index, "%.0f" % value, ha="center", va="center", fontsize=7,
                        color="#101828")
    ax.set_title("Где живёт каждая боль: доля сообщений группы по городам, %", fontsize=12,
                 fontweight="bold")
    fig.colorbar(image, ax=ax, shrink=0.8, label="доля группы, %")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "heat_group_city.png"), dpi=170)
    plt.close(fig)
    log("карта «группы × города» готова")

    # 2. группы × площадки
    matrix = np.zeros((len(top_groups), len(platforms)))
    for row_index, cluster in enumerate(top_groups):
        total = sum(group_hub.get((cluster, platform), 0) for platform in platforms) or 1
        for col_index, platform in enumerate(platforms):
            matrix[row_index, col_index] = group_hub.get((cluster, platform), 0) / float(total) * 100
    fig, ax = plt.subplots(figsize=(10, 6.4), dpi=170)
    image = ax.imshow(matrix, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(platforms)))
    ax.set_xticklabels(platforms, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(top_groups)))
    ax.set_yticklabels([titles[cluster][:38] for cluster in top_groups], fontsize=8)
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            if value >= 5:
                ax.text(col_index, row_index, "%.0f" % value, ha="center", va="center", fontsize=7,
                        color="#101828")
    ax.set_title("Какая боль на какой площадке: доля сообщений группы по площадкам, %", fontsize=12,
                 fontweight="bold")
    fig.colorbar(image, ax=ax, shrink=0.8, label="доля группы, %")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "heat_group_platform.png"), dpi=170)
    plt.close(fig)
    log("карта «группы × площадки» готова")

    # 3. динамика по площадкам: отзывы против соцсетей и мессенджеров, по кварталам, топ-6 групп
    chosen = top_groups[:6]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), dpi=170, sharex=True)
    for ax, cluster in zip(axes.ravel(), chosen):
        reviews = [group_bucket_quarter.get((cluster, "Отзывы", key), 0) for key in quarters]
        others = [group_bucket_quarter.get((cluster, "Соцсети и мессенджеры", key), 0)
                  + group_bucket_quarter.get((cluster, "СМИ и прочее", key), 0) for key in quarters]
        ax.plot(range(len(quarters)), reviews, marker="o", markersize=3, label="отзывы (карты, отзовики)")
        ax.plot(range(len(quarters)), others, marker="o", markersize=3,
                label="соцсети, мессенджеры и СМИ")
        ax.set_title(titles[cluster][:40], fontsize=9)
        ax.set_xticks(range(0, len(quarters), max(1, len(quarters) // 4)))
        ax.set_xticklabels([quarters[index] for index in range(0, len(quarters), max(1, len(quarters) // 4))],
                           fontsize=7, rotation=45, ha="right")
        ax.grid(alpha=0.25)
    axes.ravel()[0].legend(fontsize=7)
    fig.suptitle("Динамика болей по кварталам: объём сообщений", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "platform_dynamics.png"), dpi=170)
    plt.close(fig)
    log("динамика по кварталам готова")

    # 4. кампании против болей
    shown = [row for row in campaign_rows if row["grew"]][:8]
    if shown:
        fig, ax = plt.subplots(figsize=(11, 5.6), dpi=170)
        labels_x = ["%s (%s)" % (row["campaign"][:26], row["from"][:7]) for row in shown]
        positions = np.arange(len(shown))
        width = 0.26
        for offset, index in enumerate((0, 1, 2)):
            values = []
            for row in shown:
                item = row["grew"][index] if len(row["grew"]) > index else {"growth": 1.0}
                values.append(item["growth"])
            ax.bar(positions + (offset - 1) * width, values, width, label="боль №%d" % (index + 1))
        ax.axhline(1.0, color="#98a2b3", linewidth=0.8)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels_x, rotation=35, ha="right", fontsize=7.5)
        ax.set_ylabel("во сколько раз больше сообщений, чем до кампании")
        ax.set_title("Кампании и боли: какие жалобы растут во время кампании", fontsize=12,
                     fontweight="bold")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "campaign_pains.png"), dpi=170)
        plt.close(fig)
    log("график кампаний готов")
    log("готово")


if __name__ == "__main__":
    main()
