#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Сводит смысловые группы жалоб в устойчивые темы.

Кластеризация даёт много близких групп: «Долгое ожидание заказов», «Долгое время ожидания
заказа», «долгое ожидание заказа» — это одна боль, разложенная на несколько кластеров.
Здесь близкие группы объединяются в понятные темы с честным пересчётом объёмов, динамики
и географии по сообщениям.

Результат: /tmp/kfc_topics/clusters_themes.json + theme_bars.png
"""
from __future__ import annotations

import collections
import datetime
import io
import json
import os
import re

import numpy as np

OUT = "/tmp/kfc_topics"

# Правила отнесения группы к теме: проверяются сверху вниз, первое совпадение выигрывает.
# Порядок важен: конкретные случаи (ребрендинг, безопасность, техсбои) идут раньше общих.
RULES = [
    (r"позитивн|отсутствие жалоб", "positive"),
    (r"отравлен|сальмонелл|тухл", "safety"),
    (r"приложен|it-систем|техническ\w* сбой|сбой", "app"),
    (r"оплат", "payment"),
    (r"напит|морожен|кофе", "drinks"),
    (r"ребрендинг|переход|несоответствие ожиданий|сравнение с прежн", "rebrand"),
    (r"гряз|антисанитар|санитар", "clean"),
    (r"комплектац|недостаток|картошк|\bзаказ\w* не", "order"),
    (r"\bед[аыуеой]\b|ед[ыуае]|продукц|свежест|крыл|стрипс|холодн|невкусн|качеств\w* еды", "food"),
    (r"ожидан|медленн|очеред|отмен", "wait"),
    (r"обслужив|персонал|грубост|хамств|атмосфер|график\w* работы|подростк", "service"),
]

THEMES = {
    "wait": ("Ожидание и скорость обслуживания",
             "Долгое ожидание заказа, очереди, медленная выдача, отмены из-за задержек."),
    "food": ("Качество и вкус еды",
             "Невкусные и остывшие блюда, испорченные продукты, нарекания к конкретным позициям меню."),
    "service": ("Обслуживание и персонал",
                "Грубость, равнодушие, нехватка смены, атмосфера в зале, поведение других посетителей."),
    "clean": ("Чистота и санитария",
              "Грязь в зале и туалетах, немытые столы, следы антисанитарии."),
    "order": ("Комплектация заказов",
              "Забыли позицию, положили не то, неполный заказ, ошибки в составе."),
    "rebrand": ("Ребрендинг и сравнение с прежним KFC",
                "Сравнение с KFC, несоответствие ожиданий, «раньше было лучше»."),
    "app": ("Приложение и технические сбои",
            "Не работает приложение, сбои касс и оплаты, невозможно оформить заказ."),
    "safety": ("Отравления и безопасность",
               "Жалобы на отравления, сальмонелла, просрочка — темы с высоким риском."),
    "payment": ("Оплата",
                "Отказ принимать наличные, проблемы с оплатой на кассе."),
    "drinks": ("Напитки и десерты",
               "Напитки, кофе, мороженое — отдельная линия жалоб."),
    "positive": ("Положительные отзывы",
                 "Сообщения без претензий: похвала, благодарности, нейтральные отзывы."),
}


def theme_of(title: str) -> str:
    """Тема группы по её названию."""
    low = (title or "").lower()
    for pattern, theme in RULES:
        if re.search(pattern, low):
            return theme
    return ""


def log(message: str) -> None:
    print("[%s] %s" % (datetime.datetime.now().strftime("%H:%M:%S"), message), flush=True)


def rows():
    with io.open(os.path.join(OUT, "negatives.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> None:
    labels = np.load(os.path.join(OUT, "labels_final.npy"))
    clusters = json.load(io.open(os.path.join(OUT, "clusters_final.json"), encoding="utf-8"))
    items = clusters.get("clusters") or []
    group_title = {int(item["cluster"]): (item.get("title") or "группа %d" % item["cluster"])
                   for item in items}
    group_size = {int(item["cluster"]): item["size"] for item in items}

    profiles = {}
    unknown = collections.Counter()
    for index, row in enumerate(rows()):
        if index >= len(labels):
            break
        cluster = int(labels[index])
        theme = theme_of(group_title.get(cluster, ""))
        if not theme:
            unknown[group_title.get(cluster, str(cluster))] += 1
            continue
        profile = profiles.get(theme)
        if profile is None:
            profile = profiles[theme] = {
                "size": 0, "reach": 0, "months": collections.Counter(),
                "cities": collections.Counter(), "hubs": collections.Counter(),
                "groups": collections.Counter(), "texts": [],
            }
        profile["size"] += 1
        profile["reach"] += row.get("reach") or 0
        if row.get("month"):
            profile["months"][row["month"]] += 1
        if row.get("city"):
            profile["cities"][row["city"]] += 1
        if row.get("hub"):
            profile["hubs"][row["hub"]] += 1
        profile["groups"][group_title.get(cluster, str(cluster))] += 1
        if len(profile["texts"]) < 3 and len(row.get("text") or "") > 60:
            profile["texts"].append({"text": row["text"][:300], "hub": row.get("hub") or "",
                                     "url": row.get("url") or "", "month": row.get("month") or ""})

    total = sum(profile["size"] for profile in profiles.values()) or 1
    result = []
    for theme, profile in profiles.items():
        title, essence = THEMES.get(theme, (theme, ""))
        months = profile["months"]
        result.append({
            "theme": theme, "title": title, "essence": essence, "size": profile["size"],
            "share": round(profile["size"] / float(total), 4),
            "reach": profile["reach"],
            "peak_month": max(months.items(), key=lambda kv: kv[1])[0] if months else "",
            "from_2026": sum(count for month, count in months.items() if month >= "2026-01"),
            "months": dict(months),
            "cities": profile["cities"].most_common(8),
            "hubs": profile["hubs"].most_common(5),
            "groups": profile["groups"].most_common(12),
            "examples": profile["texts"][:3],
        })
    result.sort(key=lambda row: -row["size"])
    log("тем получилось: %d" % len(result))
    for row in result:
        log("   %-44s %7d  %5.1f%%  групп внутри: %d"
            % (row["title"][:42], row["size"], row["share"] * 100, len(row["groups"])))
    if unknown:
        log("не отнесено ни к одной теме: %s" % dict(unknown.most_common(5)))

    with io.open(os.path.join(OUT, "clusters_themes.json"), "w", encoding="utf-8") as fh:
        city_themes = collections.defaultdict(collections.Counter)
        for row in result:
            for city, count in row["cities"]:
                city_themes[city][row["theme"]] += count
        json.dump({"themes": result, "total": total,
                   "city_themes": {city: counter.most_common(6) for city, counter in city_themes.items()},
                   "groups": [{"cluster": item["cluster"], "title": item.get("title"),
                               "size": item["size"], "theme": theme_of(item.get("title") or "")}
                              for item in items],
                   "built_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}, fh,
                  ensure_ascii=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pains = [row for row in result if row["theme"] != "positive"]
    fig, ax = plt.subplots(figsize=(10.5, 6.2), dpi=170)
    shown = pains[::-1]
    ax.barh([("%s — %s%%" % (row["title"][:44], ("%.1f" % (row["share"] * 100)).replace(".", ",")))
             for row in shown], [row["size"] for row in shown], color="#1760e8")
    for index, row in enumerate(shown):
        ax.text(row["size"] + max(r["size"] for r in shown) * 0.01, index,
                "с 2026 года: %s" % "{:,}".format(row["from_2026"]).replace(",", " "),
                va="center", fontsize=7.5, color="#475467")
    ax.set_xlabel("негативных сообщений")
    ax.set_title("Темы жалоб: объём, доля и сколько пришло в 2026 году", fontsize=12, fontweight="bold")
    ax.margins(x=0.22)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "theme_bars.png"), dpi=170)
    plt.close(fig)
    log("график тем: theme_bars.png")

    # карта «тема × город»: в каком городе какая боль звучит громче
    city_order = collections.Counter()
    for row in pains:
        for city, count in row["cities"]:
            city_order[city] += count
    cities = [city for city, _ in city_order.most_common(12)]
    matrix = np.zeros((len(pains), len(cities)))
    for row_index, row in enumerate(pains):
        mapping = dict(row["cities"])
        total_theme = sum(mapping.values()) or 1
        for col_index, city in enumerate(cities):
            matrix[row_index, col_index] = mapping.get(city, 0) / float(total_theme) * 100
    fig, ax = plt.subplots(figsize=(11, 5.6), dpi=170)
    image = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(cities)))
    ax.set_xticklabels(cities, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(pains)))
    ax.set_yticklabels([row["title"][:40] for row in pains], fontsize=8)
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            if value >= 4:
                ax.text(col_index, row_index, "%.0f" % value, ha="center", va="center", fontsize=7,
                        color="#101828")
    ax.set_title("Где живёт каждая тема: доля сообщений темы по городам, %", fontsize=12,
                 fontweight="bold")
    fig.colorbar(image, ax=ax, shrink=0.8, label="доля темы, %")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "theme_city.png"), dpi=170)
    plt.close(fig)
    log("карта «тема × город»: theme_city.png")
    log("готово")


if __name__ == "__main__":
    main()
