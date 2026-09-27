# -*- coding: utf-8 -*-
"""Сжатый аналитический документ по бренду: три раздела, сильные числа, два визуала.

Документ самодостаточный: несколько страниц, главное для решения, ничего лишнего.
"""
from __future__ import annotations

import os
import re

from kfc_build_crossyear_v2 import TABLE

import kfc_blocks2 as B2


def _ru(value: str) -> str:
    """Русская запись чисел: десятичная запятая вместо точки."""
    return str(value).replace(".", ",")


def _cut(text: str, limit: int) -> str:
    """Обрезка по границе слова, чтобы не рвать фразу посередине."""
    text = str(text or "").strip()
    if len(text) <= limit:
        return text
    head = text[:limit]
    space = head.rfind(" ")
    if space > limit * 0.6:
        head = head[:space]
    return head.rstrip(" ,;:—-(") + "…"


def _tidy_action(text: str) -> str:
    """Убирает скобки, которые повторяют уже названное в пункте число."""
    text = str(text or "—").strip()

    def drop(match):
        digits = re.findall(r"\d+", match.group(1))
        if digits and all(digit in text[:match.start()] for digit in digits):
            return ""
        return match.group(0)

    text = re.sub(r"\s{2,}", " ", re.sub(r"\s*\(([^()]*)\)", drop, text)).strip()
    return re.sub(r"(\d)\.(\d)", r"\1,\2", text)


def _share(value) -> str:
    return _ru(B2.share(value))


def _service_campaigns(campaigns: list) -> int:
    """Сколько окон кампаний дали рост именно сервисных тем."""
    words = ("ожидан", "обслужив", "персонал")
    count = 0
    for item in campaigns:
        top = B2.themes_of_growth(item.get("grew"))[:1]
        if top and any(word in (top[0]["title"] or "").lower() for word in words):
            count += 1
    return count


def _key_numbers(facts: dict, themes: list, authors: dict, cuts: dict) -> list:
    totals = facts.get("totals") or {}
    concentration = facts.get("concentration") or {}
    campaigns = (cuts or {}).get("campaigns") or []
    service = _service_campaigns(campaigns)
    years = facts.get("years") or {}
    pains = [row for row in themes if row.get("theme") != "positive"]
    top = pains[0] if pains else {}
    return [
        ["2,9 млн сообщений", "Столько раз о бренде писали за 28 месяцев: май 2024 — август 2026, "
                              "все площадки сразу."],
        ["%s негатива" % _share(totals.get("negative_share")),
         "Каждое десятое сообщение негативное. Хуже всего был 2025 год: %s."
         % _share((years.get("2025") or {}).get("negative_share"))],
        ["%d тем жалоб" % len(pains),
         "Весь негатив разложен по смыслу: половина приходится на две темы — «%s» (%s) и «%s» (%s)."
         % (pains[0].get("title", "—") if pains else "—", _share(pains[0].get("share")) if pains else "—",
            pains[1].get("title", "—") if len(pains) > 1 else "—",
            _share(pains[1].get("share")) if len(pains) > 1 else "—")],
        ["%s" % _share((authors.get("share_top100") or 0) / 100.0),
         "Столько негатива дают 100 самых активных авторов из %s: это массовая реакция клиентов, "
         "а не организованная кампания." % B2.big(authors.get("authors_total"))],
        ["%s из %s кампаний" % (service, len(campaigns)) if campaigns else "—",
         "Во время кампаний растут сервисные жалобы: ожидание, отмены заказов, качество обслуживания."],
        ["%s" % _share(concentration.get("top10_cities_share")),
         "Столько негатива приходится на десять городов — работу можно вести адресно, "
         "вплоть до конкретного ресторана."],
    ]


def build_sections(facts: dict, insights: dict, ctx) -> list:
    themes_data = B2.load("clusters_themes")
    themes = themes_data.get("themes") or []
    pains = [row for row in themes if row.get("theme") != "positive"]
    all_clusters = B2.load("clusters_final").get("clusters") or []
    clusters = all_clusters[:8]
    author_global = (B2.load("author_core").get("global") or {})
    cuts = B2.load("cluster_cuts")
    waves = B2.load("propagation").get("stories") or []
    sections = []

    # ---------------------------------------------------------------- раздел 1
    items = sorted(insights.get("insights") or [], key=lambda row: row.get("priority") or row.get("приоритет") or 9)[:5]
    conclusions = [[B2.report_text(item.get("заголовок") or "—"),
                    B2.report_text(item.get("что_делать") or "—")] for item in items]
    chart_ids = []
    bars = os.path.join(B2.TOPICS, "theme_bars.png")
    if os.path.isfile(bars):
        chart_ids.append(B2.register_image(ctx, "lead_themes", "Темы жалоб: объём, доля и что растёт", bars))
    sections.append({
        "heading": "Главное",
        "text": "Всё посчитано по всем сообщениям о бренде за 28 месяцев — без выборок и оценок на глаз. "
                "Ниже то, что объясняет картину целиком и определяет решения.",
        "chart_ids": chart_ids,
        "tables": [
            TABLE("Шесть чисел, которые всё объясняют", ["Число", "Что за ним стоит"],
                  _key_numbers(facts, themes, author_global, cuts), layout="portrait"),
            TABLE("Пять выводов и действия", ["Вывод", "Что делать"], conclusions, layout="landscape"),
        ],
    })

    # ---------------------------------------------------------------- раздел 2
    chart_ids = []
    for name, chart_title in (
            ("cluster_map_themes.png", "Кластеризация жалоб: сообщения, смысловые группы и темы"),
            ("theme_city.png", "Где живёт каждая тема: темы и города")):
        path = os.path.join(B2.TOPICS, name)
        if os.path.isfile(path):
            chart_ids.append(B2.register_image(ctx, "lead_" + name.split(".")[0], chart_title, path))
    group_rows = [[row.get("title") or "—", B2.big(row["size"]), _share(row["share"]),
                   B2.big(row.get("from_2026")), "%d" % len(row.get("groups") or [])]
                  for row in pains]
    city_rows = [[row["city"], B2.big(row["negative"]), _share(row["share"]),
                  _ru(B2.mult(row.get("excess_norm")))] for row in (facts.get("cities") or [])[:6]]
    rest_rows = [[row["city"], "%s из %s" % (B2.big(row["negative"]), B2.big(row["count"])),
                  str(row.get("rating") or "—").replace(".", ",")] for row in (facts.get("restaurants") or [])[:4]]
    sections.append({
        "heading": "Где именно болит",
        "text": "%s негативных сообщений разложены по смыслу и сведены в %d понятных тем: близкие "
                "формулировки («долгое ожидание заказа», «долгое время ожидания») — это одна боль, "
                "а не десять. На карте кластеризации каждая точка — одно сообщение: чем ближе точки, "
                "тем ближе смысл жалобы, цвет — тема. «С 2026 года» — сколько сообщений темы пришло "
                "в этом году, «групп внутри» — из скольких кластеров собрана тема. «Перевес» — "
                "во сколько раз негатива больше, чем ожидалось при таком же наборе площадок."
                % (B2.big((facts.get("totals") or {}).get("negative")), len(pains)),
        "chart_ids": chart_ids,
        "tables": [
            TABLE("Темы жалоб", ["Тема", "Сообщений", "Доля негатива", "С 2026 года", "Групп внутри"],
                  group_rows, layout="portrait"),
            TABLE("Города: где негатива больше, чем должно быть",
                  ["Город", "Негатив", "Доля негатива", "Перевес"], city_rows, layout="portrait"),
            TABLE("Рестораны, которые тянут картину вниз",
                  ["Город", "Негативных отзывов", "Рейтинг"], rest_rows, layout="portrait",
                  note="Рейтинг — средняя оценка по всем отзывам заведения, а не только по жалобам. "
                       "Такие точки видно поимённо в системе Tellscope и можно разбирать адресно."),
        ],
    })

    # ---------------------------------------------------------------- раздел 3
    wave_bullets = []
    for item in waves[:3]:
        days = item.get("days_to_peak") or 0
        when = ("пик — в день первой публикации" if not days
                else "пик — через %d дн. после первой публикации" % days)
        wave_bullets.append("Волна «%s»: %s сообщений, %s."
                            % (_cut(item["story"], 70), B2.big(item["messages"]), when))
    campaign_bullets = []
    for item in (cuts.get("campaigns") or [])[:3]:
        grew = "; ".join("%s — x%s" % (_cut(row["title"], 56), str(row["growth"]).replace(".", ","))
                         for row in B2.themes_of_growth(item.get("grew"))[:2])
        if grew:
            campaign_bullets.append("Кампания «%s» (%s): %s."
                                    % (_cut(item["campaign"], 52), item["from"][:7], grew))
    plan = (B2.load("insights", base=B2.HOT).get("plan") or {}).get("план") or {}
    plan_rows = []
    for key, horizon in (("30_дней", "первые 30 дней"), ("90_дней", "первые 90 дней")):
        for item in (plan.get(key) or [])[:3]:
            plan_rows.append([horizon, _tidy_action(item.get("действие")),
                              str(item.get("владелец") or "—")])
    sections.append({
        "heading": "Кто пишет, как быстро разгорается и что делать",
        "text": "%s авторов написали %s негативных сообщений; верхушка из 100 авторов даёт лишь %s "
                "негатива, а %s авторов написали по одному сообщению. Значит, дело в процессах, "
                "а не в отдельных людях."
                % (B2.big(author_global.get("authors_total")), B2.big(author_global.get("messages_total")),
                   _share((author_global.get("share_top100") or 0) / 100.0),
                   B2.big(author_global.get("authors_with_one"))),
        "tables": [TABLE("Что делать: первые 30 и 90 дней", ["Горизонт", "Действие", "Владелец"],
                         plan_rows, layout="landscape",
                         note="Владелец указан по функции, чтобы на рабочей встрече у каждого "
                              "пункта был ответственный.")],
        "bullets": wave_bullets + campaign_bullets + [
            "Что ещё видно в этих данных: какая боль где живёт по городам и ресторанам; кто разгоняет "
            "волну — цепочки от первой публикации до пересказов; что происходит после кампаний в равных "
            "окнах до и после; как меняется язык аудитории год к году.",
        ],
    })
    return [B2.polish(section) for section in sections]
