# -*- coding: utf-8 -*-
"""Краткая версия отчёта для конференции: главное, боли, авторы, что делать.

Отдельный компактный документ из той же аналитики: числа, карта боли, сообщества авторов,
волны, план действий и один экран системы. Рассчитан на 10–14 страниц.
"""
from __future__ import annotations

import os

from kfc_build_crossyear_v2 import TABLE, CHART_NOTE, mkchart

import kfc_blocks2 as B2


def build_brief_sections(facts: dict, insights: dict, ctx) -> list:
    sections = []

    # 1. главное в числах
    headline = B2.build_headline_block(facts, insights)
    headline["heading"] = "Главное в числах"
    headline["text"] = ("Всё посчитано по %s сообщениям о бренде за 28 месяцев "
                        "(13.05.2024 — 31.08.2026)." % B2.big((facts.get("totals") or {}).get("messages")))
    sections.append(headline)

    # 2. восемь выводов
    items = sorted(insights.get("insights") or [], key=lambda row: row.get("приоритет") or 9)[:8]
    rows = []
    for index, item in enumerate(items, 1):
        rows.append([str(index), item.get("заголовок") or "—", B2.report_text(item.get("цифра") or "—"),
                     B2.report_text(item.get("что_делать") or "—")])
    sections.append({
        "heading": "Восемь выводов, которые стоит обсудить",
        "text": "Каждый вывод опирается на числа полного отчёта; порядок — по приоритету действий.",
        "tables": [TABLE("Выводы и действия", ["№", "Вывод", "Число", "Что делать"], rows,
                         layout="landscape")],
    })

    # 3. карта боли
    data = B2.load("clusters_final")
    clusters = data.get("clusters") or []
    if clusters:
        chart_ids = []
        path = os.path.join(B2.TOPICS, "cluster_map.png")
        if os.path.isfile(path):
            chart_ids.append(B2.register_image(ctx, "brief_map", "Карта смыслов: боли бренда", path))
        rows = [[item.get("title") or "группа %s" % item["cluster"], B2.big(item["size"]),
                 B2.share(item["share"]), B2.big(item.get("from_2026")),
                 (item.get("essence") or "")[:150]] for item in clusters[:10]]
        sections.append({
            "heading": "Карта боли: что нашли эмбеддинги",
            "text": ("%s негативных сообщений превращены в векторы моделью bge-m3 и разделены "
                     "на %d смысловых групп. Так видно боли, которых нет ни в одном списке "
                     "ключевых слов: от «несоответствия ожиданий» до «бесконтрольных подростков "
                     "в зале»." % (B2.big(data.get("total")), len(clusters))),
            "chart_ids": chart_ids,
            "tables": [TABLE("Крупнейшие смысловые группы жалоб",
                             ["Группа", "Сообщений", "Доля негатива", "С 2026 года", "В чём суть"],
                             rows, layout="landscape")],
        })

    # 4. авторы
    authors = B2.load("author_core")
    global_stats = authors.get("global") or {}
    if global_stats:
        chart_ids = []
        path = os.path.join(B2.TOPICS, "author_map.png")
        if os.path.isfile(path):
            chart_ids.append(B2.register_image(ctx, "brief_authors",
                                               "Боли и их авторы: массовая реакция", path))
        rows = [
            ["Авторов в негативе", B2.big(global_stats.get("authors_total"))],
            ["Негативных сообщений", B2.big(global_stats.get("messages_total"))],
            ["Верхушка 100 авторов", B2.share((global_stats.get("share_top100") or 0) / 100.0) + " негатива"],
            ["Авторов с одним сообщением", B2.big(global_stats.get("authors_with_one"))],
        ]
        sections.append({
            "heading": "Кто создаёт негатив: массовая реакция, а не кампания",
            "text": ("Проверка на организованный негатив: на %s негативных сообщений приходится "
                     "%s авторов, и верхушка из 100 авторов даёт лишь %s негатива. Это меняет "
                     "стратегию: работать нужно с причинами в процессах, а не с отдельными авторами."
                     % (B2.big(global_stats.get("messages_total")),
                        B2.big(global_stats.get("authors_total")),
                        B2.share((global_stats.get("share_top100") or 0) / 100.0))),
            "chart_ids": chart_ids,
            "tables": [TABLE("Аудитория негатива", ["Показатель", "Значение"], rows, layout="portrait")],
        })

    # 5. города, волны, кампании
    cities = facts.get("cities") or []
    waves = B2.load("propagation").get("stories") or []
    cuts = B2.load("cluster_cuts")
    rows_city = [[row["city"], B2.big(row["negative"]), B2.share(row["share"]),
                  B2.mult(row.get("excess_norm")), B2.mult(row["growth"])] for row in cities[:10]]
    rows_wave = [[item["story"][:70], B2.big(item["messages"]), item.get("first_day") or "—",
                  item.get("peak_day") or "—", "%s" % (item.get("days_to_peak") or 0)] for item in waves[:5]]
    tables = [TABLE("Города-очаги", ["Город", "Негатив", "Доля негатива", "Перевес", "Рост"], rows_city,
                    layout="portrait"),
              TABLE("Волны: скорость и пик", ["Повод", "Сообщений", "Первый день", "Пик",
                                              "Дней до пика"], rows_wave, layout="landscape")]
    bullets = []
    campaigns = (cuts or {}).get("campaigns") or []
    service = sum(1 for item in campaigns for row in (item.get("grew") or [])
                  if any(word in row["title"].lower() for word in ("ожидан", "обслужив", "заказ", "атмосфер")))
    if campaigns:
        bullets.append("Во время кампаний растут сервисные жалобы: в %d из %d окон в первых строках "
                       "роста — ожидание, отмены заказов и качество обслуживания. Кампанию нужно "
                       "планировать вместе с усилением смен." % (service, len(campaigns)))
    if waves:
        fast = [item for item in waves if (item.get("days_to_peak") or 0) <= 1]
        bullets.append("Ответ нужен в первые сутки: у %d из %d разобранных поводов пик пришёл "
                       "в первый-второй день." % (len(fast), len(waves)))
    sections.append({
        "heading": "Где, когда и как быстро",
        "text": "География очагов, скорость волн и связь кампаний с ростом жалоб.",
        "tables": tables,
        "bullets": bullets,
    })

    # 6. что делать
    plan = (B2.load("insights", base=B2.HOT).get("plan") or {}).get("план") or {}
    plan_block = B2.load("insights3")
    sections.append(B2.polish(_plan_section(plan)))

    # 7. график из системы
    screens = []
    candidates = []
    for folder, pattern in (("/tmp/kfc_shots4", "view_"), ("/tmp/kfc_shots3", "info_"),
                            ("/tmp/kfc_shots2", "chart_")):
        if os.path.isdir(folder):
            candidates += [os.path.join(folder, name) for name in sorted(os.listdir(folder))
                           if name.startswith(pattern) and name.endswith(".png")]
    for index, path in enumerate(candidates[:2], 1):
        if os.path.getsize(path) > 60000:
            screens.append(B2.register_image(ctx, "brief_shot_%d" % index,
                                             "График из системы Tellscope по теме KFC", path))
    if screens:
        sections.append({
            "heading": "Как это видно в Tellscope",
            "text": "Те же данные доступны в интерфейсе: тему, период и площадки можно менять "
                    "и пересчитывать в любой момент.",
            "chart_ids": screens,
        })
    return [B2.polish(section) for section in sections]


def _plan_section(horizons: dict) -> dict:
    horizons = horizons or {}
    labels = [("30_дней", "Первые 30 дней"), ("90_дней", "Первые 90 дней"), ("год", "Год")]
    tables = []
    for key, title in labels:
        items = horizons.get(key) or []
        rows = [[str(item.get("действие") or "—"), str(item.get("владелец") or "—"),
                 str(item.get("срок") or "—"), str(item.get("kpi") or "—")] for item in items]
        if rows:
            tables.append(TABLE(title, ["Действие", "Владелец", "Срок", "По какому числу видно результат"],
                                rows, layout="landscape"))
    bullets = [str(item) for item in (horizons.get("как_мерить") or [])]
    return {"heading": "Что делать: 30, 90 дней и год",
            "text": ("Действия по приоритету очагов. Владелец указан по функции, чтобы на рабочей "
                     "встрече у каждого пункта был ответственный."),
            "tables": tables, "bullets": bullets}
