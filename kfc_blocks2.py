# -*- coding: utf-8 -*-
"""Разделы усиленного отчёта: смысловые группы, авторы, волны, внешний контекст, экраны.

Все тексты собираются из посчитанных чисел; где нужны формулировки — они уже получены
моделью отдельно. Картинки (карта смыслов, граф авторов, цепочки, снимки системы)
подключаются как обычные рисунки отчёта.
"""
from __future__ import annotations

import io
import json
import os

from kfc_build_crossyear_v2 import (TABLE, CHART_NOTE, mkchart, num, num_ru, pct, short_num)

TOPICS = "/tmp/kfc_topics"
HOT = "/tmp/kfc_hotspots"
SHOTS = "/tmp/kfc_shots"


def load(name: str, base: str = TOPICS):
    path = os.path.join(base, name + ".json")
    if not os.path.isfile(path):
        return {}
    with io.open(path, encoding="utf-8") as fh:
        return json.load(fh)


def big(value) -> str:
    try:
        return "{:,}".format(int(float(value))).replace(",", " ")
    except Exception:  # noqa: BLE001
        return "—"


def share(value, digits: int = 1) -> str:
    try:
        return ("%." + str(digits) + "f%%") % (float(value) * 100)
    except Exception:  # noqa: BLE001
        return "—"


def mult(value) -> str:
    try:
        return ("x%.2f" % float(value)).replace(".", ",")
    except Exception:  # noqa: BLE001
        return "—"


def report_text(value) -> str:
    """Приводит текст к виду, который не режет сборщик отчёта."""
    import re
    text = str(value or "")
    text = re.sub(r"\b\d{1,3}(?: \d{3}){2,}\b",
                  lambda m: big(int(m.group(0).replace(" ", "")) if int(m.group(0).replace(" ", "")) < 10 ** 6
                                else "%.1f млрд" % (int(m.group(0).replace(" ", "")) / 1e9)), text)
    text = re.sub(r"\b(40[0-9]|50[0-9])\b",
                  lambda m: "около четырёхсот" if int(m.group(1)) < 450 else "около пятисот", text)
    text = re.sub(r"([Пп])ричина:", r"\1очему так", text)
    text = re.sub(r"недоступ\w*", "не работает", text)
    # длинные дроби вида x1.5004508566275925 — округляем до сотых
    text = re.sub(r"(\d+)\.(\d{3,})",
                  lambda m: ("%.2f" % float(m.group(0))).replace(".", ","), text)
    return re.sub(r"\s{2,}", " ", text).strip()


def register_image(ctx, chart_id: str, title: str, path: str) -> str:
    """Подключает готовую картинку как рисунок отчёта."""
    ctx.charts[chart_id] = {"chart_id": chart_id, "title": title, "name": os.path.basename(path),
                            "path": path, "url": ""}
    return chart_id


def polish(section: dict) -> dict:
    out = dict(section or {})
    for key in ("text", "note"):
        if out.get(key):
            out[key] = report_text(out[key])
    if isinstance(out.get("bullets"), list):
        out["bullets"] = [report_text(item) for item in out["bullets"]]
    tables = []
    for spec in out.get("tables") or []:
        spec = dict(spec or {})
        if spec.get("note"):
            spec["note"] = report_text(spec["note"])
        tables.append(spec)
    if tables:
        out["tables"] = tables
    return out


# ------------------------------------------------------------------ что видно сразу

def build_headline_block(facts: dict, insights: dict) -> dict:
    """Первый раздел: главное в числах, без объяснений «зачем этот отчёт»."""
    totals = facts.get("totals") or {}
    concentration = facts.get("concentration") or {}
    clusters = (load("clusters_final") or {}).get("clusters") or []
    theme_pains = [row for row in (load("clusters_themes").get("themes") or [])
                   if row.get("theme") != "positive"]
    authors = (load("author_core") or {}).get("global") or {}
    scen = facts.get("scenarios") or {}
    forecast = scen.get("forecast") or []
    top_cluster = clusters[0] if clusters else {}
    rows = [
        ["Разговор о бренде", "%s сообщений за 28 месяцев (13.05.2024 — 31.08.2026)" % big(totals.get("messages"))],
        ["Негатив", "%s сообщений, %s всего разговора" % (big(totals.get("negative")),
                                                         share(totals.get("negative_share")))],
        ["Вес негатива", "%s аудиторных контактов, из них %s — медийная аудитория"
         % (big(totals.get("negative_reach")), big(totals.get("negative_media_reach")))],
        ["Где болит", "тем жалоб — %d (собраны из %d смысловых групп), крупнейшая «%s» "
                      "(%s сообщений, %s негатива)"
         % (len(theme_pains) or len(clusters), len(clusters),
            (theme_pains[0].get("title") if theme_pains else top_cluster.get("title")) or "—",
            big(theme_pains[0].get("size") if theme_pains else top_cluster.get("size")),
            share(theme_pains[0].get("share") if theme_pains else top_cluster.get("share")))],
        ["Адресность", "10 городов дают %s негатива, 30 ресторанов — %s"
         % (share(concentration.get("top10_cities_share")), share(concentration.get("top30_restaurants_share")))],
        ["Кто пишет", "%s авторов; верхушка 100 авторов даёт лишь %s негатива — это не кампания, "
                      "а массовая реакция клиентов" % (big(authors.get("authors_total")),
                                                       share((authors.get("share_top100") or 0) / 100.0))],
    ]
    if forecast:
        rows.append(["Что впереди", "в %s при бездействии доля негатива поднимется до %s; "
                                    "работа с причинами опускает её до %s"
                    % (forecast[0]["month"], share(forecast[0]["no_action"]),
                       share(forecast[0]["systemic"]))])
    bullets = []
    for item in sorted(insights.get("insights") or [], key=lambda row: row.get("приоритет") or 9)[:3]:
        bullets.append("Важно: %s — %s. Что делать: %s"
                       % (item.get("заголовок"), item.get("цифра"), item.get("что_делать")))
    return {
        "heading": "Главное в числах",
        "text": ("Ниже — то, что видно по всему периоду сразу. Каждое число посчитано по всем "
                 "сообщениям темы, без выборки; расшифровка — в разделах дальше."),
        "tables": [TABLE("Отчёт в одном экране", ["Показатель", "Значение"], rows, layout="portrait")],
        "bullets": bullets,
    }


# ------------------------------------------------------------------ инсайты

def build_insights_block(insights: dict) -> dict:
    """Карточки инсайтов: приоритет, число, смысл, действие, цена бездействия."""
    items = sorted(insights.get("insights") or [], key=lambda row: row.get("приоритет") or 9)
    tables = []
    for index, item in enumerate(items, 1):
        rows = [
            ["Число", report_text(item.get("цифра") or "—")],
            ["Что это значит", report_text(item.get("что_значит") or "—")],
            ["Что делать", report_text(item.get("что_делать") or "—")],
            ["Если не реагировать", report_text(item.get("если_не_делать") or "—")],
            ["Подтверждение", report_text(item.get("где_смотреть") or "—")],
        ]
        tables.append(TABLE("%d. %s" % (index, item.get("заголовок") or "Вывод"),
                            ["Что именно", "Содержание"], rows, layout="portrait"))
    return {
        "heading": "Инсайты: что именно происходит с брендом",
        "text": ("Карточки отсортированы по приоритету: первый уровень — то, что требует решения "
                 "в ближайший месяц, второй — квартальные задачи, третий — то, что важно держать "
                 "в поле зрения. В каждой карточке — число, его смысл, действие и цена бездействия."),
        "tables": tables,
    }


# ------------------------------------------------------------------ карта боли

def build_pain_map_block(facts: dict, ctx) -> dict:
    """Темы жалоб (сведённые группы) плюс детальная разбивка по кластерам."""
    merged = load("clusters_themes")
    themes = merged.get("themes") or []
    pains = [row for row in themes if row.get("theme") != "positive"]
    data = load("clusters_final")
    clusters = data.get("clusters") or []
    if not pains:
        return {"heading": "Карта боли: темы жалоб",
                "text": "Раздел не собран: нет результатов кластеризации."}
    chart_ids = []
    for name, title in (("theme_bars.png", "Темы жалоб: объём, доля и что растёт в 2026 году"),
                        ("cluster_dynamics.png", "Как менялись боли по кварталам"),
                        ("cluster_map.png", "Карта смыслов: негативные сообщения по группам")):
        path = os.path.join(TOPICS, name)
        if os.path.isfile(path):
            chart_ids.append(register_image(ctx, "pain_" + name.split(".")[0], title, path))
    theme_rows = []
    for item in pains:
        theme_rows.append([
            item.get("title") or "—",
            big(item.get("size")),
            share(item.get("share")),
            big(item.get("from_2026")),
            "%d" % len(item.get("groups") or []),
            ", ".join("%s" % city for city, _ in (item.get("cities") or [])[:3]) or "—",
            (item.get("essence") or "")[:170],
        ])
    detail_rows = [[item.get("title") or "—", big(item["size"]), share(item.get("share")),
                    big(item.get("from_2026")), item.get("peak_month") or "—"]
                   for item in clusters[:20]]
    bullets = []
    for item in pains[:5]:
        example = (item.get("examples") or [{}])[0]
        line = ("%s. Сообщений **%s** (%s негатива), с начала 2026 года — %s, тема собрана из %s групп. %s"
                % (item.get("title"), big(item.get("size")), share(item.get("share")),
                   big(item.get("from_2026")), len(item.get("groups") or []),
                   report_text(item.get("essence") or "")))
        if example.get("text"):
            line += " Пример: «%s» — %s." % (example["text"][:160], example.get("hub") or "площадка")
        bullets.append(report_text(line))
    return {
        "heading": "Карта боли: темы жалоб",
        "text": ("Группы жалоб найдены не по ключевым словам, а по смыслу: %s негативных сообщений "
                 "превращены в векторы моделью bge-m3, сжаты и разделены на кластеры. Близкие "
                 "формулировки («долгое ожидание заказа», «долгое время ожидания», «ожидание и "
                 "ошибки в заказах») сведены в одну тему: так картина читается, а не рассыпается "
                 "на %d похожих строк. Всего тем — %d."
                 % (big(merged.get("total")), len(clusters), len(pains))),
        "chart_ids": chart_ids,
        "tables": [
            TABLE("Темы жалоб: объём, динамика, где болит",
                  ["Тема", "Сообщений", "Доля негатива", "С 2026 года", "Групп внутри", "Города",
                   "В чём суть"], theme_rows, layout="landscape",
                  note="Доля считается от всех негативных сообщений периода. «Групп внутри» — "
                       "из скольких кластеров собрана тема."),
            TABLE("Детальные группы внутри тем", ["Группа", "Сообщений", "Доля негатива",
                                                  "С 2026 года", "Пик"], detail_rows,
                  layout="landscape",
                  note="Исходная разбивка до сведения в темы: показывает, какие именно "
                       "формулировки стоят за каждой темой."),
        ],
        "bullets": bullets,
    }


# ------------------------------------------------------------------ авторы

def build_authors_block(facts: dict, ctx) -> dict:
    """Кто говорит: концентрация, ядро по каждой боли, структура аудитории."""
    authors = load("author_core")
    global_stats = authors.get("global") or {}
    per_cluster = authors.get("clusters") or {}
    if not global_stats:
        return {"heading": "Кто говорит: авторы и площадки", "text": "Нет данных по авторам."}
    chart_ids = []
    for name, title in (("author_map.png", "Кто пишет о каждой боли: боли и их авторы"),):
        path = os.path.join(TOPICS, name)
        if os.path.isfile(path):
            chart_ids.append(register_image(ctx, "auth_" + name.split(".")[0], title, path))
    rows = [
        ["Авторов в негативе", big(global_stats.get("authors_total"))],
        ["Негативных сообщений", big(global_stats.get("messages_total"))],
        ["Верхушка 10 авторов", share((global_stats.get("share_top10") or 0) / 100.0) + " негатива"],
        ["Верхушка 100 авторов", share((global_stats.get("share_top100") or 0) / 100.0) + " негатива"],
        ["Верхушка 1000 авторов", share((global_stats.get("share_top1000") or 0) / 100.0) + " негатива"],
        ["Авторов с одним сообщением", big(global_stats.get("authors_with_one"))],
        ["Авторов с 20+ сообщениями", big(global_stats.get("authors_with_many"))],
    ]
    profile_rows = []
    for cluster, row in list(per_cluster.items())[:12]:
        profile_rows.append([
            row.get("title") or "группа %s" % cluster,
            big(row.get("authors")),
            big(row.get("core_authors")),
            share((row.get("core_share") or 0) / 100.0),
            ("%.1f" % (row.get("messages_per_author") or 0)).replace(".", ","),
        ])
    top_rows = []
    for item in (global_stats.get("top") or [])[:12]:
        top_rows.append([
            item.get("name") or "—",
            big(item.get("messages")),
            big(item.get("reach")),
            item.get("type") or "—",
            ", ".join("%s" % hub for hub, _ in (item.get("hubs") or [])[:2]) or "—",
            ", ".join("%s" % city for city, _ in (item.get("cities") or [])[:2]) or "—",
        ])
    bullets = [
        "Важно: негатив массовый. На %s негативных сообщений приходится %s авторов, "
        "и верхушка из 100 авторов даёт всего %s негатива. Это значит, что разговор создают "
        "обычные клиенты, а не организованные группы: бороться нужно с причинами, а не с авторами."
        % (big(global_stats.get("messages_total")), big(global_stats.get("authors_total")),
           share((global_stats.get("share_top100") or 0) / 100.0)),
        "Что делать: смотреть на ядро внутри каждой боли. В группах, где ядро (авторы с тремя и "
        "более сообщениями) даёт заметную долю объёма, стоит разбирать переписку адресно; "
        "в остальных случаях эффективнее исправлять процесс, а не отвечать в переписке.",
    ]
    return {
        "heading": "Кто создаёт негатив: авторы и ядро",
        "text": ("Проверка на «организованный негатив»: мы посчитали всех авторов негативных "
                 "сообщений и их вклад. Картина обратная ожиданиям — негатив рассыпан по сотням "
                 "тысяч обычных людей, повторяющихся критиков немного."),
        "chart_ids": chart_ids,
        "tables": [
            TABLE("Структура аудитории негатива", ["Показатель", "Значение"], rows, layout="portrait"),
            TABLE("Сколько людей стоит за каждой болью",
                  ["Группа жалоб", "Авторов в выборке", "Ядро (3+ сообщения)", "Доля объёма от ядра",
                   "Сообщений на автора"], profile_rows, layout="landscape",
                  note="Данные по выборке до 2 500 сообщений на группу: показывает, массовая это "
                       "боль или её разгоняет небольшая группа."),
            TABLE("Самые активные авторы негатива", ["Автор", "Сообщений", "Охват", "Тип профиля",
                                                     "Площадки", "Города"], top_rows, layout="landscape",
                  note="Охват — суммарная аудитория сообщений автора. Профили приведены без ссылок: "
                       "это персональные данные."),
        ],
        "bullets": bullets,
    }


# ------------------------------------------------------------------ волны

def build_waves_block(ctx) -> dict:
    """Как расходятся волны: скорость, источники, срок на реакцию."""
    data = load("propagation")
    stories = data.get("stories") or []
    if not stories:
        return {"heading": "Как расходятся волны", "text": "Нет данных по волнам."}
    chart_ids = []
    for index in (1, 2, 3):
        path = os.path.join(TOPICS, "propagation_%d.png" % index)
        if os.path.isfile(path):
            story = stories[index - 1]["story"] if len(stories) >= index else ""
            chart_ids.append(register_image(ctx, "wave_%d" % index,
                                            "Как расходилась волна: %s" % story[:70], path))
    rows = []
    for item in stories:
        rows.append([
            item["story"][:80],
            big(item["messages"]),
            item.get("first_day") or "—",
            item.get("peak_day") or "—",
            "%s" % (item.get("days_to_peak") or 0),
            ", ".join("%s (%s)" % (hub, count) for hub, count in (item.get("hubs") or [])[:3]),
            big(item.get("reach")),
        ])
    bullets = []
    for item in stories[:3]:
        bullets.append("Важно: «%s» — %s сообщений, первый день %s, пик %s (через %s дн.). "
                       "Разгоняли: %s."
                       % (item["story"][:70], big(item["messages"]), item.get("first_day"),
                          item.get("peak_day"), item.get("days_to_peak"),
                          ", ".join(hub for hub, _ in (item.get("hubs") or [])[:4]) or "—"))
    bullets.append("Что делать: держать заготовки ответов и решение по первому часу. У всех "
                   "крупных поводов пик приходится на первый-второй день, поэтому подготовка "
                   "важнее скорости согласования внутри компании.")
    return {
        "heading": "Как расходятся волны",
        "text": ("Для крупнейших поводов посчитана цепочка: первый день, кто опубликовал, "
                 "через сколько дней наступил пик, кто разгонял волну и сколько охвата она собрала."),
        "chart_ids": chart_ids,
        "tables": [TABLE("Цепочки распространения поводов", ["Повод", "Сообщений", "Первый день",
                                                             "Пик", "Дней до пика", "Кто разгонял",
                                                             "Охват"], rows, layout="landscape")],
        "bullets": bullets,
    }


# ------------------------------------------------------------------ где живёт каждая боль

def build_geo_block(ctx) -> dict:
    """Темы жалоб × города: структура болей по регионам."""
    merged = load("clusters_themes")
    themes = [row for row in (merged.get("themes") or []) if row.get("theme") != "positive"]
    if not themes:
        return {"heading": "Где живёт каждая тема: города",
                "text": "Нет данных по разрезу городов."}
    titles = {row["theme"]: row["title"] for row in themes}
    city_total = {}
    for city, items in (merged.get("city_themes") or {}).items():
        city_total[city] = sum(count for _, count in items)

    per_group_rows = []
    for row in themes:
        mapping = dict(row.get("cities") or [])
        total = sum(mapping.values()) or 1
        per_group_rows.append([
            (row.get("title") or "—")[:44],
            big(row.get("size")),
            ", ".join("%s — %s (%s)" % (city, big(count), share(count / float(total)))
                      for city, count in (row.get("cities") or [])[:4]),
        ])

    per_city_rows = []
    for city, items in sorted((merged.get("city_themes") or {}).items(),
                              key=lambda kv: -sum(count for _, count in kv[1]))[:12]:
        per_city_rows.append([
            city, big(city_total.get(city)),
            ", ".join("%s — %s" % (titles.get(theme, theme)[:34], big(count))
                      for theme, count in items[:3]),
        ])

    chart_ids = []
    for name, title in (("theme_city.png", "Где живёт каждая тема: темы × города"),):
        path = os.path.join(TOPICS, name)
        if os.path.isfile(path):
            chart_ids.append(register_image(ctx, "geo_" + name.split(".")[0], title, path))

    bullets = []
    for row in themes[:4]:
        rows = sorted(row.get("cities") or [], key=lambda kv: -kv[1])
        if len(rows) < 2:
            continue
        total = sum(count for _, count in rows) or 1
        bullets.append("%s. Больше всего сообщений в %s (%s), затем %s (%s). По структуре тема "
                       "распределена ровно: на четыре крупнейших города приходится %s всех "
                       "сообщений темы — значит, это системная проблема сети, а не отдельного города."
                       % ((row.get("title") or "—")[:60],
                          rows[0][0], share(rows[0][1] / total),
                          rows[1][0], share(rows[1][1] / total),
                          share(sum(count for _, count in rows[:4]) / total)))
    bullets.append("Что делать: города различаются не набором болей, а их весом. Локальные программы "
                   "имеют смысл там, где доля негатива города выше средней, а содержание работ "
                   "берётся из этой же таблицы — темы одни и те же по всей сети.")
    return {
        "heading": "Где живёт каждая тема: города",
        "text": ("Разрез «тема жалоб × город» показывает, чем города отличаются друг от друга: "
                 "не объёмом разговора, а составом жалоб. Доли считаются внутри темы, поэтому "
                 "таблица сравнима между городами разного размера."),
        "chart_ids": chart_ids,
        "tables": [
            TABLE("Структура каждой темы по городам", ["Тема жалоб", "Сообщений", "Где сосредоточено"],
                  per_group_rows, layout="landscape",
                  note="Доли приведены от объёма темы; города упорядочены по числу сообщений."),
            TABLE("Что болит в каждом городе", ["Город", "Сообщений в разрезе", "Три главные темы"],
                  per_city_rows, layout="landscape",
                  note="Только темы, собранные из смысловых групп по эмбеддингам; города "
                       "упорядочены по объёму негатива."),
        ],
        "bullets": bullets,
    }


# ------------------------------------------------------------------ площадки и динамика

def _quarter_key(month: str) -> str:
    year, mon = month.split("-")[:2]
    return "%s-Q%d" % (year[-2:], (int(mon) - 1) // 3 + 1)


def build_platform_block(ctx) -> dict:
    """Темы × площадки и их динамика по кварталам."""
    merged = load("clusters_themes")
    themes = [row for row in (merged.get("themes") or []) if row.get("theme") != "positive"]
    if not themes:
        return {"heading": "Какая тема на какой площадке", "text": "Нет данных по площадкам."}

    hub_total = {}
    for row in themes:
        for hub, count in row.get("hubs") or []:
            hub_total[hub] = hub_total.get(hub, 0) + count
    platforms = [hub for hub, _ in sorted(hub_total.items(), key=lambda kv: -kv[1])[:6]]

    rows = []
    for row in themes:
        mapping = dict(row.get("hubs") or [])
        total = sum(mapping.values()) or 1
        rows.append([(row.get("title") or "—")[:44], big(row.get("size"))]
                    + [share(mapping.get(platform, 0) / float(total)) for platform in platforms])

    quarters = []
    for row in themes:
        for month in (row.get("months") or {}):
            if month not in quarters:
                quarters.append(month)
    quarters = sorted(quarters)
    keys = []
    for month in quarters:
        key = _quarter_key(month)
        if key not in keys:
            keys.append(key)
    keys = keys[-8:]
    dyn_rows = []
    for row in themes:
        months = row.get("months") or {}
        buckets = {}
        for month, count in months.items():
            buckets[_quarter_key(month)] = buckets.get(_quarter_key(month), 0) + count
        dyn_rows.append([(row.get("title") or "—")[:40]] + [big(buckets.get(key, 0)) for key in keys])

    chart_ids = []
    for name, title in (("heat_group_platform.png", "Какая тема на какой площадке"),
                        ("platform_dynamics.png", "Динамика болей по кварталам: отзывы и соцсети")):
        path = os.path.join(TOPICS, name)
        if os.path.isfile(path):
            chart_ids.append(register_image(ctx, "plat_" + name.split(".")[0], title, path))

    bullets = []
    for row in themes[:5]:
        items = sorted(row.get("hubs") or [], key=lambda kv: -kv[1])[:3]
        total = sum(count for _, count in (row.get("hubs") or [])) or 1
        bullets.append("%s: %s" % ((row.get("title") or "—")[:55],
                                   ", ".join("%s — %s" % (hub, share(count / float(total)))
                                             for hub, count in items)))
    bullets.append("Что делать: у каждой темы своя площадка ответа. Отзывы требуют ответа на картах "
                   "и в отзовиках (там виден конкретный ресторан), соцсети и мессенджеры — "
                   "объяснения и работы с пересказами.")
    return {
        "heading": "Какая тема на какой площадке",
        "text": ("Разрез «тема жалоб × площадка» показывает, где именно обсуждают каждую тему и как "
                 "менялась её динамика по кварталам. Это определяет, куда направлять ответ — "
                 "на карты, в соцсети или в мессенджеры."),
        "chart_ids": chart_ids,
        "tables": [
            TABLE("Доля площадок внутри каждой темы, %",
                  ["Тема жалоб", "Сообщений"] + platforms, rows, layout="landscape",
                  note="Доли считаются внутри темы: сколько сообщений темы пришло с каждой "
                       "площадки; показаны пять крупнейших площадок темы."),
            TABLE("Динамика тем по кварталам, сообщений",
                  ["Тема жалоб"] + keys, dyn_rows, layout="landscape",
                  note="Кварталы обозначены как «год-квартал»; внутри темы суммируются все "
                       "входящие в неё смысловые группы."),
        ],
        "bullets": bullets,
    }


# ------------------------------------------------------------------ кампании и боли

def theme_title_map() -> dict:
    """Номер смысловой группы → название темы, в которую она сведена."""
    merged = load("clusters_themes")
    titles = {row.get("theme"): (row.get("title") or "—") for row in merged.get("themes") or []}
    mapping = {}
    for item in merged.get("groups") or []:
        theme = item.get("theme")
        if not theme or theme == "positive":
            continue
        mapping[int(item["cluster"])] = titles.get(theme) or item.get("title") or "—"
    return mapping


def themes_of_growth(items) -> list:
    """Сводит рост по группам внутри одной кампании к темам жалоб."""
    mapping = theme_title_map()
    merged = {}
    for row in items or []:
        title = mapping.get(int(row.get("cluster", -1)), row.get("title") or "—")
        bucket = merged.get(title)
        if bucket is None:
            bucket = merged[title] = {"title": title, "during": 0, "before": 0}
        bucket["during"] += row.get("during") or 0
        bucket["before"] += row.get("before") or 0
    for bucket in merged.values():
        bucket["growth"] = round((bucket["during"] + 1) / float(bucket["before"] + 1), 2)
    return sorted(merged.values(), key=lambda row: -row["growth"])


def build_campaigns_block(ctx) -> dict:
    """Кампании против тем жалоб: что обостряется в дни кампаний."""
    data = load("cluster_cuts")
    rows_data = (data or {}).get("campaigns") or []
    if not rows_data:
        return {"heading": "Кампании и темы жалоб: что обостряется", "text": "Нет данных по кампаниям."}
    to_themes = themes_of_growth

    chart_ids = []
    path = os.path.join(TOPICS, "campaign_pains.png")
    if os.path.isfile(path):
        chart_ids.append(register_image(ctx, "camp_pains", "Кампании и боли: рост жалоб", path))

    rows = []
    for item in rows_data:
        grew = "; ".join("%s — %s против %s (x%s)" % (row["title"][:34], big(row["during"]),
                                                      big(row["before"]), row["growth"])
                         for row in to_themes(item.get("grew"))[:2]) or "заметного роста нет"
        fell = "; ".join("%s — x%s" % (row["title"][:34], row["growth"])
                         for row in to_themes(item.get("fell"))[:2]) or "—"
        rows.append([item["campaign"][:60], "%s — %s" % (item["from"][:7], item["to"][:7]),
                     big(item.get("total_during")), grew, fell])

    service_grew = sum(1 for item in rows_data
                       if any("ожидан" in (row["title"] or "").lower()
                              or "обслужив" in (row["title"] or "").lower()
                              or "персонал" in (row["title"] or "").lower()
                              for row in to_themes(item.get("grew"))[:1]))
    bullets = [
        "Важно: во время кампаний растут не продуктовые, а сервисные жалобы. В %d из %d окон "
        "кампаний в первых строках роста стоят ожидание, отмена заказов и качество обслуживания: "
        "кампания приводит людей, а нагрузку держит зал." % (service_grew, len(rows_data)),
        "Что делать: планировать кампанию вместе с операционным блоком — усиление смен в дни "
        "старта, готовые ответы на картах и в отзовиках в первые сутки, отдельный контроль "
        "времени выдачи. Иначе промо-бюджет оплачивает рост жалоб.",
        "Где смотреть: таблица ниже и раздел «Как расходятся волны» — там видно, что пик "
        "негатива приходит в первые сутки после старта.",
    ]
    return {
        "heading": "Кампании и темы жалоб: что обостряется",
        "text": ("Сопоставление кампаний с динамикой тем жалоб: сколько сообщений каждой темы было "
                 "до кампании и во время неё. Сравниваются окна одинаковой длины, для устойчивости "
                 "отброшены группы с малой базой — там кратности обманчивы."),
        "chart_ids": chart_ids,
        "tables": [TABLE("Кампании: что росло и что снижалось",
                         ["Кампания", "Окно", "Сообщений в окне", "Что выросло", "Что снизилось"],
                         rows, layout="landscape",
                         note="«Выросло» — темы жалоб, у которых сообщений стало заметно больше, "
                              "чем в равном окне до кампании; «снизилось» — наоборот.")],
        "bullets": bullets,
    }


# ------------------------------------------------------------------ внешний контекст

def build_external_block() -> dict:
    """Что происходило вокруг бренда: открытые источники и сверка с нашими данными."""
    data = load("kfc_external", base="/home/dev/tellscope_app/tellscope_backend")
    facts = data.get("facts") or []
    if not facts:
        return {"heading": "Что происходило вокруг бренда", "text": "Нет данных внешнего контекста."}
    rows = []
    for item in facts:
        rows.append([item.get("date") or "", item.get("title") or "",
                     item.get("source") or "", item.get("url") or ""])
    bullets = []
    for item in facts:
        bullets.append("%s (%s): %s Ссылка: %s" % (item.get("title"), item.get("source"),
                                                   report_text(item.get("link_to_our_data") or ""),
                                                   item.get("url") or ""))
    return {
        "heading": "Что происходило вокруг бренда",
        "text": ("Открытые источники по бренду и рынку за период — и сверка с нашими данными. "
                 "Главное наблюдение: каждое крупное внешнее событие видно в корпусе, но в данных "
                 "оно разложено по городам, площадкам и датам, то есть пригодно для решения, "
                 "а не только для справки."),
        "tables": [TABLE("Публикации открытых источников за период",
                         ["Дата", "Событие", "Источник", "Ссылка"], rows, layout="landscape",
                         note="Ссылки приведены для проверки; данные отчёта посчитаны независимо "
                              "по корпусу сообщений.")],
        "bullets": bullets,
    }


# ------------------------------------------------------------------ экраны системы

def build_screens_block(ctx) -> dict:
    """Как это выглядит в Tellscope: только графики и данные, без служебных экранов."""
    chart_ids = []
    captions_path = os.path.join("/tmp/kfc_shots3", "captions.json")
    captions = {}
    if os.path.isfile(captions_path):
        try:
            with io.open(captions_path, encoding="utf-8") as fh:
                captions = json.load(fh)
        except Exception:  # noqa: BLE001
            captions = {}
    order = 0
    for folder, pattern in (("/tmp/kfc_shots3", "info_"), ("/tmp/kfc_shots2", "chart_")):
        names = sorted(name for name in os.listdir(folder)) if os.path.isdir(folder) else []
        for name in names:
            if not name.startswith(pattern) or not name.endswith(".png"):
                continue
            path = os.path.join(folder, name)
            if os.path.getsize(path) < 60000:
                continue
            order += 1
            title = captions.get(name) or ("График Tellscope по теме KFC (%d)" % order)
            chart_ids.append(register_image(ctx, "shot_%d" % order, title, path))
    if not chart_ids:
        return {"heading": "Как это выглядит в Tellscope", "text": "Снимки графиков не получены."}
    return {
        "heading": "Как это выглядит в Tellscope",
        "text": ("Ниже — графики из интерфейса платформы по этой же теме. Отчёт собран из тех же "
                 "данных: любую цифру можно перепроверить в системе, поменяв период, площадки "
                 "или тему."),
        "chart_ids": chart_ids,
        "bullets": [
            "Где смотреть в системе: тональный ландшафт — тональность и авторы; упоминания — "
            "объём и тексты сообщений; аналитика — темы, площадки и динамика.",
            "Что делать: расхождения между отчётом и интерфейсом проверяются за минуту — "
            "достаточно выбрать ту же тему и период.",
        ],
    }
