#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Управленческие инсайты нового отчёта: 16 карточек, два прохода внешней модели.

Модель получает только посчитанные факты (база, очаги, смысловые кластеры, авторы,
волны, внешний контекст) и обязана опираться на их числа. Формат каждой карточки:
заголовок — цифра — что значит — что делать — если не реагировать — приоритет.
В текстах допускается **выделение** ключевых чисел: при сборке оно станет жирным.

Результат: /tmp/kfc_topics/insights2.json
"""
from __future__ import annotations

import datetime
import io
import json
import os
import sys

OUT = "/tmp/kfc_topics"
HOT = "/tmp/kfc_hotspots"
BACKEND = "/home/dev/tellscope_app/tellscope_backend"
sys.path.insert(0, BACKEND)


def log(message: str) -> None:
    print("[%s] %s" % (datetime.datetime.now().strftime("%H:%M:%S"), message), flush=True)


def load(path: str, default=None):
    if not os.path.isfile(path):
        return default if default is not None else {}
    with io.open(path, encoding="utf-8") as fh:
        return json.load(fh)


def digest() -> str:
    facts = load(os.path.join(HOT, "facts.json"))
    clusters = load(os.path.join(OUT, "clusters_final.json"))
    authors = load(os.path.join(OUT, "author_core.json"))
    propagation = load(os.path.join(OUT, "propagation.json"))
    external = load(os.path.join(BACKEND, "kfc_external.json"))
    lines = []

    totals = facts.get("totals") or {}
    lines.append("ПЕРИОД И ОБЪЁМ: 13.05.2024–31.08.2026, 28 месяцев. Сообщений %s, негативных %s (%s%%). "
                 "Аудиторные контакты негатива: %s, медийная аудитория %s."
                 % (totals.get("messages"), totals.get("negative"),
                    round((totals.get("negative_share") or 0) * 100, 1),
                    totals.get("negative_reach"), totals.get("negative_media_reach")))
    years = facts.get("years") or {}
    lines.append("ГОДЫ: " + "; ".join("%s — %s сообщений, доля негатива %s%%"
                                      % (year, row["total"], round(row["negative_share"] * 100, 1))
                                      for year, row in sorted(years.items())))
    hubs = facts.get("hubtype") or {}
    lines.append("ПЛОЩАДКИ (негатив): " + "; ".join("%s — %s (%s%% внутри площадки)"
                                                    % (name, row["negative"], round(row["share"] * 100, 1))
                                                    for name, row in sorted(hubs.items(),
                                                                            key=lambda kv: -kv[1]["negative"])[:5]))
    ratings = {k: v for k, v in (facts.get("ratings") or {}).items() if k in ("1", "2", "3", "4", "5")}
    lines.append("ОЦЕНКИ В ОТЗЫВАХ: " + "; ".join("%s — %s" % (key, ratings[key]) for key in sorted(ratings)))
    conc = facts.get("concentration") or {}
    lines.append("КОНЦЕНТРАЦИЯ: 10 городов дают %s%% негатива, 30 заведений — %s%%, 5 тем — %s%%; "
                 "городов с ростом %s, заведений с ростом %s."
                 % (round((conc.get("top10_cities_share") or 0) * 100, 1),
                    round((conc.get("top30_restaurants_share") or 0) * 100, 1),
                    round((conc.get("top5_themes_share") or 0) * 100, 1),
                    conc.get("cities_with_growth"), conc.get("restaurants_with_growth")))
    scen = facts.get("scenarios") or {}
    forecast = scen.get("forecast") or []
    if forecast:
        first = forecast[0]
        lines.append("СЦЕНАРИИ на %s: без действий %s%%, адресная работа %s%%, системная работа %s%% "
                     "(средняя доля негатива %s%%)."
                     % (first["month"], round(first["no_action"] * 100, 1), round(first["targeted"] * 100, 1),
                        round(first["systemic"] * 100, 1), round((scen.get("mean_share") or 0) * 100, 1)))

    lines.append("")
    lines.append("ГОРОДА-ОЧАГИ (город | негатив | доля негатива | перевес | рост | о чём пишут):")
    for row in (facts.get("cities") or [])[:10]:
        lines.append("- %s | %s | %s%% | x%s | x%s | %s"
                     % (row["city"], row["negative"], round(row["share"] * 100, 1),
                        row.get("excess_norm"), row["growth"],
                        "; ".join((row.get("what_people_say") or [])[:2]) or "—"))
    lines.append("")
    lines.append("ЗАВЕДЕНИЯ (город | негативных из всего | доля | рейтинг | рост):")
    for row in (facts.get("restaurants") or [])[:10]:
        lines.append("- %s | %s из %s | %s%% | %s | x%s"
                     % (row["city"], row["negative"], row["count"], round(row["share"] * 100, 1),
                        row.get("rating"), row["growth"]))

    clusters = clusters.get("clusters") or []
    lines.append("")
    lines.append("СМЫСЛОВЫЕ ГРУППЫ ЖАЛОБ (эмбеддинги + кластеризация; объём | доля негатива | пик | "
                 "с 2026 года | суть):")
    for row in clusters[:16]:
        lines.append("- %s | %s | %s%% | пик %s | с 2026: %s | %s"
                     % (row.get("title") or "группа %s" % row["cluster"], row["size"],
                        round((row.get("share") or 0) * 100, 1), row.get("peak_month"),
                        row.get("from_2026"), (row.get("essence") or "")[:150]))

    global_authors = (authors.get("global") or {})
    lines.append("")
    lines.append("АВТОРЫ: всего %s авторов на %s негативных сообщений; верхушка 10 авторов — %s%% негатива, "
                 "100 авторов — %s%%, 1000 авторов — %s%%; авторов с одним сообщением — %s."
                 % (global_authors.get("authors_total"), global_authors.get("messages_total"),
                    global_authors.get("share_top10"), global_authors.get("share_top100"),
                    global_authors.get("share_top1000"), global_authors.get("authors_with_one")))
    for cluster, row in list((authors.get("clusters") or {}).items())[:6]:
        lines.append("- боль «%s»: в выборке %s авторов, ядро (3+ сообщения) — %s человек, это %s%% объёма"
                     % (row.get("title"), row.get("authors"), row.get("core_authors"), row.get("core_share")))
    lines.append("ТОП АВТОРОВ ПО НЕГАТИВУ: " + "; ".join(
        "%s (%s сообщений, %s)" % (item["name"], item["messages"],
                                   ", ".join(hub for hub, _ in item.get("hubs") or [])[:40])
        for item in (global_authors.get("top") or [])[:6]))

    lines.append("")
    lines.append("ВОЛНЫ ИНЦИДЕНТОВ (повод | сообщений | первый день | пик | дней до пика | кто разгонял):")
    for row in (propagation.get("stories") or [])[:6]:
        lines.append("- %s | %s | %s | %s | %s | %s"
                     % (row["story"][:60], row["messages"], row.get("first_day"), row.get("peak_day"),
                        row.get("days_to_peak"),
                        ", ".join(hub for hub, _ in (row.get("hubs") or [])[:4])))

    lines.append("")
    lines.append("ВНЕШНИЙ КОНТЕКСТ (публикации открытых источников):")
    for item in (external.get("facts") or [])[:9]:
        lines.append("- %s (%s): %s" % (item["title"], item["source"], item.get("link_to_our_data", "")[:150]))
    text = "\n".join(lines)
    log("выжимка фактов: %d знаков" % len(text))
    return text[:26000]


def main() -> None:
    import kfc_ai

    facts = digest()
    system = ("Ты ведущий аналитик медиаполя сети фастфуда Rostic's (бывший KFC) и готовишь отчёт "
              "для руководства компании. Пишешь по-русски, деловым языком, без канцелярита и без "
              "общих слов. Опираешься только на переданные числа: свои цифры, даты и факты "
              "придумывать запрещено. Ключевые числа в тексте выделяй двойными звёздочками, "
              "например: рост **x1,63** за квартал.")
    focus_a = ("Первые восемь инсайтов: где именно болит и почему. Опирайся на смысловые группы "
               "жалоб, города, заведения, оценки и волны инцидентов. Покажи то, что неочевидно: "
               "что стоит за цифрами, какие группы растут, где причины повторяются.")
    focus_b = ("Вторые восемь инсайтов: кто говорит, что вокруг бренда и что делать. Опирайся на "
               "авторов и площадки, на структуру негатива, на внешний контекст и на плюсы бренда. "
               "Обязательно включи: почему негатив массовый и что это значит для стратегии; что "
               "происходит с кампаниями и позитивом; что даст работа с адресными очагами; какой "
               "скрытый резерв есть у бренда.")
    schema = ("Верни строго JSON:\n"
              '{"инсайты": [{"заголовок": "до 8 слов", "цифра": "главное число с коротким пояснением", '
              '"что_значит": "2-3 предложения, с выделением чисел **жирным**", '
              '"что_делать": "1-2 предложения, конкретное действие", '
              '"если_не_делать": "1 предложение", "приоритет": 1, '
              '"где_смотреть": "какой раздел отчёта подтверждает"}]}')

    result = {"built_at": "", "insights": []}
    for index, focus in enumerate((focus_a, focus_b), 1):
        prompt = ("ФАКТЫ О БРЕНДЕ:\n%s\n\n%s\n\n%s" % (facts, focus, schema))
        data = kfc_ai.ask_json(prompt, system=system, model="deepseek-chat", max_tokens=6000, attempts=3)
        items = (data or {}).get("инсайты") or (data or {}).get("insights") or []
        log("проход %d: инсайтов %d" % (index, len(items)))
        for item in items:
            if not isinstance(item, dict):
                continue
            result["insights"].append({
                "заголовок": str(item.get("заголовок") or "").strip(),
                "цифра": str(item.get("цифра") or "").strip(),
                "что_значит": str(item.get("что_значит") or item.get("что_значает") or "").strip(),
                "что_делать": str(item.get("что_делать") or "").strip(),
                "если_не_делать": str(item.get("если_не_делать") or "").strip(),
                "приоритет": int(item.get("приоритет") or 2),
                "где_смотреть": str(item.get("где_смотреть") or "").strip(),
            })
        with io.open(os.path.join(OUT, "insights2.json"), "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False)
    result["built_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    with io.open(os.path.join(OUT, "insights2.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False)
    for item in result["insights"]:
        log("%d. %s — %s" % (item["приоритет"], item["заголовок"][:60], item["цифра"][:70]))
    log("готово: инсайтов %d" % len(result["insights"]))


if __name__ == "__main__":
    main()
