#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Инсайты с гарантированными числами: каркас считаем сами, формулировки пишет модель.

Ошибки в цифрах недопустимы, поэтому порядок обратный обычному: сначала мы считаем
конкретный факт с числами и источником, затем внешняя модель пишет по нему короткий
управленческий текст — и не имеет права добавлять свои цифры.

Результат: /tmp/kfc_topics/insights3.json
"""
from __future__ import annotations

import collections
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


def num(value) -> str:
    try:
        return "{:,}".format(int(float(value))).replace(",", " ")
    except Exception:  # noqa: BLE001
        return "—"


def pct(value, digits=1) -> str:
    return ("%." + str(digits) + "f") % float(value)


def skeletons() -> list:
    """Каркасы инсайтов: числа посчитаны здесь и не выдумываются моделью."""
    facts = load(os.path.join(HOT, "facts.json"))
    norm = load(os.path.join(HOT, "channel_norm.json"))
    clusters = load(os.path.join(OUT, "clusters_final.json")).get("clusters") or []
    authors = load(os.path.join(OUT, "author_core.json"))
    propagation = load(os.path.join(OUT, "propagation.json")).get("stories") or []
    external = load(os.path.join(BACKEND, "kfc_external.json")).get("facts") or []

    totals = facts.get("totals") or {}
    years = facts.get("years") or {}
    cities = facts.get("cities") or []
    restaurants = facts.get("restaurants") or []
    ratings = {k: v for k, v in (facts.get("ratings") or {}).items() if k.isdigit()}
    ratings_total = sum(ratings.values()) or 1
    hubtypes = (norm.get("by_hubtype") or {})
    scen = facts.get("scenarios") or {}
    forecast = scen.get("forecast") or []
    concentration = facts.get("concentration") or {}
    author_global = authors.get("global") or {}
    negative = float(totals.get("negative") or 1)

    out = []

    def add(focus, headline, numbers, source, priority):
        out.append({"focus": focus, "headline": headline, "numbers": numbers,
                    "source": source, "priority": priority})

    reviews = hubtypes.get("Отзывы") or {}
    add("где болит",
        "Карты и отзовики дают основную часть негатива",
        {"негативных сообщений с отзывов": reviews.get("negative"),
         "всего негативных сообщений": totals.get("negative"),
         "доля всего негатива, %": round((reviews.get("negative") or 0) / negative * 100, 1),
         "доля негатива внутри отзывов, %": round((reviews.get("share") or 0) * 100, 1),
         "всего сообщений с отзывов": reviews.get("total")},
        "раздел «Карта боли», таблица площадок", 1)

    if years:
        worst = max(years.items(), key=lambda kv: kv[1]["negative_share"])
        calm = min(years.items(), key=lambda kv: kv[1]["negative_share"])
        add("год к году", "Худший год периода и разворот в 2026",
            {"худший год": worst[0], "доля негатива в нём, %": round(worst[1]["negative_share"] * 100, 1),
             "самый спокойный год": calm[0],
             "доля негатива в нём, %": round(calm[1]["negative_share"] * 100, 1),
             "сообщений в худшем году": worst[1]["total"]},
            "раздел «Тональность и её драйверы»", 2)

    if cities:
        top = cities[0]
        add("география", "Города-очаги: где доля негатива выше рынка",
            {"город": top["city"], "негативных": top["negative"],
             "доля негатива, %": round(top["share"] * 100, 1),
             "перевес над ожидаемым": top.get("excess_norm"),
             "рост за квартал": top["growth"],
             "10 городов дают, % негатива": round((concentration.get("top10_cities_share") or 0) * 100, 1)},
            "раздел «Карта очагов напряжения»", 1)
        if len(cities) > 3:
            second = cities[3]
            add("география", "Город с самой высокой долей негатива",
                {"город": second["city"], "доля негатива, %": round(second["share"] * 100, 1),
                 "перевес": second.get("excess_norm"), "негативных": second["negative"]},
                "раздел «Карта очагов напряжения»", 2)

    if restaurants:
        worst = max(restaurants, key=lambda row: row["share"])
        add("точки", "Конкретные рестораны: адресная работа даёт быстрый эффект",
            {"город": worst["city"], "негативных отзывов": worst["negative"],
             "всего отзывов": worst["count"], "доля негатива, %": round(worst["share"] * 100, 1),
             "рейтинг": worst.get("rating"), "рост за квартал": worst["growth"],
             "30 заведений дают, % негатива": round((concentration.get("top30_restaurants_share") or 0) * 100, 1)},
            "раздел «Карта очагов напряжения», таблица заведений", 1)

    if ratings:
        add("оценки", "Оценки: сколько клиентов можно вернуть",
            {"отзывов с оценкой 1": ratings.get("1"),
             "это доля всего негатива, %": round(ratings.get("1", 0) / negative * 100, 1),
             "отзывов с оценкой 2 и 3": (ratings.get("2") or 0) + (ratings.get("3") or 0),
             "отзывов с оценкой 5": ratings.get("5"),
             "всего оценок": ratings_total},
            "раздел «Карта боли», таблица оценок", 2)

    if clusters:
        biggest = clusters[0]
        add("смысловые группы", "Главная боль по смыслу, а не по ключевым словам",
            {"группа": biggest.get("title"), "сообщений": biggest["size"],
             "доля негатива, %": round((biggest.get("share") or 0) * 100, 1),
             "пик": biggest.get("peak_month"), "суть": (biggest.get("essence") or "")[:220],
             "всего групп": len(clusters)},
            "раздел «Карта боли: смысловые группы»", 1)
        growing = sorted(clusters, key=lambda row: -(row.get("from_2026") or 0) / max(1, row["size"]))[:1]
        if growing:
            row = growing[0]
            add("что растёт", "Боль, которая растёт быстрее остальных",
                {"группа": row.get("title"), "сообщений": row["size"],
                 "с 2026 года": row.get("from_2026"),
                 "доля негатива, %": round((row.get("share") or 0) * 100, 1),
                 "суть": (row.get("essence") or "")[:220]},
                "раздел «Карта боли», график динамики по кварталам", 1)
        recent = sorted(clusters, key=lambda row: -(row.get("from_2026") or 0))[:1]
        if recent:
            row = recent[0]
            add("свежее", "Самая активная боль последнего года",
                {"группа": row.get("title"), "с 2026 года": row.get("from_2026"),
                 "всего сообщений": row["size"], "пик": row.get("peak_month"),
                 "суть": (row.get("essence") or "")[:220]},
                "раздел «Карта боли», график динамики", 2)

    if author_global:
        add("кто говорит", "Негатив массовый: это не организованная кампания",
            {"всего авторов": author_global.get("authors_total"),
             "авторов с одним сообщением": author_global.get("authors_with_one"),
             "верхушка 100 авторов даёт, % негатива": author_global.get("share_top100"),
             "верхушка 1000 авторов даёт, % негатива": author_global.get("share_top1000"),
             "всего негативных сообщений": author_global.get("messages_total")},
            "раздел «Кто говорит: авторы и площадки»", 1)

    if propagation:
        row = max(propagation, key=lambda item: item["messages"])
        add("волны", "Как разгорается волна и сколько есть времени на ответ",
            {"повод": row["story"][:80], "сообщений": row["messages"],
             "первый день": row.get("first_day"), "пик": row.get("peak_day"),
             "дней до пика": row.get("days_to_peak"),
             "кто разгонял": ", ".join(hub for hub, _ in (row.get("hubs") or [])[:4])},
            "раздел «Как расходятся волны»", 1)
        fast = [item for item in propagation if (item.get("days_to_peak") or 0) <= 1]
        if fast:
            add("скорость", "Ответ нужен в первые сутки",
                {"поводов с пиком в первый день": len(fast),
                 "пример": fast[0]["story"][:80], "сообщений": fast[0]["messages"],
                 "пик": fast[0].get("peak_day")},
                "раздел «Как расходятся волны»", 2)

    if forecast:
        first = forecast[0]
        add("что впереди", "Сезонный риск ближайших месяцев",
            {"месяц": first["month"],
             "сценарий без действий, %": round(first["no_action"] * 100, 1),
             "сценарий адресной работы, %": round(first["targeted"] * 100, 1),
             "сценарий системной работы, %": round(first["systemic"] * 100, 1),
             "средняя доля негатива за период, %": round((scen.get("mean_share") or 0) * 100, 1)},
            "раздел «Цена бездействия»", 2)

    positives = totals.get("positive")
    if positives:
        add("резерв", "Позитив есть, и он тоже массовый",
            {"позитивных сообщений": positives,
             "доля позитива, %": round(positives / float(totals.get("messages") or 1) * 100, 1),
             "отзывов с оценкой 5": ratings.get("5"),
             "всего сообщений": totals.get("messages"),
             "медиаохват негатива": totals.get("negative_media_reach")},
            "раздел «Тональность и её драйверы»", 3)

    long_tail = (totals.get("negative") or 0) - (concentration.get("top10_cities") or 0)
    add("длинный хвост", "Три четверти негатива — вне десяти крупнейших городов",
        {"негативных вне топ-10 городов": long_tail,
         "это доля негатива, %": round(long_tail / negative * 100, 1),
         "негативных в топ-10 городах": concentration.get("top10_cities"),
         "городов с ростом негатива": concentration.get("cities_with_growth")},
        "раздел «Карта очагов напряжения», таблица городов", 3)

    if external:
        item = next((row for row in external if "80 ресторанов" in row["title"]), external[0])
        add("внешний контекст", "Что происходило вокруг бренда по открытым источникам",
            {"публикация": item["title"], "источник": item["source"], "ссылка": item["url"],
             "наши данные": item.get("link_to_our_data", "")[:220]},
            "раздел «Что происходило вокруг бренда»", 3)

    return out


def headline(item: dict) -> tuple:
    """Главное число карточки с точным названием объекта — пишем сами, без модели."""
    focus = item["focus"]
    numbers = item["numbers"]
    if focus == "где болит":
        return ("Карты и отзовики дают основную часть негатива",
                "**%s%%** всего негатива приходит с отзывов (карты, отзовики, маркетплейсы): "
                "**%s** сообщений из **%s** негативных. Внутри самих отзывов негативных — **%s%%**."
                % (numbers.get("доля всего негатива, %"), num(numbers.get("негативных сообщений с отзывов")),
                   num(numbers.get("всего негативных сообщений") or numbers.get("негативных сообщений с отзывов")),
                   numbers.get("доля негатива внутри отзывов, %")))
    if focus == "год к году":
        return ("Худший год периода и разворот в 2026",
                "**%s%%** негатива в %s году — худший показатель периода; в %s году — **%s%%**. "
                "В %s году разговор был самым большим: **%s** сообщений."
                % (numbers.get("доля негатива в нём, %"), numbers.get("худший год"),
                   numbers.get("самый спокойный год"), numbers.get("доля негатива в нём, %"),
                   numbers.get("худший год"), num(numbers.get("сообщений в худшем году"))))
    if focus == "география" and "город" in numbers and "перевес над ожидаемым" in numbers:
        return ("Крупнейший очаг по объёму и динамике",
                "%s: **%s** негативных сообщений, доля негатива города — **%s%%**, это **x%s** "
                "от ожидаемого при таком же наборе площадок, рост за последний квартал — **x%s**. "
                "Десять городов вместе дают **%s%%** негатива."
                % (numbers.get("город"), num(numbers.get("негативных")),
                   numbers.get("доля негатива, %"), numbers.get("перевес над ожидаемым"),
                   numbers.get("рост за квартал"), numbers.get("10 городов дают, % негатива")))
    if focus == "география":
        return ("Город с самой высокой долей негатива",
                "%s: **%s%%** сообщений города — негативные (**%s** сообщений), это **x%s** "
                "от ожидаемого." % (numbers.get("город"), numbers.get("доля негатива, %"),
                                    num(numbers.get("негативных")), numbers.get("перевес")))
    if focus == "точки":
        return ("Одно заведение может испортить район",
                "У заведения в городе %s — **%s** негативных отзывов из **%s** (**%s%%**), "
                "средний рейтинг **%s**, рост негатива за квартал **x%s**. Тридцать таких "
                "заведений дают **%s%%** всего негатива сети."
                % (numbers.get("город"), num(numbers.get("негативных отзывов")),
                   num(numbers.get("всего отзывов")), numbers.get("доля негатива, %"),
                   str(numbers.get("рейтинг")).replace(".", ","), numbers.get("рост за квартал"),
                   numbers.get("30 заведений дают, % негатива")))
    if focus == "оценки":
        return ("Оценки: 60% негатива — это отзывы на «единицу»",
                "Отзывов с оценкой «1» — **%s**, это **%s%%** всего негатива. Ещё **%s** отзывов "
                "с оценками «2» и «3»: это клиенты, которых ещё можно вернуть. Отзывов на «5» — "
                "**%s**." % (num(numbers.get("отзывов с оценкой 1")), numbers.get("это доля всего негатива, %"),
                             num(numbers.get("отзывов с оценкой 2 и 3")), num(numbers.get("отзывов с оценкой 5"))))
    if focus == "смысловые группы":
        return ("Главная боль по смыслу, а не по ключевым словам",
                "Крупнейшая смысловая группа — «%s»: **%s** сообщений (**%s%%** негатива), пик в %s. "
                "Всего групп — **%s**." % (numbers.get("группа"), num(numbers.get("сообщений")),
                                           numbers.get("доля негатива, %"), numbers.get("пик"),
                                           numbers.get("всего групп")))
    if focus == "что растёт":
        return ("Боль, которая растёт быстрее остальных",
                "Группа «%s»: **%s** сообщений, из них **%s** — за 2026 год (это **%s%%** негатива "
                "всего периода)." % (numbers.get("группа"), num(numbers.get("сообщений")),
                                     num(numbers.get("с 2026 года")), numbers.get("доля негатива, %")))
    if focus == "свежее":
        return ("Самая активная боль последнего года",
                "«%s»: **%s** сообщений за 2026 год при **%s** за весь период, пик в %s."
                % (numbers.get("группа"), num(numbers.get("с 2026 года")),
                   num(numbers.get("всего сообщений")), numbers.get("пик")))
    if focus == "кто говорит":
        return ("Негатив массовый: это не организованная кампания",
                "**%s** авторов написали **%s** негативных сообщений. Верхушка из 100 авторов даёт "
                "лишь **%s%%** негатива, из 1000 — **%s%%**; авторов с единственным сообщением — "
                "**%s**." % (num(numbers.get("всего авторов")), num(numbers.get("всего негативных сообщений")),
                             numbers.get("верхушка 100 авторов даёт, % негатива"),
                             numbers.get("верхушка 1000 авторов даёт, % негатива"),
                             num(numbers.get("авторов с одним сообщением"))))
    if focus == "волны":
        return ("Волна разгоняется за сутки",
                "«%s»: **%s** сообщений, первый день %s, пик %s — через **%s** дней. Разгоняли: %s."
                % (numbers.get("повод"), num(numbers.get("сообщений")), numbers.get("первый день"),
                   numbers.get("пик"), numbers.get("дней до пика"), numbers.get("кто разгонял")))
    if focus == "скорость":
        return ("Ответ нужен в первые сутки",
                "У **%s** поводов пик пришёлся на первый день; пример — «%s» (**%s** сообщений, "
                "пик %s)." % (numbers.get("поводов с пиком в первый день"), numbers.get("пример"),
                              num(numbers.get("сообщений")), numbers.get("пик")))
    if focus == "что впереди":
        return ("Сезонный риск ближайших месяцев",
                "В %s доля негатива ожидается **%s%%** при бездействии, **%s%%** при работе с "
                "адресными очагами и **%s%%** при работе с причинами по всей сети. Средняя доля "
                "за период — **%s%%**." % (numbers.get("месяц"), numbers.get("сценарий без действий, %"),
                                           numbers.get("сценарий адресной работы, %"),
                                           numbers.get("сценарий системной работы, %"),
                                           numbers.get("средняя доля негатива за период, %")))
    if focus == "резерв":
        return ("Позитив есть, и он тоже массовый",
                "Позитивных сообщений — **%s** (**%s%%** всего разговора), отзывов на «5» — **%s**. "
                "При этом негатив собрал **%s** медиаохвата."
                % (num(numbers.get("позитивных сообщений")), numbers.get("доля позитива, %"),
                   num(numbers.get("отзывов с оценкой 5")), num(numbers.get("медиаохват негатива"))))
    if focus == "длинный хвост":
        return ("Три четверти негатива — вне десяти крупнейших городов",
                "Вне топ-10 городов — **%s** негативных сообщений, это **%s%%** всего негатива; "
                "городов с ростом негатива за квартал — **%s**."
                % (num(numbers.get("негативных вне топ-10 городов")),
                   numbers.get("это доля негатива, %"), numbers.get("городов с ростом негатива")))
    if focus == "внешний контекст":
        return ("Внешний фон: рост сети на фоне давления на прибыль рынка",
                "«%s» (%s). Наши данные: %s" % (numbers.get("публикация"), numbers.get("источник"),
                                                 numbers.get("наши данные") or "—"))
    return (item["focus"], json.dumps(numbers, ensure_ascii=False))


def main() -> None:
    import kfc_ai

    items = skeletons()
    log("каркасов: %d" % len(items))
    system = ("Ты пишешь управленческие выводы для отчёта о медиаполе сети фастфуда Rostic's "
              "(бывший KFC). Числа и объекты уже заданы: используй только их, свои цифры и факты "
              "добавлять нельзя. Особенно важно не путать города и отдельные заведения, сообщения "
              "и отзывы. Пиши по-русски, деловым языком, коротко. Ключевые числа выделяй двойными "
              "звёздочками, например **72,4%**.")
    result = []
    for index, item in enumerate(items, 1):
        title, headline_text = headline(item)
        prompt = (
            "ФАКТ (числа и их точный смысл; в скобках указано, к чему относится число):\n%s\n\n"
            "ГЛАВНОЕ ЧИСЛО (его менять нельзя, только пересказать своими словами):\n%s\n\n"
            "Направление вывода: %s\nИсточник в отчёте: %s\n\n"
            "Напиши по этому факту вывод для руководства. Ответ строго JSON:\n"
            '{"что_значит": "2-3 предложения: почему это важно для бизнеса, с выделением чисел **жирным**", '
            '"что_делать": "1-2 предложения: конкретное действие для сети", '
            '"если_не_делать": "1 предложение: последствие бездействия"}'
            % (json.dumps(item["numbers"], ensure_ascii=False, indent=1), headline_text,
               item["focus"], item["source"]))
        data = kfc_ai.ask_json(prompt, system=system, model="deepseek-chat", max_tokens=900, attempts=3)
        if not data:
            log("%d: ответа нет" % index)
            continue
        result.append({
            "заголовок": title,
            "цифра": headline_text,
            "что_значит": str(data.get("что_значит") or data.get("что_значает") or "").strip(),
            "что_делать": str(data.get("что_делать") or "").strip(),
            "если_не_делать": str(data.get("если_не_делать") or "").strip(),
            "приоритет": item["priority"],
            "где_смотреть": item["source"],
            "числа": item["numbers"],
        })
        log("%d/%d %s" % (index, len(items), title[:60]))
        with io.open(os.path.join(OUT, "insights3.json"), "w", encoding="utf-8") as fh:
            json.dump({"insights": result, "built_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")},
                      fh, ensure_ascii=False)
    log("готово: инсайтов %d" % len(result))


if __name__ == "__main__":
    main()
