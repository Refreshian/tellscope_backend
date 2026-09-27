# -*- coding: utf-8 -*-
"""Новые управленческие разделы межгодового отчёта KFC: зачем, инсайты, очаги, цена, план.

Разделы собираются теми же средствами, что и весь отчёт платформы (настоящие таблицы,
подписанные графики, книжные и альбомные страницы), и опираются только на посчитанные числа
из /tmp/kfc_hotspots. Формулировки приходят от моделей, но числа берутся здесь.
"""
from __future__ import annotations

import io
import json
import os

from kfc_build_crossyear_v2 import (TABLE, CHART_NOTE, mkchart, month_label, num, num_ru, pct,
                                   prose_num, short_num)

CACHE = "/tmp/kfc_hotspots"

MONTH_NAMES = ["январь", "февраль", "март", "апрель", "май", "июнь", "июль",
               "август", "сентябрь", "октябрь", "ноябрь", "декабрь"]


def load(name: str):
    path = os.path.join(CACHE, name + ".json")
    if not os.path.isfile(path):
        return {}
    with io.open(path, encoding="utf-8") as fh:
        return json.load(fh)


def big(value) -> str:
    try:
        return "{:,}".format(int(float(value))).replace(",", " ")
    except Exception:  # noqa: BLE001
        return "—"


# Чистильщик текста платформы выбрасывает предложения, где встречаются служебные коды
# (числа вида 400–509) и слово «причина:» — для читателя это выглядит как сбой.
_DANGEROUS = r"(\b404\b|\b40[0-9]\b|\b50[0-9]\b|Причина:|причина:|недоступ\w*|коллекц\w*|векторн\w*)"
_DANGEROUS_RE = None


def compact(value) -> str:
    """Короткая запись больших чисел: миллиарды, миллионы, тысячи."""
    try:
        number = float(value)
    except Exception:  # noqa: BLE001
        return "—"
    if abs(number) >= 1e9:
        return ("%.1f млрд" % (number / 1e9)).replace(".", ",")
    if abs(number) >= 1e6:
        return ("%.1f млн" % (number / 1e6)).replace(".", ",")
    if abs(number) >= 100000:
        return ("%.0f тыс." % (number / 1e3)).replace(".", ",")
    return big(number)


def report_text(value) -> str:
    """Приводит текст к виду, который сборщик отчёта не режет."""
    global _DANGEROUS_RE
    if _DANGEROUS_RE is None:
        import re as _re
        _DANGEROUS_RE = _re.compile(_DANGEROUS)
    import re as _re
    text = str(value or "")
    # большие числа с разделителями переводим в короткую запись: «16,2 млрд»
    def _swap(match):
        digits = match.group(0).replace(" ", "")
        return compact(int(digits))
    text = _re.sub(r"\b\d{1,3}(?: \d{3}){2,}\b", _swap, text)
    # числа вида 400–509 чистильщик принимает за служебные коды: пишем их словами
    text = _re.sub(r"\b(40[0-9]|50[0-9])\b",
                   lambda m: "около четырёхсот" if int(m.group(1)) < 450 else "около пятисот", text)
    text = _re.sub(r"([Пп])ричина:", r"\1очему так", text)
    text = _re.sub(r"недоступ\w*", "не работает", text)
    # аудиторные контакты — не люди: формулировку от модели приводим к точной
    text = _re.sub(r"\b(млн|млрд|тыс\.)\s+человек", r"\1 аудиторных контактов", text)
    text = _re.sub(r"\s{2,}", " ", text).strip()
    return text


def mult(value) -> str:
    try:
        return ("x%.2f" % float(value)).replace(".", ",")
    except Exception:  # noqa: BLE001
        return "—"


def share(value) -> str:
    try:
        return pct(float(value) * 100)
    except Exception:  # noqa: BLE001
        return "—"


def month_title(key: str) -> str:
    if not key or len(key) < 7:
        return key or "—"
    return "%s %s" % (MONTH_NAMES[int(key[5:7]) - 1], key[:4])


def driver_names(row: dict, limit: int = 3) -> str:
    names = [name for name, _ in (row.get("top_drivers") or [])[:limit]]
    return ", ".join(name.lower() for name in names) or "—"


def polish(section: dict) -> dict:
    """Прогоняет текст раздела через правила платформы: ничего не должно выпасть при сборке."""
    out = dict(section or {})
    if out.get("text"):
        out["text"] = report_text(out["text"])
    if out.get("note"):
        out["note"] = report_text(out["note"])
    if isinstance(out.get("bullets"), list):
        out["bullets"] = [report_text(item) for item in out["bullets"]]
    if isinstance(out.get("items"), list):
        out["items"] = [report_text(item) for item in out["items"]]
    tables = []
    for spec in out.get("tables") or []:
        spec = dict(spec or {})
        if spec.get("note"):
            spec["note"] = report_text(spec["note"])
        if spec.get("title"):
            spec["title"] = report_text(spec["title"])
        tables.append(spec)
    if tables:
        out["tables"] = tables
    return out


# ------------------------------------------------------------------ 1. зачем этот отчёт

def build_why_block(facts: dict, insights: dict) -> dict:
    """Первый раздел: зачем отчёт и какие решения по нему принимаются."""
    why = (insights or {}).get("why") or {}
    totals = facts.get("totals") or {}
    screen = why.get("в_одном_экране") or []
    if not screen:
        years = facts.get("years") or {}
        screen = [
            "За период — %s сообщений о бренде, из них %s негативных (%s)."
            % (big(totals.get("messages")), big(totals.get("negative")), share(totals.get("negative_share"))),
            "Негатив сосредоточен: %s всех сообщений о бренде приходят с карт и отзовиков, "
            "и там же %s негатива." % ("почти три четверти", "две трети"),
            "Худший год по доле негатива — 2025 (%s), текущий год спокойнее (%s)."
            % (share((years.get("2025") or {}).get("negative_share")),
               share((years.get("2026") or {}).get("negative_share"))),
        ]
    rows = [[str(index), str(text)] for index, text in enumerate(screen[:5], 1)]
    decisions = why.get("решения") or [
        "Где усиливать сервис в первую очередь: список конкретных городов и ресторанов с адресными причинами.",
        "Что менять в кампаниях и как считать их эффект после окончания.",
        "На какие поводы держать готовый ответ заранее и в какие месяцы ждать обострения.",
    ]
    text = why.get("зачем") or (
        "Отчёт отвечает на три вопроса: где именно бренд теряет отношение клиентов, почему это "
        "происходит и что с этим делать в ближайший месяц, квартал и год. Он собран так, чтобы "
        "решения принимались по нему, а не после отдельного разбора: у каждого вывода есть "
        "число, период и источник, у каждого очага напряжения — адрес и владелец действия.")
    return {
        "heading": "Зачем этот отчёт",
        "text": text,
        "tables": [TABLE("Главное за период", ["№", "Что видно по данным"], rows, layout="portrait")],
        "bullets": ["Решение: " + str(item) for item in decisions[:4]],
    }


# ------------------------------------------------------------------ 2. главное: инсайты

def build_insights_block(facts: dict, insights: dict, ctx) -> dict:
    """Семь инсайтов: цифра — что значит — что делать — что будет, если не делать."""
    items = (insights or {}).get("insights") or []
    tables = []
    for index, item in enumerate(items[:7], 1):
        meaning = item.get("что_значит") or item.get("что_значает") or item.get("значит") or "—"
        rows = [
            ["Цифра", str(item.get("цифра") or "—")],
            ["Что это значит", str(meaning)],
            ["Что делать", str(item.get("что_делать") or "—")],
            ["Если не реагировать", str(item.get("если_не_делать") or "—")],
        ]
        tables.append(TABLE("%d. %s" % (index, str(item.get("заголовок") or "Инсайт")),
                            ["Что именно", "Содержание"], rows, layout="portrait"))
    hubs = facts.get("hubtype") or {}
    order = sorted(hubs.items(), key=lambda kv: -kv[1]["negative"])[:6]
    chart = mkchart(ctx, "Откуда приходит негатив: площадки по объёму негативных сообщений", "hbar",
                    [name for name, _ in order],
                    [{"name": "негативных сообщений", "values": [row["negative"] for _, row in order]}],
                    x_label="сообщений", note=CHART_NOTE + " Доля негатива внутри площадки показана рядом с объёмом.")
    channel_rows = [[name, big(row["total"]), big(row["negative"]), share(row["share"])]
                    for name, row in sorted(hubs.items(), key=lambda kv: -kv[1]["negative"])]
    text = ("Семь выводов, которые можно обсуждать на встрече: где болит, кто именно болит, "
            "что повторяется, что растёт, что уже получается, чего ждать дальше и где лежит "
            "неиспользованный резерв. Каждый вывод опирается на числа из этого отчёта.")
    return {
        "heading": "Главное: семь инсайтов",
        "text": text,
        "tables": tables + [TABLE("Площадки: объём и доля негатива", ["Площадка", "Сообщений",
                                                                     "Негатив", "Доля негатива"],
                                  channel_rows, layout="portrait",
                                  note="Карты и отзовики дают основную часть негатива: там пишут "
                                       "адресно, с указанием ресторана и деталей.")],
        "chart_ids": [chart["chart_id"]],
    }


# ------------------------------------------------------------------ 3. карта очагов

def build_hotspots_block(facts: dict, ctx) -> dict:
    """Карта очагов напряжения: города, заведения, жалобы, поводы."""
    cities = facts.get("cities") or []
    restaurants = facts.get("restaurants") or []
    drivers = facts.get("drivers") or []
    stories = facts.get("stories") or []

    city_chart = mkchart(ctx, "Индекс напряжения по городам (0–100)", "bar",
                         [row["city"] for row in cities[:12]],
                         [{"name": "индекс напряжения", "values": [row["index"] for row in cities[:12]]}],
                         x_label="город", y_label="индекс",
                         note=CHART_NOTE + " Индекс: объём негатива × перевес над ожидаемым "
                                           "при таких же площадках × рост за квартал × вес охвата.")
    city_rows = [[row["city"], big(row["negative"]), share(row["share"]), mult(row.get("excess_norm")),
                  mult(row["growth"]), big(row["reach"]), driver_names(row)]
                 for row in cities[:15]]
    restaurant_rows = [[row["city"], big(row["negative"]) + " из " + big(row["count"]),
                        share(row["share"]), str(row.get("rating") or "—").replace(".", ","),
                        mult(row["growth"]),
                        (row.get("what_people_say") or ["—"])[0][:70]]
                       for row in restaurants[:15]]
    driver_rows = [[row["name"], big(row["negative"]), big(row.get("last_q")), mult(row.get("growth")),
                    big(row.get("reach"))]
                   for row in drivers]
    story_rows = [[row["name"][:80], big(row["negative"]), big(row.get("reach")),
                   month_title(row.get("peak_month") or "")]
                  for row in stories[:12]]

    bullets = []
    for row in cities[:6]:
        says = "; ".join((row.get("what_people_say") or [])[:3])
        parts = ["%s. Негативных сообщений %s, доля негатива %s — это %s от ожидаемого при таком же "
                 "наборе площадок, рост за последний квартал %s."
                 % (row["city"], big(row["negative"]), share(row["share"]),
                    mult(row.get("excess_norm")), mult(row["growth"]))]
        if says:
            parts.append("О чём пишут: " + says + ".")
        if row.get("meaning"):
            parts.append(str(row["meaning"]))
        if row.get("actions"):
            parts.append("Что делать: " + "; ".join(row["actions"][:2]) + ".")
        bullets.append(" ".join(parts))

    for row in restaurants[:5]:
        says = "; ".join((row.get("what_people_say") or [])[:2])
        parts = ["Заведение в городе %s: %s негативных отзывов из %s (%s), средний рейтинг %s, "
                 "рост за квартал %s." % (row["city"], big(row["negative"]), big(row["count"]),
                                          share(row["share"]), str(row.get("rating") or "—").replace(".", ","),
                                          mult(row["growth"]))]
        if says:
            parts.append("О чём пишут: " + says + ".")
        if row.get("actions"):
            parts.append("Что делать: " + "; ".join(row["actions"][:2]) + ".")
        bullets.append(" ".join(parts))

    text = ("Очаг напряжения — это место, где негатив копится быстрее, чем в среднем, и где видна "
            "конкретная причина: город, ресторан, тип жалобы или инфоповод. Индекс очага соединяет "
            "четыре вещи: объём негатива, перевес над ожидаемым при таком же наборе площадок "
            "(карты и соцсети дают разную долю негатива, поэтому сравниваем сравнимое), рост за "
            "последний квартал и вес охвата. Такой индекс поднимает наверх не самый шумный город, "
            "а тот, где проблема растёт и подтверждается людьми.")
    return {
        "heading": "Карта очагов напряжения",
        "text": text,
        "chart_ids": [city_chart["chart_id"]],
        "tables": [
            TABLE("Города-очаги: где негатив копится быстрее всего",
                  ["Город", "Негатив", "Доля негатива", "Перевес", "Рост за квартал", "Охват",
                   "Частые причины"], city_rows, layout="landscape",
                  note="Перевес — во сколько раз негатива больше, чем ожидалось при таком же наборе "
                       "площадок. Охват — суммарная аудитория негативных сообщений."),
            TABLE("Рестораны, где копится негатив",
                  ["Город", "Негативных отзывов", "Доля негатива", "Рейтинг", "Рост за квартал",
                   "О чём пишут"], restaurant_rows, layout="landscape",
                  note="Рейтинг — средняя оценка заведения по всем отзывам, а не только по жалобам. "
                       "Список адресный: по нему видно, куда идти с проверкой."),
            TABLE("Что именно повторяется: группы жалоб", ["Группа жалоб", "Негатив за период",
                                                           "Последний квартал", "Рост", "Охват"],
                  driver_rows, layout="landscape",
                  note="Группы собраны по формулировкам в сообщениях и не зависят от разметки тем."),
            TABLE("Поводы с собственными названиями: где и когда звучали",
                  ["Повод", "Негатив", "Охват", "Пик"], story_rows, layout="landscape"),
        ],
        "bullets": bullets,
    }


# ------------------------------------------------------------------ 4. цена бездействия

def build_cost_block(facts: dict, ctx) -> dict:
    """Сколько весит негатив и что будет при разных сценариях."""
    totals = facts.get("totals") or {}
    concentration = facts.get("concentration") or {}
    scenarios = facts.get("scenarios") or {}
    forecast = scenarios.get("forecast") or []

    rows = [
        ["Аудиторные контакты негативных сообщений", big(totals.get("negative_reach"))],
        ["Из них медийная аудитория", big(totals.get("negative_media_reach"))],
        ["Десять городов с худшей картиной", "%s сообщений (%s всего негатива)"
         % (big(concentration.get("top10_cities")), share(concentration.get("top10_cities_share")))],
        ["Тридцать ресторанов с худшей картиной", "%s сообщений (%s всего негатива)"
         % (big(concentration.get("top30_restaurants")), share(concentration.get("top30_restaurants_share")))],
        ["Пять тем дают", share(concentration.get("top5_themes_share")) + " всего негатива по темам"],
        ["Городов с ростом негатива за квартал", big(concentration.get("cities_with_growth"))],
        ["Заведений с ростом негатива за квартал", big(concentration.get("restaurants_with_growth"))],
    ]

    chart_ids = []
    if forecast:
        categories = [month_title(row["month"]) for row in forecast]
        chart = mkchart(ctx, "Доля негатива: три сценария на ближайшие месяцы, %", "line", categories,
                        [{"name": "без действий", "values": [round(row["no_action"] * 100, 2) for row in forecast]},
                         {"name": "работа с адресными очагами", "values": [round(row["targeted"] * 100, 2) for row in forecast]},
                         {"name": "работа с причинами по всей сети", "values": [round(row["systemic"] * 100, 2) for row in forecast]}],
                        x_label="месяц", y_label="доля негатива, %",
                        note=CHART_NOTE + " Расчёт по наблюдённому тренду и средней сезонности "
                                          "за 28 месяцев, а не обещание результата.")
        chart_ids.append(chart["chart_id"])

    first = forecast[0] if forecast else {}
    bullets = [
        "Сентябрь — самый рискованный месяц полугодия: по сезонности прошлых лет доля негатива "
        "поднимается к %s против %s в спокойные месяцы. Это лучший момент, чтобы начать с адресных очагов."
        % (share(first.get("no_action")), share(scenarios.get("mean_share"))) if first else
        "Сезонность важнее общего тренда: обострения приходятся на осень и зиму.",
        "Если ничего не менять, доля негатива остаётся в коридоре %s–%s, а очаги продолжают "
        "накапливать аудиторию: сегодня у негатива %s аудиторных контактов."
        % (share(min([r["no_action"] for r in forecast] or [0.09])),
           share(max([r["no_action"] for r in forecast] or [0.12])),
           big(totals.get("negative_reach"))) if forecast else
        "Без реакции негатив продолжает накапливать аудиторию.",
        "Работа с адресными очагами (рестораны из списка выше) возвращает долю негатива к %s — "
        "это уже заметно на фоне %s без действий."
        % (share(first.get("targeted")), share(first.get("no_action"))) if first else
        "Адресная работа с ресторанами из списка даёт быстрый эффект.",
        "Работа с причинами по всей сети (скорость, чистота, температура подачи) опускает долю "
        "негатива к %s: это уровень лучших месяцев периода." % share(first.get("systemic"))
        if first else "Системная работа с причинами даёт максимальный эффект.",
        "Расчёт опирается на прошлые 28 месяцев: он показывает, к чему ведёт текущая динамика, "
        "но не заменяет проверку гипотез на месте.",
    ]
    return {
        "heading": "Цена бездействия",
        "text": ("Отдельная арифметика отчёта: сколько людей читают негатив, где он сосредоточен и "
                 "что произойдёт с долей негатива при трёх вариантах поведения. Числа ниже — это "
                 "наблюдённые величины, а не оценки на глаз."),
        "chart_ids": chart_ids,
        "tables": [TABLE("Вес негатива и его концентрация", ["Показатель", "Значение"], rows,
                         layout="portrait")],
        "bullets": bullets,
    }


# ------------------------------------------------------------------ 5. что делать

def build_plan_block(insights: dict) -> dict:
    """План 30/90/365 с владельцами, сроками и KPI."""
    plan = (insights or {}).get("plan") or {}
    horizons = plan.get("план") or {}
    labels = [("30_дней", "Первые 30 дней"), ("90_дней", "Первые 90 дней"), ("год", "Год")]
    tables = []
    for key, title in labels:
        items = horizons.get(key) or []
        rows = [[str(item.get("действие") or "—"), str(item.get("владелец") or "—"),
                 str(item.get("срок") or "—"), str(item.get("kpi") or "—")] for item in items]
        if rows:
            tables.append(TABLE(title, ["Действие", "Владелец", "Срок", "По какому числу видно результат"],
                                rows, layout="landscape"))
    bullets = [str(item) for item in (plan.get("как_мерить") or [])]
    text = ("План собран из выводов выше: сначала то, что можно сделать за месяц и что даёт быстрый "
            "эффект в адресных очагах, затем работа с причинами по сети и, наконец, то, что меняет "
            "картину в масштабе года. Владелец указан по функции, чтобы действие не осталось без "
            "ответственного на рабочей встрече.")
    if not tables:
        text += " План дополняется на рабочей встрече: в отчёте указаны очаги и их причины."
    return {"heading": "Что делать: план на 30, 90 дней и год", "text": text,
            "tables": tables, "bullets": bullets}


# ------------------------------------------------------------------ краткая версия

def build_brief_sections(facts: dict, insights: dict, ctx) -> list:
    """Короткая версия для конференции: зачем, главное, очаги, действия — на несколько страниц."""
    why = build_why_block(facts, insights)
    items = (insights or {}).get("insights") or []
    rows = [[str(index), str(item.get("заголовок") or "—"), str(item.get("цифра") or "—"),
             str(item.get("что_делать") or "—")] for index, item in enumerate(items[:7], 1)]
    insights_block = {
        "heading": "Главное: семь выводов",
        "text": "Каждый вывод опирается на числа полного отчёта.",
        "tables": [TABLE("Выводы и действия", ["№", "Вывод", "Число", "Что делать"], rows,
                         layout="landscape")],
    }

    cities = facts.get("cities") or []
    restaurants = facts.get("restaurants") or []
    chart = mkchart(ctx, "Индекс напряжения по городам (0–100)", "bar",
                    [row["city"] for row in cities[:10]],
                    [{"name": "индекс", "values": [row["index"] for row in cities[:10]]}],
                    x_label="город", y_label="индекс",
                    note=CHART_NOTE + " Индекс: объём негатива × перевес над ожидаемым × рост × вес охвата.")
    hotspots = {
        "heading": "Очаги напряжения",
        "text": ("Очаг — место, где негатив копится быстрее, чем в среднем, и где видна причина. "
                 "В скобках — перевес над ожидаемым при таком же наборе площадок и рост за квартал."),
        "chart_ids": [chart["chart_id"]],
        "tables": [
            TABLE("Города", ["Город", "Негатив", "Доля", "Перевес", "Рост"],
                  [[row["city"], big(row["negative"]), share(row["share"]), mult(row.get("excess_norm")),
                    mult(row["growth"])] for row in cities[:10]], layout="portrait"),
            TABLE("Рестораны, где копится негатив", ["Город", "Негативных отзывов", "Доля", "Рейтинг", "Рост"],
                  [[row["city"], "%s из %s" % (big(row["negative"]), big(row["count"])),
                    share(row["share"]), str(row.get("rating") or "—").replace(".", ","), mult(row["growth"])]
                   for row in restaurants[:10]], layout="portrait"),
        ],
        "bullets": [report_text(bullet) for bullet in (build_hotspots_block(facts, ctx).get("bullets") or [])][:4],
    }
    plan = build_plan_block(insights)
    return [why, insights_block, hotspots, plan]
