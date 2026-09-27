# -*- coding: utf-8 -*-
"""Создание задач на годовые и межгодовой отчёты KFC (скрипт на сервере, без импорта main).

Задача — обычный прогон в режиме run: агент обязан первым шагом вызвать read_reports и
строить отчёт только по месячным структурным итогам. В текст задачи встроена проверенная
по датасету таблица объёмов/тональности/площадок (агрегации Elasticsearch), потому что в
месячных итогах messages.in_slice — это срез чтения, и доли негатива там искажены.
"""
import io
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

BASE = "http://127.0.0.1:5000"
B = "/home/dev/tellscope_app/tellscope_backend"
R = B + "/data/1/reports_directory"
DATASET = "kfc_13.05.2024-22.09.2026 Отчёты"
ANNUAL_FOLDER = "kfc_13.05.2024-22.09.2026 Годовые Агент"
ANNUAL_TOOLS = ["read_reports", "build_report"]
LIST = B + "/data/1/kfc_annual_list.json"
PROBE = "/tmp/kfc_month_probe.json"

MONTH_STEM = ["январ", "феврал", "март", "апрел", "ма", "июн", "июл",
              "август", "сентябр", "октябр", "ноябр", "декабр"]
SUMMARY_NAME = re.compile(r"^(\d{4}-\d{2})_summary\.json$")

BAN = (
    "ТЕХНИЧЕСКОЕ ОГРАНИЧЕНИЕ ЗАДАЧИ: у прогона включены ровно два инструмента — read_reports и "
    "build_report, вызов любого другого движок отклонит с ошибкой «Инструмент недоступен. Доступны: "
    "build_report, read_reports». Вся фактура уже есть в ответе read_reports (период, сообщения, темы "
    "с частотами и пояснениями, цитаты со ссылками, авторы) и в проверенной таблице ниже — "
    "дополнительный разбор данных не нужен и невозможен.\n\n"
)

BUILD_HINT = (
    "ВЫЗОВ build_report — аргументы объектом: {\"title\": \"<заголовок>\", "
    "\"folder\": \"kfc_13.05.2024-22.09.2026 Годовые Агент\", \"sections\": [{\"heading\": \"...\", "
    "\"text\": \"...\", \"bullets\": [\"...\"], \"findings\": [{\"topic\": \"<тема>\", \"essence\": \"<о чём>\", "
    "\"count\": <частота>, \"tone\": \"<тональность>\", \"quotes\": [{\"text\": \"<цитата>\", \"url\": \"<ссылка>\", "
    "\"hub\": \"<площадка>\", \"date\": \"<дата>\", \"author\": \"<автор>\"}]}]}]}.\n"
    "findings с цитатами обязательны (бери из topics в ответе read_reports): без них прогон помечается "
    "неуспешным — «в срезе N негативных сообщений, но в отчёте нет ни одной темы с цитатой».\n"
    "В КАЖДОМ разделе обязательно заполни bullets (список строк) или findings: раздел без bullets, items и "
    "findings платформа считает пустым и закрывает прогон ошибкой «данные за период не найдены».\n\n"
)

REQUIRED = (
    "Разделы (заголовки ровно такие): 1. «Как построен отчёт» — что отчёт агрегирует готовые месячные отчёты "
    "(DOCX + <ГГГГ-ММ>_summary.json), перечисли использованные месяцы и исключённые. 2. «Динамика по месяцам» — "
    "объём, число тем и доля негатива по месяцам, таблица и пояснение пиков. 3. «Тематики года» — темы с пояснением "
    "сути и частотой, устойчивые и разовые. 4. «Тональность» — абсолют и доли по месяцам (месяц, негатив, "
    "нейтрал, позитив, всего, доля негатива в %). 5. «Площадки» — где обсуждали и как менялась структура. "
    "6. «Активные авторы» — кто чаще в цитатах месячных отчётов. 7. «Ключевые инфоповоды года» — с месяцем. "
    "8. «Выводы» — 5–8 пунктов. 9. «Таблица: месяц → основные темы, доля негатива, ключевой инфоповод». "
    "10. «Примеры и ссылки» — 5–8 цитат со ссылками из месячных итогов.\n"
)


def token():
    body = urllib.parse.urlencode({"username": "test@test.ru", "password": "1245"}).encode()
    req = urllib.request.Request(BASE + "/auth/login", data=body,
                                 headers={"Content-Type": "application/x-www-form-urlencoded"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)["access_token"]


def call(path, payload=None, method="GET", tok=""):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"Authorization": "Bearer " + tok,
                                          "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            return resp.status, json.load(resp)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")[:300]


def available_months():
    """Месяцы, у которых в датасетной папке есть полный комплект (docx+pdf+summary)."""
    fdir = os.path.join(R, DATASET)
    kinds = {}
    for name in sorted(os.listdir(fdir)):
        match = SUMMARY_NAME.match(name)
        if match:
            kinds.setdefault(match.group(1), set()).add("summary")
            continue
        low = name.lower().replace("ё", "е")
        if not low.endswith((".docx", ".pdf")):
            continue
        month = None
        stamp = re.search(r"(20\d{2})-(\d{2})-\d{2}", low)
        if stamp:
            month = "%s-%s" % (stamp.group(1), stamp.group(2))
        else:
            for index, stem in enumerate(MONTH_STEM):
                found = re.search(stem + r"[а-я]*\s+(20\d{2})", low)
                if found:
                    month = "%s-%02d" % (found.group(1), index + 1)
                    break
        if month:
            kinds.setdefault(month, set()).add("docx" if low.endswith(".docx") else "pdf")
    return sorted(m for m, k in kinds.items() if {"docx", "pdf", "summary"} <= k)


def month_end(month):
    """Последний день месяца: '2025-02' → '2025-02-28'."""
    import calendar
    year, num = int(month[:4]), int(month[5:7])
    return "%s-%02d" % (month, calendar.monthrange(year, num)[1])


def probe_block(months, probe, platforms=2, compact=False):
    """Компактная справка по объёмам/тональности/площадкам из агрегаций датасета."""
    by_month = {}
    for row in probe:
        by_month[row["month"]] = row
    lines = ["ПРОВЕРЕННЫЕ ЧИСЛА (агрегации по всему датасету за месяц; используй их в разделах 2, 4 и 5; "
             "в месячных итогах messages.in_slice — только срез чтения, поэтому доля негатива там искажена):"]
    for month in months:
        row = by_month.get(month)
        if not row:
            continue
        if compact:
            lines.append("%s: %s/%s (%s%%)" % (month, row.get("total"), row.get("negative"),
                                              row.get("negative_share")))
            continue
        part = "%s: всего %s, негатив %s (%s%%)" % (month, row.get("total"), row.get("negative"),
                                                    row.get("negative_share"))
        if platforms:
            hubs = ", ".join("%s %s" % (p.get("hub"), p.get("count"))
                             for p in (row.get("platforms") or [])[:platforms])
            if hubs:
                part += ", площадки: " + hubs
        lines.append(part)
    if compact:
        lines[0] = ("ПРОВЕРЕННЫЕ ЧИСЛА (агрегации по датасету; формат «месяц: всего сообщений/негатив (доля "
                    "негатива)»; в месячных итогах messages.in_slice — срез чтения, доля негатива там искажена):")
    return "\n".join(lines)


def volume_series(months, probe):
    by_month = {row["month"]: row for row in probe}
    return ", ".join("%s=%s" % (month, by_month[month].get("total"))
                     for month in months if month in by_month)


def share_series(months, probe):
    by_month = {row["month"]: row for row in probe}
    parts = []
    for month in months:
        row = by_month.get(month)
        if row:
            parts.append("%s=%s%%" % (month[5:], row.get("negative_share")))
    return ", ".join(parts)


def year_totals(months, probe):
    by_month = {row["month"]: row for row in probe}
    total = neg = neu = pos = 0
    hubs = {}
    for month in months:
        row = by_month.get(month)
        if not row:
            continue
        total += int(row.get("total") or 0)
        neg += int(row.get("negative") or 0)
        neu += int(row.get("neutral") or 0)
        pos += int(row.get("positive") or 0)
        for item in row.get("platforms") or []:
            hubs[item.get("hub")] = hubs.get(item.get("hub"), 0) + int(item.get("count") or 0)
    top = sorted(hubs.items(), key=lambda kv: -kv[1])[:6]
    return total, neg, neu, pos, top


def exclusion_line(missing):
    if not missing:
        return "Все месяцы периода представлены, исключённых месяцев нет."
    names = []
    for month in missing:
        names.append("%s %s" % (MONTH_NAMES[int(month[5:7]) - 1], month[:4]))
    return ("ВНИМАНИЕ, обязательная строка в отчёте: «за %s данные не собраны, месяцы исключены "
            "из сравнения»." % " и ".join(names))


def main():
    apply = "--apply" in sys.argv
    probe = json.load(io.open(PROBE, encoding="utf-8"))
    have = available_months()
    years = {2024: [m for m in have if m.startswith("2024")],
             2025: [m for m in have if m.startswith("2025")],
             2026: [m for m in have if m.startswith("2026")]}
    full = {2024: ["2024-%02d" % m for m in range(5, 13)],
            2025: ["2025-%02d" % m for m in range(1, 13)],
            2026: ["2026-%02d" % m for m in range(1, 9)]}

    tasks = []
    for year in (2024, 2025, 2026):
        months = years[year]
        missing = [m for m in full[year] if m not in months]
        total, neg, neu, pos, top = year_totals(months, probe)
        label = "май–декабрь" if year == 2024 else ("январь–декабрь" if year == 2025 else "январь–август")
        text = (
            "Собери ГОДОВОЙ отчёт по теме KFC за %d год (%s) на основе уже готовых месячных отчётов.\n\n"
            "ПЕРВЫЙ ШАГ (обязательно и первым): вызови инструмент read_reports с параметрами "
            "folder=\"%s\", limit=50. Он вернёт месячные структурные итоги (период, сообщения, темы с частотами "
            "и пояснениями, цитаты, авторов, ссылки).\n\n"
            "ВТОРОЙ ШАГ: собери документ инструментом build_report: "
            "title=\"Годовой отчёт по теме KFC за %d год (по месячным отчётам)\", "
            "folder=\"%s\".\n\n%s%s%s\n%s\n\n"
            "Сводка года по проверенным числам: всего %s сообщений, негатив %s (%s%%), нейтрал %s, "
            "позитив %s. Топ площадок: %s.\n\n%s"
            % (year, label, DATASET, year, ANNUAL_FOLDER, BAN, BUILD_HINT, REQUIRED,
               exclusion_line(missing), total, neg, round(neg / float(total or 1) * 100, 2), neu, pos,
               ", ".join("%s %s" % (hub, count) for hub, count in top),
               probe_block(months, probe, platforms=0))
        )
        tasks.append({"year": year, "mode": "run", "index": 1102, "model": "deepseek",
                      "tools": ANNUAL_TOOLS,
                      "min_date": ("2024-05-13" if year == 2024 else "%s-01" % months[0]) if months else None,
                      "max_date": month_end(months[-1]) if months else None,
                      "text": text, "months": months, "missing": missing})

    all_months = sorted(set(years[2024] + years[2025] + years[2026]))
    all_missing = sorted(set(m for obj in tasks for m in obj["missing"]))
    totals = []
    for year in (2024, 2025, 2026):
        total, neg, neu, pos, top = year_totals(years[year], probe)
        totals.append("%d: всего %s, негатив %s (%s%%), площадки %s" % (
            year, total, neg, round(neg / float(total or 1) * 100, 2),
            ", ".join("%s %s" % (hub, count) for hub, count in top[:4])))
    text = (
        "Собери ЕДИНЫЙ МЕЖГОДОВОЙ отчёт по теме KFC за 2024, 2025 и 2026 годы на основе уже "
        "готовых месячных отчётов.\n\n"
        "ПЕРВЫЙ ШАГ (обязательно и первым): вызови инструмент read_reports с параметрами "
        "folder=\"%s\", limit=50 — это месячные структурные итоги за все три года.\n\n"
        "ВТОРОЙ ШАГ: собери документ инструментом build_report: "
        "title=\"Межгодовой отчёт по теме KFC: 2024 / 2025 / 2026 (по месячным отчётам)\", "
        "folder=\"%s\".\n\n%s%s"
        "Разделы (заголовки ровно такие): 1. «Как построен отчёт» — что отчёт агрегирует месячные отчёты трёх "
        "лет, перечисли использованные и исключённые месяцы. 2. «Общие и различающиеся темы» — что обсуждают все "
        "три года, что появилось и что исчезло. 3. «Тренд тональности» — доля негатива по годам и по месяцам, "
        "таблица и пояснение пиков. 4. «Сезонность» — одинаковые месяцы разных лет. 5. «Авторы и площадки» — как "
        "менялись. 6. «Инфоповоды: разовые и повторяющиеся». 7. «Таблица: год → месяц → основные темы, доля "
        "негатива, ключевой инфоповод». 8. «Сводные выводы и рекомендации» — 5–8 выводов и 3–5 рекомендаций.\n\n%s\n\n"
        "Сводка по годам: %s. Доля негатива по годам: %s.\n\n%s"
        % (DATASET, ANNUAL_FOLDER, BAN, BUILD_HINT, exclusion_line(all_missing), " | ".join(totals),
           "; ".join("%d — %s%%" % (year, round(year_totals(years[year], probe)[1] /
                                                float(year_totals(years[year], probe)[0] or 1) * 100, 2))
                     for year in (2024, 2025, 2026)),
           probe_block(all_months, probe, platforms=0, compact=True))
    )
    tasks.append({"year": "inter", "mode": "run", "index": 1102, "model": "deepseek",
                  "tools": ANNUAL_TOOLS,
                  "min_date": "2024-05-13",
                  "max_date": month_end(all_months[-1]) if all_months else None,
                  "text": text, "months": all_months, "missing": all_missing})

    print("месяцев с комплектом: %d — %s" % (len(have), ", ".join(have)))
    for obj in tasks:
        assert len(obj["text"]) <= 3900, "текст задачи %s длиннее лимита: %d" % (obj["year"], len(obj["text"]))
        print("\n===== %s (длина текста %d, месяцев %d) =====" % (
            obj["year"], len(obj["text"]), len(obj["months"])))
        print(obj["text"][-1500:])
        print("... [показан хвост текста] ...")

    if not apply:
        print("\n(сухой прогон: задачи не созданы; запустите с --apply)")
        return

    only = ""
    for arg in sys.argv:
        if arg.startswith("--only="):
            only = arg.split("=", 1)[1]
    wanted = {part.strip() for part in only.split(",") if part.strip()}
    tok = token()
    made = []
    for obj in tasks:
        if wanted and str(obj["year"]) not in wanted:
            print("пропускаю %s (не в списке --only)" % obj["year"])
            continue
        status, resp = call("/harness/task", {
            "text": obj["text"], "mode": obj["mode"], "index": obj["index"],
            "min_date": obj["min_date"], "max_date": obj["max_date"], "model": obj["model"],
            "tools": obj.get("tools"),
        }, method="POST", tok=tok)
        task = (resp or {}).get("task") if isinstance(resp, dict) else None
        print("задача на %s: HTTP %s, id=%s, tools=%s" % (obj["year"], status, (task or {}).get("id"),
                                                          (task or {}).get("tools")))
        if task:
            made.append({"label": str(obj["year"]), "id": task["id"], "months": obj["months"],
                         "missing": obj["missing"]})
    if made:
        json.dump(made, io.open(LIST, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("список сохранён: %s (%d задач)" % (LIST, len(made)))


if __name__ == "__main__":
    main()
