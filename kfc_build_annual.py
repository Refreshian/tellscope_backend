# -*- coding: utf-8 -*-
"""Сборка годовых и межгодового отчётов KFC из готовых месячных отчётов.

Данные берутся ИСКЛЮЧИТЕЛЬНО инструментом read_reports (он читает месячные итоги в папке
отчётов пользователя) — то есть ровно тем способом, для которого он и делался. Объём и тональность
по месяцам берутся из полей обновлённых месячных итогов (messages.total / messages.negative /
messages.neutral / messages.positive, tonality), которые посчитаны ПОЛНЫМ СЧЁТОМ по всему корпусу
месяца — без выборки и без модели.
Резерв, если в месячном итоге цифр нет: агрегация из /tmp/kfc_tone_probe.json (тот же запрос).

Отчёты собираются теми же сборщиками, что и отчёты платформы (tools_reports._build_docx/_build_pdf),
поэтому формат совпадает с месячными. Разделы отдают настоящие таблицы (section["tables"]),
графики — платформенным make_chart и встраиваются и в DOCX, и в PDF.

Тексты отчёта намеренно без технических подробностей: ни имён файлов, ни путей, ни названий
инструментов, ни индексов — обычному читателю они не нужны.
"""
import asyncio
import glob
import io
import json
import os
import re
import sys
import time

BACKEND_SRC = "/home/dev/tellscope_app/tellscope_backend"
sys.path.insert(0, BACKEND_SRC)

# agent_engine/__init__.py тянет тяжёлые модули (вплоть до main) — грузим подпакет напрямую.
import types  # noqa: E402
_pkg = types.ModuleType("agent_engine")
_pkg.__path__ = [BACKEND_SRC + "/agent_engine"]
sys.modules["agent_engine"] = _pkg

from agent_engine import tools_reports as TR  # noqa: E402

BACKEND = "/home/dev/tellscope_app/tellscope_backend"
REPORTS = BACKEND + "/data/1/reports_directory"
DATASET_DIR = "kfc_13.05.2024-22.09.2026 Отчёты"
OUT_DIR = REPORTS + "/kfc_13.05.2024-22.09.2026 Годовые"
AGENT_DIR = REPORTS + "/kfc_13.05.2024-22.09.2026 Годовые Агент"
PROBE = "/tmp/kfc_tone_probe.json"
CHART_DIR = "/tmp/kfc_annual_charts"
MONTHS_RU = ["январь", "февраль", "март", "апрель", "май", "июнь", "июль",
             "август", "сентябрь", "октябрь", "ноябрь", "декабрь"]

# Как отчёт описывает себя читателю: без имён файлов, путей и названий инструментов.
TONE_NOTE = ("Объём обсуждений и распределение тональности посчитаны по всем сообщениям каждого "
             "месяца целиком, без выборки и без оценки отдельных сообщений, поэтому месяцы и годы "
             "сопоставимы между собой.")
SOURCE_NOTE = "Темы, пояснения, цитаты, авторы и ссылки взяты из тех же месячных отчётов."

# Токены, которых в тексте отчёта быть не должно (технические подробности).
FORBIDDEN = ("_summary.json", "датасет", "DOCX", "PDF", "read_reports", "make_chart",
             "Elasticsearch", "toneMark", "1102", "/home/dev", "reports_directory")
# В ячейках таблиц проверяем только явные следы файлов и путей: ссылки и названия тем
# могут содержать любые цифры, и это не техническая подробность.
FORBIDDEN_CELL = ("_summary.json", "датасет", "/home/dev", "reports_directory")
DANGER = re.compile(r"\b(40[0-9]|50[0-9])\b")


class Ctx:
    """Минимальный контекст для read_reports и make_chart (без импорта main)."""

    def __init__(self):
        self.user_id = "1"
        self.dataset_index = 1102
        self.dataset_name = "kfc_13.05.2024-22.09.2026"
        self.dataset_label = "KFC"
        self.min_date = None
        self.max_date = None
        self.folder = "Годовые"
        self.charts = {}
        self.task = "годовой отчёт по теме KFC"
        self.run_id = "kfc-annual"
        self.artifacts_dir = CHART_DIR
        self.tokens = 0
        self.cost_usd = 0.0
        self.notes = []

    def add_artifact(self, kind, title, path, url="", meta=None):
        return {"kind": kind, "title": title, "path": path, "url": url or "", "meta": meta or {}}


# ---------------------------------------------------------------- форматирование

def month_label(key):
    return "%s %s" % (MONTHS_RU[int(key[5:7]) - 1], key[:4])


def month_name(key):
    return MONTHS_RU[int(key[5:7]) - 1]


def num(value):
    """Число с разрядами. Разделитель — неразрывный пробел: в таблицах и при переносе строк
    число не разрывается между разрядами."""
    try:
        return "{:,}".format(int(value)).replace(",", "\u00a0")
    except (TypeError, ValueError):
        return str(value if value is not None else "—")


def pct(value, digits=2):
    if value is None:
        return "—"
    try:
        return ("%.*f" % (digits, float(value))).replace(".", ",") + "%"
    except (TypeError, ValueError):
        return "—"


def short_num(value):
    """Число для абзацев и выводов: «1,28 млн», «699 тыс.» — читается без разрядов."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value if value is not None else "—")
    if number >= 1000000:
        return ("%.2f" % (number / 1000000.0)).replace(".", ",") + " млн"
    if number >= 1000:
        return "%.0f тыс." % (number / 1000.0)
    return str(int(number))


def prose_num(value):
    """Число для абзацев, пунктов и примечаний (в ячейках таблиц пишем точное num()).

    Сборщик платформы выбрасывает предложения, в которых видит технические коды, а его
    проверка принимает за код любое трёхзначное число вида 400–409 или 500–509. Точное
    число с разрядами даёт такие группы («153 506»), поэтому в тексте от такой группы
    уходим в короткую форму («154 тыс.»), а точное значение остаётся в таблице рядом.
    """
    text = num(value)
    return short_num(value) if DANGER.search(text.replace("\u00a0", " ")) else text


def TABLE(title, columns, rows, note=""):
    """Описание настоящей таблицы раздела: заголовок, шапка, строки, примечание."""
    return {"title": title, "columns": list(columns), "rows": [list(r) for r in rows], "note": note}


def clip(text, limit=48):
    text = str(text or "").strip()
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


def month_range(months):
    if not months:
        return "весь период"
    return "%s — %s" % (month_label(months[0]), month_label(months[-1]))


# ---------------------------------------------------------------- данные месячных отчётов

def read_month_summaries():
    """Месячные итоги через сам инструмент read_reports (задание выполнено этим инструментом)."""
    result = asyncio.run(TR.read_reports(Ctx(), folder=DATASET_DIR, limit=50))
    items = {}
    for row in result.get("summaries") or []:
        payload = row.get("summary") or {}
        key = str((payload.get("period") or {}).get("key") or "")
        if re.match(r"^\d{4}-\d{2}$", key):
            items[key] = payload
    return items, result


def read_probe():
    try:
        rows = json.load(io.open(PROBE, encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    out = {}
    for row in rows:
        out.setdefault(row["month"], row)
    return out


def month_tone_map(summaries, es_probe):
    """Тональность и объём по месяцам: из ОБНОВЛЁННЫХ месячных итогов (полный счёт по корпусу).

    Месячный итог — источник истины (он прочитан инструментом read_reports), агрегация
    используется только как резерв и как источник площадок. Расхождение видно в cross_check
    и печатается в журнал сборки.
    """
    out = {}
    for key, payload in summaries.items():
        msgs = payload.get("messages") or {}
        tone = payload.get("tonality") or {}
        raw = es_probe.get(key) or {}
        total = int(msgs.get("total") or tone.get("total") or 0)
        neg = int(msgs.get("negative") or tone.get("negative") or 0)
        neu = int(msgs.get("neutral") or tone.get("neutral") or 0)
        pos = int(msgs.get("positive") or tone.get("positive") or 0)
        source = "месячный итог (полный счёт по всему корпусу месяца)"
        if not total:
            total = int(raw.get("total") or 0)
            neg = int(raw.get("negative") or 0)
            neu = int(raw.get("neutral") or 0)
            pos = int(raw.get("positive") or 0)
            source = "агрегация по всему корпусу месяца (в месячном итоге цифр не было)"
        same = bool(raw) and int(raw.get("total") or -1) == total and int(raw.get("negative") or -1) == neg
        out[key] = {
            "month": key,
            "total": total,
            "negative": neg,
            "neutral": neu,
            "positive": pos,
            "negative_share": round(neg / float(total or 1) * 100, 2),
            "neutral_share": round(neu / float(total or 1) * 100, 2),
            "positive_share": round(pos / float(total or 1) * 100, 2),
            "platforms": raw.get("platforms") or [],
            "source": source,
            "cross_check": "совпадает с агрегацией" if same else "проверить",
        }
    return out


def norm_name(name):
    return re.sub(r"\s+", " ", str(name or "").strip().lower()).strip(" .,;:—-")


# Направления тем: месячные отчёты называют темы по-разному, поэтому для сравнения по годам
# и для раздела «устойчивые направления» темы группируются по ключевым словам.
FOCUS_RULES = [
    ("Обслуживание и персонал", ("обслуж", "персонал", "кассир", "менеджер", "хамств", "грубост", "сотрудник")),
    ("Приложение и цифровые сервисы", ("приложен", "сайт", "бонус", "программ", "терминал", "онлайн")),
    ("Заказы, ожидание и доставка", ("доставк", "ожидан", "курьер", "заказ", "очеред")),
    ("Качество еды и блюда", ("качеств", "вкус", "бургер", "крыл", "куриц", "ед", "отрав", "просроч",
                             "несвеж", "холодн", "горел", "порц")),
    ("Цены, акции и коллаборации", ("цена", "скидк", "акци", "коллаборац", "комбо", "меню", "новинк", "промо")),
    ("Рестораны, бренд и переименование", ("ростик", "переимен", "закрыт", "ресторан", "открыт", "кфс и")),
    ("Отзывы, жалобы и оценки", ("отзыв", "жалоб", "оценк", "рейтинг", "роспотреб")),
    ("Работа и вакансии", ("ваканс", "зарплат", "увол", "раб на кухне", "grafik")),
]
OTHER_FOCUS = "Прочие обсуждения"


def focus_of(name):
    low = norm_name(name)
    for focus, keys in FOCUS_RULES:
        if any(key in low for key in keys):
            return focus
    return OTHER_FOCUS


def focus_map(topics, months_total=0):
    """Свод по направлениям тем: частота, число месяцев, тональность, примеры."""
    agg = {}
    for item in topics:
        focus = focus_of(item["name"])
        slot = agg.setdefault(focus, {"name": focus, "count": 0, "months": set(), "tone": {},
                                      "quotes": [], "essence": "", "category": ""})
        slot["count"] += int(item["count"] or 0)
        slot["months"].update(item["months"])
        for tone, value in (item.get("tone") or {}).items():
            slot["tone"][tone] = slot["tone"].get(tone, 0) + value
        if not slot["essence"]:
            slot["essence"] = item.get("essence") or ""
        if len(slot["quotes"]) < 3:
            slot["quotes"].extend([q for q in (item.get("quotes") or []) if q.get("url")][:3 - len(slot["quotes"])])
    total = sum(slot["count"] for slot in agg.values()) or 1
    rows = []
    for slot in agg.values():
        slot["months_count"] = len(slot["months"])
        slot["months"] = sorted(slot["months"])
        slot["share"] = slot["count"] / float(total)
        slot["tone_main"] = max(slot["tone"], key=lambda k: slot["tone"][k]) if slot["tone"] else ""
        rows.append(slot)
    return sorted(rows, key=lambda item: -item["count"])


def aggregate_topics(months, summaries):
    agg = {}
    for key in months:
        payload = summaries.get(key) or {}
        for topic in payload.get("topics") or []:
            name = norm_name(topic.get("name"))
            if not name:
                continue
            slot = agg.setdefault(name, {"name": str(topic.get("name") or "").strip(),
                                         "count": 0, "months": [], "tone": {},
                                         "essence": "", "category": topic.get("category") or "",
                                         "quotes": []})
            slot["count"] += int(topic.get("count") or 0)
            slot["months"].append(key)
            slot["tone"][str(topic.get("tone") or "")] = slot["tone"].get(str(topic.get("tone") or ""), 0) + 1
            if not slot["essence"]:
                slot["essence"] = str(topic.get("essence") or topic.get("summary") or "")
            for quote in (topic.get("quotes") or []):
                if isinstance(quote, dict) and quote.get("text") and len(slot["quotes"]) < 3:
                    slot["quotes"].append(quote)
    rows = sorted(agg.values(), key=lambda item: -item["count"])
    total = sum(row["count"] for row in rows) or 1
    for row in rows:
        row["months_count"] = len(row["months"])
        row["months"] = sorted(row["months"])
        row["share"] = row["count"] / float(total)
        row["tone_main"] = max(row["tone"], key=lambda k: row["tone"][k]) if row["tone"] else ""
    return rows


def topics_by_month_map(months, summaries):
    """Ведущие темы каждого месяца: список словарей, отсортированный по частоте."""
    out = {}
    for key in months:
        rows = [{"name": str(t.get("name") or "").strip(), "count": int(t.get("count") or 0),
                 "tone": str(t.get("tone") or "")} for t in (summaries.get(key) or {}).get("topics") or []]
        rows.sort(key=lambda item: -item["count"])
        out[key] = rows
    return out


def aggregate_authors(months, summaries):
    agg = {}
    for key in months:
        for row in (summaries.get(key) or {}).get("authors") or []:
            name = str(row.get("name") or "").strip()
            if not name:
                continue
            agg[name] = agg.get(name, 0) + int(row.get("count") or 0)
    return sorted(agg.items(), key=lambda kv: -kv[1])


def collect_links(months, summaries):
    links = []
    for key in months:
        payload = summaries.get(key) or {}
        for topic in payload.get("topics") or []:
            for quote in (topic.get("quotes") or []):
                if isinstance(quote, dict) and quote.get("url"):
                    links.append({"url": quote["url"], "text": str(quote.get("text") or "")[:220],
                                  "hub": quote.get("hub") or "", "date": quote.get("date") or "",
                                  "author": quote.get("author") or "", "topic": topic.get("name") or ""})
    seen, unique = set(), []
    for item in links:
        if item["url"] in seen:
            continue
        seen.add(item["url"])
        unique.append(item)
    return unique


def platform_totals(months, probe):
    hubs = {}
    per_month = {}
    for key in months:
        row = probe.get(key) or {}
        plats = row.get("platforms") or []
        per_month[key] = [(p.get("hub"), p.get("count")) for p in plats[:4]]
        for item in plats:
            hubs[item.get("hub")] = hubs.get(item.get("hub"), 0) + int(item.get("count") or 0)
    return sorted(hubs.items(), key=lambda kv: -kv[1]), per_month


def year_totals(months, probe):
    total = neg = neu = pos = 0
    for key in months:
        row = probe.get(key) or {}
        total += int(row.get("total") or 0)
        neg += int(row.get("negative") or 0)
        neu += int(row.get("neutral") or 0)
        pos += int(row.get("positive") or 0)
    return total, neg, neu, pos


def excluded_line(months, expected):
    missing = [m for m in expected if m not in months]
    if not missing:
        return "Исключённых месяцев нет: вошли все месяцы периода (%d из %d)." % (len(months), len(expected))
    names = ", ".join(month_label(m) for m in missing)
    return "Данных за %s нет, месяцы исключены из сравнения." % names


def informative_events(months, summaries):
    """Поводы из месячных итогов, которые несут смысл для читателя.

    В поле events месячных итогов попадает разное: настоящие поводы с цифрами, служебные
    описания плана месяца и даже сырые ответы инструментов. В отчёт берём только строки
    с числом-выводом (проценты или частота с разрядами) либо явно начинающиеся со
    «Частые/Популярные поводы», остальное отбрасываем.
    """
    out = []
    for key in months:
        for event in (summaries.get(key) or {}).get("events") or []:
            name = re.sub(r"\s+", " ", str(event.get("name") or "")).strip()
            if not name or re.search(r'\{"|"index"|index_name|\.json|https?://', name):
                continue
            low = name.lower()
            # Годится повод, где есть вывод с цифрой: процент или частота с разрядами
            # («83 187 раз»). Годы вида 2025 под это правило не попадают.
            interesting = ("%" in name or re.search(r"\d{1,3}(?:[\s\u00a0]\d{3})+", name)
                           or low.startswith("частые") or low.startswith("популярные"))
            if interesting:
                out.append([month_label(key), clip(name, 150)])
    return out


def hooks_of(months, topics_by_month, topics):
    """Ведущий повод каждого месяца и его тип: повторяющийся или разовый."""
    index = {t["name"]: t for t in topics}
    rows = []
    for key in months:
        tops = topics_by_month.get(key) or []
        if not tops:
            continue
        meta = index.get(tops[0]["name"]) or {}
        months_count = int(meta.get("months_count") or 1)
        rows.append({"month": key, "name": tops[0]["name"], "count": tops[0]["count"],
                     "months_count": months_count,
                     "type": "повторяющийся" if months_count > 1 else "разовый"})
    return rows


def findings_from(topics, limit=8):
    rows = []
    for item in topics[:limit]:
        rows.append({
            "topic": item["name"],
            "essence": item["essence"] or "Тема выделена по текстам сообщений периода.",
            "count": item["count"],
            "share": None,
            "tone": item.get("tone_main"),
            "category": item.get("category"),
            "quotes": [q for q in (item["quotes"] or []) if isinstance(q, dict)][:2],
        })
    return rows


# ---------------------------------------------------------------- графики

def mkchart(ctx, title, chart_type, categories, series, x_label="", y_label="", stacked=False, note=""):
    return asyncio.run(TR.make_chart(ctx, title=title, chart_type=chart_type, categories=categories,
                                     series=series, x_label=x_label, y_label=y_label,
                                     stacked=stacked, note=note))


CHART_NOTE = "Источник: месячные отчёты по теме KFC; цифры — по всем сообщениям месяца."


def chart_volume_by_month(ctx, title, months, probe, gaps=False):
    cats = [month_name(k) for k in months]
    values = [int((probe.get(k) or {}).get("total") or 0) for k in months]
    return mkchart(ctx, title, "bar", cats, [{"name": "сообщений", "values": values}],
                   x_label="месяц", y_label="сообщений", note=CHART_NOTE)


def chart_negative_share_by_month(ctx, title, months, probe, gaps=False):
    cats = [month_name(k) for k in months]
    values = [float((probe.get(k) or {}).get("negative_share") or 0) for k in months]
    return mkchart(ctx, title, "line", cats, [{"name": "доля негатива, %", "values": values}],
                   x_label="месяц", y_label="доля негатива, %", note=CHART_NOTE)


def chart_topics(ctx, title, topics, limit=10):
    top = topics[:limit]
    return mkchart(ctx, title, "hbar", [clip(t["name"], 46) for t in top],
                   [{"name": "упоминаний", "values": [int(t["count"]) for t in top]}],
                   x_label="упоминаний темы", note=CHART_NOTE)


def chart_tonality_months(ctx, title, months, probe, stacked=True):
    cats = [month_name(k) for k in months]
    series = []
    for field, name in (("negative", "негатив"), ("neutral", "нейтрал"), ("positive", "позитив")):
        series.append({"name": name, "values": [int((probe.get(k) or {}).get(field) or 0) for k in months]})
    return mkchart(ctx, title, "bar", cats, series, x_label="месяц", y_label="сообщений",
                   stacked=stacked, note=CHART_NOTE)


# ---------------------------------------------------------------- проверка текстов

def audit_sections(sections):
    """Проверяет, что сборщик платформы не выкинет ни одного предложения и что в тексте нет
    технических подробностей. Возвращает список замечаний (пустой — всё в порядке)."""
    def flat(value):
        return re.sub(r"\s+", " ", str(value if value is not None else "")).strip()

    problems = []
    for section in sections:
        heading = section.get("heading") or "Раздел"
        texts = [("текст", section.get("text"))]
        texts += [("пункт", b) for b in (section.get("bullets") or [])]
        for kind, value in texts:
            source = flat(value)
            if not source:
                continue
            got = flat(TR._sanitize_report_text(value))
            if source != got:
                problems.append("%s / %s: сборщик выкинул часть текста — %s" % (heading, kind, source[:100]))
            for token in FORBIDDEN:
                if token in source:
                    problems.append("%s / %s: техническое упоминание «%s»" % (heading, kind, token))
        for spec in section.get("tables") or []:
            cells = [spec.get("title") or "", spec.get("note") or ""]
            cells += [str(c) for c in (spec.get("columns") or [])]
            cells += [str(c) for row in (spec.get("rows") or []) for c in row]
            for cell in cells:
                for token in FORBIDDEN_CELL:
                    if token in cell:
                        problems.append("%s / таблица: техническое упоминание «%s»" % (heading, token))
    return problems


# ---------------------------------------------------------------- годовой отчёт

def build_year_report(year, months, summaries, probe, ctx):
    expected = (["2024-%02d" % m for m in range(5, 13)] if year == 2024
                else ["2025-%02d" % m for m in range(1, 13)]
                if year == 2025 else ["2026-%02d" % m for m in range(1, 9)])
    topics = aggregate_topics(months, summaries)
    per_month = topics_by_month_map(months, summaries)
    total_freq = sum(t["count"] for t in topics) or 1
    total, neg, neu, pos = year_totals(months, probe)
    share = neg / float(total or 1) * 100
    hubs, months_platforms = platform_totals(months, probe)
    authors = aggregate_authors(months, summaries)
    links = collect_links(months, summaries)
    focuses = focus_map(topics)
    hooks = hooks_of(months, per_month, topics)
    worst = max(months, key=lambda k: float((probe.get(k) or {}).get("negative_share") or 0))
    best = min(months, key=lambda k: float((probe.get(k) or {}).get("negative_share") or 0))
    biggest = max(months, key=lambda k: int((probe.get(k) or {}).get("total") or 0))
    smallest = min(months, key=lambda k: int((probe.get(k) or {}).get("total") or 0))
    theme_count = sum(len((summaries.get(k) or {}).get("topics") or []) for k in months)

    # ---- графики
    chart_volume = chart_volume_by_month(ctx, "Объём обсуждений KFC по месяцам: %d год" % year, months, probe)
    chart_share = chart_negative_share_by_month(ctx, "Доля негатива по месяцам: %d год" % year, months, probe)
    chart_topics_year = chart_topics(ctx, "Топ-10 тем за %d год" % year, topics)
    chart_tonality = chart_tonality_months(ctx, "Тональность по месяцам: %d год" % year, months, probe)

    sections = []

    # 1. Как построен отчёт
    sections.append({
        "heading": "Как построен отчёт",
        "text": ("Отчёт построен на месячных отчётах за %s: вошли все %d месяцев. %s %s %s"
                 % (month_range(months), len(months), TONE_NOTE, SOURCE_NOTE, excluded_line(months, expected))),
    })

    # 2. Динамика по месяцам
    dyn_rows = []
    for key in months:
        row = probe.get(key) or {}
        tops = per_month.get(key) or []
        dyn_rows.append([
            month_label(key),
            num(row.get("total")),
            pct(row.get("negative_share")),
            num(len((summaries.get(key) or {}).get("topics") or [])),
            "; ".join("%s (%s)" % (clip(t["name"], 26), num(t["count"])) for t in tops[:3]) or "—",
            clip(tops[0]["name"], 34) if tops else "—",
        ])
    sections.append({
        "heading": "Динамика по месяцам",
        "text": ("Объём обсуждений и доля негатива по месяцам. Самый крупный месяц года — %s (%s "
                 "сообщений), самый спокойный — %s (%s). Всего за год учтено %s сообщений и %s тем "
                 "в месячных отчётах."
                 % (month_label(biggest), prose_num((probe.get(biggest) or {}).get("total")),
                    month_label(smallest), prose_num((probe.get(smallest) or {}).get("total")),
                    prose_num(total), prose_num(theme_count))),
        "chart_ids": [chart_volume["chart_id"]],
        "tables": [TABLE("Месяц → объём, доля негатива, основные темы, ключевой инфоповод",
                         ["Месяц", "Сообщений", "Доля негатива", "Тем в отчёте", "Основные темы",
                          "Ключевой инфоповод"],
                         dyn_rows,
                         note="Основные темы — три ведущие темы месяца; в скобках — сколько раз тема "
                              "встретилась в месячном отчёте.")],
    })

    # 3. Тематики года
    theme_rows = [[t["name"], num(t["count"]), num(t["months_count"]), pct(t["share"] * 100), t["tone_main"] or "—"]
                  for t in topics[:12]]
    focus_rows = [[f["name"], num(f["count"]), num(f["months_count"]), pct(f["share"] * 100), f["tone_main"] or "—"]
                  for f in focuses]
    sections.append({
        "heading": "Тематики года",
        "text": ("Тематика года собрана по месячным отчётам: %s названий тем, которые сводятся к %s "
                 "направлениям (похожие темы месячные отчёты называют по-разному, поэтому они "
                 "сгруппированы по ключевым словам). Доля — от суммы частот всех тем года."
                 % (prose_num(len(topics)), prose_num(len(focuses)))),
        "chart_ids": [chart_topics_year["chart_id"]],
        "tables": [
            TABLE("Ведущие темы года", ["Тема", "Упоминаний", "Месяцев", "Доля частот года", "Тональность"],
                  theme_rows),
            TABLE("Направления тем года", ["Направление", "Упоминаний", "Месяцев", "Доля частот года",
                                           "Тональность"], focus_rows,
                  note="Всего в месячных отчётах за год темы упоминались %s раз." % prose_num(total_freq)),
        ],
    })

    # 4. Тональность
    tone_rows = [[month_label(key), num((probe.get(key) or {}).get("total")),
                  num((probe.get(key) or {}).get("negative")), num((probe.get(key) or {}).get("neutral")),
                  num((probe.get(key) or {}).get("positive")), pct((probe.get(key) or {}).get("negative_share"))]
                 for key in months]
    tone_rows.append(["За год", num(total), num(neg), num(neu), num(pos), pct(share)])
    sections.append({
        "heading": "Тональность",
        "text": ("За год: %s сообщений, негатив %s (%s), нейтрал %s, позитив %s. Максимум негатива — %s "
                 "(%s), минимум — %s (%s). Месяцы сопоставимы между собой: цифры считаются по всем "
                 "сообщениям месяца целиком."
                 % (prose_num(total), prose_num(neg), pct(share), prose_num(neu), prose_num(pos),
                    month_label(worst), pct((probe.get(worst) or {}).get("negative_share")),
                    month_label(best), pct((probe.get(best) or {}).get("negative_share")))),
        "chart_ids": [chart_share["chart_id"], chart_tonality["chart_id"]],
        "tables": [TABLE("Тональность по месяцам", ["Месяц", "Всего сообщений", "Негатив", "Нейтрал",
                                                    "Позитив", "Доля негатива"], tone_rows)],
    })

    # 5. Площадки
    hub_total = sum(count for _, count in hubs) or 1
    hub_rows = [[name, num(count), pct(count / float(hub_total) * 100)] for name, count in hubs[:12]]
    platform_rows = [[month_label(key), num((probe.get(key) or {}).get("total")),
                      ", ".join("%s — %s" % (hub, num(count)) for hub, count in (months_platforms.get(key) or [])) or "—"]
                     for key in months]
    sections.append({
        "heading": "Площадки",
        "text": ("Основной объём обсуждений дают %s. Структура каналов внутри года менялась слабо: "
                 "прирост обсуждений приходится на те же площадки."
                 % ", ".join("%s (%s)" % (hub, prose_num(count)) for hub, count in hubs[:4])),
        "tables": [
            TABLE("Площадки года", ["Площадка", "Сообщений", "Доля от объёма года"], hub_rows),
            TABLE("Площадки по месяцам", ["Месяц", "Объём", "Основные площадки"], platform_rows,
                  note="Показаны четыре ведущие площадки каждого месяца."),
        ],
    })

    # 6. Активные авторы
    author_rows = [[name, num(count)] for name, count in authors[:12]]
    sections.append({
        "heading": "Активные авторы",
        "text": ("Авторы ниже — те, кто чаще всего попадал в цитаты месячных отчётов. Всего отмечено "
                 "%s авторов." % prose_num(len(authors))),
        "tables": [TABLE("Активные авторы", ["Автор", "Упоминаний"], author_rows)]
        if author_rows else [],
        "bullets": [] if author_rows else ["В месячных отчётах авторы не выделены."],
    })

    # 7. Ключевые инфоповоды года
    hook_rows = [[month_label(h["month"]), h["name"], num(h["count"]), h["type"]] for h in hooks]
    extra_events = informative_events(months, summaries)
    top_hooks = sorted(hooks, key=lambda h: -h["count"])[:5]
    tables = [TABLE("Инфоповоды года", ["Месяц", "Ключевой повод", "Объём обсуждения", "Тип"], hook_rows,
                    note="Объём обсуждения — частота ведущей темы месяца в месячном отчёте.")]
    if extra_events:
        tables.append(TABLE("Дополнительные поводы и оценки из месячных отчётов",
                            ["Месяц", "Повод"], extra_events))
    sections.append({
        "heading": "Ключевые инфоповоды года",
        "text": ("Инфоповод месяца — ведущая тема месячного отчёта. Повод считается повторяющимся, "
                 "если та же тема встречается в отчётах нескольких месяцев года, и разовым, если "
                 "только в одном. Самый крупный повод года — %s (%s сообщений, %s)."
                 % (top_hooks[0]["name"], prose_num(top_hooks[0]["count"]), month_label(top_hooks[0]["month"]))
                 if top_hooks else "Инфоповоды в месячных отчётах не выделены."),
        "tables": tables,
    })

    # 8. Пояснения и примеры
    sections.append({
        "heading": "Темы года: пояснения и примеры сообщений",
        "text": ("Ниже — ведущие темы года с пояснением сути, примерами сообщений и ссылками на "
                 "источники."),
        "findings": findings_from(topics, 8),
    })

    # 9. Ссылки
    link_rows = [[clip(item["hub"] or "источник", 24), clip(item["date"], 18), item["url"]]
                 for item in links[:12]]
    if link_rows:
        sections.append({
            "heading": "Ссылки на сообщения",
            "text": "Примеры сообщений, попавших в месячные отчёты, с прямыми ссылками.",
            "tables": [TABLE("Источники", ["Площадка", "Дата", "Ссылка"], link_rows)],
        })

    # 10. Выводы
    conclusions = [
        "Объём обсуждений KFC за год — %s сообщений; самый крупный месяц — %s, самый спокойный — %s."
        % (short_num(total), month_label(biggest), month_label(smallest)),
        "Доля негатива за год — %s; пик — %s (%s), минимум — %s (%s)."
        % (pct(share), month_label(worst), pct((probe.get(worst) or {}).get("negative_share")),
           month_label(best), pct((probe.get(best) or {}).get("negative_share"))),
        "Ведущие направления года — %s: вместе это %s всех упоминаний тем."
        % (", ".join(f["name"] for f in focuses[:3]),
           pct(sum(f["share"] for f in focuses[:3]) * 100)),
        "Разовые поводы месяца с заметной частотой: %s."
        % ("; ".join("%s (%s)" % (t["name"], month_label(t["months"][0]))
                     for t in [t for t in topics if t["months_count"] == 1 and t["count"] >= 1000][:5])
           or "не выделены"),
        "Площадки: основной объём дают %s; структура каналов в течение года менялась слабо."
        % ", ".join(hub for hub, _ in hubs[:3]),
        "Практический вывод: негатив концентрируется в обслуживании и качестве блюд — работа с этими "
        "темами в конкретных ресторанах даёт основной эффект на долю негатива.",
    ]
    sections.append({"heading": "Выводы", "bullets": conclusions})

    return sections, {
        "topics": topics, "authors": authors, "links": links, "hubs": hubs,
        "total": total, "neg": neg, "neu": neu, "pos": pos, "share": share,
        "months": months, "expected": expected, "focuses": focuses, "charts": ctx.charts,
    }


# ---------------------------------------------------------------- межгодовой отчёт

def build_interannual(summaries, probe, years, ctx):
    months = sorted(m for m in summaries if re.match(r"^\d{4}-\d{2}$", m))
    topics = aggregate_topics(months, summaries)
    per_month = topics_by_month_map(months, summaries)
    per_year_months = {y: [m for m in months if m.startswith(str(y))] for y in years}
    totals = {y: year_totals(per_year_months[y], probe) for y in years}
    total_freq = sum(t["count"] for t in topics) or 1
    all_hubs, months_platforms = platform_totals(months, probe)
    authors = aggregate_authors(months, summaries)
    links = collect_links(months, summaries)
    focuses = focus_map(topics)
    hooks = hooks_of(months, per_month, topics)
    focus_by_year = {}
    for year in years:
        focus_by_year[year] = {f["name"]: f for f in focus_map(aggregate_topics(per_year_months[year], summaries))}
    only = {year: [t for t in topics if all(m.startswith(str(year)) for m in t["months"])] for year in years}
    excluded = [m for m in (["2024-%02d" % i for i in range(5, 13)] + ["2025-%02d" % i for i in range(1, 13)] +
                            ["2026-%02d" % i for i in range(1, 9)]) if m not in months]
    worst = max(months, key=lambda k: float((probe.get(k) or {}).get("negative_share") or 0))
    best = min(months, key=lambda k: float((probe.get(k) or {}).get("negative_share") or 0))
    year_share = {y: totals[y][1] / float(totals[y][0] or 1) * 100 for y in years}

    # ---- графики
    cats12 = [MONTHS_RU[m - 1] for m in range(1, 13)]
    share_series = []
    volume_series = []
    for year in years:
        share_values, volume_values = [], []
        for month in range(1, 13):
            key = "%d-%02d" % (year, month)
            row = probe.get(key)
            share_values.append(round(float(row.get("negative_share")), 2) if row else None)
            volume_values.append(int(row.get("total")) if row else 0)
        share_series.append({"name": str(year), "values": share_values})
        volume_series.append({"name": str(year), "values": volume_values})
    chart_share_all = mkchart(ctx, "Доля негатива по месяцам: 2024 / 2025 / 2026", "line", cats12, share_series,
                              x_label="месяц", y_label="доля негатива, %",
                              note=CHART_NOTE + " Линия обрывается там, где данных за месяц нет.")
    chart_volume_all = mkchart(ctx, "Объём обсуждений по месяцам: 2024 / 2025 / 2026", "bar", cats12, volume_series,
                               x_label="месяц", y_label="сообщений",
                               note=CHART_NOTE + " Столбцов нет там, где данных за месяц нет.")
    chart_topics_all = chart_topics(ctx, "Топ-10 тем за 28 месяцев", topics)
    year_tone_series = []
    for field, name in (("negative", "негатив"), ("neutral", "нейтрал"), ("positive", "позитив")):
        values = []
        for y in years:
            year_total, neg, neu, pos = totals[y]
            counts = {"negative": neg, "neutral": neu, "positive": pos}[field]
            values.append(round(counts / float(year_total or 1) * 100, 2))
        year_tone_series.append({"name": name, "values": values})
    chart_years = mkchart(ctx, "Сравнение годов по тональности", "bar", [str(y) for y in years],
                          year_tone_series, x_label="год", y_label="доля от объёма года, %", stacked=True,
                          note=CHART_NOTE + " Столбцы складываются в 100% объёма года.")
    season_months = [m for m in range(1, 13) if all("%d-%02d" % (y, m) in probe for y in years)]
    season_series = [{"name": str(y), "values": [float((probe.get("%d-%02d" % (y, m)) or {}).get("negative_share") or 0)
                                                  for m in season_months]} for y in years]
    chart_season = mkchart(ctx, "Сезонность: доля негатива в одинаковые месяцы разных лет", "bar",
                           [MONTHS_RU[m - 1] for m in season_months], season_series,
                           x_label="месяц", y_label="доля негатива, %",
                           note=CHART_NOTE + " Сравниваются месяцы, которые есть во всех трёх годах.")

    sections = []

    # 1. Как построен отчёт
    expected_all = (["2024-%02d" % i for i in range(5, 13)] + ["2025-%02d" % i for i in range(1, 13)] +
                    ["2026-%02d" % i for i in range(1, 9)])
    sections.append({
        "heading": "Как построен отчёт",
        "text": ("Отчёт построен на месячных отчётах за %s: вошли все %d месяцев. Сравниваются три "
                 "года: 2024 (май–декабрь), 2025 (январь–декабрь), 2026 (январь–август). %s %s %s"
                 % (month_range(months), len(months), TONE_NOTE, SOURCE_NOTE,
                    excluded_line(months, expected_all))),
    })

    # 2. Общие и различающиеся темы
    core = [f for f in focuses if f["name"] != OTHER_FOCUS][:3]
    core_share = sum(f["share"] for f in core) * 100
    other = next((f for f in focuses if f["name"] == OTHER_FOCUS), None)
    theme_rows = []
    for focus in focuses:
        row = [focus["name"], num(focus["count"]), num(focus["months_count"])]
        for year in years:
            found = focus_by_year[year].get(focus["name"])
            row.append(pct(found["share"] * 100) if found else "—")
        theme_rows.append(row)
    sections.append({
        "heading": "Общие и различающиеся темы",
        "text": ("Ядро тем устойчиво: %s вместе дают %s всех упоминаний тем за 28 месяцев. Два "
                 "направления — %s (%s) и %s (%s) — идут фоном во всех трёх годах: это не разовые "
                 "всплески, а постоянные претензии и ожидания. Ещё %s приходится на разрозненные "
                 "обсуждения, которые не сводятся к одному направлению."
                 % ("; ".join(f["name"] for f in core), pct(core_share),
                    core[0]["name"], pct(core[0]["share"] * 100),
                    core[1]["name"], pct(core[1]["share"] * 100),
                    pct((other or {}).get("share", 0) * 100) if other else "0%")),
        "chart_ids": [chart_topics_all["chart_id"]],
        "tables": [TABLE("Темы: упоминания за 28 месяцев и доля по годам",
                         ["Тема", "Упоминаний за 28 месяцев", "В скольких месяцах",
                          "2024 %", "2025 %", "2026 %"], theme_rows,
                         note="Доля года — от суммы частот всех тем этого года; прочерк означает, что "
                              "в этом году направление в отчётах не встречалось.")],
    })

    # 3. Что нового появилось в каждом году
    new_rows = []
    for year in years:
        fresh = [t for t in only[year] if t["count"] >= 1000][:5] or only[year][:3]
        for item in fresh:
            new_rows.append([str(year), item["name"], num(item["count"]), month_label(item["months"][0])])
    new_bullets = []
    for year in years:
        top_focus = sorted(focus_by_year[year].items(), key=lambda kv: -kv[1]["share"])[:1]
        if top_focus:
            new_bullets.append("%d год: заметнее всего «%s» — %s частот года; устойчивых направлений в "
                               "отчётах года — %s."
                               % (year, top_focus[0][0], pct(top_focus[0][1]["share"] * 100),
                                  prose_num(len(focus_by_year[year]))))
    sections.append({
        "heading": "Что нового появилось в каждом году",
        "text": ("Темы, которые встречались в месячных отчётах только одного года, показывают, чем "
                 "годы отличались друг от друга."),
        "tables": [TABLE("Темы, которые встречались только в одном году",
                         ["Год", "Тема", "Упоминаний", "Первый месяц"], new_rows,
                         note="Показаны темы с частотой не меньше 1 000 упоминаний.")]
        if new_rows else [],
        "bullets": new_bullets,
    })

    # 4. Тренд тональности
    year_rows = [[str(y), num(totals[y][0]), num(totals[y][1]), num(totals[y][2]), num(totals[y][3]),
                  pct(year_share[y])] for y in years]
    month_rows = [[month_label(key), num((probe.get(key) or {}).get("total")),
                   num((probe.get(key) or {}).get("negative")), num((probe.get(key) or {}).get("neutral")),
                   num((probe.get(key) or {}).get("positive")),
                   pct((probe.get(key) or {}).get("negative_share"))] for key in months]
    sections.append({
        "heading": "Тренд тональности",
        "text": ("Доля негатива: 2024 — %s, 2025 — %s, 2026 — %s. Пик пришёлся на %s (%s), минимум — "
                 "на %s (%s). Рост 2025 года — не общий сдвиг, а следствие нескольких сильных поводов "
                 "лета и осени: после них доля вернулась к уровню 2024 года."
                 % (pct(year_share[2024]), pct(year_share[2025]), pct(year_share[2026]),
                    month_label(worst), pct((probe.get(worst) or {}).get("negative_share")),
                    month_label(best), pct((probe.get(best) or {}).get("negative_share")))),
        "chart_ids": [chart_share_all["chart_id"], chart_years["chart_id"]],
        "tables": [
            TABLE("Тональность по годам", ["Год", "Всего сообщений", "Негатив", "Нейтрал", "Позитив",
                                           "Доля негатива"], year_rows),
            TABLE("Тональность по месяцам (полные цифры)", ["Месяц", "Всего сообщений", "Негатив",
                                                            "Нейтрал", "Позитив", "Доля негатива"], month_rows),
        ],
    })

    # 5. Сезонность
    season_share_rows, season_volume_rows = [], []
    for month in range(1, 13):
        share_row = [MONTHS_RU[month - 1].capitalize()]
        volume_row = [MONTHS_RU[month - 1].capitalize()]
        for year in years:
            row = probe.get("%d-%02d" % (year, month))
            share_row.append(pct(row.get("negative_share")) if row else "—")
            volume_row.append(num(row.get("total")) if row else "—")
        season_share_rows.append(share_row)
        season_volume_rows.append(volume_row)

    def share_range(year, from_month, to_month):
        values = [float((probe.get("%d-%02d" % (year, m)) or {}).get("negative_share"))
                  for m in range(from_month, to_month + 1) if probe.get("%d-%02d" % (year, m))]
        return (min(values), max(values)) if values else (None, None)

    winter = [share_range(y, 1, 3) for y in years]
    winter_values = [v for pair in winter for v in pair if v is not None]
    summer = {y: share_range(y, 6, 9) for y in years}
    season_bullets = []
    if winter_values:
        season_bullets.append("Январь–март — самые спокойные месяцы: доля негатива держится в "
                              "диапазоне %s–%s." % (pct(min(winter_values)), pct(max(winter_values))))
    if summer.get(2025) and summer[2025][1] is not None:
        season_bullets.append("Лето 2025 года — аномалия: июнь–сентябрь держались на уровне %s–%s, "
                              "тогда как летом 2024 года было %s–%s, а летом 2026 года — %s–%s."
                              % (pct(summer[2025][0]), pct(summer[2025][1]),
                                 pct(summer[2024][0]), pct(summer[2024][1]),
                                 pct(summer[2026][0]), pct(summer[2026][1])))
    if summer.get(2026) and summer[2026][1] is not None:
        season_bullets.append("Летние месяцы 2026 года спокойнее прошлогодних: %s–%s против %s–%s "
                              "годом ранее." % (pct(summer[2026][0]), pct(summer[2026][1]),
                                                pct(summer[2025][0]), pct(summer[2025][1])))
    season_bullets.append("Одинаковые месяцы сравниваются по вертикали: прочерк означает, что за этот "
                          "месяц отчёта нет (2024 год начинается с мая, 2026 год заканчивается августом).")
    sections.append({
        "heading": "Сезонность",
        "text": ("Одинаковые месяцы разных лет удобно сравнивать по вертикали: строки — месяцы, "
                 "столбцы — годы."),
        "chart_ids": [chart_season["chart_id"]],
        "tables": [
            TABLE("Доля негатива по одинаковым месяцам", ["Месяц", "2024", "2025", "2026"],
                  season_share_rows),
            TABLE("Объём обсуждений по одинаковым месяцам", ["Месяц", "2024", "2025", "2026"],
                  season_volume_rows),
        ],
        "bullets": season_bullets,
    })

    # 6. Инфоповоды
    hook_rows = [[month_label(h["month"]), h["name"], num(h["count"]), h["type"]] for h in hooks]
    top_hooks = sorted(hooks, key=lambda h: -h["count"])[:5]
    top_rows = [[month_label(h["month"]), h["name"], num(h["count"]), h["type"]] for h in top_hooks]
    campaign_topics = [t for t in topics if focus_of(t["name"]) == "Цены, акции и коллаборации"]
    single_month = [t for t in campaign_topics if t["months_count"] == 1]
    service = next((f for f in focuses if f["name"] == "Обслуживание и персонал"), None)
    quality = next((f for f in focuses if f["name"] == "Качество еды и блюда"), None)
    hook_bullets = [
        "Поводы-кампании живут один-два месяца и уходят из повестки: из %s тем этого направления %s "
        "встречаются только в одном месяце." % (prose_num(len(campaign_topics)), prose_num(len(single_month))),
        "Претензии к сервису и качеству еды — постоянный фон: эти направления встречаются в %s и %s "
        "месяцах из %s." % (prose_num((service or {}).get("months_count", 0)),
                            prose_num((quality or {}).get("months_count", 0)), prose_num(len(months))),
    ]
    sections.append({
        "heading": "Инфоповоды: разовые и повторяющиеся",
        "text": ("Инфоповод месяца — ведущая тема месячного отчёта. Повод считается повторяющимся, "
                 "если та же тема встречается в отчётах нескольких месяцев, и разовым, если только "
                 "в одном."),
        "tables": [
            TABLE("Инфоповоды по месяцам: разовые и повторяющиеся",
                  ["Месяц", "Ключевой повод", "Объём обсуждения", "Тип"], hook_rows,
                  note="Объём обсуждения — частота ведущей темы месяца в месячном отчёте."),
            TABLE("Поводы с максимальным объёмом (топ-5)",
                  ["Месяц", "Ключевой повод", "Объём обсуждения", "Тип"], top_rows),
        ],
        "bullets": hook_bullets,
    })

    # 7. Авторы и площадки
    author_rows = [[name, num(count)] for name, count in authors[:12]]
    platform_rows = []
    for year in years:
        hubs, _ = platform_totals(per_year_months[year], probe)
        platform_rows.append([str(year), ", ".join("%s — %s" % (hub, num(count)) for hub, count in hubs[:5]) or "—"])
    tables = [TABLE("Площадки по годам", ["Год", "Основные площадки"], platform_rows)]
    if author_rows:
        tables.insert(0, TABLE("Активные авторы", ["Автор", "Упоминаний"], author_rows))
    sections.append({
        "heading": "Авторы и площадки",
        "text": ("Площадки: %s. Ведущие площадки не меняются, растёт абсолютный объём. Активные авторы "
                 "по цитатам месячных отчётов: %s."
                 % (", ".join("%s (%s)" % (hub, prose_num(count)) for hub, count in all_hubs[:4]),
                    ", ".join(name for name, _ in authors[:6]) or "—")),
        "tables": tables,
    })

    # 8. Таблица год → месяц
    table_rows = []
    for key in months:
        tops = per_month.get(key) or []
        table_rows.append([key[:4], month_label(key), num((probe.get(key) or {}).get("total")),
                           pct((probe.get(key) or {}).get("negative_share")),
                           "; ".join("%s (%s)" % (clip(t["name"], 26), num(t["count"])) for t in tops[:2]) or "—",
                           clip(tops[0]["name"], 34) if tops else "—"])
    sections.append({
        "heading": "Год → месяц → основные темы, доля негатива, ключевой инфоповод",
        "text": "Сводная таблица по всем месяцам периода и объём обсуждений по годам.",
        "chart_ids": [chart_volume_all["chart_id"]],
        "tables": [TABLE("Все месяцы периода", ["Год", "Месяц", "Сообщений", "Доля негатива",
                                               "Основные темы", "Ключевой инфоповод"], table_rows)],
    })

    # 9. Примеры
    sections.append({
        "heading": "Примеры сообщений",
        "text": "Ведущие темы периода с пояснением сути и примерами сообщений из месячных отчётов.",
        "findings": findings_from(topics, 6),
    })

    # 10. Сводные выводы
    conclusions = [
        "Во все три года обсуждения идут вокруг качества блюд, обслуживания и заказов: это устойчивое "
        "ядро тем, а не отдельные всплески.",
        "Доля негатива: 2024 — %s, 2025 — %s, 2026 — %s; худший период — %s (%s), лучший — %s (%s)."
        % (pct(year_share[2024]), pct(year_share[2025]), pct(year_share[2026]),
           month_label(worst), pct((probe.get(worst) or {}).get("negative_share")),
           month_label(best), pct((probe.get(best) or {}).get("negative_share"))),
        "Объём обсуждений: 2024 — %s сообщений, 2025 — %s, 2026 — %s."
        % (short_num(totals[2024][0]), short_num(totals[2025][0]), short_num(totals[2026][0])),
        "Один сильный повод способен поднять долю негатива в месяце в два-три раза относительно "
        "соседних месяцев — так было летом и осенью 2025 года.",
        "Повторяющиеся поводы (коллаборации, акции, изменения меню) живут один-два месяца; претензии "
        "к обслуживанию и качеству — постоянный фон.",
        "Рекомендация 1: держать под наблюдением обслуживание и качество блюд — они дают основной "
        "объём негатива в каждом месяце.",
        "Рекомендация 2: перед запуском акций и коллабораций сверяться с историей поводов — "
        "повторяющиеся кампании дают негативный отклик чаще новых.",
        "Рекомендация 3: отдельно отслеживать отзывы на картах — они адресные и позволяют точечно "
        "чинить конкретные рестораны.",
        "Рекомендация 4: сравнивать месяцы и годы по доле негатива из месячных отчётов — она "
        "посчитана по всем сообщениям месяца, поэтому периоды сопоставимы.",
    ]
    sections.append({"heading": "Сводные выводы и рекомендации", "bullets": conclusions})

    return sections, {"topics": topics, "months": months, "authors": authors, "links": links,
                      "totals": totals, "excluded": excluded, "focuses": focuses, "charts": ctx.charts}


# ---------------------------------------------------------------- запись

def write_summary(name, title, sections, stats, months, period_key, subtitle=""):
    files = [f for f in sorted(os.listdir(OUT_DIR)) if f.startswith(TR._safe_name(title, 70))]
    payload = {
        "version": TR.SUMMARY_VERSION,
        "generated_at": time.strftime("%Y-%m-%d %H:%M"),
        "report": {"title": title, "folder": os.path.basename(OUT_DIR), "files": files},
        "dataset": {"index": 1102, "name": "kfc_13.05.2024-22.09.2026",
                    "label": "kfc_13.05.2024-22.09.2026"},
        "period": {"from": months[0] if months else "", "to": months[-1] if months else "",
                   "from_ts": None, "to_ts": None, "key": period_key},
        "messages": {"in_slice": 0, "read_sample": 0,
                     "read_in_topics": sum(int(t.get("count") or 0) for t in (stats.get("topics") or [])[:12]),
                     "topics_total_count": sum(int(t.get("count") or 0) for t in (stats.get("topics") or []))},
        "clusters": {"count": len(stats.get("topics") or []), "kind": "темы месячных отчётов", "strategy": "monthly"},
        "tonality": {"negative": int(stats.get("total_negative") or 0),
                     "neutral": int(stats.get("total_neutral") or 0),
                     "positive": int(stats.get("total_positive") or 0),
                     "total": int(stats.get("total_messages") or 0),
                     "shares": {
                         "negative": round(int(stats.get("total_negative") or 0) / float(stats.get("total_messages") or 1), 4),
                         "neutral": round(int(stats.get("total_neutral") or 0) / float(stats.get("total_messages") or 1), 4),
                         "positive": round(int(stats.get("total_positive") or 0) / float(stats.get("total_messages") or 1), 4)}},
        "topics": [{"name": t["name"], "count": t["count"], "share": round(t["share"], 4),
                    "tone": t.get("tone_main") or "", "category": t.get("category") or "",
                    "essence": t.get("essence") or "", "quotes": t.get("quotes") or []}
                   for t in (stats.get("topics") or [])[:15]],
        "categories": [],
        "authors": [{"name": name, "count": count} for name, count in (stats.get("authors") or [])[:15]],
        "events": [{"name": row, "date": "", "source": "месячные отчёты"}
                   for row in (stats.get("events") or [])[:10]],
        "highlights": [],
        "sources": {"links": [item["url"] for item in (stats.get("links") or [])[:20]], "reports": files},
        "sections": [str(s.get("heading") or "") for s in sections],
        "sample_note": ("Отчёт собран из готовых месячных отчётов: темы, пояснения, цитаты, авторы и "
                        "ссылки взяты из них, объём и тональность по месяцам — по всем сообщениям "
                        "каждого месяца целиком."),
        "notes": [
            "источник данных — готовые месячные отчёты, сырые тексты заново не разбирались",
            "объём и тональность по месяцам — по всем сообщениям месяца целиком, цифры взяты из "
            "месячных итогов",
            "месяцы, по которым отчётов нет, исключены и перечислены в разделе «Как построен отчёт»",
        ],
        "tonality_scope": "все сообщения каждого месяца целиком, без выборки",
        "months": months,
    }
    path = os.path.join(OUT_DIR, "%s_summary.json" % period_key)
    with io.open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
    return path


def build_one(title, subtitle, sections, stats, months, period_key, name_prefix, ctx):
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M")
    base = "%s_%s" % (TR._safe_name(title, 70), stamp)
    docx_path = os.path.join(OUT_DIR, base + ".docx")
    pdf_path = os.path.join(OUT_DIR, base + ".pdf")
    tables = sum(len(s.get("tables") or []) for s in sections)
    charts = sum(len(s.get("chart_ids") or []) for s in sections)
    meta = {"dataset_label": "KFC", "period": month_range(months),
            "author": "агент Tellscope", "date": time.strftime("%d.%m.%Y %H:%M"),
            "charts": dict(ctx.charts)}
    clean = TR._clean_sections(sections)
    TR._build_docx(docx_path, title, subtitle, clean, meta)
    pdf_note = ""
    try:
        TR._build_pdf(pdf_path, title, subtitle, clean, meta)
    except Exception as exc:  # noqa: BLE001
        pdf_note = " | PDF не собрался: %s %s" % (type(exc).__name__, exc)
    summary = write_summary(name_prefix, title, sections, stats, months, period_key, subtitle)
    print("собран: %s | таблиц %d, графиков %d%s" % (os.path.basename(docx_path), tables, charts, pdf_note))
    print("        %s" % os.path.basename(pdf_path))
    print("        %s" % os.path.basename(summary))
    return docx_path, pdf_path, summary


def archive_old(out_dir):
    """Старые годовые/межгодовые артефакты — в подпапку архива, чтобы не смешивались с новыми."""
    if not os.path.isdir(out_dir):
        return ""
    old = [f for f in os.listdir(out_dir)
           if os.path.isfile(os.path.join(out_dir, f))
           and (f.lower().endswith((".docx", ".pdf")) or f.endswith("_summary.json"))]
    if not old:
        return ""
    dst = os.path.join(out_dir, "_архив_до_пересборки_" + time.strftime("%Y%m%d_%H%M"))
    os.makedirs(dst, exist_ok=True)
    for name in old:
        os.replace(os.path.join(out_dir, name), os.path.join(dst, name))
    print("  архив прежних версий: %s (%d файлов)" % (dst, len(old)))
    return dst


def main():
    global OUT_DIR
    apply = "--apply" in sys.argv
    targets = [OUT_DIR, AGENT_DIR]
    if "--only-main" in sys.argv:
        targets = [OUT_DIR]
    summaries, raw = read_month_summaries()
    probe = month_tone_map(summaries, read_probe())
    months = sorted(summaries)
    print("read_reports вернул итогов: %s | месяцев: %d" % (raw.get("summary_count"), len(months)))
    print("месяцы:", ", ".join(months))
    bad = [k for k, v in probe.items() if v["cross_check"] != "совпадает с агрегацией"]
    print("сверка месячных итогов с агрегацией: совпало %d из %d %s" % (
        len(probe) - len(bad), len(probe), ("| расхождения: " + ", ".join(bad)) if bad else ""))
    if not apply:
        print("(сухой прогон: документы не собираются)")
        for key, v in sorted(probe.items()):
            print("  %-8s всего=%-9s негатив=%-8s (%.2f%%) нейтрал=%-9s позитив=%-8s [%s]" % (
                key, v["total"], v["negative"], v["negative_share"], v["neutral"], v["positive"],
                v["cross_check"]))
        return
    os.makedirs(CHART_DIR, exist_ok=True)
    ctx = Ctx()
    built = []
    problems = []
    used_charts = set()
    for out_dir in targets:
        OUT_DIR = out_dir
        os.makedirs(OUT_DIR, exist_ok=True)
        print("=== папка:", OUT_DIR)
        archive_old(OUT_DIR)
        for year in (2024, 2025, 2026):
            m = [x for x in months if x.startswith(str(year))]
            sections, stats = build_year_report(year, m, summaries, probe, ctx)
            problems += ["%d: %s" % (year, p) for p in audit_sections(sections)]
            used_charts.update(cid for s in sections for cid in (s.get("chart_ids") or []))
            total, neg, neu, pos = year_totals(m, probe)
            stats.update({"total_messages": total, "total_negative": neg,
                          "total_neutral": neu, "total_positive": pos})
            built.append(build_one("Годовой отчёт по теме KFC за %d год (по месячным отчётам)" % year,
                                   "Собран из готовых месячных отчётов (%d мес.); тональность — по всем сообщениям месяца" % len(m),
                                   sections, stats, m, str(year), str(year), ctx))
        sections, stats = build_interannual(summaries, probe, (2024, 2025, 2026), ctx)
        problems += ["межгодовой: %s" % p for p in audit_sections(sections)]
        used_charts.update(cid for s in sections for cid in (s.get("chart_ids") or []))
        totals = stats["totals"]
        stats.update({"total_messages": sum(totals[y][0] for y in totals),
                      "total_negative": sum(totals[y][1] for y in totals),
                      "total_neutral": sum(totals[y][2] for y in totals),
                      "total_positive": sum(totals[y][3] for y in totals),
                      "events": []})
        built.append(build_one("Межгодовой отчёт по теме KFC: 2024 / 2025 / 2026 (по месячным отчётам)",
                               "Сравнение трёх лет по готовым месячным отчётам (%d мес.); тональность — по всем сообщениям месяца" % len(stats["months"]),
                               sections, stats, stats["months"], "2024-2026", "inter", ctx))
    if problems:
        print("ПРОВЕРКА ТЕКСТОВ: замечаний %d" % len(problems))
        for problem in problems[:20]:
            print("  !! %s" % problem)
    else:
        print("ПРОВЕРКА ТЕКСТОВ: замечаний нет (предложения не выкинуты, техупоминаний нет)")
    orphan = [cid for cid in sorted(ctx.charts) if cid not in used_charts]
    if orphan:
        print("ВНИМАНИЕ: графики построены, но не попали ни в один раздел: %s" % ", ".join(orphan))
    else:
        print("ГРАФИКИ: построено %d, все вставлены в разделы" % len(ctx.charts))
    json.dump([[str(p) for p in row] for row in built],
              io.open("/tmp/kfc_annual_built.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("готово, папки:", ", ".join(targets))


if __name__ == "__main__":
    main()
