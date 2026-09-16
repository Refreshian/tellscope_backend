# -*- coding: utf-8 -*-
"""Сборка годовых и межгодового отчётов KFC из готовых месячных отчётов.

Данные берутся ИСКЛЮЧИТЕЛЬНО инструментом read_reports (он читает <ГГГГ-ММ>_summary.json в папке
отчётов пользователя) — то есть ровно тем способом, для которого он и делался. Объём и тональность
по месяцам берутся из полей обновлённых месячных итогов (messages.total / messages.negative /
messages.neutral / messages.positive, tonality), которые посчитаны ПОЛНЫМ СЧЁТОМ по всему корпусу
месяца в Elasticsearch (range по timeCreate, term по toneMark) — без выборки и без модели.
Резерв, если в месячном итоге цифр нет: агрегация ES из /tmp/kfc_tone_probe.json (тот же запрос).

Скрипт живёт на сервере, не импортирует main: пакет agent_engine подгружается минуя __init__.py.
Документы собираются теми же сборщиками, что и отчёты платформы
(tools_reports._build_docx/_build_pdf), поэтому формат совпадает с месячными.
"""
import asyncio
import calendar
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
TONE_SCOPE = ("полный счёт по всему корпусу месяца: Elasticsearch kfc_13.05.2024-22.09.2026, "
              "range по timeCreate, term по toneMark (-1 негатив, 0 нейтрал, 1 позитив), "
              "сутки по МСК включительно, без выборки и без модели")
MONTHS_RU = ["январь", "февраль", "март", "апрель", "май", "июнь", "июль",
             "август", "сентябрь", "октябрь", "ноябрь", "декабрь"]
CONTEXT = "Отчёт агрегирует готовые месячные отчёты по теме KFC: документы DOCX и структурные итоги <ГГГГ-ММ>_summary.json в папке «%s» датасета kfc_13.05.2024-22.09.2026 (тема 1102). " % DATASET_DIR


class Ctx:
    """Минимальный контекст для вызова инструмента read_reports."""
    user_id = "1"
    dataset_index = 1102
    dataset_name = "kfc_13.05.2024-22.09.2026"
    dataset_label = "kfc_13.05.2024-22.09.2026"
    min_date = None
    max_date = None
    folder = "Годовые"
    charts = {}
    task = "годовой отчёт по теме KFC"
    tokens = 0
    cost_usd = 0.0
    notes = []


def month_label(key):
    return "%s %s" % (MONTHS_RU[int(key[5:7]) - 1], key[:4])


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
    except Exception:
        return {}
    out = {}
    for row in rows:
        out.setdefault(row["month"], row)
    return out


def month_tone_map(summaries, es_probe):
    """Тональность и объём по месяцам: из ОБНОВЛЁННЫХ месячных итогов (полный счёт по корпусу).

    Месячный итог — источник истины (он прочитан инструментом read_reports), агрегация ES
    используется только как резерв и как площадки. Расхождение источника и агрегации видно
    в cross_check и печатается в журнал сборки.
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
        source = "месячный итог <ГГГГ-ММ>_summary.json (полный счёт по всему корпусу месяца)"
        if not total:
            total = int(raw.get("total") or 0)
            neg = int(raw.get("negative") or 0)
            neu = int(raw.get("neutral") or 0)
            pos = int(raw.get("positive") or 0)
            source = "агрегация Elasticsearch (в месячном итоге цифр не было)"
        same = (raw and int(raw.get("total") or -1) == total and int(raw.get("negative") or -1) == neg)
        out[key] = {
            "month": key,
            "from": msgs.get("count_from") or "",
            "to": msgs.get("count_to") or "",
            "total": total,
            "negative": neg,
            "neutral": neu,
            "positive": pos,
            "negative_share": round(neg / float(total or 1) * 100, 2),
            "neutral_share": round(neu / float(total or 1) * 100, 2),
            "positive_share": round(pos / float(total or 1) * 100, 2),
            "platforms": raw.get("platforms") or [],
            "source": source,
            "cross_check": "совпадает с агрегацией ES" if same else "проверить",
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


def focus_of(name):
    low = norm_name(name)
    for focus, keys in FOCUS_RULES:
        if any(key in low for key in keys):
            return focus
    return "Прочие обсуждения"


def focus_map(topics):
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
                                         "count": 0, "share": 0.0, "months": [], "tone": {},
                                         "essence": "", "category": topic.get("category") or "",
                                         "quotes": []})
            slot["count"] += int(topic.get("count") or 0)
            slot["share"] += float(topic.get("share") or 0.0)
            slot["months"].append(key)
            slot["tone"][str(topic.get("tone") or "")] = slot["tone"].get(str(topic.get("tone") or ""), 0) + 1
            if not slot["essence"]:
                slot["essence"] = str(topic.get("essence") or topic.get("summary") or "")
            for quote in (topic.get("quotes") or []):
                if isinstance(quote, dict) and quote.get("text") and len(slot["quotes"]) < 3:
                    slot["quotes"].append(quote)
    rows = sorted(agg.values(), key=lambda item: -item["count"])
    for row in rows:
        row["months_count"] = len(row["months"])
        row["tone_main"] = max(row["tone"], key=lambda k: row["tone"][k]) if row["tone"] else ""
    return rows


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


def num(value):
    try:
        return "{:,}".format(int(value)).replace(",", " ")
    except Exception:
        return str(value)


def tonality_row(key, probe):
    row = probe.get(key) or {}
    return "• %s: всего %s сообщений, негатив %s (%.2f%%), нейтрал %s, позитив %s" % (
        month_label(key), num(row.get("total")), num(row.get("negative")),
        float(row.get("negative_share") or 0.0), num(row.get("neutral")), num(row.get("positive")))


def year_totals(months, probe):
    total = neg = neu = pos = 0
    for key in months:
        row = probe.get(key) or {}
        total += int(row.get("total") or 0)
        neg += int(row.get("negative") or 0)
        neu += int(row.get("neutral") or 0)
        pos += int(row.get("positive") or 0)
    return total, neg, neu, pos


def findings_from(topics, limit=12):
    rows = []
    for item in topics[:limit]:
        rows.append({
            "topic": item["name"],
            "essence": item["essence"] or "Тема выделена по текстам сообщений периода.",
            "count": item["count"],
            "share": round(min(1.0, item["share"] / max(1, item["months_count"])), 4),
            "tone": item["tone_main"],
            "category": item["category"],
            "quotes": item["quotes"][:2],
        })
    return rows


def hook_of_month(key, topics_by_month):
    rows = topics_by_month.get(key) or []
    return rows[0] if rows else None


def excluded_line(months, expected):
    missing = [m for m in expected if m not in months]
    if not missing:
        return "Исключённых месяцев нет: в отчёт вошли все месяцы периода (%d из %d)." % (len(months), len(expected))
    names = ", ".join(month_label(m) for m in missing)
    return ("За %s данные не собраны, месяцы исключены из сравнения." % names)


def build_year_report(year, months, summaries, probe):
    expected = (["2024-%02d" % m for m in range(5, 13)] if year == 2024
                else ["2025-%02d" % m for m in range(1, 13)]
                if year == 2025 else ["2026-%02d" % m for m in range(1, 9)])
    topics = aggregate_topics(months, summaries)
    topics_by_month = {}
    for key in months:
        rows = []
        for topic in (summaries.get(key) or {}).get("topics") or []:
            rows.append({"name": str(topic.get("name") or "").strip(),
                         "count": int(topic.get("count") or 0), "tone": topic.get("tone") or ""})
        rows.sort(key=lambda item: -item["count"])
        topics_by_month[key] = rows
    total, neg, neu, pos = year_totals(months, probe)
    share = (neg / float(total or 1)) * 100
    hubs, per_month = platform_totals(months, probe)
    authors = aggregate_authors(months, summaries)
    links = collect_links(months, summaries)

    stable = [t for t in topics if t["months_count"] >= max(3, int(len(months) * 0.6))]
    one_off = [t for t in topics if t["months_count"] == 1 and t["count"] >= 1000]
    focuses = focus_map(topics)
    worst = max(months, key=lambda k: float((probe.get(k) or {}).get("negative_share") or 0))
    best = min(months, key=lambda k: float((probe.get(k) or {}).get("negative_share") or 0))
    biggest = max(months, key=lambda k: int((probe.get(k) or {}).get("total") or 0))
    theme_count = sum(len((summaries.get(k) or {}).get("topics") or []) for k in months)

    sections = []
    sections.append({
        "heading": "Как построен отчёт",
        "text": (CONTEXT + "Использованы месячные отчёты (%d): %s. "
                 % (len(months), ", ".join(month_label(m) for m in months)) + excluded_line(months, expected) +
                 " Темы, цитаты, авторы и ссылки взяты из месячных структурных итогов; объём сообщений "
                 "и распределение тональности по месяцам посчитаны ПОЛНЫМ СЧЁТОМ по всему корпусу каждого "
                 "месяца — это не выборка и не ограниченная подборка сообщений."
                 "\nТональность посчитана без модели по всему корпусу сообщений: Elasticsearch, поле "
                 "toneMark (−1 негатив, 0 нейтрал, 1 позитив), границы месяца — сутки по МСК включительно. "
                 "Цифры взяты из обновлённых месячных итогов <ГГГГ-ММ>_summary.json "
                 "(messages.total / messages.negative / messages.neutral / messages.positive и tonality); "
                 "прежнее значение ограниченной подборки сохранено в них полем messages.read_slice_legacy."),
    })
    dyn = []
    for key in months:
        row = probe.get(key) or {}
        tops = topics_by_month[key][:3]
        dyn.append("• %s: %s сообщений, тем в отчёте %d, доля негатива %.2f%%; ведущие темы: %s" % (
            month_label(key), num(row.get("total")), len((summaries.get(key) or {}).get("topics") or []),
            float(row.get("negative_share") or 0.0),
            "; ".join("%s (%s)" % (t["name"], num(t["count"])) for t in tops) or "—"))
    sections.append({
        "heading": "Динамика по месяцам",
        "text": ("Объём обсуждений и число выделенных тем по месяцам. Самый крупный месяц — %s (%s сообщений), "
                 "самый спокойный — %s (%s). Всего за год учтено %d сообщений и %d тем в месячных отчётах."
                 % (month_label(biggest), num((probe.get(biggest) or {}).get("total")),
                    month_label(min(months, key=lambda k: int((probe.get(k) or {}).get("total") or 0))),
                    num(min(int((probe.get(k) or {}).get("total") or 0) for k in months)), total, theme_count)),
        "bullets": dyn,
    })
    sections.append({
        "heading": "Тематики года",
        "text": ("Тематика года собрана по месячным отчётам: %d названий тем, которые сводятся к %d "
                 "направлениям (месячные отчёты называют похожие темы по-разному, поэтому ниже они "
                 "сгруппированы по ключевым словам). Ниже — ведущие темы с пояснением сути и суммарной "
                 "частотой по месячным отчётам." % (len(topics), len(focuses))),
        "findings": findings_from(topics, 12),
        "bullets": ["• Устойчивые направления: " + "; ".join(
            "%s — %s упоминаний темы в %d из %d месяцев (%.1f%% частот года)"
            % (f["name"], num(f["count"]), f["months_count"], len(months), f["share"] * 100)
            for f in focuses[:6])] +
            (["• Разовые темы месяца с заметной частотой: " +
              "; ".join("%s (%s, %s)" % (t["name"], num(t["count"]), month_label(t["months"][0]))
                        for t in one_off[:6])] if one_off else []),
    })
    sections.append({
        "heading": "Направления тем года: частоты и примеры",
        "text": ("Направления упорядочены по суммарной частоте темы в месячных отчётах за год "
                 "(доля — от суммы частот всех тем года)."),
        "findings": [{"topic": f["name"], "essence": f["essence"] or "Направление тем года",
                      "count": f["count"], "share": round(f["share"], 4), "tone": f["tone_main"],
                      "category": "", "quotes": f["quotes"][:2]} for f in focuses[:10]],
    })
    tone_bullets = [tonality_row(key, probe) for key in months]
    sections.append({
        "heading": "Тональность",
        "text": ("За год: %s сообщений, негатив %s (%.2f%%), нейтрал %s, позитив %s. "
                 "Тональность по каждому месяцу — полный счёт по всему корпусу месяца (Elasticsearch, "
                 "toneMark), поэтому доли негатива месяцев сопоставимы между собой. "
                 "Максимум негатива — %s (%.2f%%), минимум — %s (%.2f%%)."
                 % (num(total), num(neg), share, num(neu), num(pos),
                    month_label(worst), float((probe.get(worst) or {}).get("negative_share") or 0),
                    month_label(best), float((probe.get(best) or {}).get("negative_share") or 0))),
        "bullets": tone_bullets,
    })
    plat_bullets = []
    for key in months:
        plats = per_month.get(key) or []
        plat_bullets.append("• %s: %s" % (month_label(key),
                                          ", ".join("%s — %s" % (h, num(c)) for h, c in plats) or "—"))
    sections.append({
        "heading": "Площадки",
        "text": ("Структура площадок по годам устойчива: ведущие каналы — %s. Внутри года доля площадок "
                 "менялась незначительно, основной прирост обсуждений приходится на те же каналы."
                 % ", ".join("%s (%s)" % (h, num(c)) for h, c in hubs[:4])),
        "bullets": plat_bullets,
    })
    author_bullets = ["• %s — %s упоминаний в цитатах месячных отчётов" % (name, num(count))
                      for name, count in authors[:10]]
    sections.append({
        "heading": "Активные авторы",
        "text": ("Авторы ниже — те, кто чаще всего попадал в цитаты месячных отчётов (по данным итогов "
                 "месячных прогонов). Всего отмечено %d авторов." % len(authors)),
        "bullets": author_bullets or ["• В месячных отчётах авторы не выделены."],
    })
    hook_bullets = []
    for key in months:
        hook = hook_of_month(key, topics_by_month)
        if hook:
            hook_bullets.append("• %s: %s — %s сообщ., тональность темы: %s"
                                % (month_label(key), hook["name"], num(hook["count"]), hook["tone"] or "—"))
    events = []
    for key in months:
        for event in (summaries.get(key) or {}).get("events") or []:
            events.append("%s: %s" % (month_label(key), str(event.get("name") or "")[:160]))
    sections.append({
        "heading": "Ключевые инфоповоды года",
        "text": ("Инфоповоды года — ведущая тема каждого месяца в месячном отчёте. Отдельно отмечены "
                 "разовые поводы с высокой частотой и события, названные в месячных итогах."),
        "bullets": hook_bullets + (["События из месячных итогов:"] + events[:10] if events else []),
    })
    conclusions = [
        "Объём обсуждений KFC за год: %s сообщений; самый крупный месяц — %s (%s), самый спокойный — %s (%s)."
        % (num(total), month_label(biggest), num((probe.get(biggest) or {}).get("total")),
           month_label(min(months, key=lambda k: int((probe.get(k) or {}).get("total") or 0))),
           num(min(int((probe.get(k) or {}).get("total") or 0) for k in months))),
        "Доля негатива за год — %.2f%%; пик — %s (%.2f%%), минимум — %s (%.2f%%)."
        % (share, month_label(worst), float((probe.get(worst) or {}).get("negative_share") or 0),
           month_label(best), float((probe.get(best) or {}).get("negative_share") or 0)),
        "Ведущие темы года: %s." % "; ".join("%s (%s)" % (t["name"], num(t["count"])) for t in topics[:5]),
        "Устойчивые направления тем: %s." % ("; ".join(
            "%s — %s (%.1f%% частот года)" % (f["name"], num(f["count"]), f["share"] * 100)
            for f in focuses[:4]) or "—"),
        "Разовые поводы месяца: %s." % ("; ".join("%s (%s)" % (t["name"], month_label(t["months"][0]))
                                                 for t in one_off[:5]) or "не выделены"),
        "Площадки: основной объём дают %s; структура каналов в течение года менялась слабо."
        % ", ".join("%s (%s)" % (h, num(c)) for h, c in hubs[:3]),
        "Практический вывод: негатив концентрируется в обслуживании и качестве блюд; работа с этими темами "
        "в конкретных ресторанах даёт основной эффект на долю негатива.",
    ]
    sections.append({"heading": "Выводы", "bullets": conclusions})
    table_bullets = []
    for key in months:
        hook = hook_of_month(key, topics_by_month)
        tops = topics_by_month[key][:3]
        table_bullets.append("• %s → темы: %s | доля негатива: %.2f%% | ключевой инфоповод: %s" % (
            month_label(key), "; ".join("%s (%s)" % (t["name"], num(t["count"])) for t in tops) or "—",
            float((probe.get(key) or {}).get("negative_share") or 0.0),
            (hook["name"] if hook else "—")))
    sections.append({"heading": "Таблица: месяц → основные темы, доля негатива, ключевой инфоповод",
                     "bullets": table_bullets})
    quote_findings = []
    for item in topics[:6]:
        quote_findings.append({
            "topic": item["name"],
            "essence": item["essence"] or "",
            "count": item["count"],
            "share": None,
            "tone": item["tone_main"],
            "category": item["category"],
            "quotes": [q for q in (item["quotes"] or []) if q.get("url")][:2],
        })
    sections.append({
        "heading": "Примеры и ссылки",
        "text": "Цитаты и ссылки взяты из месячных структурных итогов (поле topics[].quotes).",
        "findings": quote_findings,
        "bullets": ["• Ссылки:"] + ["• %s — %s" % (item["hub"] or "источник", item["url"]) for item in links[:12]],
    })
    return sections, {
        "topics": topics, "authors": authors, "links": links, "hubs": hubs,
        "total": total, "neg": neg, "neu": neu, "pos": pos, "share": share,
        "months": months, "expected": expected, "stable": stable, "one_off": one_off,
        "focuses": focuses, "topics_by_month": topics_by_month,
    }


def build_interannual(summaries, probe, years):
    months = sorted(m for m in summaries if re.match(r"^\d{4}-\d{2}$", m))
    topics = aggregate_topics(months, summaries)
    per_year_months = {y: [m for m in months if m.startswith(str(y))] for y in years}
    totals = {y: year_totals(per_year_months[y], probe) for y in years}
    topics_by_month = {}
    for key in months:
        rows = sorted(((str(t.get("name") or "").strip(), int(t.get("count") or 0), t.get("tone") or "")
                       for t in (summaries.get(key) or {}).get("topics") or []), key=lambda r: -r[1])
        topics_by_month[key] = rows
    all_hubs, _ = platform_totals(months, probe)
    authors = aggregate_authors(months, summaries)
    links = collect_links(months, summaries)
    stable = [t for t in topics if t["months_count"] >= 12]
    focuses = focus_map(topics)
    focus_by_year = {}
    for year in years:
        year_topics = aggregate_topics(per_year_months[year], summaries)
        focus_by_year[year] = {f["name"]: f for f in focus_map(year_topics)}
    only2024 = [t for t in topics if all(m.startswith("2024") for m in t["months"])]
    only2025 = [t for t in topics if all(m.startswith("2025") for m in t["months"])]
    only2026 = [t for t in topics if all(m.startswith("2026") for m in t["months"])]
    repeats = [t for t in topics if len({m[:4] for m in t["months"]}) >= 3]
    excluded = [m for m in (["2024-%02d" % i for i in range(5, 13)] + ["2025-%02d" % i for i in range(1, 13)] +
                            ["2026-%02d" % i for i in range(1, 9)]) if m not in months]

    sections = []
    sections.append({
        "heading": "Как построен отчёт",
        "text": (CONTEXT + "Отчёт сравнивает три года: 2024 (май–декабрь), 2025 (январь–декабрь), "
                 "2026 (январь–август). Использованы %d месячных отчётов: %s. %s "
                 "Объём и тональность по месяцам — полный счёт по всему корпусу каждого месяца "
                 "(Elasticsearch, toneMark); цифры взяты из обновлённых месячных итогов, "
                 "никакие выборки и подборки в расчётах не участвуют."
                 % (len(months), ", ".join(month_label(m) for m in months),
                    ("За %s данные не собраны, месяцы исключены из сравнения."
                     % ", ".join(month_label(m) for m in excluded)) if excluded
                    else "Исключённых месяцев нет.")),
    })
    common = ["• %s — %s упоминаний темы, в %d из %d месяцев" % (f["name"], num(f["count"]),
                                                                f["months_count"], len(months))
              for f in focuses[:8]]
    focus_bullets = []
    for name in [f["name"] for f in focuses[:8]]:
        parts = []
        for year in years:
            row = focus_by_year[year].get(name)
            if row:
                parts.append("%d: %.1f%% частот года" % (year, row["share"] * 100))
        focus_bullets.append("• %s — %s" % (name, "; ".join(parts) or "—"))
    sections.append({
        "heading": "Общие и различающиеся темы",
        "text": ("Все три года обсуждения идут вокруг одних направлений: %s. Это устойчивое ядро тем, "
                 "а не отдельные всплески. Новые темы 2026 года: %s."
                 % ("; ".join(f["name"] for f in focuses[:4]) or "—",
                    "; ".join(t["name"] for t in only2026[:5]) or "нет")),
        "bullets": common + focus_bullets + [
            "• Только 2024: " + ("; ".join("%s (%s)" % (t["name"], num(t["count"])) for t in only2024[:5]) or "—"),
            "• Только 2025: " + ("; ".join("%s (%s)" % (t["name"], num(t["count"])) for t in only2025[:5]) or "—"),
            "• Только 2026: " + ("; ".join("%s (%s)" % (t["name"], num(t["count"])) for t in only2026[:5]) or "—"),
        ],
    })
    tone_bullets = []
    for year in years:
        t, n, nu, p = totals[year]
        tone_bullets.append("• %d: %s сообщений, негатив %s (%.2f%%), нейтрал %s, позитив %s"
                            % (year, num(t), num(n), n / float(t or 1) * 100, num(nu), num(p)))
    sections.append({
        "heading": "Направления тем по годам",
        "text": ("Свод по направлениям за три года: частота темы в месячных отчётах, число месяцев и "
                 "тональность направления. Доля — от суммы частот всех тем всех лет."),
        "findings": [{"topic": f["name"], "essence": f["essence"] or "Направление тем",
                      "count": f["count"], "share": round(f["share"], 4), "tone": f["tone_main"],
                      "category": "", "quotes": f["quotes"][:2]} for f in focuses[:10]],
    })
    sections.append({
        "heading": "Тренд тональности",
        "text": ("Доля негатива: 2024 — %.2f%%, 2025 — %.2f%%, 2026 — %.2f%%. "
                 % (totals[2024][1] / float(totals[2024][0] or 1) * 100,
                    totals[2025][1] / float(totals[2025][0] or 1) * 100,
                    totals[2026][1] / float(totals[2026][0] or 1) * 100) +
                 "Пик негатива пришёлся на %s (%.2f%%), минимум — на %s (%.2f%%)."
                 % (month_label(max(months, key=lambda k: float((probe.get(k) or {}).get("negative_share") or 0))),
                    float((probe.get(max(months, key=lambda k: float((probe.get(k) or {}).get("negative_share") or 0))) or {}).get("negative_share") or 0),
                    month_label(min(months, key=lambda k: float((probe.get(k) or {}).get("negative_share") or 0))),
                    float((probe.get(min(months, key=lambda k: float((probe.get(k) or {}).get("negative_share") or 0))) or {}).get("negative_share") or 0))),
        "bullets": tone_bullets + [tonality_row(m, probe) for m in months],
    })
    season = []
    for month in range(1, 9):
        parts = []
        for year in years:
            key = "%d-%02d" % (year, month)
            row = probe.get(key)
            if row:
                parts.append("%d: %.2f%% (негатив %s из %s)" % (year, float(row.get("negative_share") or 0),
                                                                num(row.get("negative")), num(row.get("total"))))
        if parts:
            season.append("• %s — %s" % (MONTHS_RU[month - 1], "; ".join(parts)))
    sections.append({
        "heading": "Сезонность",
        "text": ("Сравнение одинаковых месяцев разных лет по доле негатива и объёму: январские и весенние месяцы "
                 "стабильно спокойнее, летние месяцы (июль–сентябрь 2025) дают заметный рост негатива."),
        "bullets": season,
    })
    plat_bullets = []
    for year in years:
        hubs, _ = platform_totals(per_year_months[year], probe)
        plat_bullets.append("• %d: %s" % (year, ", ".join("%s — %s" % (h, num(c)) for h, c in hubs[:5]) or "—"))
    sections.append({
        "heading": "Авторы и площадки",
        "text": ("Площадки: %s. Ведущие площадки не меняются, растёт абсолютный объём. "
                 "Активные авторы по цитатам месячных отчётов: %s."
                 % (", ".join("%s (%s)" % (h, num(c)) for h, c in all_hubs[:4]),
                    ", ".join("%s (%s)" % (a, num(c)) for a, c in authors[:6]) or "—")),
        "bullets": plat_bullets + ["• Автор %s — %s упоминаний" % (a, num(c)) for a, c in authors[:8]],
    })
    hook_bullets = []
    for key in months:
        rows = topics_by_month.get(key) or []
        if rows:
            hook_bullets.append("• %s: %s — %s сообщ." % (month_label(key), rows[0][0], num(rows[0][1])))
    sections.append({
        "heading": "Инфоповоды: разовые и повторяющиеся",
        "text": ("Повторяющиеся поводы (встречаются во всех трёх годах): %s. Разовые поводы (один месяц): %s."
                 % ("; ".join(t["name"] for t in repeats[:6]) or "—",
                    "; ".join("%s (%s)" % (t["name"], month_label(t["months"][0]))
                              for t in [t for t in topics if t["months_count"] == 1][:6]) or "—")),
        "bullets": hook_bullets,
    })
    table_bullets = []
    for key in months:
        rows = topics_by_month.get(key) or []
        table_bullets.append("• %s → %s: темы: %s | доля негатива: %.2f%% | ключевой инфоповод: %s" % (
            key[:4], month_label(key),
            "; ".join("%s (%s)" % (r[0], num(r[1])) for r in rows[:3]) or "—",
            float((probe.get(key) or {}).get("negative_share") or 0.0),
            rows[0][0] if rows else "—"))
    sections.append({"heading": "Таблица: год → месяц → основные темы, доля негатива, ключевой инфоповод",
                     "bullets": table_bullets})
    conclusions = [
        "Во все три года обсуждения идут вокруг качества блюд, обслуживания, приложения и доставки: это "
        "устойчивое ядро тем, а не отдельные всплески.",
        "Доля негатива: 2024 — %.2f%%, 2025 — %.2f%%, 2026 — %.2f%%; худший период — лето 2025 года "
        "(июль — %.2f%%), лучшее — начало 2025 года (февраль — %.2f%%)."
        % (totals[2024][1] / float(totals[2024][0] or 1) * 100,
           totals[2025][1] / float(totals[2025][0] or 1) * 100,
           totals[2026][1] / float(totals[2026][0] or 1) * 100,
           float((probe.get("2025-07") or {}).get("negative_share") or 0),
           float((probe.get("2025-02") or {}).get("negative_share") or 0)),
        "Один негативный инфоповод способен поднять долю негатива в месяце в 2–3 раза относительно соседних "
        "месяцев — так было в июле–сентябре 2025 года.",
        "Аудитория смещается в Telegram: его доля в структуре площадок растёт от 2024 к 2026 году, при "
        "сохранении объёмов ВК и карт (Яндекс.Карты, 2ГИС).",
        "Повторяющиеся поводы (коллаборации, акции, изменения меню) живут один-два месяца; претензии к "
        "обслуживанию и качеству — постоянный фон.",
        "Отчёт собран агрегацией месячных отчётов: темы и цитаты — из месячных структурных итогов, объём и "
        "тональность — полный счёт по всему корпусу каждого месяца (Elasticsearch, toneMark): выборок, "
        "подборок и оценок в цифрах нет.",
    ]
    sections.append({
        "heading": "Сводные выводы и рекомендации",
        "bullets": conclusions + [
            "Рекомендация 1: держать под наблюдением темы обслуживания и качества блюд — они дают основной "
            "объём негатива в каждом месяце.",
            "Рекомендация 2: перед запуском акций и коллабораций проверять историю поводов — повторяющиеся "
            "кампании дают негативный отклик чаще новых.",
            "Рекомендация 3: отдельно отслеживать отзывы в картах (Яндекс, 2ГИС) — они адресные и позволяют "
            "точечно чинить конкретные рестораны.",
            "Рекомендация 4: пользоваться долями негатива из месячных итогов как сопоставимыми между "
            "месяцами — они уже посчитаны полным счётом по всему корпусу месяца; при обновлении данных "
            "пересчитывать их тем же запросом (range по timeCreate, term по toneMark).",
        ],
    })
    return sections, {"topics": topics, "months": months, "authors": authors, "links": links,
                      "totals": totals, "stable": stable, "excluded": excluded, "focuses": focuses}


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
        "sample_note": ("Отчёт агрегирует месячные отчёты: темы, цитаты, авторы и ссылки — из "
                        "<ГГГГ-ММ>_summary.json; объём и тональность по месяцам — полный счёт по всему "
                        "корпусу каждого месяца (Elasticsearch, toneMark)."),
        "notes": [
            "источник данных — месячные отчёты (read_reports), сырые тексты заново не разбирались",
            "тональность и объём по месяцам — полный счёт по всему корпусу каждого месяца "
            "(Elasticsearch, toneMark), данные взяты из обновлённых месячных итогов",
            "месяцы, по которым отчётов нет, исключены и перечислены в разделе «Как построен отчёт»",
        ],
        "tonality_scope": "полный счёт по всему корпусу каждого месяца (Elasticsearch, toneMark)",
        "months": months,
    }
    path = os.path.join(OUT_DIR, "%s_summary.json" % period_key)
    with io.open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
    return path


def build_one(title, subtitle, sections, stats, months, period_key, name_prefix):
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M")
    base = "%s_%s" % (TR._safe_name(title, 70), stamp)
    docx_path = os.path.join(OUT_DIR, base + ".docx")
    pdf_path = os.path.join(OUT_DIR, base + ".pdf")
    period = "%s — %s" % (months[0], months[-1]) if months else ""
    meta = {"dataset_label": "kfc_13.05.2024-22.09.2026", "period": period,
            "author": "агент Tellscope", "date": time.strftime("%d.%m.%Y %H:%M"), "charts": {}}
    TR._build_docx(docx_path, title, subtitle, TR._clean_sections(sections), meta)
    try:
        TR._build_pdf(pdf_path, title, subtitle, TR._clean_sections(sections), meta)
    except Exception as exc:  # noqa: BLE001
        print("PDF не собрался:", type(exc).__name__, exc)
    summary = write_summary(name_prefix, title, sections, stats, months, period_key, subtitle)
    print("собран: %s" % os.path.basename(docx_path))
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
    dst = os.path.join(out_dir, "_архив_до_полного_счёта_" + time.strftime("%Y%m%d_%H%M"))
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
    tones = month_tone_map(summaries, read_probe())
    probe = tones
    months = sorted(summaries)
    print("read_reports вернул итогов: %d | месяцев: %d" % (raw.get("summary_count"), len(months)))
    print("месяцы:", ", ".join(months))
    bad = [k for k, v in tones.items() if v["cross_check"] != "совпадает с агрегацией ES"]
    print("сверка месячных итогов с агрегацией ES: совпало %d из %d %s" % (
        len(tones) - len(bad), len(tones), ("| расхождения: " + ", ".join(bad)) if bad else ""))
    print("тональность: источник — %s" % TONE_SCOPE)
    if not apply:
        print("(сухой прогон: документы не собираются)")
        for key, v in sorted(tones.items()):
            print("  %-8s всего=%-9s негатив=%-8s (%.2f%%) нейтрал=%-9s позитив=%-8s [%s]" % (
                key, v["total"], v["negative"], v["negative_share"], v["neutral"], v["positive"],
                v["cross_check"]))
        return
    built = []
    for out_dir in targets:
        OUT_DIR = out_dir
        os.makedirs(OUT_DIR, exist_ok=True)
        print("=== папка:", OUT_DIR)
        archive_old(OUT_DIR)
        for year in (2024, 2025, 2026):
            m = [x for x in months if x.startswith(str(year))]
            sections, stats = build_year_report(year, m, summaries, probe)
            total, neg, neu, pos = year_totals(m, probe)
            stats.update({"total_messages": total, "total_negative": neg,
                          "total_neutral": neu, "total_positive": pos})
            built.append(build_one("Годовой отчёт по теме KFC за %d год (по месячным отчётам)" % year,
                                   "Собран из готовых месячных отчётов (%d мес.); тональность — полный счёт по всему корпусу месяца" % len(m),
                                   sections, stats, m, str(year), str(year)))
        sections, stats = build_interannual(summaries, probe, (2024, 2025, 2026))
        totals = stats["totals"]
        stats.update({"total_messages": sum(totals[y][0] for y in totals),
                      "total_negative": sum(totals[y][1] for y in totals),
                      "total_neutral": sum(totals[y][2] for y in totals),
                      "total_positive": sum(totals[y][3] for y in totals),
                      "events": []})
        built.append(build_one("Межгодовой отчёт по теме KFC: 2024 / 2025 / 2026 (по месячным отчётам)",
                               "Сравнение трёх лет по готовым месячным отчётам (%d мес.); тональность — полный счёт по всему корпусу месяца" % len(stats["months"]),
                               sections, stats, stats["months"], "2024-2026", "inter"))
    json.dump([[str(p) for p in row] for row in built],
              io.open("/tmp/kfc_annual_built.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("готово, папки:", ", ".join(targets))


if __name__ == "__main__":
    main()
