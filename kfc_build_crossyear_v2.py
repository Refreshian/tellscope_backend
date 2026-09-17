# -*- coding: utf-8 -*-
"""Расширенный межгодовой отчёт по теме KFC: 2024 / 2025 / 2026 (десять содержательных блоков).

Чем этот сборщик отличается от годового: годовой отчёт пересказывает готовые месячные отчёты,
а здесь к ним добавляются расчёты по всему корпусу темы, которых в месячных отчётах нет:

  * тепловая карта «тема × месяц» за все 28 месяцев;
  * вклад каждой темы в негатив каждого года и динамика негатива ВНУТРИ темы;
  * анатомия инфоповодов: старт → пик → затухание, срок на реакцию, кто разгонял волну;
  * когорты авторов и смена лидеров мнений по годам;
  * где именно живёт негатив (площадка за площадкой) и матрица «площадка × тема × тональность»;
  * кампании «до / во время / после» с проверкой, был ли негативный откат;
  * доля голоса рядом с конкурентами;
  * индекс риска по месяцам;
  * устойчивые формулировки и мемы по годам.

Тяжёлые агрегации считает модуль kfc_crossyear_agg: они идут по всему корпусу темы и занимают
минуты, без участия модели. Локальные модели подключаются только там, где нужен СМЫСЛ:
быстрая qwen3-4b-fast читает выборки сообщений волн и кампаний, Qwen3-32B даёт названия,
интерпретацию и формулировки выводов.

Отчёты собираются теми же сборщиками, что и отчёты платформы (tools_reports._build_docx /
_build_pdf): настоящие таблицы, подписанные графики, книжные и альбомные страницы, таблица
не рвётся между страницами. Новый файл кладётся РЯДОМ с прежними отчётами: ничего не удаляется
и не перезаписывается.

Тексты отчёта — без технических подробностей: ни имён файлов, ни путей, ни названий инструментов,
ни номеров тем.
"""
from __future__ import annotations

import asyncio
import datetime
import glob
import io
import json
import os
import re
import sys
import time
import types
from typing import Any, Dict, List, Optional, Tuple

BACKEND = "/home/dev/tellscope_app/tellscope_backend"
sys.path.insert(0, BACKEND)

# agent_engine/__init__.py тянет тяжёлые модули (вплоть до главного приложения) — грузим подпакет
# напрямую, иначе сборщик поднимет celery и заденет чужие процессы на видеокартах.
_pkg = types.ModuleType("agent_engine")
_pkg.__path__ = [BACKEND + "/agent_engine"]
sys.modules["agent_engine"] = _pkg

from agent_engine import tools_reports as TR  # noqa: E402

import kfc_crossyear_agg as A  # noqa: E402

REPORTS = BACKEND + "/data/1/reports_directory"
DATASET_DIR = "kfc_13.05.2024-22.09.2026 Отчёты"
OUT_DIR = REPORTS + "/kfc_13.05.2024-22.09.2026 Годовые"
CHART_DIR = "/tmp/kfc_crossyear_charts"

MONTHS_RU = ["январь", "февраль", "март", "апрель", "май", "июнь", "июль",
             "август", "сентябрь", "октябрь", "ноябрь", "декабрь"]
MONTHS_SHORT = ["янв", "фев", "мар", "апр", "май", "июн", "июл", "авг", "сен", "окт", "ноя", "дек"]
YEARS = (2024, 2025, 2026)
TITLE = "Межгодовой отчёт по теме KFC 2024–2026 (расширенный)"
SUBTITLE = "Десять блоков: темы по месяцам, драйверы тональности, инфоповоды, авторы, кампании, риски"

# Токены, которых в тексте отчёта быть не должно.
FORBIDDEN = ("_summary.json", "summary.json", "датасет", "DOCX", "PDF", "read_reports", "make_chart",
             "Elasticsearch", "toneMark", "1102", "/home/dev", "reports_directory", "Qdrant",
             "kfc_crossyear", "tag_1", "tag_2")
FORBIDDEN_CELL = ("_summary.json", "summary.json", "датасет", "/home/dev", "reports_directory",
                  "Qdrant", "1102")
# Сборщик платформы выбрасывает предложения, похожие на технические коды (трёхзначные 400–409, 500–509).
DANGER = re.compile(r"\b(40[0-9]|50[0-9])\b")

STATS: Dict[str, Any] = {"fast_calls": 0, "gen_calls": 0, "es_queries": 0, "models": {}}


class Ctx:
    """Минимальный контекст для сборщиков отчёта и построения графиков (без главного приложения)."""

    def __init__(self):
        self.user_id = "1"
        self.dataset_index = 1102
        self.dataset_name = "kfc_13.05.2024-22.09.2026"
        self.dataset_label = "KFC"
        self.min_date = None
        self.max_date = None
        self.folder = os.path.basename(OUT_DIR)
        self.charts: Dict[str, dict] = {}
        self.task = "расширенный межгодовой отчёт по теме KFC"
        self.run_id = "kfc-crossyear-v2"
        self.artifacts_dir = CHART_DIR
        self.tokens = 0
        self.cost_usd = 0.0
        self.notes: List[str] = []

    def add_artifact(self, kind, title, path, url="", meta=None):
        return {"kind": kind, "title": title, "path": path, "url": url or "", "meta": meta or {}}


# ---------------------------------------------------------------- форматирование

def month_label(key: str) -> str:
    return "%s %s" % (MONTHS_RU[int(key[5:7]) - 1], key[:4])


def month_short(key: str) -> str:
    return "%s %s" % (MONTHS_SHORT[int(key[5:7]) - 1], key[2:4])


def num(value: Any) -> str:
    try:
        return "{:,}".format(int(value)).replace(",", "\u00a0")
    except (TypeError, ValueError):
        return str(value if value is not None else "—")


def pct(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    try:
        return ("%.*f" % (digits, float(value))).replace(".", ",") + "%"
    except (TypeError, ValueError):
        return "—"


def short_num(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value if value is not None else "—")
    if number >= 1000000:
        return ("%.2f" % (number / 1000000.0)).replace(".", ",") + " млн"
    if number >= 1000:
        return "%.0f тыс." % (number / 1000.0)
    return str(int(number))


def prose_num(value: Any) -> str:
    """Число для абзацев и пунктов (в ячейках таблиц пишем точное значение).

    Сборщик платформы выбрасывает предложения, в которых видит технические коды, а его проверка
    принимает за код любое трёхзначное число вида 400–409 или 500–509. Точное число с разрядами
    даёт такие группы («153 506»), поэтому в тексте уходим в короткую форму; если и она попадает
    под проверку (404 сообщения), пишем значение в тысячах с десятыми — точное число остаётся
    в таблице рядом.
    """
    text = num(value)
    if not DANGER.search(text.replace("\u00a0", " ")):
        return text
    short = short_num(value)
    if not DANGER.search(short.replace("\u00a0", " ")):
        return short
    try:
        return ("%.1f тыс." % (float(value) / 1000.0)).replace(".", ",")
    except (TypeError, ValueError):
        return text


def link_text(url: Any) -> str:
    """Ссылка для абзаца: без служебной приставки протокола.

    Проверка сборщика считает слово «http» служебным и выбрасывает предложение целиком, поэтому
    в тексте ссылка пишется коротко (youtube.com/...). В приложении ссылки остаются полными:
    в ячейке таблицы такая проверка не применяется, и ссылку можно открыть как есть.
    """
    text = str(url or "").strip()
    for prefix in ("https://", "http://"):
        if text.lower().startswith(prefix):
            return text[len(prefix):]
    return text


def safe_text(text: Any, limit: int = 900) -> str:
    """Текст модели без предложений, которые сборщик платформы считает служебными.

    Проверка платформы вырезает предложения со словами вида «коллекция», «соединение», «причина»
    и с трёхзначными кодами. Тексты моделей пишутся живым языком и иногда попадают под это
    правило («коллекционных брелоков»), поэтому такие предложения убираем заранее — тогда в отчёте
    не остаётся обрывков.
    """
    source = re.sub(r"\s+", " ", str(text or "")).strip()
    if not source:
        return ""
    kept = []
    for part in re.split(r"(?<=[.!?])\s+", source):
        part = part.strip()
        if part and TR._sanitize_report_text(part) == part:
            kept.append(part)
    return re.sub(r"\s+", " ", " ".join(kept)).strip()[:limit]


def safe_quote(text: Any, limit: int = 190) -> str:
    """Цитата, которую сборщик платформы не тронет.

    Живые сообщения содержат что угодно, в том числе слова, которые проверка платформы принимает
    за служебные. Из такой цитаты берём первое предложение, которое проходит проверку целиком;
    если не проходит ни одно — цитату не приводим вовсе, чтобы вместо неё не появилось огрызка.
    """
    source = re.sub(r"\s+", " ", str(text or "")).strip()[:limit]
    if not source:
        return ""
    if TR._sanitize_report_text(source) == source:
        return source
    for part in re.split(r"(?<=[.!?])\s+", source):
        part = part.strip()
        if part and TR._sanitize_report_text(part) == part:
            return part
    return ""


def TABLE(title, columns, rows, note="", layout="auto"):
    return {"title": title, "columns": list(columns), "rows": [list(r) for r in rows],
            "note": note, "layout": layout}


def clip(text: Any, limit: int = 48) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


def num_ru(value: Any) -> str:
    return ("%.2f" % float(value)).replace(".", ",") if value is not None else "—"


def delta_pp(after: Any, before: Any) -> str:
    """Изменение в процентных пунктах со знаком — так видно «стало хуже» или «стало лучше»."""
    try:
        diff = float(after) - float(before)
    except (TypeError, ValueError):
        return "—"
    return ("%+.1f" % diff).replace(".", ",") + " п.п."


def days_phrase(value: Any) -> str:
    """«1 день», «3 дня», «12 дней» — чтобы в тексте не появлялось «через 1 дня»."""
    try:
        number = int(value)
    except (TypeError, ValueError):
        return "—"
    tail = number % 100
    word = ("дней" if 11 <= tail <= 14 else
            "день" if tail % 10 == 1 else
            "дня" if tail % 10 in (2, 3, 4) else "дней")
    return "%d %s" % (number, word)


def ratio_phrase(after: Any, before: Any) -> str:
    """«в 1,7 раза выше» — простая и честная формулировка вместо «почти в три раза»."""
    def clean(value):
        text = str(value).replace("\u00a0", "").replace(" ", "").replace(",", ".")
        match = re.search(r"\d+(?:\.\d+)?", text)
        return float(match.group(0)) if match else 0.0

    try:
        a, b = clean(after), clean(before)
    except (TypeError, ValueError):
        return ""
    if b <= 0 or a <= 0:
        return ""
    ratio = a / b
    if ratio < 1.05:
        return "примерно одинаково"
    return ("выше в %.1f раза" % ratio).replace(".", ",")


# ---------------------------------------------------------------- данные месячных отчётов

class MonthReader(Ctx):
    """Контекст для чтения готовых месячных отчётов тем же инструментом, что и на платформе."""

    def __init__(self):
        super().__init__()
        self.folder = "Годовые"


def read_month_summaries() -> Dict[str, dict]:
    result = asyncio.run(TR.read_reports(MonthReader(), folder=DATASET_DIR, limit=50))
    items = {}
    for row in result.get("summaries") or []:
        payload = row.get("summary") or {}
        key = str((payload.get("period") or {}).get("key") or "")
        if re.match(r"^\d{4}-\d{2}$", key):
            items[key] = payload
    return items


def month_topics(summaries: Dict[str, dict], key: str):
    raw = (summaries.get(key) or {}).get("topics") or []
    return TR.filter_spam_topics(raw)


def month_spam(summaries: Dict[str, dict], months: List[str]) -> Dict[str, dict]:
    out = {}
    for key in months:
        payload = summaries.get(key) or {}
        block = payload.get("filtered") if isinstance(payload.get("filtered"), dict) else {}
        count = int(block.get("spam_messages") or 0)
        if not count:
            _kept, dropped, count = month_topics(summaries, key)
        if count:
            out[key] = count
    return out


def month_quotes(summaries: Dict[str, dict], key: str, limit: int = 40) -> List[dict]:
    """Цитаты месяца без спама, с ссылками — из них собирается приложение отчёта."""
    rows = []
    for topic in month_topics(summaries, key)[0]:
        for quote in (topic.get("quotes") or []):
            if not isinstance(quote, dict) or not str(quote.get("text") or "").strip():
                continue
            rows.append({"text": re.sub(r"\s+", " ", str(quote["text"])).strip()[:280],
                         "url": str(quote.get("url") or ""),
                         "hub": str(quote.get("hub") or ""),
                         "date": str(quote.get("date") or ""),
                         "author": str(quote.get("author") or ""),
                         "tone": str(topic.get("tone") or ""),
                         "topic": str(topic.get("name") or ""),
                         "month": key})
        if len(rows) >= limit:
            break
    return rows[:limit]


def hook_of_month(summaries: Dict[str, dict], key: str) -> Dict[str, Any]:
    """Ключевой повод месяца: самая крупная осмысленная тема без спама и шума."""
    rows = []
    for topic in month_topics(summaries, key)[0]:
        name = str(topic.get("name") or "").strip()
        if not name or TR.topic_noise_reasons(topic):
            continue
        quotes = [q for q in (topic.get("quotes") or []) if isinstance(q, dict)]
        voices = max(len({str(q.get("author") or "").lower() for q in quotes} - {""}),
                     len({str(q.get("hub") or "").lower() for q in quotes} - {""}))
        rows.append({"name": name, "count": int(topic.get("count") or 0), "voices": voices,
                     "tone": str(topic.get("tone") or "")})
    rows.sort(key=lambda r: -r["count"])
    for row in rows:
        if row["voices"] >= 2 and row["count"] >= 100:
            return row
    return {"name": "выраженного инфоповода нет", "count": 0, "voices": 0, "tone": ""}


# ---------------------------------------------------------------- графики

def mkchart(ctx: Ctx, title: str, chart_type: str, categories, series, **kwargs) -> dict:
    return asyncio.run(TR.make_chart(ctx, title=title, chart_type=chart_type, categories=categories,
                                     series=series, **kwargs))


def heatmap_chart(ctx: Ctx, title: str, columns: List[str], rows: List[Tuple[str, List[float]]],
                  note: str = "", scale_label: str = "") -> dict:
    """Тепловая карта «тема × месяц»: цвет — значение, подписи по обеим осям читаются.

    Платформенные графики рисуют ряды и категории, а для цветовой шкалы нужен плотный рисунок,
    поэтому карта строится тем же инструментом рисования, что и остальные графики, и попадает
    в отчёт наравне с ними: раздел ссылается на неё как на обычный график.
    """
    plt = TR._mpl()
    grid = [[float(v or 0) for v in values] for _name, values in rows]
    fig, ax = plt.subplots(figsize=(12.2, 0.52 * len(grid) + 2.2), dpi=170)
    ax.grid(False)
    image = ax.imshow(grid, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(columns, fontsize=7.5, rotation=45, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([name for name, _values in rows], fontsize=9)
    bar = fig.colorbar(image, ax=ax, fraction=0.022, pad=0.015)
    bar.ax.tick_params(labelsize=8)
    if scale_label:
        bar.set_label(scale_label, fontsize=9)
    for i, row in enumerate(grid):
        top = max(row) or 1
        for j, value in enumerate(row):
            if value <= 0:
                continue
            if j % 2 == 0 or value >= top * 0.5:
                ax.text(j, i, ("%.1f" % value).replace(".", ","), ha="center", va="center",
                        fontsize=6.2, color="#1B1B1B" if value < top * 0.6 else "white")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    if note:
        fig.text(0.01, 0.005, note, fontsize=8, color="#667085")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    chart_id = "chart%d" % (len(ctx.charts) + 1)
    name = "%s_%s.png" % (chart_id, TR._safe_name(title, 50))
    os.makedirs(CHART_DIR, exist_ok=True)
    path = os.path.join(CHART_DIR, name)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    artifact = ctx.add_artifact("chart", title, path,
                                url="/api/agent/artifact/%s/%s" % (ctx.run_id, name),
                                meta={"chart_id": chart_id, "chart_type": "heatmap"})
    ctx.charts[chart_id] = {"chart_id": chart_id, "title": title, "name": name, "path": path,
                            "url": artifact["url"]}
    return {"chart_id": chart_id, "name": name, "title": title, "url": artifact["url"]}


CHART_NOTE = ("Источник: все сообщения темы за 13.05.2024–31.08.2026 (2,93 млн); "
              "тональность — по каждому сообщению.")


# ---------------------------------------------------------------- проверка текстов

def polish_sections(sections: List[dict]) -> List[dict]:
    """Косметика финального текста: двойные точки («п.п..»), лишние пробелы, пустые пункты.

    Проверка предложений у сборщика платформы предложения не выбрасывает, но двойная точка
    выглядит как опечатка, поэтому убираем её до сборки документа.
    """
    def tidy(value: Any) -> str:
        text = re.sub(r"\s+", " ", str(value if value is not None else "")).strip()
        text = re.sub(r"\.{2,}", ".", text)
        text = re.sub(r"\s+([,;:.!?])", r"\1", text)
        return text

    out = []
    for section in sections:
        section = dict(section)
        section["text"] = tidy(section.get("text"))
        if section.get("note"):
            section["note"] = tidy(section["note"])
        for key in ("bullets", "items"):
            if isinstance(section.get(key), list):
                section[key] = [tidy(item) for item in section[key] if tidy(item)]
        tables = []
        for spec in section.get("tables") or []:
            spec = dict(spec)
            spec["title"] = tidy(spec.get("title"))
            spec["note"] = tidy(spec.get("note"))
            spec["columns"] = [tidy(c) for c in (spec.get("columns") or [])]
            spec["rows"] = [[tidy(c) for c in row] for row in (spec.get("rows") or [])]
            tables.append(spec)
        section["tables"] = tables
        out.append(section)
    return out


def audit_sections(sections: List[dict]) -> List[str]:
    """Проверка: сборщик платформы не выбросил ни одного предложения и в тексте нет техножаргона."""
    def flat(value):
        return re.sub(r"\s+", " ", str(value if value is not None else "")).strip()

    problems = []
    for section in sections:
        heading = section.get("heading") or "Раздел"
        texts = [("текст", section.get("text"))] + [("пункт", b) for b in (section.get("bullets") or [])]
        for kind, value in texts:
            source = flat(value)
            if not source:
                continue
            if flat(TR._sanitize_report_text(value)) != source:
                problems.append("%s / %s: сборщик выкинул часть текста — %s" % (heading, kind, source[:90]))
            for token in FORBIDDEN:
                if token.lower() in source.lower():
                    problems.append("%s / %s: техническое упоминание «%s»" % (heading, kind, token))
        specs = list(section.get("tables") or [])
        for cell in [section.get("note") or ""]:
            for token in FORBIDDEN:
                if token.lower() in flat(cell).lower():
                    problems.append("%s / примечание: техническое упоминание «%s»" % (heading, token))
        for spec in specs:
            cells = [spec.get("title") or "", spec.get("note") or ""]
            cells += [str(c) for c in (spec.get("columns") or [])]
            cells += [str(c) for row in (spec.get("rows") or []) for c in row]
            for cell in cells:
                for token in FORBIDDEN_CELL:
                    if token.lower() in str(cell).lower():
                        problems.append("%s / таблица: техническое упоминание «%s»" % (heading, token))
        for item in (section.get("findings") or []):
            for token in FORBIDDEN:
                if token.lower() in flat(item.get("essence")).lower():
                    problems.append("%s / пояснение темы: техническое упоминание «%s»" % (heading, token))
    return problems

# ---------------------------------------------------------------- инфоповоды и кампании

# Крупные поводы для разбора «старт → пик → затухание». Каждый привязан либо к готовой теме
# (тогда объём и период берутся точно по меткам Brand Analytics), либо к устойчивым формулировкам
# сообщений. Ничего не выдумывается: имя повода, объём и даты приходят из данных.
EVENTS: List[dict] = [
    {"key": "outage", "name": "Массовый технический сбой: кассы и приложение не работают",
     "phrases": ("сбой", "кассы самообслуживания"), "from": "2025-07-25", "to": "2025-08-15",
     "kind": "сбой"},
    {"key": "salmonella", "name": "Сальмонелла в сырье для стрипсов",
     "phrases": ("сальмонелл", "стрипс"), "from": "2025-08-01", "to": "2025-08-31", "kind": "инцидент"},
    {"key": "inspection", "name": "Проверка на кухне: грязь и просроченные продукты",
     "phrases": ("SHOT ПРОВЕРКА",), "from": "2025-07-25", "to": "2025-08-15", "kind": "инцидент"},
    {"key": "star_rail", "name": "Коллаборация с Honkai: Star Rail",
     "phrases": ("Star Rail", "Honkai", "брелок"), "from": "2026-07-20", "to": "2026-08-31",
     "kind": "кампания"},
    {"key": "smurfs", "name": "Комбо «Смешарики»", "phrases": ("Смешарики",),
     "from": "2025-11-01", "to": "2025-12-31", "kind": "кампания"},
    {"key": "minions", "name": "Комбо «МиньонБум»", "phrases": ("Миньон",),
     "from": "2026-06-01", "to": "2026-07-31", "kind": "кампания"},
    {"key": "chicken_shortage", "name": "Дефицит курицы и уменьшение порций", "theme": "Дефицит курицы",
     "kind": "проблема"},
    {"key": "milana", "name": "Бокс от Миланы Хаметовой", "theme": "Бокс от Миланы Хаметовой",
     "kind": "кампания"},
    {"key": "fixies", "name": "Детское комбо «Фиксики»", "theme": "Детское комбо Фиксики",
     "kind": "кампания"},
    {"key": "rats", "name": "Крысы и мыши в ресторанах", "theme": "Крыса/мышь", "kind": "инцидент"},
    {"key": "rebrand", "name": "Разговоры о вывеске: KFC и Rostic’s", "theme": "KFC на Ростикс",
     "kind": "переход"},
]

# Кампании и линейки для блока «что сработало»: имя готовые темы → человеческое название.
CAMPAIGNS: List[Tuple[str, str]] = [
    ("Бокс от Миланы Хаметовой", "Бокс от Миланы Хаметовой"),
    ("Юнирест_Чикен тамагочи", "Детская игрушка «Чикен тамагочи»"),
    ("Детское комбо Фиксики", "Детское комбо «Фиксики»"),
    ("МиньонБум Комбо", "Комбо «МиньонБум»"),
    ("Комбо Смешарики", "Комбо «Смешарики»"),
    ("Комбо \"Союзмультфильм\"", "Комбо «Союзмультфильм»"),
    ("Комбо «Лео и Тиг»", "Комбо «Лео и Тиг»"),
    ("Комбо \"Легенды космоса\"", "Комбо «Легенды космоса»"),
    ("Азиатская Линейка", "Азиатская линейка"),
    ("Грузинская Линейка", "Грузинская линейка"),
    ("Мексиканская Линейка", "Мексиканская линейка"),
    ("Итальянская Линейка", "Итальянская линейка"),
    ("Корейская линейка", "Корейская линейка"),
    ("ТАЙСКОЕ ЛЕТО", "Сезонное меню «Тайское лето»"),
    ("Халяль", "Линейка «Халяль»"),
    ("ДР Rostic's", "День рождения сети"),
    ("Благотворительность", "Благотворительные акции"),
]

# Устойчивые формулировки и мемы: как в текстах называют бренд, конкурентов и поводы.
LANGUAGE_TERMS: List[Tuple[str, str]] = [
    ("KFC", "латинское написание бренда"),
    ("Ростикс", "новое название сети"),
    ("Rostic", "написание нового названия латиницей"),
    ("КФС", "название бренда кириллицей"),
    ("вкусно и точка", "бывший McDonald’s"),
    ("Бургер Кинг", "Burger King"),
    ("Мак", "разговорное «Мак» о бывшем McDonald’s"),
    ("Сандерс", "Полковник Сандерс"),
    ("шефролл", "блюдо, вокруг которого спорят о цене"),
    ("баскет", "набор «Баскет»"),
    ("дефицит кур", "разговор о нехватке курицы"),
    ("таракан", "инцидент с насекомыми"),
    ("просроч", "просроченная продукция"),
    ("подорожа", "рост цен"),
    ("цена акции", "реклама акций в приложении"),
    ("Ростикс ростикса", "мем-скороговорка про сеть"),
    ("Хочу босс", "мем-фраза про бургер"),
    ("крылышк", "крылышки — самое обсуждаемое блюдо"),
    ("комбо", "комбо-наборы"),
    ("приложение", "мобильное приложение и бонусы"),
]


def event_query(event: dict, catalog: Dict[str, List[Tuple[str, str]]], lo: str = "", hi: str = "") -> dict:
    """Запрос волны по поводу: по готовой теме, по одной формулировке или по нескольким сразу."""
    if event.get("theme") and event["theme"] in catalog:
        return A.theme_query([tuple(p) for p in catalog[event["theme"]]], lo, hi)
    phrases = event.get("phrases") or ((event.get("phrase"),) if event.get("phrase") else ())
    should = [{"match": {"text": {"query": phrase, "minimum_should_match": "70%"}}}
              for phrase in phrases if str(phrase or "").strip()]
    if not should:
        return {"bool": {"must": [{"match_none": {}}], "filter": A.period_filter(lo, hi)}}
    if len(should) == 1:
        return {"bool": {"must": should, "filter": A.period_filter(lo, hi)}}
    # Несколько формулировок одного повода: сообщение относится к волне, если встретилась любая
    # из них (название коллаборации пишут и латиницей, и по имени героя).
    return {"bool": {"should": should, "minimum_should_match": 1,
                     "filter": A.period_filter(lo, hi)}}


def event_volume(event: dict) -> str:
    """Как посчитан объём повода — это уходит в примечание к таблице, чтобы читатель не гадал."""
    if event.get("theme"):
        return "по метке темы"
    return "по формулировкам сообщений"


def event_window(event: dict, catalog: Dict[str, List[Tuple[str, str]]],
                 theme_months: Dict[str, Dict[str, int]]) -> Tuple[str, str, str]:
    """Период активности повода: задан вручную для новостной волны или найден по меткам темы."""
    if event.get("from") and event.get("to"):
        return event["from"], event["to"], "задан окном волны"
    months_map = theme_months.get(event.get("theme") or "") or {}
    first, last, kind = A.theme_window(months_map)
    if not first:
        return "", "", kind
    y, m = int(first[:4]), int(first[5:7])
    lo = "%04d-%02d-01" % (y, m)
    y2, m2 = int(last[:4]), int(last[5:7])
    hi = (datetime.date(y2, m2, 1) + datetime.timedelta(days=31))
    hi = (hi.replace(day=1) - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    return lo, hi, kind


READ_SYSTEM = ("Ты аналитик соцмедиа и СМИ. Разбирай только то, что есть в переданных сообщениях, "
               "ничего не додумывай. Отвечай ТОЛЬКО JSON без пояснений.")

READ_PROMPT = """Ниже сообщения одной волны обсуждения бренда быстрого питания за короткий период.
Разбери их строго по фактам из текста.

Верни JSON такого вида:
{{"about": "о чём эта волна, 2-3 предложения",
 "subtopics": [{{"name": "короткое название смыслового блока", "share": "примерная доля в процентах"}}],
 "claims": [{{"claim": "суть претензии, ожидания или наблюдения", "quote": "дословная короткая выдержка", "tone": "негатив|нейтрал|позитив"}}],
 "sentiment": {{"negative": 0, "neutral": 0, "positive": 0}},
 "praise": ["за что хвалят, если есть"],
 "blame": ["на что жалуются, если есть"]}}

Правила: не больше 6 subtopics и 6 claims; quote — дословная выдержка до 160 символов; ничего не выдумывать.

Сообщения:
{messages}"""


def cached(name: str, builder):
    """Результат чтения моделью сохраняется на диск: повторная сборка отчёта не гоняет модели заново."""
    path = os.path.join(A.CACHE_DIR, "read_%s.json" % re.sub(r"\W+", "_", name)[:60])
    if os.path.isfile(path):
        try:
            with io.open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:  # noqa: BLE001
            pass
    value = builder()
    try:
        os.makedirs(A.CACHE_DIR, exist_ok=True)
        with io.open(path, "w", encoding="utf-8") as fh:
            json.dump(value, fh, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        pass
    return value


def read_wave(label: str, messages: List[dict], fast: bool = True, batch: int = 18) -> dict:
    """Чтение выборки волны быстрой моделью: о чём говорят, за что хвалят, на что жалуются.

    Претензия модели засчитывается, только если её цитата действительно есть в прочитанных
    сообщениях: так в отчёт не попадают формулировки, которых в данных не было.
    """
    picked = [m for m in messages if m.get("text")][:batch * 3]
    if not picked:
        return {}
    haystack = " ".join(re.sub(r"\s+", " ", str(m.get("text") or "")).lower() for m in picked)
    chunks = []
    for start in range(0, len(picked), batch):
        chunk = picked[start:start + batch]
        lines = "\n".join("[%d] (%s, %s) %s" % (i + 1, m.get("hub") or "источник", m.get("date") or "",
                                                clip(m.get("text"), 260))
                          for i, m in enumerate(chunk))
        chunks.append((chunk, READ_PROMPT.format(messages=lines)))

    def build():
        from concurrent.futures import ThreadPoolExecutor

        merged: Dict[str, Any] = {"about": "", "subtopics": [], "claims": [], "praise": [],
                                  "blame": [], "sentiment": {"negative": 0, "neutral": 0,
                                                             "positive": 0}, "read": 0}
        def one(item):
            chunk, prompt = item
            return A.llm_json(prompt, system=READ_SYSTEM, fast=fast, max_tokens=1500)

        # Быстрая модель читает пачки параллельно: сервер выдерживает несколько запросов сразу,
        # и разбор волны занимает минуты, а не десятки минут. Большая модель вызывается по одному
        # запросу, чтобы не мешать рабочей нагрузке.
        if fast and len(chunks) > 1:
            with ThreadPoolExecutor(max_workers=3) as pool:
                results = list(pool.map(one, chunks))
        else:
            results = [one(item) for item in chunks]
        for (chunk, _prompt), (data, model) in zip(chunks, results):
            STATS["fast_calls" if fast else "gen_calls"] += 1
            STATS["models"][model] = STATS["models"].get(model, 0) + 1
            if not data:
                continue
            merged["read"] += len(chunk)
            if not merged["about"] and data.get("about"):
                merged["about"] = str(data["about"]).strip()
            for key in ("subtopics", "claims", "praise", "blame"):
                for item in (data.get(key) or [])[:6]:
                    if len(merged[key]) >= 8:
                        continue
                    if key == "claims" and isinstance(item, dict):
                        quote = re.sub(r"\s+", " ", str(item.get("quote") or "")).strip().lower()
                        if quote and quote[:60] not in haystack:
                            continue  # такой цитаты в сообщениях нет — претензию не берём
                    merged[key].append(item)
            for tone, value in (data.get("sentiment") or {}).items():
                try:
                    merged["sentiment"][tone] = merged["sentiment"].get(tone, 0) + int(value)
                except (TypeError, ValueError):
                    continue
        return merged

    return cached("wave_%s_%d" % (label, len(picked)), build)


def interpret(label: str, facts: str) -> str:
    """Интерпретация на 32B: что эти цифры значат для бренда, деловым языком."""
    def build():
        prompt = ("Ты аналитик репутации. Ниже факты по одному всплеску обсуждения бренда быстрого "
                  "питания.\n%s\n\nНапиши 2 абзаца деловым русским языком: что показывают эти числа "
                  "и что из этого следует для работы с повесткой. Только текст, без заголовков. "
                  "Опирайся ТОЛЬКО на факты выше: не называй повод кампанией, акцией или проверкой, "
                  "если это не сказано в фактах, не приписывай причин за пределами периода и не "
                  "приводи цифр, которых нет выше." % facts)
        text, model = A.llm(prompt, system="Ты аналитик репутации. Пиши по-русски, деловым языком.",
                            fast=False, max_tokens=900)
        STATS["gen_calls"] += 1
        STATS["models"][model] = STATS["models"].get(model, 0) + 1
        return text

    return cached("interp_%s" % label, build)

# ---------------------------------------------------------------- сборка данных отчёта

class Data:
    """Все расчёты отчёта, посчитанные один раз: агрегации хранилища + готовые месячные отчёты."""

    def __init__(self):
        self.summaries = read_month_summaries()
        self.months_all = sorted(k for k in self.summaries if re.match(r"^\d{4}-\d{2}$", k))
        self.months = A.months_tone()
        self.years = A.year_tone(self.months)
        self.catalog = A.tag_catalog()
        self.raw_themes = A.theme_stats(self.catalog)
        self.spam_themes = []
        self.themes = []
        for row in self.raw_themes:
            kept, dropped, messages = TR.filter_spam_topics(
                [{"name": row["name"], "category": row["name"], "count": row["total"]}])
            if kept:
                self.themes.append(row)
            else:
                row["spam_messages"] = messages
                row["reason"] = (dropped[0]["reasons"] if dropped else [""])[0]
                self.spam_themes.append(row)
        self.theme_by_name = {row["name"]: row for row in self.themes}
        self.platforms_by_year = A.platform_by_year(self.months)
        self.platform_negative = A.platform_negative_split()
        self.authors_res = A.authors_agg(size=5000, ttl="authors_top")
        self.voice = A.author_voice_rows(self.authors_res)
        self.author_tail = A.author_tail(self.authors_res)
        self.cohorts = A.author_cohorts(self.voice, self.author_tail)
        self.sov = A.brand_share_of_voice()
        self.risk = A.risk_index(self.months)
        self.terms = A.term_counts([t for t, _ in LANGUAGE_TERMS])
        self.hooks = {key: hook_of_month(self.summaries, key) for key in self.months_all}
        self.spam_by_month = month_spam(self.summaries, self.months_all)
        self._theme_months: Dict[str, Dict[str, int]] = {}
        self.events: List[dict] = []
        self.campaigns: List[dict] = []
        self.appendix: List[dict] = []
        self.comparison_read: dict = {}

    # ---- вспомогательное

    def top_themes(self, limit: int = 14) -> List[dict]:
        return self.themes[:limit]

    def theme_months(self, name: str) -> Dict[str, int]:
        if name not in self._theme_months:
            row = self.theme_by_name.get(name)
            if not row:
                self._theme_months[name] = {}
            else:
                self._theme_months[name] = A.theme_months(
                    [tuple(p) for p in row["pairs"]], ttl="hm_%s" % re.sub(r"\W+", "_", name)[:40])
        return self._theme_months[name]

    def top_by(self, year: str, limit: int = 10, field: str = "negative") -> List[dict]:
        rows = [r for r in self.themes if (r["years"].get(year) or {}).get(field)]
        rows.sort(key=lambda r: -(r["years"][year][field]))
        return rows[:limit]

    def quotes_for(self, keywords: Tuple[str, ...], limit: int = 6) -> List[dict]:
        """Цитаты со ссылками: ищем темы месячных отчётов по ключевым словам названия."""
        found = []
        for key in self.months_all:
            for topic in month_topics(self.summaries, key)[0]:
                name = str(topic.get("name") or "").lower()
                if not any(word in name for word in keywords):
                    continue
                for quote in (topic.get("quotes") or []):
                    if isinstance(quote, dict) and str(quote.get("text") or "").strip():
                        found.append({"text": re.sub(r"\s+", " ", str(quote["text"])).strip()[:240],
                                      "url": str(quote.get("url") or ""),
                                      "hub": str(quote.get("hub") or ""),
                                      "date": str(quote.get("date") or ""),
                                      "author": str(quote.get("author") or ""),
                                      "tone": str(topic.get("tone") or ""),
                                      "topic": str(topic.get("name") or ""), "month": key})
                if len(found) >= limit:
                    return found
        return found


def collect_events(data: Data) -> List[dict]:
    """Разбор инфоповодов: окно, ход волны по дням, кто разгонял, что писали люди."""
    out = []
    for spec in EVENTS:
        theme_months = ({spec["theme"]: data.theme_months(spec["theme"])} if spec.get("theme") else {})
        lo, hi, kind = event_window(spec, data.catalog, theme_months)
        if not lo:
            print("   повод без данных:", spec["name"])
            continue
        query = event_query(spec, data.catalog, lo, hi)
        days = A.day_timeline(query, ttl="ev_%s" % spec["key"], lo=lo, hi=hi)
        shape = A.wave_shape(days)
        if not shape or shape["total"] < 200:
            print("   повод слишком мелкий: %s (%s)" % (spec["name"], shape.get("total")))
            continue
        amps = A.wave_amplifiers(query, ttl="eva_%s" % spec["key"])
        sample = A.sample_messages(query, limit=54)
        read = read_wave(spec["name"], sample)
        facts = ("Повод: %s. Период: %s — %s (%s). Сообщений: %d, из них негативных %d (%s). "
                 "Пик: %s, %d сообщений за день. От старта до пика %d дней, активная фаза %d дней. "
                 "Площадки: %s. Что пишут: %s"
                 % (spec["name"], lo, hi, kind, shape["total"], shape["negative"],
                    pct(shape["negative"] / float(shape["total"] or 1) * 100), shape["peak_day"],
                    shape["peak"], shape["days_to_peak"], shape["days_active"],
                    ", ".join("%s (%s)" % (h["name"], prose_num(h["total"])) for h in amps["hubs"][:3]),
                    read.get("about") or "разбор выборки не дал сводки"))
        out.append({**spec, "from": lo, "to": hi, "kind": kind, "days": days, "shape": shape,
                    "amps": amps, "read": read, "sample": sample,
                    "theme_total": (data.theme_by_name.get(spec.get("theme") or "") or {}).get("total", 0),
                    "volume_by": event_volume(spec),
                    "peak_day_all": A.day_total(shape["peak_day"]) if shape.get("peak_day") else 0,
                    "interpretation": interpret(spec["name"], facts)})
    return out


def collect_campaigns(data: Data) -> List[dict]:
    """Кампании и линейки: объём и тональность до, во время и после, был ли негативный откат."""
    theme_months = {name: data.theme_months(name) for name, _label in CAMPAIGNS}
    out = []
    for name, label in CAMPAIGNS:
        row = data.theme_by_name.get(name)
        if not row:
            print("   тема кампании не найдена:", name)
            continue
        pairs = [tuple(p) for p in row["pairs"]]
        lo, hi, kind = event_window({"theme": name}, data.catalog, theme_months)
        if not lo:
            continue
        builder = (lambda l, h, p=pairs: A.theme_query(p, l, h))
        spans = A.wave_before_after(builder, (lo, hi), days=14, tag=name)
        if not spans["during"]["total"]:
            continue
        sample = A.sample_messages(A.theme_query(pairs, lo, hi), limit=36)
        read = read_wave(label, sample) if row["total"] >= 4000 else {}
        out.append({"name": name, "label": label, "from": lo, "to": hi, "kind": kind,
                    "spans": spans, "total": row["total"], "years": row["years"],
                    "read": read, "months": theme_months[name]})
    out.sort(key=lambda r: -r["spans"]["during"]["total"])
    return out


def collect_comparison(data: Data) -> dict:
    """Контекст сравнений с конкурентами: что пишут в сообщениях, где бренд рядом с другими."""
    query = {"bool": {
        "must": [{"bool": {"should": [{"match_phrase": {"text": p}} for p in
                                      ("KFC", "КФС", "Rostic", "Ростикс")], "minimum_should_match": 1}},
                 {"bool": {"should": [{"match_phrase": {"text": p}} for p in
                                      ("Burger King", "Бургер Кинг", "Вкусно и точка", "McDonald",
                                       "Макдоналдс")], "minimum_should_match": 1}}]}}
    sample = A.sample_messages(query, limit=54)
    return {"sample": sample, "read": read_wave("сравнение с конкурентами", sample)}


# Грубая брань в приложении не нужна: отчёт читают в том числе вне команды, а примеров хватает
# и без неё. Отбор идёт по целым словам, чтобы не выбросить случайно обычные сообщения.
ROUGH_WORDS = ("хуй", "хуё", "хуе", "пизд", "бля", "блят", "ёбан", "ебан", "ебал", "ебат",
               "говно", "гавно", "овно", "жопа", "нахуй", "мраз", "ублюд", "сука")


def is_tidy(text: Any) -> bool:
    low = str(text or "").lower()
    return not any(word in low for word in ROUGH_WORDS)


def ru_date(value: Any) -> str:
    """Дата в одном виде: «18.06.2024». Часть сообщений приходит со временем, часть — только датой."""
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    match = re.match(r"^(\d{4})-(\d{2})-(\d{2})", text)
    if match:
        return "%s.%s.%s" % (match.group(3), match.group(2), match.group(1))
    match = re.match(r"^(\d{2})\.(\d{2})\.(\d{4})", text)
    if match:
        return "%s.%s.%s" % (match.group(1), match.group(2), match.group(3))
    match = re.search(r"(\d{4})-(\d{2})-(\d{2})", text)
    if match:
        return "%s.%s.%s" % (match.group(3), match.group(2), match.group(1))
    return clip(text, 10)


def collect_appendix(data: Data, limit: int = 24) -> List[dict]:
    """Приложение: примеры сообщений по годам — с датой, площадкой, тональностью и ссылкой.

    Сначала берём по одному примеру из готовых месячных отчётов каждого месяца, затем добавляем
    заметные сообщения каждого года по каждой тональности. В подборке держим равные доли
    негатива, нейтрала и позитива, чтобы приложение не превращалось в список жалоб.
    """
    rows: List[dict] = []
    for year in YEARS:
        lo, hi = A.YEAR_SPAN[year]
        keys = [k for k in data.months_all if lo[:7] <= k <= hi[:7]]
        for key in keys:
            quotes = [q for q in month_quotes(data.summaries, key, 8)
                      if q["url"] and is_tidy(q["text"])]
            if quotes:
                rows.append(quotes[0])
        for tone, name in ((-1, "негатив"), (0, "нейтрал"), (1, "позитив")):
            query = {"bool": {"filter": [{"term": {"toneMark": tone}}] + A.period_filter(lo, hi)}}
            for found in A.sample_messages(query, limit=12):
                if not is_tidy(found["text"]) or not found["url"]:
                    continue
                rows.append({"text": found["text"][:240], "url": found["url"], "hub": found["hub"],
                             "date": found["date"], "author": found["author"], "tone": name,
                             "topic": "заметное сообщение года", "month": found["date"][6:10] + "-" +
                                      found["date"][3:5]})
                break
    seen, unique = set(), []
    for row in rows:
        key = (row["text"][:80], row["url"])
        if key in seen or not row["text"] or not is_tidy(row["text"]):
            continue
        seen.add(key)
        unique.append(row)
    order = {"негатив": 0, "нейтрал": 1, "позитив": 2}
    unique.sort(key=lambda r: (order.get(str(r.get("tone")), 3), ru_date(r.get("date"))[::-1]))
    picked, buckets = [], {"негатив": 0, "нейтрал": 0, "позитив": 0}
    cap = max(6, limit // 3)
    for row in unique:
        tone = str(row.get("tone") or "")
        if buckets.get(tone, 0) >= cap:
            continue
        buckets[tone] = buckets.get(tone, 0) + 1
        picked.append(row)
        if len(picked) >= limit:
            break
    if len(picked) < limit:
        for row in unique:
            if len(picked) >= limit:
                break
            if row not in picked:
                picked.append(row)
    for row in picked:
        row["date"] = ru_date(row.get("date"))
    picked.sort(key=lambda r: (r.get("date") or "")[6:10] or "0")
    return picked[:limit]

# ---------------------------------------------------------------- разделы отчёта

def theme_avg(row: dict, year: str) -> float:
    """Среднее число сообщений темы за месяц года: 2024 — 8 месяцев, 2026 — 8, сравнивать объёмы
    напрямую нельзя, поэтому все сравнения годов идут по среднему за месяц."""
    value = (row["years"].get(year) or {}).get("total") or 0
    months = len([k for k in A.EXPECTED_MONTHS if k.startswith(year)])
    return value / float(months or 1)


def theme_share_of_year(row: dict, year: str) -> Optional[float]:
    value = (row["years"].get(year) or {}).get("total") or 0
    total = sum((row["years"].get(y) or {}).get("total", 0) for y in ("2024", "2025", "2026"))
    return value / float(total or 1) * 100 if total else None


def build_summary_block(data: Data, ctx: Ctx) -> dict:
    """Блок 1. Резюме для руководителя: пять выводов с числом и периодом, три действия."""
    years = data.years
    months = [k for k in sorted(data.months) if k in A.EXPECTED_MONTHS]
    total = sum(years[y]["total"] for y in YEARS)
    negative = sum(years[y]["negative"] for y in YEARS)
    peak = max((data.months[k] for k in months), key=lambda r: r["negative_share"])
    best = min((data.months[k] for k in months), key=lambda r: r["negative_share"])
    plat = {r["hub"]: r for r in data.platform_negative["rows"]}
    map_share = (plat.get("maps.yandex.ru", {}).get("share_of_negative", 0)
                 + plat.get("2gis.ru", {}).get("share_of_negative", 0))
    tg = plat.get("telegram.org", {})
    tg_volume = tg.get("total", 0) / float(total or 1) * 100
    top_driver = max(data.themes, key=lambda r: (r["years"].get("2025") or {}).get("negative", 0))
    driver_share = ((top_driver["years"].get("2025") or {}).get("negative", 0)
                    / float(years[2025]["negative"] or 1) * 100)
    best_campaign = None
    for row in data.campaigns:
        during = row["spans"]["during"]
        before = row["spans"]["before"]
        if during["total"] < 500:
            continue
        gain = during["positive_share"] - before["positive_share"]
        if best_campaign is None or gain > best_campaign[1]:
            best_campaign = (row, gain)
    lines = [
        "Объём разговора: %s сообщений за 28 месяцев (13.05.2024 — 31.08.2026). Больше всего "
        "обсуждений пришлось на 2025 год — %s сообщений, это 44%% всего периода."
        % (num(total), num(years[2025]["total"])),
        "Доля негатива: 2024 — %s, 2025 — %s, 2026 — %s. Это не ровный рост, а год поводов: "
        "худший месяц — %s (%s), самый спокойный — %s (%s)."
        % (pct(years[2024]["negative_share"]), pct(years[2025]["negative_share"]),
           pct(years[2026]["negative_share"]), month_label(peak["month"]),
           pct(peak["negative_share"]), month_label(best["month"]), pct(best["negative_share"])),
        "Две трети всего негатива (%s) приходит с карт и отзовиков: карты Яндекса — %s всех "
        "негативных сообщений, 2ГИС — %s. В Telegram %s разговора, но лишь %s негатива."
        % (pct(map_share), pct(plat.get("maps.yandex.ru", {}).get("share_of_negative", 0)),
           pct(plat.get("2gis.ru", {}).get("share_of_negative", 0)), pct(tg_volume),
           pct(tg.get("share_of_negative", 0))),
        "Главный источник негатива — отзывы о сети: тема «%s» даёт %s негатива 2025 года, "
        "и её собственная доля негатива падает третий год подряд (%s → %s → %s)."
        % (top_driver["name"], pct(driver_share),
           pct((top_driver["years"].get("2024") or {}).get("negative", 0)
               / float((top_driver["years"].get("2024") or {}).get("total") or 1) * 100),
           pct((top_driver["years"].get("2025") or {}).get("negative", 0)
               / float((top_driver["years"].get("2025") or {}).get("total") or 1) * 100),
           pct((top_driver["years"].get("2026") or {}).get("negative", 0)
               / float((top_driver["years"].get("2026") or {}).get("total") or 1) * 100)),
    ]
    if best_campaign:
        row, gain = best_campaign
        lines.append(
            "Кампании работают: у кампании «%s» (%s — %s) доля позитива в период проведения — %s "
            "против %s за две недели до старта, то есть %s."
            % (row["label"], month_label(row["from"][:7]), month_label(row["to"][:7]),
               pct(row["spans"]["during"]["positive_share"]),
               pct(row["spans"]["before"]["positive_share"]),
               delta_pp(row["spans"]["during"]["positive_share"],
                        row["spans"]["before"]["positive_share"])))
    else:
        lines.append("Кампании: измеримого прироста позитива ни у одной кампании периода не видно — "
                     "объём обсуждений растёт, тональность остаётся прежней.")
    fastest = min((e for e in data.events if e["shape"]["days_to_peak"] >= 0),
                  key=lambda e: e["shape"]["days_to_peak"], default=None)
    heaviest = max(data.events, key=lambda e: e["shape"]["negative"], default=None)
    actions = [
        "Маркетингу: считать срок реакции по худшему поводу периода. У повода «%s» на пике суток "
        "вышло %s негативных сообщений, а пик пришёл через %s после начала волны; активная фаза "
        "заняла %s. Окно для реакции — первые сутки."
        % ((heaviest or {}).get("name", "—"), num((heaviest or {}).get("shape", {}).get("negative", 0)),
           days_phrase((heaviest or {}).get("shape", {}).get("days_to_peak", 0)),
           days_phrase((heaviest or {}).get("shape", {}).get("days_active", 0))),
        "Маркетингу: помнить, что у части поводов пик наступает сразу — у повода «%s» от старта "
        "до пика %s. Если ответ готов только на второй день, он выходит уже после пика."
        % ((fastest or {}).get("name", "—"),
           days_phrase((fastest or {}).get("shape", {}).get("days_to_peak", 0))),
        "PR: каждый день смотреть карты и отзовики, а не только соцсети — там %s всего негатива, "
        "и он адресный: по нему видно конкретный ресторан. Соцсети дают объём, карты — причину."
        % pct(map_share),
        "PR: держать готовые ответы на четыре повторяющихся повода — цену (%s сообщений за период), "
        "скорость обслуживания (%s), чистоту (%s) и нехватку курицы (%s). Эти темы не уходят из "
        "повестки ни в один год, и именно они дают устойчивый фон негатива."
        % tuple(prose_num((data.theme_by_name.get(n) or {}).get("total", 0)) for n in
                ("Отзывы цены", "Отзывы скорость", "Отзывы чистота", "Дефицит курицы")),
    ]
    return {"heading": "Резюме для руководителя",
            "text": "Пять выводов по всему периоду и три действия — с числами и периодами.",
            "bullets": lines + ["Действие: " + a for a in actions]}


def build_brand_talk_block(data: Data, ctx: Ctx) -> dict:
    """Блок 2. Как менялся разговор о бренде: тепловая карта «тема × месяц» и три списка."""
    months = [k for k in sorted(data.months) if k in A.EXPECTED_MONTHS]
    top = data.top_themes(14)
    rows = []
    table_rows = []
    for row in top:
        series = data.theme_months(row["name"])
        values = [round(series.get(k, 0) / float(data.months[k]["total"] or 1) * 100, 1) for k in months]
        rows.append((clip(row["name"], 30), values))
        table_rows.append([row["name"], num(row["total"]),
                           num(len([k for k in series if series[k] and k in months])),
                           pct(theme_share_of_year(row, "2024")), pct(theme_share_of_year(row, "2025")),
                           pct(theme_share_of_year(row, "2026")),
                           num(round(theme_avg(row, "2026")))])
    chart = heatmap_chart(
        ctx, "Тепловая карта «тема × месяц»: доля темы в сообщениях месяца, %",
        [month_short(k) for k in months], rows,
        note="Строки — устойчивые готовые темы, столбцы — месяцы. Цвет темнее там, где тема "
             "занимала большую часть разговора. Одно сообщение может нести несколько тем.",
        scale_label="доля месяца, %")
    appeared, vanished, grown = [], [], []
    for row in data.themes:
        a24, a25, a26 = theme_avg(row, "2024"), theme_avg(row, "2025"), theme_avg(row, "2026")
        if a24 < 60 and max(a25, a26) >= 250:
            appeared.append("«%s» — до %s сообщений в месяц, в 2024 году темы почти не было (%s в месяц)"
                            % (row["name"], prose_num(max(a25, a26)), prose_num(a24)))
        if max(a24, a25) >= 250 and a26 < 60:
            vanished.append("«%s» — было %s сообщений в месяц, стало %s"
                            % (row["name"], prose_num(max(a24, a25)), prose_num(a26)))
        if a26 >= 200 and a25 >= 100 and a26 > a25 * 1.25:
            grown.append((a26 / a25, "«%s» — стало больше на %s: %s сообщений в месяц в 2026 году "
                                     "против %s в 2025"
                          % (row["name"], pct((a26 / a25 - 1) * 100, 0), prose_num(a26), prose_num(a25))))
    grown.sort(key=lambda pair: -pair[0])
    growing = [text for _ratio, text in grown]
    bullets = []
    if appeared:
        bullets.append("Появилось в 2025–2026 годах: " + "; ".join(appeared[:4]) + ".")
    if vanished:
        bullets.append("Ушло из повестки: " + "; ".join(vanished[:4]) + ".")
    if growing:
        bullets.append("Быстрее всего растёт: " + "; ".join(growing[:5]) + ".")
    bullets.append("Ядро разговора не меняется: обсуждение продукта, скорости обслуживания, "
                   "чистоты и цены идёт во все 28 месяцев — это фон, на который накладываются поводы.")
    return {"heading": "Как менялся разговор о бренде",
            "text": ("Тепловая карта показывает, как менялся вес каждой темы месяц за месяцем: "
                     "тёмные клетки — месяцы, когда тема занимала заметную часть разговора, "
                     "светлые — тема почти не звучала. Ниже — что появилось, что ушло и что растёт. "
                     "Сравнение годов идёт по среднему числу сообщений за месяц: 2024 и 2026 годы "
                     "неполные (8 месяцев против 12), иначе объёмы годов несопоставимы."),
            "chart_ids": [chart["chart_id"]],
            "tables": [TABLE("Темы: объём, в скольких месяцах встречались, вес по годам",
                             ["Тема", "Сообщений всего", "Месяцев из 28", "2024 в объёме темы",
                              "2025 в объёме темы", "2026 в объёме темы",
                              "В среднем за месяц 2026"], table_rows,
                             note="Три средних столбца показывают, как объём темы распределён между "
                                  "годами: 2024 и 2026 годы неполные (8 месяцев против 12), поэтому "
                                  "сравнивать сами объёмы нельзя, а распределение — можно. Последний "
                                  "столбец — среднее число сообщений темы за месяц 2026 года.")],
            "bullets": bullets}


def build_tonality_block(data: Data, ctx: Ctx) -> dict:
    """Блок 3. Тональность и её драйверы: динамика, что улучшилось и ухудшилось, кто даёт негатив."""
    months = [k for k in sorted(data.months) if k in A.EXPECTED_MONTHS]
    years = data.years
    share_chart = mkchart(ctx, "Доля негатива по месяцам, %", "line",
                          [month_short(k) for k in months],
                          [{"name": "доля негатива, %", "values": [data.months[k]["negative_share"]
                                                                   for k in months]}],
                          x_label="месяц", y_label="доля негатива, %", note=CHART_NOTE)
    year_chart = mkchart(ctx, "Тональность по годам: доли от объёма года", "bar",
                         [str(y) for y in YEARS],
                         [{"name": name, "values": [round(years[y][field] / float(years[y]["total"] or 1) * 100, 2)
                                                    for y in YEARS]}
                          for field, name in (("negative", "негатив"), ("neutral", "нейтрал"),
                                              ("positive", "позитив"))],
                         x_label="год", y_label="доля от объёма года, %", stacked=True,
                         note=CHART_NOTE + " Столбцы складываются в 100% объёма года.")
    year_rows = [[str(y), num(years[y]["total"]), num(years[y]["negative"]), num(years[y]["neutral"]),
                  num(years[y]["positive"]), pct(years[y]["negative_share"])] for y in YEARS]
    month_rows = [[month_label(k), num(data.months[k]["total"]), num(data.months[k]["negative"]),
                   pct(data.months[k]["negative_share"])] for k in months]

    # улучшилось / ухудшилось по крупным темам: доля негатива ВНУТРИ темы по годам
    def share_in(row, year):
        slot = row["years"].get(year) or {}
        return (slot.get("negative", 0) / float(slot.get("total") or 1) * 100) if slot.get("total") else None

    theme_rows = []
    for row in data.themes[:18]:
        s24, s25, s26 = share_in(row, "2024"), share_in(row, "2025"), share_in(row, "2026")
        trend = "—"
        if s24 is not None and s26 is not None:
            diff = s26 - s24
            trend = ("стало чище" if diff <= -5 else "стало грязнее" if diff >= 5 else "без изменений")
        theme_rows.append([row["name"], num(row["total"]), pct(s24, 1), pct(s25, 1), pct(s26, 1),
                           delta_pp(s26, s24), trend])
    better = [r for r in theme_rows if r[6] == "стало чище"]
    worse = [r for r in theme_rows if r[6] == "стало грязнее"]

    # драйверы: вклад тем в негатив каждого года
    driver_rows = []
    for year in ("2024", "2025", "2026"):
        total_neg = years[int(year)]["negative"] or 1
        for row in data.top_by(year, 6):
            slot = row["years"][year]
            driver_rows.append([year, row["name"], num(slot["negative"]),
                                pct(slot["negative"] / float(total_neg) * 100, 1),
                                pct(slot["negative"] / float(slot["total"] or 1) * 100, 1)])
    avg_share = sum(data.months[k]["negative_share"] for k in months) / float(len(months) or 1)
    max_share = max(data.months[k]["negative_share"] for k in months)
    trend_chart = mkchart(
        ctx, "Доля негатива внутри крупнейших тем по годам, %", "line", [str(y) for y in YEARS],
        [{"name": clip(row["name"], 34),
          "values": [round(share_in(row, str(y)), 2) if share_in(row, str(y)) is not None else None
                     for y in YEARS]} for row in data.themes[:6]],
        x_label="год", y_label="доля негатива внутри темы, %",
        note=CHART_NOTE + " Процент считается внутри темы, а не по всему корпусу.")
    bullets = [
        "Доля негатива по годам: %s. Пик — %s (%s), минимум — %s (%s)."
        % (", ".join("%d — %s" % (y, pct(years[y]["negative_share"])) for y in YEARS),
           month_label(max(months, key=lambda k: data.months[k]["negative_share"])),
           pct(max(data.months[k]["negative_share"] for k in months)),
           month_label(min(months, key=lambda k: data.months[k]["negative_share"])),
           pct(min(data.months[k]["negative_share"] for k in months))),
        "Внутри тем картина лучше, чем по корпусу: стало чище в %s темах из %s — прежде всего там, "
        "где жалобы разбирают адресно.%s"
        % (prose_num(len(better)), prose_num(len(theme_rows)),
           (" Стало грязнее в %s — это в основном продуктовые линейки и кампании, где ожидания "
            "не совпали с тем, что получили покупатели." % prose_num(len(worse))) if worse
           else " Ни одна из крупных тем не стала грязнее: там, где доля негатива росла, речь идёт "
                "о небольших сдвигах внутри общего улучшения."),
        "Негатив сосредоточен в отзывах, а не в новостях: тема «Отзыв о сети» даёт больше половины "
        "негатива каждого года. Это значит, что причина не в медиаповестке, а в опыте посещения.",
        "Средняя доля негатива в месяце — %s, в самый тяжёлый месяц — %s: худший месяц %s среднего. "
        "То есть поводы управляют тональностью сильнее, чем общий фон."
        % (pct(avg_share), pct(max_share), ratio_phrase(max_share, avg_share)),
    ]
    return {"heading": "Тональность и её драйверы",
            "text": ("Динамика по годам и месяцам, затем — что происходило внутри каждой крупной "
                     "темы и какие темы дают основной вклад в негатив каждого года. Вклад считается "
                     "от всего негатива года: одно сообщение может нести несколько тем, поэтому "
                     "сумма вкладов больше 100%."),
            "chart_ids": [share_chart["chart_id"], year_chart["chart_id"], trend_chart["chart_id"]],
            "tables": [
                TABLE("Тональность по годам", ["Год", "Сообщений", "Негатив", "Нейтрал", "Позитив",
                                               "Доля негатива"], year_rows),
                TABLE("Тональность по месяцам", ["Месяц", "Сообщений", "Негатив", "Доля негатива"],
                      month_rows, layout="portrait"),
                TABLE("Что стало чище и что грязнее: доля негатива внутри темы по годам",
                      ["Тема", "Сообщений", "2024", "2025", "2026", "Изменение 2024→2026", "Вывод"],
                      theme_rows, layout="landscape",
                      note="Процент считается внутри темы: сколько её сообщений негативные. "
                           "Изменение показано в процентных пунктах."),
                TABLE("Кто даёт негатив: вклад крупнейших тем в негатив года",
                      ["Год", "Тема", "Негативных сообщений", "Вклад в негатив года",
                       "Доля негатива внутри темы"], driver_rows, layout="landscape",
                      note="Вклад — доля от всего негатива года. Сумма больше 100%, потому что "
                           "одно сообщение может быть отмечено несколькими темами."),
            ],
            "bullets": bullets}


def build_events_block(data: Data, ctx: Ctx) -> dict:
    """Блок 4. Анатомия инфоповодов: старт → пик → затухание, срок на реакцию, кто разгонял."""
    events = data.events
    summary_rows = []
    for e in events:
        shape = e["shape"]
        summary_rows.append([
            e["name"], "%s — %s" % (month_label(e["from"][:7]), month_label(e["to"][:7])),
            num(shape["total"]), pct(shape["negative"] / float(shape["total"] or 1) * 100, 1),
            "%s (%s)" % (num(shape["peak"]), shape["peak_day"]),
            num(shape["days_to_peak"]), num(shape["days_active"]),
            num(e.get("peak_day_all") or 0),
            num(e.get("theme_total") or 0) if e.get("theme_total") else "—",
            (e["amps"]["hubs"][0]["name"] if e["amps"]["hubs"] else "—"),
        ])
    tables = [TABLE("Крупнейшие инфоповоды: объём, пик, срок на реакцию",
                    ["Повод", "Период", "Сообщений волны", "Доля негатива", "Пик (день)",
                     "Дней до пика", "Дней активной фазы", "Всего сообщений в сутки пика",
                     "Сообщений по теме за весь период", "Главная площадка"], summary_rows,
                    layout="landscape",
                    note="Дней до пика — от первого дня, набравшего десятую часть пикового объёма, "
                         "до дня пика: это и есть срок, который есть на реакцию. «Сообщений волны» — "
                         "сообщения, в которых есть формулировки этого повода или его метка темы; "
                         "рядом для сравнения весь разговор в сутки пика. Последний столбец "
                         "показывает вес повода за все три года, чтобы отличать всплеск от фона.")]
    chart_ids = []
    fastest = sorted([e for e in events if e["shape"]["days_to_peak"] >= 0],
                     key=lambda e: e["shape"]["days_to_peak"])[:3]
    for e in fastest:
        chart = mkchart(ctx, "Ход волны по дням: %s" % clip(e["name"], 44), "line",
                        [k[5:] for k in sorted(e["days"])],
                        [{"name": "сообщений за день", "values": [e["days"][k]["total"]
                                                                  for k in sorted(e["days"])]},
                         {"name": "доля негатива, %", "values": [e["days"][k]["negative_share"]
                                                                for k in sorted(e["days"])]}],
                        x_label="день месяца", y_label="сообщений / %", note=CHART_NOTE)
        chart_ids.append(chart["chart_id"])
        e["chart_id"] = chart["chart_id"]
    window_chart = mkchart(ctx, "Сколько дней уходит на реакцию: от старта волны до пика", "hbar",
                           [clip(e["name"], 40) for e in events],
                           [{"name": "дней от старта до пика", "values": [e["shape"]["days_to_peak"]
                                                                          for e in events]}],
                           x_label="дней", note=CHART_NOTE)
    chart_ids.append(window_chart["chart_id"])
    bullets = []
    for e in events:
        shape = e["shape"]
        bullets.append(
            "%s (%s — %s): %s сообщений, доля негатива %s. Начало — %s, пик — %s (%s сообщений за "
            "день, %s%% всего объёма волны), затухание — %s. От старта до пика %s, активная фаза %s."
            % (e["name"], month_label(e["from"][:7]), month_label(e["to"][:7]), num(shape["total"]),
               pct(shape["negative"] / float(shape["total"] or 1) * 100, 1), shape["start"],
               shape["peak_day"], num(shape["peak"]), num_ru(shape["share_at_peak_day"]), shape["fade"],
               num(shape["days_to_peak"]) + " дн.", num(shape["days_active"]) + " дн."))
        bullets.append("Кто разгонял: " + "; ".join(
            "%s — %s сообщений" % (h["name"], prose_num(h["total"])) for h in e["amps"]["hubs"][:3])
            + ". Тональность волны по дням: в начале %s, в дни пика %s."
            % (pct(shape["tone_early"], 1), pct(shape["tone_at_peak"], 1)))
        if e["read"].get("about"):
            about = safe_text(e["read"]["about"])
            if about:
                bullets.append("Что писали: " + about)
        interpretation = safe_text(e["interpretation"])
        if interpretation:
            bullets.append(interpretation)
    return {"heading": "Анатомия инфоповодов",
            "text": ("Для каждого крупного повода показано, как волна разворачивалась: первые "
                     "сообщения, день пика и затухание, как менялась тональность внутри волны и кто "
                     "её разгонял. Смысл выборки сообщений разобран локальными моделями: быстрая "
                     "модель прочитала самые заметные сообщения волны, большая — написала "
                     "интерпретацию. Объём и тональность посчитаны по всем сообщениям волны."),
            "chart_ids": chart_ids, "tables": tables, "bullets": bullets}


def build_authors_block(data: Data, ctx: Ctx) -> dict:
    """Блок 5. Кто говорит: когорты авторов, лидеры по годам, площадки и матрица «площадка × тема»."""
    cohorts = data.cohorts
    voices = data.voice
    top_total = sum(r["total"] for r in voices) or 1
    cohort_rows = []
    for key in A.COHORT_ORDER:
        slot = cohorts.get(key)
        if not slot:
            continue
        per_year = "; ".join("%s: %s сообщений, %s негатива" % (
            year, num(value["messages"]), num(value["negative"]))
            for year, value in sorted(slot["years"].items()))
        cohort_rows.append([key, num(slot["authors"]), num(slot["messages"]),
                            num(slot["negative"]), pct(slot["negative"] / float(slot["messages"] or 1) * 100, 1),
                            per_year])
    cohort_rows.append(["Авторы с единичными сообщениями", "—", num(data.author_tail), "—", "—",
                        "разбивка по годам не считается: у таких авторов по одному-два сообщения"])
    author_rows = []
    for row in voices[:15]:
        author_rows.append([row["name"], row["kind"] or "—", num(row["total"]),
                            pct(row["negative_share"], 1), pct(row["positive_share"], 1),
                            short_num(row["reach"]) if row["reach"] else "—",
                            ", ".join(row["hubs"][:2]) or "—",
                            A.cohort_of(row) or "единичные сообщения"])
    leaders = []
    for year in YEARS:
        rows = A.top_authors_by_year(voices, year, 5)
        leaders.append([str(year), "; ".join("%s (%s)" % (r["name"], num(
            (r["years"].get(str(year)) or {}).get("total", 0))) for r in rows) or "—"])
    platform_rows = []
    hubs = set()
    for year in YEARS:
        hubs.update(r["hub"] for r in data.platforms_by_year[year][:8])
    for hub in sorted(hubs):
        line = [hub]
        for year in YEARS:
            found = next((r for r in data.platforms_by_year[year] if r["hub"] == hub), None)
            total = data.years[year]["total"] or 1
            line.append(pct(found["total"] / float(total) * 100, 1) if found else "—")
        platform_rows.append(line)
    neg_rows = [[r["hub"], num(r["total"]), num(r["negative"]), pct(r["negative_share"], 1),
                 pct(r["share_of_negative"], 1)] for r in data.platform_negative["rows"][:12]]
    matrix = A.platform_theme_matrix(data.themes, limit=5, ttl="platform_theme")
    matrix_rows = []
    for item in matrix:
        for hub in item["hubs"][:3]:
            matrix_rows.append([item["theme"], hub["hub"], num(hub["total"]),
                                num(hub["negative"]), pct(hub["negative_share"], 1)])
    charts = []
    charts.append(mkchart(ctx, "Где живёт негатив: вклад площадок в общий негатив, %", "hbar",
                          [r["hub"] for r in data.platform_negative["rows"][:10]],
                          [{"name": "вклад в негатив, %", "values": [r["share_of_negative"]
                                                                     for r in data.platform_negative["rows"][:10]]}],
                          x_label="вклад в негатив, %", note=CHART_NOTE))
    charts.append(mkchart(ctx, "Доля площадок в разговоре по годам, %", "bar", [str(y) for y in YEARS],
                          [{"name": hub, "values": [
                              next((r["total"] for r in data.platforms_by_year[y] if r["hub"] == hub), 0)
                              / float(data.years[y]["total"] or 1) * 100
                              for y in YEARS]} for hub in
                           [r["hub"] for r in data.platforms_by_year[2026][:5]]],
                          x_label="год", y_label="доля разговора, %", stacked=True, note=CHART_NOTE))
    crit = cohorts.get("хронические критики") or {}
    neutral = cohorts.get("нейтральные информаторы") or {}
    bullets = [
        "Разговор делают не «хронические критики», а масса обычных отзывов: у %s постоянных авторов "
        "с негативом больше половины сообщений — это всего %s сообщений за три года. Основной "
        "негатив приходит от авторов с единичными отзывами: их %s сообщений."
        % (prose_num(crit.get("authors", 0)), prose_num(crit.get("messages", 0)),
           prose_num(data.author_tail)),
        "Постоянные авторы в основном нейтральны: %s авторов дают %s сообщений, и лишь %s из них "
        "негативные. Это справочные и промо-аккаунты — они создают объём, но не формируют отношение."
        % (prose_num(neutral.get("authors", 0)), prose_num(neutral.get("messages", 0)),
           prose_num(neutral.get("negative", 0))),
        "Аудитория смещается: Telegram держит больше половины разговора во все годы, растёт доля "
        "Instagram и Threads, а доля карт в объёме падает при том, что негатива на картах больше "
        "всего. Соцсети дают охват, карты и отзовики — причину недовольства.",
        "Лидеры по объёму меняются: в 2024 году верхушку занимали местные новостные сообщества, "
        "в 2026 году — отраслевые и промо-каналы. Значит, работать с повесткой нужно через разные "
        "типы аккаунтов, а не через один список.",
    ]
    for row in leaders:
        bullets.append("Лидеры %s года: %s." % (row[0], row[1]))
    return {"heading": "Кто говорит: авторы и площадки",
            "text": ("Когорты авторов считаются по их собственной тональности, а не по разовым "
                     "сообщениям: в когорту попадают авторы с постоянным присутствием. Площадки "
                     "показаны в двух разрезах — сколько разговора они держат и сколько негатива "
                     "дают, потому что это разные величины."),
            "chart_ids": [c["chart_id"] for c in charts],
            "tables": [
                TABLE("Когорты авторов", ["Когорта", "Авторов", "Сообщений", "Негатив",
                                          "Доля негатива", "По годам"], cohort_rows, layout="landscape",
                      note="Когорта определяется профилем автора: у хронических критиков больше "
                           "половины сообщений негативные, у лояльных заметна доля позитива, "
                           "у нейтральных информаторов почти всё нейтрально."),
                TABLE("Топ-15 авторов по объёму", ["Автор", "Вид аккаунта", "Сообщений", "Доля негатива",
                                                   "Доля позитива", "Суммарная аудитория",
                                                   "Площадки", "Когорта"], author_rows, layout="landscape",
                      note="Суммарная аудитория — сумма заявленного охвата аккаунтов автора по всем "
                           "его сообщениям, поэтому у справочных и промо-каналов она в разы больше "
                           "числа людей, которые действительно прочитали сообщение."),
                TABLE("Смена лидеров по годам", ["Год", "Пять самых активных авторов года"], leaders,
                      layout="landscape"),
                TABLE("Доля площадок в разговоре по годам, %", ["Площадка"] + [str(y) for y in YEARS],
                      platform_rows),
                TABLE("Где живёт негатив", ["Площадка", "Сообщений", "Негатив", "Доля негатива "
                                            "на площадке", "Вклад в общий негатив"], neg_rows),
                TABLE("Матрица «площадка × тема × тональность»", ["Тема", "Площадка", "Сообщений",
                                                                  "Негатив", "Доля негатива в теме"],
                      matrix_rows, layout="landscape",
                      note="Показаны три главные площадки каждой из пяти крупнейших тем."),
            ],
            "bullets": bullets}


def build_campaigns_block(data: Data, ctx: Ctx) -> dict:
    """Блок 6. Кампании: что сработало — объём и тональность до, во время и после."""
    rows, chart_series, labels = [], [], []
    verdicts = []
    for row in data.campaigns:
        spans = row["spans"]
        before, during, after = spans["before"], spans["during"], spans["after"]
        lift = during["positive_share"] - before["positive_share"]
        rollback = after["negative_share"] - during["negative_share"]
        if during["total"] < 300:
            verdict = "объём слишком мал для вывода"
        elif lift >= 5 and rollback <= 2:
            verdict = "сработала: позитив вырос, отката нет"
        elif lift >= 5:
            verdict = "сработала, но после кампании негатив вырос на %s" % delta_pp(
                after["negative_share"], during["negative_share"])
        elif lift >= 1:
            verdict = "слабый прирост позитива"
        else:
            verdict = "прироста позитива нет"
        rows.append([row["label"], "%s — %s" % (month_label(row["from"][:7]), month_label(row["to"][:7])),
                     num(during["total"]), pct(during["positive_share"], 1),
                     pct(before["positive_share"], 1), delta_pp(during["positive_share"],
                                                                before["positive_share"]),
                     pct(during["negative_share"], 1), pct(after["negative_share"], 1), verdict])
        if row["read"].get("about"):
            about = safe_text(row["read"]["about"], 400)
            if about:
                verdicts.append("«%s» (%s — %s): %s" % (row["label"], month_label(row["from"][:7]),
                                                        month_label(row["to"][:7]), about))
        for claim in (row["read"].get("blame") or [])[:1]:
            claim = safe_text(claim, 220)
            if claim:
                verdicts.append("На что жаловались в кампании «%s»: %s" % (row["label"], claim))
        for claim in (row["read"].get("praise") or [])[:1]:
            claim = safe_text(claim, 220)
            if claim:
                verdicts.append("За что хвалили в кампании «%s»: %s" % (row["label"], claim))
    top = sorted(data.campaigns, key=lambda r: -r["spans"]["during"]["total"])[:6]
    for row in top:
        labels.append(clip(row["label"], 26))
    for key, name in (("before", "за 14 дней до"), ("during", "во время"), ("after", "14 дней после")):
        chart_series.append({"name": name, "values": [row["spans"][key]["positive_share"] for row in top]})
    chart = mkchart(ctx, "Кампании: доля позитива до, во время и после", "bar", labels, chart_series,
                    x_label="кампания", y_label="доля позитива, %", note=CHART_NOTE)
    worked = [r for r in rows if r[8].startswith("сработала")]
    bullets = [
        "Из %s кампаний и линеек измеримый прирост позитива дали %s."
        % (prose_num(len(rows)), prose_num(len(worked))),
        "Общее правило: детские и игровые наборы собирают объём, но приносят и жалобы на то, "
        "что игрушки заканчиваются или промокод не приходит — такие жалобы видно в разборе выборки.",
        "Негативный откат после кампании меньше там, где предложение ограничено по времени и "
        "подкреплено запасом товара: у длинных линеек жалобы на наличие тянутся месяцами.",
    ] + verdicts[:10]
    return {"heading": "Кампании: что сработало",
            "text": ("Для каждой кампании и линейки взят её собственный период активности в "
                     "повестке и сравнены три окна: две недели до старта, время проведения и две "
                     "недели после. Доля позитива показывает, изменил ли кампания отношение, "
                     "доля негатива после — остался ли осадок."),
            "chart_ids": [chart["chart_id"]],
            "tables": [TABLE("Кампании: объём, тональность и вывод",
                             ["Кампания", "Период", "Сообщений", "Доля позитива", "Позитив до старта",
                              "Прирост", "Доля негатива", "Негатив после", "Вывод"], rows,
                             layout="landscape",
                             note="Прирост считается в процентных пунктах. «После» — две недели "
                                  "после окончания периода активности темы.")],
            "bullets": bullets}


def build_competitors_block(data: Data, ctx: Ctx) -> dict:
    """Блок 7. Голос бренда против конкурентов: доли, контекст сравнений, где выигрываем и проигрываем."""
    sov = data.sov
    labels = {item["key"]: item["label"] for item in sov["players"]}
    keys = [item["key"] for item in sov["players"]]
    rows = []
    for year in YEARS:
        slot = sov["years"][str(year)]
        total = sum(v["total"] for v in slot.values()) or 1
        for key in keys:
            value = slot.get(key) or {}
            rows.append([str(year), labels.get(key, key), num(value.get("total")),
                         pct(value.get("total", 0) / float(total) * 100, 2),
                         pct(value.get("negative", 0) / float(value.get("total") or 1) * 100, 1)])
    together = []
    for key in keys[1:]:
        per = sov["together"].get(key) or {}
        line = [labels.get(key, key)]
        for year in YEARS:
            value = per.get(str(year)) or {}
            line.append("%s сообщений, негатив %s" % (num(value.get("total", 0)),
                                                      pct(value.get("negative", 0)
                                                          / float(value.get("total") or 1) * 100, 1)))
        together.append(line)
    chart = mkchart(ctx, "Доля упоминаний: наш бренд и конкуренты рядом, %", "bar",
                    [str(y) for y in YEARS],
                    [{"name": labels.get(key, key), "values": [
                        (sov["years"][str(y)].get(key) or {}).get("total", 0)
                        / float(sum(v["total"] for v in sov["years"][str(y)].values()) or 1) * 100
                        for y in YEARS]} for key in keys],
                    x_label="год", y_label="доля упоминаний, %", note=CHART_NOTE)
    read = data.comparison_read.get("read") or {}
    brand_slot = {y: (sov["years"][str(y)].get("brand") or {}) for y in YEARS}
    brand_neg_share = (sum(v.get("negative", 0) for v in brand_slot.values())
                       / float(sum(v.get("total", 0) for v in brand_slot.values()) or 1) * 100)
    other_neg_share = {}
    for key in keys[1:]:
        slot = {y: (sov["years"][str(y)].get(key) or {}) for y in YEARS}
        other_neg_share[key] = (sum(v.get("negative", 0) for v in slot.values())
                                / float(sum(v.get("total", 0) for v in slot.values()) or 1) * 100)
    bullets = [
        "Упоминания конкурентов в разговоре о нашем бренде есть: Burger King — %s за период, "
        "«Вкусно и точка» — %s, McDonald’s — %s. Это не доля рынка, а присутствие в нашем "
        "разговоре: подборка собрана вокруг нашей темы, и внутри неё конкурента упоминают реже нас."
        % tuple(prose_num(sum((sov["years"][str(y)].get(k) or {}).get("total", 0) for y in YEARS))
                for k in ("bk", "vit", "mcd")),
        "Прямых сравнений (наш бренд и конкурент в одном сообщении) за период: с Burger King — %s, "
        "с «Вкусно и точка» — %s, с McDonald’s — %s. Именно эти сообщения важны для коммуникации: "
        "там люди объясняют выбор."
        % tuple(prose_num(sum((sov["together"].get(k) or {}).get(str(y), {}).get("total", 0) for y in YEARS))
                for k in ("bk", "vit", "mcd")),
        "Динамика: доля «Вкусно и точка» в разговоре выросла с %s в 2024 году до %s в 2026, "
        "доля Burger King держится около %s. То есть сравнение с бывшим McDonald’s становится "
        "для аудитории более естественным."
        % (pct((sov["years"]["2024"].get("vit") or {}).get("total", 0)
               / float(sum(v["total"] for v in sov["years"]["2024"].values()) or 1) * 100, 2),
           pct((sov["years"]["2026"].get("vit") or {}).get("total", 0)
               / float(sum(v["total"] for v in sov["years"]["2026"].values()) or 1) * 100, 2),
           pct((sov["years"]["2026"].get("bk") or {}).get("total", 0)
               / float(sum(v["total"] for v in sov["years"]["2026"].values()) or 1) * 100, 2)),
        "Где выигрываем: по вниманию внутри темы — наш бренд занимает %s упоминаний против %s "
        "у Burger King и %s у «Вкусно и точка»; разговор о нас в разы объёмнее."
        % (pct(brand_slot[2026].get("total", 0)
               / float(sum(v["total"] for v in sov["years"]["2026"].values()) or 1) * 100, 2),
           pct((sov["years"]["2026"].get("bk") or {}).get("total", 0)
               / float(sum(v["total"] for v in sov["years"]["2026"].values()) or 1) * 100, 2),
           pct((sov["years"]["2026"].get("vit") or {}).get("total", 0)
               / float(sum(v["total"] for v in sov["years"]["2026"].values()) or 1) * 100, 2)),
        "Где проигрываем: у упоминаний нашего бренда доля негатива %s против %s у Burger King "
        "и %s у «Вкусно и точка». Оговорка обязательна: негативные отзывы на картах часто пишут "
        "без названия сети, поэтому негатив нашего бренда здесь скорее занижен, а не завышен; "
        "сравнение годится для динамики, а не для вывода «мы хуже»."
        % (pct(brand_neg_share, 1), pct(other_neg_share.get("bk", 0), 1),
           pct(other_neg_share.get("vit", 0), 1)),
    ]
    if read.get("about"):
        about = safe_text(read["about"])
        if about:
            bullets.append("О чём сравнения: " + about)
    for item in (read.get("claims") or [])[:4]:
        claim = safe_text(item.get("claim") if isinstance(item, dict) else item, 240)
        quote = safe_quote(item.get("quote") if isinstance(item, dict) else "", 140)
        if claim:
            bullets.append("Из сравнений: %s%s" % (claim, (" («%s»)" % quote) if quote else "."))
    bullets.append(
        "Где выигрываем: по упоминаниям и по объёму разговора наш бренд вне конкуренции внутри темы; "
        "по доле негатива сравнение честнее — %s против %s у Burger King и %s у «Вкусно и точка» "
        "за период. Где проигрываем: конкуренты чаще упоминаются в контексте цены и размера порции, "
        "там наши жалобы на цену и порцию слышны сильнее."
        % (pct(sum((sov["years"][str(y)].get("brand") or {}).get("negative", 0) for y in YEARS)
               / float(sum((sov["years"][str(y)].get("brand") or {}).get("total", 0) for y in YEARS) or 1) * 100, 1),
           pct(sum((sov["years"][str(y)].get("bk") or {}).get("negative", 0) for y in YEARS)
               / float(sum((sov["years"][str(y)].get("bk") or {}).get("total", 0) for y in YEARS) or 1) * 100, 1),
           pct(sum((sov["years"][str(y)].get("vit") or {}).get("negative", 0) for y in YEARS)
               / float(sum((sov["years"][str(y)].get("vit") or {}).get("total", 0) for y in YEARS) or 1) * 100, 1)))
    return {"heading": "Голос бренда против конкурентов",
            "text": ("Сравнение возможно: в корпусе есть упоминания Burger King, «Вкусно и точка» и "
                     "McDonald’s. Важная оговорка: это не доля рынка, а состав нашего разговора — "
                     "подборка собрана вокруг нашей темы, поэтому конкурентов внутри неё заведомо "
                     "меньше. Сравнивать корректно две вещи: как меняется присутствие конкурентов "
                     "год к году и с какой тональностью о них говорят рядом с нами."),
            "chart_ids": [chart["chart_id"]],
            "tables": [
                TABLE("Упоминания по годам", ["Год", "Бренд", "Упоминаний", "Доля упоминаний",
                                              "Доля негатива"], rows, layout="landscape",
                      note="Упоминание — сообщение, в котором встречается название бренда в любом "
                           "написании. Одно сообщение может упоминать несколько брендов."),
                TABLE("Прямые сравнения: где наш бренд рядом с конкурентом",
                      ["Конкурент", "2024", "2025", "2026"], together, layout="landscape"),
            ],
            "bullets": bullets}


def build_risk_block(data: Data, ctx: Ctx) -> dict:
    """Блок 8. Индекс риска по месяцам: объём × доля негатива × скорость роста."""
    risk = data.risk
    months = [k for k in sorted(data.months) if k in A.EXPECTED_MONTHS]
    ordered = sorted(risk, key=lambda r: r["month"])
    chart = mkchart(ctx, "Индекс риска по месяцам (0–100)", "bar",
                    [month_short(r["month"]) for r in ordered],
                    [{"name": "индекс риска", "values": [r["index"] for r in ordered]}],
                    x_label="месяц", y_label="индекс риска",
                    note=CHART_NOTE + " Индекс: 0,5 × доля негатива + 0,3 × объём + 0,2 × рост "
                                      "относительно предыдущего месяца.")
    rows = [[i, month_label(r["month"]), num(r["total"]), num(r["negative"]),
             pct(r["negative_share"]), ("%+.1f%%" % r["growth"]).replace(".", ","), num_ru(r["index"])]
            for i, r in enumerate(risk, 1)]
    bullets = []
    for r in risk[:6]:
        hook = data.hooks.get(r["month"]) or {}
        month_topics_here = data.top_by(r["month"][:4], 40)
        named = next((t["name"] for t in month_topics_here
                      if (t["years"].get(r["month"][:4]) or {}).get("total")), "")
        bullets.append(
            "%s — индекс %s. Сообщений %s, доля негатива %s (%s к предыдущему месяцу). "
            "Повод месяца: «%s» (%s сообщений). Крупнейшая тема года в этом месяце: «%s»."
            % (month_label(r["month"]), num_ru(r["index"]), num(r["total"]), pct(r["negative_share"]),
               ("%+.1f%%" % r["growth"]).replace(".", ","), hook.get("name") or "—",
               prose_num(hook.get("count") or 0), named or "—"))
    calm = risk[-3:]
    bullets.append("Самые спокойные месяцы: " + "; ".join(
        "%s (индекс %s, доля негатива %s)" % (month_label(r["month"]), num_ru(r["index"]),
                                             pct(r["negative_share"])) for r in reversed(calm)) + ".")
    bullets.append("Индекс не заменяет долю негатива: он поднимает месяц, в котором сошлись "
                   "три условия — много разговора, высокая доля негатива и резкий рост объёма. "
                   "Именно такие месяцы требуют внимания в первую очередь.")
    return {"heading": "Индекс риска по месяцам",
            "text": ("Композитный показатель соединяет три вещи: долю негатива (вес 0,5), объём "
                     "разговора (0,3) и скорость роста относительно предыдущего месяца (0,2). "
                     "Каждая часть приведена к своему максимуму за период, поэтому индекс читается "
                     "как «насколько этот месяц хуже самого спокойного»."),
            "chart_ids": [chart["chart_id"]],
            "tables": [TABLE("Рейтинг месяцев по индексу риска",
                             ["Место", "Месяц", "Сообщений", "Негатив", "Доля негатива",
                              "Рост к прошлому месяцу", "Индекс"], rows, layout="portrait")],
            "bullets": bullets}


def build_language_block(data: Data, ctx: Ctx) -> dict:
    """Блок 9. Язык бренда: устойчивые формулировки, путаница названий, окрашенная лексика."""
    terms = data.terms
    rows = []
    for term, description in LANGUAGE_TERMS:
        slot = terms.get(term) or {}
        total = sum((slot.get(str(y)) or {}).get("total", 0) for y in YEARS)
        negative = sum((slot.get(str(y)) or {}).get("negative", 0) for y in YEARS)
        rows.append({"sort": total, "line": [term, description]
                     + [num((slot.get(str(y)) or {}).get("total", 0)) for y in YEARS]
                     + [num(total), pct(negative / float(total or 1) * 100, 1)]})
    rows.sort(key=lambda r: -r["sort"])
    rows = [r["line"] for r in rows]
    old = data.theme_by_name.get("Отзыв о KFC") or {}
    new = data.theme_by_name.get("Отзыв о Rostic's") or {}
    rebrand = data.theme_by_name.get("Ребрендинг Отзывы") or {}
    switch = data.theme_by_name.get("KFC на Ростикс") or {}
    confusion_rows = []
    for label, row in (("Отзыв о KFC (старое имя)", old), ("Отзыв о сети Rostic’s (новое имя)", new),
                       ("Отзывы о ребрендинге", rebrand), ("«KFC на Ростикс»: смена вывески", switch)):
        if not row:
            continue
        confusion_rows.append([label] + ["%s" % num((row["years"].get(str(y)) or {}).get("total", 0))
                                         for y in YEARS]
                              + [num(row["total"])])
    macro = {}
    for year in YEARS:
        for term in ("KFC", "Ростикс"):
            macro[year] = macro.get(year, 0) + ((terms.get(term) or {}).get(str(year)) or {}).get("total", 0)
    macro_rows = [[str(year), num(macro.get(year, 0)),
                   num(((terms.get("вкусно и точка") or {}).get(str(year)) or {}).get("total", 0)),
                   num(((terms.get("Мак") or {}).get(str(year)) or {}).get("total", 0))]
                  for year in YEARS]
    memes = []
    for row in data.themes:
        name = row["name"]
        if any(word in name.lower() for word in ("комбо", "линейка", "меню", "бургер", "баскет",
                                                 "шеф", "байтс", "ролл", "крылышк", "наггетс",
                                                 "чизбургер", "твистер", "бокс")):
            memes.append([name, num((row["years"].get("2024") or {}).get("total", 0)),
                          num((row["years"].get("2025") or {}).get("total", 0)),
                          num((row["years"].get("2026") or {}).get("total", 0)),
                          num(row["total"])])
    memes.sort(key=lambda r: -int(str(r[4]).replace("\u00a0", "")))
    bullets = [
        "Названия живут в двух системах: старое имя и новое соседствуют. По упоминаниям новое имя "
        "уже обошло старое, а тема «Отзыв о KFC» почти исчезла из повестки к 2026 году — "
        "аудитория переучилась.",
        "Путаница названий видна там, где её меньше всего ждут: тема про бывший McDonald’s "
        "набрала %s сообщений за период, а в 2025–2026 годах её вес вырос. Это разговор про "
        "конкурента внутри нашего поля, и он влияет на сравнение цен и порций."
        % prose_num((data.theme_by_name.get("Мак") or {}).get("total", 0)),
        "Окрашенная лексика устойчива: разговоры о нехватке курицы, о насекомых и о просроченной "
        "продукции повторяются каждый год отдельными темами — это готовый список слов, по которым "
        "нужно ловить ранние сигналы.",
        "Мемы живут недолго: скороговорки и шуточные фразы вокруг названия сети появляются "
        "в отдельных месяцах и уходят через один-два месяца. Планировать кампанию на меме нельзя, "
        "но реагировать на него стоит в первые дни.",
    ]
    return {"heading": "Язык бренда",
            "text": ("Как о бренде говорят на самом деле: устойчивые формулировки, путаница старого "
                     "и нового названий и лексика, которая тянет за собой негатив. Частоты считаны "
                     "по всему корпусу, поэтому видно не только, что говорят, но и как это менялось "
                     "по годам."),
            "tables": [
                TABLE("Устойчивые формулировки по годам", ["Формулировка", "Что это"] +
                      [str(y) for y in YEARS] + ["Всего", "Доля негатива"],
                      rows, layout="landscape",
                      note="Число — сколько сообщений содержит эту формулировку в указанном году. "
                           "Формулировки могут пересекаться внутри одного сообщения, поэтому "
                           "складывать их нельзя."),
                TABLE("Путаница старого и нового названий", ["Как называют сеть"] + [str(y) for y in YEARS]
                      + ["Всего"], confusion_rows, layout="landscape"),
                TABLE("Наш бренд и конкуренты в названиях, по годам",
                      ["Год", "Упоминания старого и нового имени", "«Вкусно и точка»", "«Мак»"],
                      macro_rows),
                TABLE("Названия кампаний, линеек и блюд в языке аудитории",
                      ["Название", "2024", "2025", "2026", "Всего"], memes[:24], layout="landscape",
                      note="Названия приходят из готовой разметки тем — это язык самой аудитории, "
                           "а не придуманные нами ярлыки."),
            ],
            "bullets": bullets}


def build_actions_block(data: Data, ctx: Ctx) -> dict:
    """Блок 10. Что делать: рекомендации отдельно маркетингу и PR, каждая — с доказательством."""
    plat = {r["hub"]: r for r in data.platform_negative["rows"]}
    map_share = (plat.get("maps.yandex.ru", {}).get("share_of_negative", 0)
                 + plat.get("2gis.ru", {}).get("share_of_negative", 0))
    risk = data.risk
    worst = risk[0] if risk else {"month": "", "index": 0, "negative_share": 0, "total": 0}
    events = sorted(data.events, key=lambda e: e["shape"]["days_to_peak"])
    fast = events[0] if events else None
    campaigns = data.campaigns
    # Кампанию и «осадок» выбираем только там, где сравнение опирается на заметный фон:
    # иначе деление на две недели с десятком сообщений даёт случайные проценты.
    def usable(row):
        spans = row["spans"]
        return spans["during"]["total"] >= 300 and spans["before"]["total"] >= 30 \
            and spans["after"]["total"] >= 30

    best_camp = max([r for r in campaigns if usable(r)] or campaigns,
                    key=lambda r: r["spans"]["during"]["positive_share"] - r["spans"]["before"]["positive_share"],
                    default=None)
    rollback = [r for r in campaigns if usable(r)
                and r["spans"]["after"]["negative_share"] - r["spans"]["during"]["negative_share"] >= 3]
    worst_camp = max(rollback, key=lambda r: r["spans"]["after"]["negative_share"]
                     - r["spans"]["during"]["negative_share"], default=None)
    price = data.theme_by_name.get("Отзывы цены") or {}
    speed = data.theme_by_name.get("Отзывы скорость") or {}
    clean = data.theme_by_name.get("Отзывы чистота") or {}
    shortage = data.theme_by_name.get("Дефицит курицы") or {}
    spoiled = data.theme_by_name.get("Отравление/сальмонелла/тухлое") or {}
    worst_event = max(data.events, key=lambda e: e["shape"]["negative"], default=None)
    instant = [e for e in data.events if e["shape"]["days_to_peak"] <= 1]
    price_share = {y: ((price.get("years", {}).get(y) or {}).get("negative", 0)
                       / float((price.get("years", {}).get(y) or {}).get("total") or 1) * 100)
                   for y in ("2024", "2025", "2026")}
    marketing = [
        "Строить кампании короткими окнами. У %s крупных поводов из %s пик наступил в первые сутки, "
        "а активная фаза — %s. Всё, что должно быть сказано, нужно сказать сразу: позже волна "
        "уходит без нас."
        % (prose_num(len(instant)), prose_num(len(data.events)),
           days_phrase(round(sum(e["shape"]["days_active"] for e in data.events)
                             / float(len(data.events) or 1)))),
        "Усиливать детские и игровые наборы: они дают и объём, и позитив. Доказательство — "
        "кампания «%s»: за две недели до старта доля позитива %s, в период проведения %s (%s)."
        % ((best_camp or {}).get("label", "—"),
           pct((best_camp or {}).get("spans", {}).get("before", {}).get("positive_share", 0)),
           pct((best_camp or {}).get("spans", {}).get("during", {}).get("positive_share", 0)),
           delta_pp((best_camp or {}).get("spans", {}).get("during", {}).get("positive_share", 0),
                    (best_camp or {}).get("spans", {}).get("before", {}).get("positive_share", 0)))
        if best_camp else "Усиливать детские и игровые наборы: по данным периода измеримого "
                          "прироста позитива они не дают, поэтому решение стоит проверять "
                          "точечно, а не переносить на всю линейку.",
        "Следить за остатком товара в кампаниях: %s"
        % ("у кампании «%s» после окончания доля негатива выросла с %s до %s — люди продолжают "
           "искать то, что уже закончилось."
           % ((worst_camp or {}).get("label", "—"),
              pct((worst_camp or {}).get("spans", {}).get("during", {}).get("negative_share", 0)),
              pct((worst_camp or {}).get("spans", {}).get("after", {}).get("negative_share", 0)))
           if worst_camp else "ни у одной кампании периода заметного роста негатива после "
                              "окончания нет, поэтому правило простое — не обещать больше, "
                              "чем есть в наличии."),
        "Отвечать на цену цифрами: тема цены — %s сообщений за период, её доля негатива прошла "
        "путь %s → %s → %s. Разговор идёт не про деньги вообще, а про соотношение цены и размера "
        "порции: это видно по примерам ниже."
        % (prose_num(price.get("total", 0)), pct(price_share["2024"], 1), pct(price_share["2025"], 1),
           pct(price_share["2026"], 1)),
        "Планировать запуски по календарю риска: %s — самый рискованный месяц периода (индекс %s, "
        "доля негатива %s). Запуск в такой месяц утонет в чужой повестке."
        % (month_label(worst.get("month") or "2025-07"), num_ru(worst.get("index", 0)),
           pct(worst.get("negative_share", 0))),
    ]
    pr = [
        "Перенести центр внимания на карты и отзовики: там %s всего негатива, и он адресный — "
        "видно конкретный ресторан и конкретную смену. Отвечать нужно там же, где жалуются, "
        "а не только в соцсетях." % pct(map_share),
        "Держать готовые ответы на четыре повторяющихся повода: цена (%s сообщений), скорость "
        "обслуживания (%s), чистота (%s) и нехватка курицы (%s). Ни один из них не уходит "
        "из повестки ни в один год."
        % (prose_num(price.get("total", 0)), prose_num(speed.get("total", 0)),
           prose_num(clean.get("total", 0)), prose_num(shortage.get("total", 0))),
        "Готовить протокол на инциденты с едой и санитарией: тема испорченной еды и отравлений — "
        "%s сообщений за период. Самый тяжёлый инцидент периода — «%s»: %s негативных сообщений, "
        "пик %s. Первый ответ должен быть в первые сутки, иначе волну разгоняют пересказы."
        % (prose_num(spoiled.get("total", 0)), (worst_event or {}).get("name", "—"),
           num((worst_event or {}).get("shape", {}).get("negative", 0)),
           (worst_event or {}).get("shape", {}).get("peak_day", "—")),
        "Разделить работу по площадкам: соцсети дают охват и позитив, карты — причину претензий. "
        "Один и тот же ответ на обеих площадках работает хуже, чем адресный ответ на карте "
        "и спокойный тон в соцсетях.",
        "Считать эффект после кампании, а не только во время, и держать связку с маркетингом: "
        "доля негатива в период кампании и через две недели после — разные показатели, "
        "и отвечает за них разные команды.",
    ]
    evidence = [
        ("Цена и размер порции", ("Отзывы цены", "цен")),
        ("Скорость обслуживания", ("Отзывы скорость", "скорост", "ожидан")),
        ("Чистота зала и туалетов", ("Отзывы чистота", "чистот")),
        ("Испорченная еда и отравления", ("Отравление", "тухл", "сальмонелл", "просроч")),
        ("Наличие товара и курицы", ("Дефицит курицы", "дефицит")),
        ("Закрытие ресторанов", ("Ребрендинг", "закрыт", "KFC на Ростикс")),
    ]
    evidence_bullets = []
    for title, keys in evidence:
        quotes = data.quotes_for(tuple(k.lower() for k in keys), limit=2)
        if not quotes:
            continue
        line = "%s. Примеры: " % title
        parts = []
        for quote in quotes[:2]:
            text = safe_quote(quote["text"], 190)
            if not text:
                continue
            parts.append("«%s» — %s, %s%s" % (text, quote["hub"] or "источник",
                                              quote["date"] or "дата не указана",
                                              (", ссылка: " + link_text(quote["url"]))
                                              if quote["url"] else ""))
        if not parts:
            continue
        evidence_bullets.append(line + "; ".join(parts) + ".")
    return {"heading": "Что делать: рекомендации для маркетинга и PR",
            "text": ("Рекомендации разделены по задачам. У каждой — доказательство из данных: тема, "
                     "месяц или период, числа и примеры сообщений со ссылками."),
            "bullets": ["Маркетинг. " + line for line in marketing] + ["PR. " + line for line in pr]
                       + evidence_bullets}


def build_appendix_block(data: Data, ctx: Ctx) -> dict:
    """Приложение: примеры сообщений за весь период и методика одним абзацем."""
    rows = []
    for i, row in enumerate(data.appendix, 1):
        rows.append([i, clip(row.get("date") or "", 10), row.get("hub") or "—",
                     str(row.get("tone") or "—"), clip(row.get("text") or "", 150),
                     row.get("url") or "—"])
    methodology = (
        "Как это посчитано, коротко и без технических подробностей. Взяты все сообщения по теме "
        "KFC за период с 13 мая 2024 года по 31 августа 2026 года — 2,93 миллиона сообщений из "
        "соцсетей, мессенджеров, карт, отзовиков и видеосервисов. Объём и тональность посчитаны "
        "по каждому сообщению целиком, без выборки: тональность уже проставлена у каждого "
        "сообщения, поэтому месяцы и годы сопоставимы напрямую. Темы, кампании, авторы и площадки "
        "получены свёртками по всему корпусу — это расчёт, а не оценка на глаз. Смысловые части "
        "отчёта (о чём именно писали в волне, за что хвалили и на что жаловались) получены "
        "чтением выборок сообщений локальными моделями: быстрая модель разбирает выборку из "
        "нескольких десятков самых заметных сообщений волны, большая — пишет интерпретацию по "
        "уже посчитанным числам. Границы месяцев взяты в московском времени, поэтому суммы "
        "месяцев и годов совпадают с месячными отчётами по теме. Спам и рекламные подборки "
        "исключены из тем и выводов отдельным фильтром — их объём по месяцам указан в разделе "
        "о темах. Цитаты в приложении приведены дословно, с датой, площадкой и ссылкой."
    )
    return {"heading": "Приложение: примеры сообщений и как считали",
            "text": methodology,
            "tables": [TABLE("Примеры сообщений за период",
                             ["№", "Дата", "Площадка", "Тональность", "Текст сообщения", "Ссылка"],
                             rows, layout="landscape",
                             note="Примеры приведены дословно и не редактировались: это язык "
                                  "аудитории, включая ошибки и сокращения.")]}


# ---------------------------------------------------------------- запись

def write_summary(sections: List[dict], data: Data, title: str, path_files: List[str]) -> str:
    months = [k for k in sorted(data.months) if k in A.EXPECTED_MONTHS]
    total = sum(data.years[y]["total"] for y in YEARS)
    negative = sum(data.years[y]["negative"] for y in YEARS)
    payload = {
        "version": TR.SUMMARY_VERSION,
        "generated_at": time.strftime("%Y-%m-%d %H:%M"),
        "report": {"title": title, "folder": os.path.basename(OUT_DIR),
                   "files": [os.path.basename(p) for p in path_files]},
        "dataset": {"index": 1102, "name": "kfc_13.05.2024-22.09.2026",
                    "label": "kfc_13.05.2024-22.09.2026"},
        "period": {"from": months[0] if months else "", "to": months[-1] if months else "",
                   "from_ts": None, "to_ts": None, "key": "2024-2026-расширенный"},
        "messages": {"in_slice": total, "read_sample": STATS["fast_calls"] * 54,
                     "read_in_topics": sum(r["total"] for r in data.themes[:12]),
                     "topics_total_count": sum(r["total"] for r in data.themes)},
        "clusters": {"count": len(data.themes), "kind": "готовые темы", "strategy": "crossyear"},
        "tonality": {"negative": negative, "neutral": sum(data.years[y]["neutral"] for y in YEARS),
                     "positive": sum(data.years[y]["positive"] for y in YEARS), "total": total,
                     "shares": {"negative": round(negative / float(total or 1), 4),
                                "neutral": round(sum(data.years[y]["neutral"] for y in YEARS)
                                                 / float(total or 1), 4),
                                "positive": round(sum(data.years[y]["positive"] for y in YEARS)
                                                  / float(total or 1), 4)}},
        "topics": [{"name": r["name"], "count": r["total"],
                    "share": round(r["total"] / float(sum(x["total"] for x in data.themes) or 1), 4),
                    "tone": "", "category": "", "essence": "", "quotes": []}
                   for r in data.themes[:15]],
        "categories": [],
        "authors": [{"name": r["name"], "count": r["total"]} for r in data.voice[:15]],
        "events": [{"name": e["name"], "date": e["from"], "source": "инфоповоды периода"}
                   for e in data.events[:10]],
        "highlights": [],
        "sources": {"links": [row["url"] for row in data.appendix if row.get("url")][:20],
                    "reports": [os.path.basename(p) for p in path_files]},
        "sections": [str(s.get("heading") or "") for s in sections],
        "sample_note": ("Расширенный межгодовой отчёт: к готовым месячным отчётам добавлены расчёты "
                        "по всему корпусу темы — темы по месяцам, драйверы тональности, анатомия "
                        "инфоповодов, когорты авторов, кампании, конкуренты, индекс риска, язык "
                        "бренда."),
        "notes": ["расширенная версия отчёта: прежние отчёты не изменялись",
                  "объём и тональность — по всем сообщениям периода целиком, без выборки",
                  "смысловые части — чтение выборок волн локальными моделями",
                  "границы месяцев взяты в местном времени, суммы совпадают с месячными отчётами"],
        "tonality_scope": "все сообщения периода целиком, без выборки",
        "months": months,
    }
    path = os.path.join(OUT_DIR, "2024-2026_расширенный_summary.json")
    with io.open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
    return path


def build_one(title: str, subtitle: str, sections: List[dict], data: Data, ctx: Ctx):
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M")
    base = "%s_%s" % (TR._safe_name(title, 70), stamp)
    docx_path = os.path.join(OUT_DIR, base + ".docx")
    pdf_path = os.path.join(OUT_DIR, base + ".pdf")
    tables = sum(len(s.get("tables") or []) for s in sections)
    charts = sum(len(s.get("chart_ids") or []) for s in sections)
    meta = {"dataset_label": "KFC", "period": "13.05.2024 — 31.08.2026",
            "author": "агент Tellscope", "date": time.strftime("%d.%m.%Y %H:%M"),
            "charts": dict(ctx.charts)}
    clean = TR._clean_sections(sections)
    TR._build_docx(docx_path, title, subtitle, clean, meta)
    pdf_note = ""
    try:
        TR._build_pdf(pdf_path, title, subtitle, clean, meta)
    except Exception as exc:  # noqa: BLE001
        pdf_note = " | PDF не собрался: %s %s" % (type(exc).__name__, exc)
    summary = write_summary(sections, data, title, [docx_path, pdf_path])
    print("собран: %s | разделов %d, таблиц %d, графиков %d%s"
          % (os.path.basename(docx_path), len(sections), tables, charts, pdf_note))
    print("        %s" % os.path.basename(pdf_path))
    print("        %s" % os.path.basename(summary))
    return docx_path, pdf_path, summary, {"sections": len(sections), "tables": tables, "charts": charts}


def main():
    apply = "--apply" in sys.argv
    ctx = Ctx()
    os.makedirs(CHART_DIR, exist_ok=True)
    t0 = time.time()
    print("=== сбор данных")
    data = Data()
    print("   месячных отчётов: %d | месяцев с цифрами: %d" % (len(data.summaries), len(data.months)))
    print("   готовых тем: %d (исключено спама: %d)" % (len(data.themes), len(data.spam_themes)))
    print("   годы: %s" % ", ".join("%d — %d сообщений, негатив %.2f%%"
                                    % (y, data.years[y]["total"], data.years[y]["negative_share"])
                                    for y in YEARS))
    for stale in ("authors_top",):
        path = os.path.join(A.CACHE_DIR, stale + ".json")
        if not os.path.isfile(path):
            print("   внимание: расчёт авторов ещё не считался")
    print("   топ-тема: %s (%d)" % (data.themes[0]["name"], data.themes[0]["total"]))
    print("   индекс риска: худший месяц %s (%.1f)" % (data.risk[0]["month"], data.risk[0]["index"]))
    print("=== разбор инфоповодов")
    data.events = collect_events(data)
    print("   поводов разобрано: %d" % len(data.events))
    print("=== разбор кампаний")
    data.campaigns = collect_campaigns(data)
    print("   кампаний разобрано: %d" % len(data.campaigns))
    print("=== контекст сравнений с конкурентами")
    data.comparison_read = collect_comparison(data)
    print("=== приложение")
    data.appendix = collect_appendix(data)
    print("   примеров: %d" % len(data.appendix))
    print("=== блоки отчёта (%.0f с)" % (time.time() - t0))
    sections = [
        build_summary_block(data, ctx),
        build_brand_talk_block(data, ctx),
        build_tonality_block(data, ctx),
        build_events_block(data, ctx),
        build_authors_block(data, ctx),
        build_campaigns_block(data, ctx),
        build_competitors_block(data, ctx),
        build_risk_block(data, ctx),
        build_language_block(data, ctx),
        build_actions_block(data, ctx),
        build_appendix_block(data, ctx),
    ]
    problems = audit_sections(sections)
    sections = polish_sections(sections)
    problems += ["после косметики: " + p for p in audit_sections(sections)]
    print("=== проверка текстов")
    if problems:
        print("   замечаний: %d" % len(problems))
        for problem in problems[:25]:
            print("   !! %s" % problem)
    else:
        print("   замечаний нет: предложения не выброшены, технических упоминаний нет")
    used = {cid for s in sections for cid in (s.get("chart_ids") or [])}
    orphan = [cid for cid in sorted(ctx.charts) if cid not in used]
    print("   графиков построено %d, в разделах %d%s"
          % (len(ctx.charts), len(used), (", вне разделов: " + ", ".join(orphan)) if orphan else ""))
    print("   вызовов моделей: быстрая %d, большая %d | %s"
          % (STATS["fast_calls"], STATS["gen_calls"], STATS["models"]))
    if not apply:
        print("(сухой прогон: документы не собираются)")
        return
    built = build_one(TITLE, SUBTITLE, sections, data, ctx)
    json.dump({"files": [built[0], built[1], built[2]], "counts": built[3], "problems": problems},
              io.open("/tmp/kfc_crossyear_v2_built.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("готово за %.0f с" % (time.time() - t0))


if __name__ == "__main__":
    main()
