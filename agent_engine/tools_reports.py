# -*- coding: utf-8 -*-
"""Инструменты отчётов: графики, сборка DOCX/PDF, сохранение во вкладке «Отчёты»."""
from __future__ import annotations

import json
import os
import re
import textwrap
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from .context import compact
from .registry import ToolError, tool

REPORTS_DIR_NAME = "reports_directory"
BACKEND_ROOT = "/home/dev/tellscope_app/tellscope_backend"

MATPLOTLIB_READY = False
PALETTE = ["#2F6BFF", "#E4002B", "#12B76A", "#F79009", "#7A5AF8", "#06AED4", "#EE46BC", "#98A2B3"]


def _mpl():
    """Настраивает matplotlib один раз: шрифт с кириллицей, светлая тема."""
    global MATPLOTLIB_READY
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not MATPLOTLIB_READY:
        plt.rcParams["font.family"] = "DejaVu Sans"
        plt.rcParams["axes.unicode_minus"] = False
        plt.rcParams["figure.facecolor"] = "white"
        plt.rcParams["axes.facecolor"] = "white"
        plt.rcParams["axes.edgecolor"] = "#D0D5DD"
        plt.rcParams["axes.grid"] = True
        plt.rcParams["grid.color"] = "#EAECF0"
        plt.rcParams["grid.linewidth"] = 0.8
        MATPLOTLIB_READY = True
    return plt


def _safe_name(text: str, limit: int = 60) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|\n\r\t]+', " ", str(text or "")).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return (cleaned[:limit] or "report").strip()


def _reports_dir(user_id: str, folder: str) -> str:
    path = os.path.join(BACKEND_ROOT, "data", str(user_id), REPORTS_DIR_NAME, _safe_name(folder, 40))
    os.makedirs(path, exist_ok=True)
    return path


RU_MONTHS = ("января", "февраля", "марта", "апреля", "мая", "июня",
             "июля", "августа", "сентября", "октября", "ноября", "декабря")
BRAND_CASE = {
    "kfc": "KFC", "кфс": "KFC", "rostics": "Rostic's", "rostic": "Rostic's", "ростикс": "Rostic's",
    "platon": "Platon", "ozon": "Ozon", "ba": "", "brand": "", "analytics": "",
}
PERIOD_RE = re.compile(r"(\d{2})\.(\d{2})\.(\d{4})\s*[-—–]\s*(\d{2})\.(\d{2})\.(\d{4})")
STAMP_RE = re.compile(r"\b20\d{6}(?:[_ ]?\d{4,6})?\b")
# Технические подробности, которым не место в отчёте: их видит только журнал запуска.
TECHNICAL_RE = re.compile(
    r"(\b404\b|\b40[0-9]\b|\b50[0-9]\b|Unexpected Response|doesn'?t exist|\bCollection\b|"
    r"\bQdrant\b|Traceback|\bHTTP\b|\bTimeout\b|\bException\b|\bError\b|Connection|"
    r"недоступ\w*|Причина:|коллекц\w*|векторн\w*|traceback)",
    re.IGNORECASE,
)


def _ru_date(day: str, month: str, year: str) -> str:
    try:
        return "%d %s %s" % (int(day), RU_MONTHS[int(month) - 1], year)
    except Exception:  # noqa: BLE001
        return "%s.%s.%s" % (day, month, year)


def _pretty_topic_name(value: Any) -> str:
    """Название темы человеческим языком: 'kfc_13.05.2024-22.09.2026' → 'KFC',
    'ba_озон_отзывы_20260912_150434' → 'Озон отзывы', 'platon_13.10.2025-30.11.2025' → 'Platon'."""
    raw = str(value or "").strip()
    if not raw:
        return ""
    text = PERIOD_RE.sub(" ", raw)
    text = STAMP_RE.sub(" ", text)
    text = re.sub(r"[_]+", " ", text)
    words: List[str] = []
    for word in re.split(r"\s+", text):
        cleaned = word.strip(" -–—·,")
        if not cleaned:
            continue
        low = cleaned.lower()
        if low in BRAND_CASE:
            prefix = BRAND_CASE[low]
            if prefix:
                words.append(prefix)
            continue
        if cleaned[:1].isdigit():
            continue
        words.append(cleaned[:1].upper() + cleaned[1:])
    name = " ".join(words).strip()
    if not name:
        name = re.sub(r"[_]+", " ", raw).strip()
    return name


RU_MONTH_WORDS = {
    "январ": 1, "феврал": 2, "март": 3, "апрел": 4, "ма": 5, "июн": 6, "июл": 7,
    "август": 8, "сентябр": 9, "октябр": 10, "ноябр": 11, "декабр": 12,
}
EN_MONTH_WORDS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _period_key_from_text(*values: Any) -> str:
    """Ключ периода ГГГГ-ММ из текста отчёта, когда в контексте периода нет.

    Цепочки и часть прогонов не передают период, и итог назывался по текущей дате
    (февральский отчёт сохранялся как 2026-09_summary.json). Поэтому ищем месяц и год в
    названии отчёта, папке и заголовках разделов: «за февраль 2026», «feb 2026», «02.2026».
    """
    haystack = " ".join(str(value or "") for value in values).lower().replace("ё", "е")
    if not haystack.strip():
        return ""
    for word, month in RU_MONTH_WORDS.items():
        match = re.search(word + r"[а-я]*\s*[._-]?\s*(20\d{2})", haystack)
        if match:
            return "%s-%02d" % (match.group(1), month)
    for word, month in EN_MONTH_WORDS.items():
        match = re.search(r"\b" + word + r"[a-z]*[._\s-]*(20\d{2})", haystack)
        if match:
            return "%s-%02d" % (match.group(1), month)
    match = re.search(r"\b(20\d{2})[._-](\d{2})\b", haystack)
    if match and 1 <= int(match.group(2)) <= 12:
        return "%s-%s" % (match.group(1), match.group(2))
    match = re.search(r"\b(\d{2})[.\-/](20\d{2})\b", haystack)
    if match and 1 <= int(match.group(1)) <= 12:
        return "%s-%s" % (match.group(2), match.group(1))
    return ""


def _topic_period(name: Any) -> str:
    """Период из имени датасета по-русски: '13 мая 2024 — 22 сентября 2026'. Нет периода — пусто."""
    match = PERIOD_RE.search(str(name or ""))
    if not match:
        stamp = re.search(r"\b(20\d{2})(\d{2})(\d{2})\b", str(name or ""))
        if stamp:
            return "%d %s %s" % (int(stamp.group(3)), RU_MONTHS[int(stamp.group(2)) - 1], stamp.group(1))
        return ""
    day1, mon1, year1, day2, mon2, year2 = match.groups()
    return "%s — %s" % (_ru_date(day1, mon1, year1), _ru_date(day2, mon2, year2))


def topic_header(*values: Any) -> str:
    """Шапка отчёта: тема + период, если он читается из имени датасета."""
    name, period = "", ""
    for value in values:
        if not name:
            name = _pretty_topic_name(value)
        if not period:
            period = _topic_period(value)
        if name and period:
            break
    if name and period:
        return "%s %s" % (name, period)
    return name or period or "—"


def _sanitize_report_text(value: Any) -> str:
    """Убирает из текста отчёта технические сообщения (404, Qdrant, «коллекция не существует»).

    Это служебные подробности: пользователю они не нужны, а в документе выглядят как ошибка.
    Цитаты сообщений не трогаем — они должны оставаться дословными.
    """
    text = str(value or "")
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", text)
    kept = [part for part in parts if part.strip() and not TECHNICAL_RE.search(part)]
    cleaned = " ".join(kept).strip()
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned


def _clean_sections(sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Чистит пояснительный текст разделов; пустые после чистки разделы без данных убираем."""
    out: List[Dict[str, Any]] = []
    for section in sections:
        if not isinstance(section, dict):
            continue
        section = dict(section)
        section["text"] = _sanitize_report_text(section.get("text"))
        if section.get("note"):
            section["note"] = _sanitize_report_text(section["note"])
        for key in ("bullets", "items"):
            if isinstance(section.get(key), list):
                section[key] = [item for item in (_sanitize_report_text(x) for x in section[key]) if item]
        section["tables"] = _clean_tables(section.get("tables"))
        for finding in section.get("findings") or []:
            if isinstance(finding, dict) and finding.get("essence"):
                finding["essence"] = _sanitize_report_text(finding["essence"])
        has_data = bool(section.get("findings") or section.get("highlights") or section.get("chart_ids")
                        or section.get("citations") or section.get("text") or section.get("bullets")
                        or section.get("tables"))
        if has_data:
            out.append(section)
    return out or sections


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:,.2f}".replace(",", " ").replace(".", ",")
    if isinstance(value, int):
        return f"{value:,}".replace(",", " ")
    return str(value)


# Маркеры списка, которые инструмент иногда присылает уже внутри текста. Стиль «List Bullet»
# рисует свой маркер, поэтому второй («• • пункт») нужно убрать.
BULLET_CHARS = "•·▪●◦‣∙"


def _strip_bullet(value: Any) -> str:
    """Убирает ведущие маркеры списка из строки пункта (может быть несколько подряд)."""
    text = str(value if value is not None else "").strip()
    while text and text[0] in BULLET_CHARS:
        text = text[1:].strip()
    return text


def _clean_cell(value: Any) -> str:
    """Ячейка таблицы: только косметика — переводы строк, табы, лишние пробелы.

    Фильтр технических сообщений (TECHNICAL_RE) к ячейкам не применяется: он режет текст по
    предложениям, а в ячейке лежит одно значение — иначе пропадали бы числа вида 404 или 507.
    """
    text = str(value if value is not None else "").strip()
    text = re.sub(r"[\r\n\t]+", " ", text)
    return re.sub(r"\s{2,}", " ", text)


def _clean_tables(tables: Any) -> List[Dict[str, Any]]:
    """Готовит таблицы разделов к сборке: колонки, строки, заголовок, примечание и раскладка.

    ``layout`` переносится как есть («auto» по умолчанию): по нему PDF выбирает книжную или
    альбомную страницу для конкретной таблицы.
    """
    out: List[Dict[str, Any]] = []
    for spec in (tables or []):
        if not isinstance(spec, dict):
            continue
        columns = [_clean_cell(c) for c in (spec.get("columns") or [])]
        rows = [[_clean_cell(c) for c in row] for row in (spec.get("rows") or [])
                if isinstance(row, (list, tuple))]
        rows = [row for row in rows if any(row)]
        if not columns and not rows:
            continue
        layout = str(spec.get("layout") or "").strip().lower()
        out.append({
            "title": _clean_cell(spec.get("title")),
            "columns": columns,
            "rows": rows,
            "note": _sanitize_report_text(spec.get("note")) if spec.get("note") else "",
            "layout": layout if layout in ("auto", "portrait", "landscape") else "auto",
        })
    return out


NUMERIC_HINTS = ("сообщени", "упоминани", "доля", "месяцев", "месяц", "объём", "объем", "%", "всего")


def _is_numeric_label(label: Any) -> bool:
    low = str(label or "").lower()
    return any(hint in low for hint in NUMERIC_HINTS)


def _is_numeric_cell(value: Any) -> bool:
    text = (str(value if value is not None else "")
            .replace(" ", "").replace("\u00a0", "").replace("%", "").replace("—", "").strip())
    if not text:
        return True  # пустая ячейка и прочерк не мешают считать колонку числовой
    return bool(re.fullmatch(r"-?\d+([.,]\d+)?", text))


def _set_table_align(table, labels: List[str]) -> None:
    """Числовые колонки — по правому краю, текстовые — по левому (единый стиль таблиц)."""
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    right = [_is_numeric_label(label) for label in labels]
    if not any(right):
        return
    for row in table.rows:
        for idx, cell in enumerate(row.cells):
            if idx < len(right) and right[idx]:
                for para in cell.paragraphs:
                    para.alignment = WD_ALIGN_PARAGRAPH.RIGHT


def _opt_float(value: Any) -> Optional[float]:
    """None и пустая строка → None («нет данных»), остальное → float."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_list(value: Any, name: str) -> List[Any]:
    """Приводит параметр к списку. Строки с JSON принимаются: так параметры приходят
    из Dify, MCP и обычных HTTP-вызовов, где массивы передаются текстом."""
    if value is None or value == "":
        return []
    if isinstance(value, str):
        text = value.strip()
        try:
            value = json.loads(text)
        except Exception as exc:  # noqa: BLE001
            raise ToolError(f"{name}: ожидается список или JSON-строка со списком ({exc})") from exc
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        raise ToolError(f"{name}: ожидается список, а пришло {type(value).__name__}")
    out: List[Any] = []
    for item in value:
        if isinstance(item, str):
            stripped = item.strip()
            if stripped[:1] in ("{", "["):
                try:
                    item = json.loads(stripped)
                except Exception:  # noqa: BLE001
                    pass
        out.append(item)
    return out


@tool(
    "list_reports",
    title="Готовые отчёты",
    description="Список уже сохранённых отчётов пользователя во вкладке «Отчёты»: папки-датасеты и файлы.",
    parameters={"type": "object", "properties": {}},
    group="reports",
)
async def list_reports(ctx):
    root = os.path.join(BACKEND_ROOT, "data", str(ctx.user_id), REPORTS_DIR_NAME)
    values: List[Dict[str, Any]] = []
    if os.path.isdir(root):
        for folder in sorted(os.listdir(root)):
            fdir = os.path.join(root, folder)
            if not os.path.isdir(fdir):
                continue
            files = []
            for name in sorted(os.listdir(fdir)):
                fp = os.path.join(fdir, name)
                if os.path.isfile(fp):
                    files.append({"name": name, "size": os.path.getsize(fp)})
            if files:
                values.append({"folder": folder, "files": files})
    return {"reports": compact(values, max_items=20), "reports_root": root}


@tool(
    "make_chart",
    title="Построить график",
    description=(
        "Строит график по переданным данным и сохраняет его как артефакт запуска. "
        "Поддерживаются столбцы (bar), горизонтальные столбцы (hbar), линии (line), области (area) и круговая (pie). "
        "Возвращает chart_id, который затем указывается в build_report."
    ),
    parameters={
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "заголовок графика"},
            "chart_type": {"type": "string", "enum": ["bar", "hbar", "line", "area", "pie"], "description": "тип графика"},
            "categories": {"type": "array", "items": {"type": "string"}, "description": "подписи по оси X (месяцы, годы, площадки)"},
            "series": {
                "type": "array",
                "description": "ряды данных",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "values": {"type": "array", "items": {"type": "number"}},
                    },
                    "required": ["name", "values"],
                },
            },
            "x_label": {"type": "string"},
            "y_label": {"type": "string"},
            "stacked": {"type": "boolean", "description": "составные столбцы/области"},
            "note": {"type": "string", "description": "подпись-источник под графиком"},
        },
        "required": ["title", "chart_type", "categories", "series"],
    },
    group="reports",
    timeout=120.0,
)
async def make_chart(
    ctx,
    title: str,
    chart_type: str,
    categories: List[str],
    series: List[Dict[str, Any]],
    x_label: str = "",
    y_label: str = "",
    stacked: bool = False,
    note: str = "",
):
    plt = _mpl()
    categories = _coerce_list(categories, "categories")
    series = _coerce_list(series, "series")
    if not categories:
        raise ToolError("categories не может быть пустым")
    if not series:
        raise ToolError("нужен хотя бы один ряд данных (series)")
    series = [s for s in series if isinstance(s, dict) and s.get("values") is not None]
    if not series:
        raise ToolError('series: ожидается список объектов вида [{"name": "ряд", "values": [1, 2, 3]}]')
    cats = [str(c) for c in categories]
    fig, ax = plt.subplots(figsize=(10, 5.4), dpi=170)
    idx = list(range(len(cats)))
    ctype = (chart_type or "bar").lower()

    if ctype == "pie":
        values = [float(v or 0) for v in (series[0].get("values") or [])]
        ax.clear()
        ax.grid(False)
        colors = [PALETTE[i % len(PALETTE)] for i in range(len(values))]
        ax.pie(values, labels=cats, autopct=lambda p: f"{p:.1f}%", colors=colors, textprops={"fontsize": 9})
        ax.axis("equal")
    elif ctype == "line" or ctype == "area":
        for i, s in enumerate(series):
            # None в ряду означает «нет данных»: линия рвётся, а не рисует ложный ноль.
            values = [_opt_float(v) for v in (s.get("values") or [])]
            color = PALETTE[i % len(PALETTE)]
            if ctype == "area":
                ax.fill_between(idx, [0.0 if v is None else v for v in values], alpha=0.18, color=color)
            ax.plot(idx, values, marker="o", markersize=4, linewidth=2, color=color, label=str(s.get("name") or f"ряд {i+1}"))
    elif ctype == "hbar":
        values = [float(v or 0) for v in (series[0].get("values") or [])]
        order = sorted(range(len(values)), key=lambda k: values[k])
        ax.barh([cats[k] for k in order], [values[k] for k in order], color=PALETTE[0], height=0.6)
    else:
        width = 0.8 / max(1, len(series)) if len(series) > 1 else 0.62
        bottom = [0.0] * len(cats)
        for i, s in enumerate(series):
            values = [float(v or 0) for v in (s.get("values") or [])]
            pos = [k - (0.8 - width) / 2 + i * width for k in idx] if len(series) > 1 else idx
            ax.bar(pos, values, width=width, label=str(s.get("name") or f"ряд {i+1}"), color=PALETTE[i % len(PALETTE)], bottom=bottom if stacked else None)
            if stacked:
                bottom = [b + v for b, v in zip(bottom, values)]
        ax.set_xticks(idx)
        ax.set_xticklabels(cats, rotation=30 if max(len(c) for c in cats) > 6 else 0, ha="right" if max(len(c) for c in cats) > 6 else "center", fontsize=9)

    if ctype != "pie" and ctype != "hbar":
        ax.set_xticks(idx)
        ax.set_xticklabels(cats, rotation=30 if max(len(c) for c in cats) > 6 else 0, ha="right" if max(len(c) for c in cats) > 6 else "center", fontsize=9)
    if x_label:
        ax.set_xlabel(x_label, fontsize=10)
    if y_label:
        ax.set_ylabel(y_label, fontsize=10)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    if len(series) > 1 or ctype == "pie":
        ax.legend(fontsize=9, frameon=False)
    if note:
        fig.text(0.01, 0.01, note, fontsize=8, color="#667085")
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    chart_id = f"chart{len(ctx.charts) + 1}"
    name = f"{chart_id}_{_safe_name(title, 50)}.png"
    path = os.path.join(ctx.artifacts_dir, name)
    os.makedirs(ctx.artifacts_dir, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)

    art = ctx.add_artifact("chart", title, path, url=f"/api/agent/artifact/{ctx.run_id}/{name}", meta={"chart_id": chart_id, "chart_type": ctype})
    ctx.charts[chart_id] = {"chart_id": chart_id, "title": title, "name": name, "path": path, "url": art["url"]}
    return {"chart_id": chart_id, "name": name, "title": title, "url": art["url"], "artifact": compact(art)}


def _docx_hyperlink(paragraph, url: str, text: str) -> None:
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    from docx.opc.constants import RELATIONSHIP_TYPE as RT

    try:
        r_id = paragraph.part.relate_to(url, RT.HYPERLINK, is_external=True)
    except Exception:
        paragraph.add_run(f"{text} ({url})")
        return
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), r_id)
    run = OxmlElement("w:r")
    rpr = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "2F6BFF")
    underline = OxmlElement("w:u")
    underline.set(qn("w:val"), "single")
    rpr.append(color)
    rpr.append(underline)
    run.append(rpr)
    text_el = OxmlElement("w:t")
    text_el.text = text
    run.append(text_el)
    hyperlink.append(run)
    paragraph._p.append(hyperlink)


def _fmt_share(value: Any) -> str:
    """0.4634 → «46,3%»; уже готовый процент (46.3) не переводим повторно."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value or "")
    if number <= 1.0:
        number *= 100.0
    return f"{number:.1f}".replace(".", ",") + "%"


def _quote_text(quote: Any) -> str:
    if isinstance(quote, dict):
        return str(quote.get("text") or "").strip()
    return str(quote or "").strip()


def _quote_attribution(quote: Any) -> str:
    """«otzovik.com · 2026-09-10 14:20 · Zhenyas123» — площадка, дата, автор."""
    if not isinstance(quote, dict):
        return ""
    parts = [str(quote.get("hub") or "").strip(), str(quote.get("date") or "").strip(), str(quote.get("author") or "").strip()]
    if quote.get("rating") is not None:
        try:
            parts.append("оценка %g" % float(quote["rating"]))
        except (TypeError, ValueError):
            pass
    return " · ".join(part for part in parts if part)


def _finding_rows(findings: Any) -> List[Dict[str, Any]]:
    return [item for item in (findings or []) if isinstance(item, dict)]


def _section_note(section: Dict[str, Any]) -> str:
    """Мягкая пометка раздела: что именно построено по выборке, а что — по всему срезу.

    Текст готовит инструмент (analyze_texts), без конкретных чисел: он объясняет смысл,
    чтобы читатель отчёта не решил, будто прочитаны все сообщения периода.
    """
    note = section.get("note")
    return str(note).strip() if isinstance(note, str) else ""


def _findings_blocks(section: Dict[str, Any]) -> List[str]:
    """Текстовое представление findings для PDF: тема, число/доля, цитаты с атрибуцией."""
    blocks: List[str] = []
    if _section_note(section):
        blocks.append("(" + _section_note(section) + ")")
    for item in _finding_rows(section.get("findings")):
        topic = str(item.get("topic") or "Тема")
        head = f"{topic} — {item.get('count')} сообщ."
        if item.get("share") is not None:
            head += f" ({_fmt_share(item.get('share'))} среза)"
        if item.get("tone"):
            head += f", тональность: {item.get('tone')}"
        if item.get("category"):
            head += f", категория: {item.get('category')}"
        blocks.append("• " + head)
        explanation = item.get("essence") or item.get("summary")
        if explanation:
            blocks.append(f"   пояснение: {explanation}")
        for quote in (item.get("quotes") or []):
            text = _quote_text(quote)
            if not text:
                continue
            attribution = _quote_attribution(quote)
            blocks.append(f"   «{text}»" + (f" — {attribution}" if attribution else ""))
    return blocks


def _highlight_blocks(section: Dict[str, Any]) -> List[str]:
    blocks: List[str] = []
    for item in _finding_rows(section.get("highlights")):
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        attribution = " · ".join(
            part for part in (
                str(item.get("hub") or "").strip(),
                str(item.get("date") or "").strip(),
                str(item.get("author") or "").strip(),
            ) if part
        )
        line = f"— {text}"
        if attribution:
            line += f" ({attribution})"
        if item.get("why"):
            line += f" — важно: {item['why']}"
        blocks.append(line)
    return blocks


def _docx_add_table(doc, spec: Dict[str, Any]) -> None:
    """Настоящая таблица DOCX: шапка, выровненные колонки, единый стиль со таблицами тем."""
    columns = [str(c) for c in (spec.get("columns") or [])]
    rows = [list(row) for row in (spec.get("rows") or [])]
    ncols = max([len(columns)] + [len(row) for row in rows] or [0])
    if ncols <= 0:
        return
    columns += [""] * (ncols - len(columns))
    if spec.get("title"):
        doc.add_heading(str(spec["title"]), level=2)
    table = doc.add_table(rows=1, cols=ncols)
    table.style = "Light Grid Accent 1"
    header = table.rows[0].cells
    for cell, label in zip(header, columns):
        cell.text = label
    for row in rows:
        cells = table.add_row().cells
        for idx in range(ncols):
            cells[idx].text = str(row[idx]) if idx < len(row) else ""
    _set_table_align(table, columns)
    if spec.get("note"):
        note = doc.add_paragraph()
        note.add_run(str(spec["note"])).italic = True


def _docx_add_charts(doc, section: Dict[str, Any], meta: Dict[str, Any], figure_no: List[int]) -> None:
    """Графики раздела с нумерованной подписью: «Рисунок 1. Заголовок»."""
    from docx.shared import Inches

    for chart_id in section.get("chart_ids") or []:
        chart = (meta.get("charts") or {}).get(chart_id)
        if not chart or not os.path.isfile(chart.get("path") or ""):
            continue
        figure_no[0] += 1
        doc.add_picture(chart["path"], width=Inches(6.3))
        caption = doc.add_paragraph()
        caption.add_run(f"Рисунок {figure_no[0]}. {chart.get('title') or chart_id}").italic = True


def _build_docx(path: str, title: str, subtitle: str, sections: List[Dict[str, Any]], meta: Dict[str, Any]) -> None:
    from docx import Document
    from docx.shared import Pt

    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)
    doc.add_heading(title, 0)
    if subtitle:
        doc.add_paragraph(subtitle)
    info = doc.add_paragraph()
    info.add_run(
        "Тема: {dataset}\nПериод: {period}\nПодготовлено: {author}, {date}".format(
            dataset=meta.get("dataset_label") or "—",
            period=meta.get("period") or "весь период",
            author=meta.get("author") or "агент Tellscope",
            date=meta.get("date") or datetime.now().strftime("%d.%m.%Y %H:%M"),
        )
    ).italic = True
    figure_no = [0]
    for section in sections:
        heading = section.get("heading") or "Раздел"
        doc.add_heading(heading, level=1)
        text = section.get("text") or ""
        for para in [p.strip() for p in str(text).split("\n") if p.strip()]:
            doc.add_paragraph(para)
        # -------- графики: сразу после вводного текста, с подписью --------
        _docx_add_charts(doc, section, meta, figure_no)
        # -------- настоящие таблицы раздела --------
        for spec in section.get("tables") or []:
            _docx_add_table(doc, spec)
        for bullet in section.get("bullets") or []:
            line = _strip_bullet(bullet)
            if line:
                doc.add_paragraph(line, style="List Bullet")
        # -------- текстовые находки: темы с частотами, долями и цитатами --------
        findings = _finding_rows(section.get("findings"))
        if findings:
            doc.add_heading("Темы из текстов сообщений", level=2)
            show_share = any(item.get("share") is not None for item in findings[:12])
            labels = ["Тема", "Пояснение", "Сообщений"] + (["Доля среза"] if show_share else []) + ["Тональность"]
            table = doc.add_table(rows=1, cols=len(labels))
            table.style = "Light Grid Accent 1"
            header = table.rows[0].cells
            for cell, label in zip(header, labels):
                cell.text = label
            for item in findings[:12]:
                row = table.add_row().cells
                values = [
                    str(item.get("topic") or "—"),
                    # Пояснение — от модели (Qwen3-32B): о чём эта тема, без цифр и перечислений.
                    str(item.get("essence") or item.get("summary") or "—"),
                    _fmt(item.get("count")),
                ]
                if show_share:
                    values.append(_fmt_share(item.get("share")) if item.get("share") is not None else "—")
                values.append(str(item.get("tone") or "—"))
                for cell, value in zip(row, values):
                    cell.text = value
            _set_table_align(table, labels)
            for item in findings[:12]:
                topic = str(item.get("topic") or "Тема")
                head = doc.add_paragraph()
                head.add_run(f"{topic} — {_fmt(item.get('count'))} сообщ.").bold = True
                if item.get("share") is not None:
                    head.add_run(f", {_fmt_share(item.get('share'))} среза")
                if item.get("category"):
                    head.add_run(f", категория: {item['category']}")
                if item.get("essence"):
                    doc.add_paragraph(str(item["essence"]))
                for quote in (item.get("quotes") or [])[:3]:
                    text = _quote_text(quote)
                    if not text:
                        continue
                    para = doc.add_paragraph(style="List Bullet")
                    run = para.add_run(f"«{text}»")
                    run.italic = True
                    attribution = _quote_attribution(quote)
                    if attribution:
                        para.add_run(f" — {attribution}")
                    if isinstance(quote, dict) and quote.get("url"):
                        para.add_run(" ")
                        _docx_hyperlink(para, str(quote["url"]), "ссылка")
            if _section_note(section):
                # Пометка о способе чтения: темы/цитаты — по выборке, счётчики — по всему срезу.
                note_para = doc.add_paragraph()
                note_para.add_run(_section_note(section)).italic = True
        highlights = _finding_rows(section.get("highlights"))
        if highlights:
            doc.add_heading("Ключевые сообщения", level=2)
            for item in highlights[:8]:
                text = str(item.get("text") or "").strip()
                if not text:
                    continue
                para = doc.add_paragraph(style="List Bullet")
                para.add_run(text)
                attribution = " · ".join(
                    part for part in (
                        str(item.get("hub") or "").strip(),
                        str(item.get("date") or "").strip(),
                        str(item.get("author") or "").strip(),
                    ) if part
                )
                if attribution:
                    para.add_run(f" — {attribution}").italic = True
                if item.get("why"):
                    para.add_run(f" (важно: {item['why']})")
                if item.get("url"):
                    para.add_run(" ")
                    _docx_hyperlink(para, str(item["url"]), "ссылка")
        # Графики раздела уже вставлены выше (перед таблицами) через _docx_add_charts.
        citations = section.get("citations") or []
        if citations:
            doc.add_paragraph("Источники:")
            for cite in citations[:20]:
                url = cite.get("url") or ""
                label = cite.get("title") or url or "источник"
                p = doc.add_paragraph(style="List Bullet")
                if url:
                    _docx_hyperlink(p, url, label)
                else:
                    p.add_run(label)
    doc.save(path)


class _PageCounter:
    """Сквозная нумерация страниц PDF: «стр. 7 из 29» вместо номера внутри блока."""

    def __init__(self):
        self.no = 0
        self.total = 0

    def mark(self) -> str:
        self.no += 1
        return f"стр. {self.no} из {self.total}" if self.total else f"стр. {self.no}"

    def reset(self):
        self.no = 0


# --- Текстовые страницы PDF: поля листа и полоса набора ---------------------
# Раньше текст начинался от x=0.08, а строка резалась по ЧИСЛУ символов (104). Широкие буквы
# давали строку шире листа, и хвост строки обрезался по краю бумаги — правое поле выходило 0 pt.
# Теперь у текстовой страницы есть явные поля, а строка переносится по ИЗМЕРЕННОЙ ширине (тем же
# шрифтом, которым печатается), поэтому за полосу набора не выходит ничего.
PDF_PAGE_W_IN = 8.27      # A4 в дюймах: ширина
PDF_PAGE_H_IN = 11.69     # A4 в дюймах: высота
TEXT_MARGIN = 0.045       # поле текстовой страницы слева и справа, доля ширины листа
TEXT_BODY_FONT = 9.5      # кегль основного текста
TEXT_LINE_STEP = 0.0155   # шаг строки, доля высоты листа
TEXT_PAGE_LINES = 52      # строк основного текста на странице
TEXT_BODY_WIDTH_IN = PDF_PAGE_W_IN * (1.0 - 2 * TEXT_MARGIN)   # 7,53 дюйма = 542 pt
_FALLBACK_CHAR_W = 0.62   # оценка ширины символа, если шрифт измерить не удалось


def _wrap_pdf_line(text: str, width_in: float, font: float = TEXT_BODY_FONT) -> List[str]:
    """Переносит строку по ИЗМЕРЕННОЙ ширине: строка не выходит за полосу набора.

    Ширина меряется тем же шрифтом (DejaVu Sans), которым строка печатается; замер кешируется
    по слову. Если замер недоступен, берётся заведомо широкая оценка «на символ» — строка тогда
    переносится чуть раньше. Слова длиннее полосы (ссылки, склейки) режутся по символам: целиком
    они не влезают и иначе уехали бы за правый край листа.
    """
    raw = str(text or "")
    if not raw.strip():
        return [raw]
    cache: Dict[str, float] = {}

    def width_pt(chunk: str) -> float:
        if not chunk:
            return 0.0
        if chunk not in cache:
            measured = _text_width_in(chunk, font)
            cache[chunk] = (measured * 72.0 if measured is not None
                            else len(chunk) * _FALLBACK_CHAR_W * font)
        return cache[chunk]

    limit = width_in * 72.0
    gap = width_pt(" ")
    words = raw.split()
    lines: List[str] = []
    current = words[0]
    used = width_pt(current)
    for word in words[1:]:
        width = width_pt(word)
        if used + gap + width <= limit:
            current = f"{current} {word}"
            used += gap + width
        else:
            lines.append(current)
            current, used = word, width
    lines.append(current)
    out: List[str] = []
    for line in lines:
        while len(line) > 1 and width_pt(line) > limit:
            cut = len(line) - 1
            while cut > 1 and width_pt(line[:cut]) > limit:
                cut -= 1
            out.append(line[:cut])
            line = line[cut:]
        out.append(line)
    return out or [""]


def _pdf_text_pages(pdf, title: str, blocks: List[str], meta: Dict[str, Any],
                    counter: "_PageCounter" = None) -> None:
    plt = _mpl()
    from matplotlib.backends.backend_pdf import PdfPages  # noqa: F401  (тип для аннотации)

    counter = counter or _PageCounter()
    page_lines = TEXT_PAGE_LINES
    lines: List[str] = []
    for block in blocks:
        for raw in str(block).split("\n"):
            lines.extend(_wrap_pdf_line(raw, TEXT_BODY_WIDTH_IN))
    pages = [lines[i : i + page_lines] for i in range(0, len(lines), page_lines)] or [[""]]
    # Кегль заголовка подбирается под ширину полосы: длинное название темы иначе уезжает за край.
    title_lines, title_font = _page_title_lines(title, TEXT_BODY_WIDTH_IN, 15.0, 10.0)
    for page_no, chunk in enumerate(pages):
        fig = plt.figure(figsize=(PDF_PAGE_W_IN, PDF_PAGE_H_IN), dpi=140)
        label = counter.mark()
        header_y = 0.94
        for line in title_lines:
            fig.text(TEXT_MARGIN, header_y, line, fontsize=title_font, fontweight="bold", va="top")
            header_y -= 0.032
        if page_no == 0:
            meta_y = header_y - 0.005
            fig.text(
                TEXT_MARGIN,
                meta_y,
                "Тема: {d}   |   Период: {p}   |   {a}, {dt}".format(
                    d=meta.get("dataset_label") or "—",
                    p=meta.get("period") or "весь период",
                    a=meta.get("author") or "агент Tellscope",
                    dt=meta.get("date") or datetime.now().strftime("%d.%m.%Y %H:%M"),
                ),
                fontsize=8.5,
                color="#667085",
                va="top",
            )
            start_y = meta_y - 0.04
        else:
            start_y = header_y - 0.02
        y = start_y
        for line in chunk:
            fig.text(TEXT_MARGIN, y, line, fontsize=TEXT_BODY_FONT, va="top", family="DejaVu Sans")
            y -= 0.0155
        fig.text(0.5, 0.03, label, fontsize=8, color="#98A2B3", ha="center")
        pdf.savefig(fig)
        plt.close(fig)


# Ширина табличного блока в символах моноширинного шрифта. Книжная страница — 108 символов
# при кегле 7,5 (6,8" из 7,4" полезной ширины), альбомная — 164 символа (10,3" из 10,4").
TABLE_LINE_WIDTH = 108
TABLE_LINE_WIDTH_WIDE = 164

# Полоса набора страницы таблицы: сверху — заголовок, снизу — номер страницы.
TABLE_TOP = 0.905
TABLE_BOTTOM = 0.055
TABLE_CAPTION_H = 0.035
TABLE_NOTE_H = 0.024
TABLE_MARGIN = 0.055

# Кегль таблицы подбирается так, чтобы таблица уместилась ЦЕЛИКОМ на одной странице:
# от читаемого 7,5 к 4,8 (кегль, доля базового шага строки от шага при 7,5).
TABLE_FIT_STEPS = (
    (7.5, 1.00), (7.2, 0.96), (7.0, 0.93), (6.8, 0.90), (6.5, 0.86),
    (6.2, 0.82), (6.0, 0.78), (5.8, 0.74), (5.5, 0.70), (5.2, 0.66),
    (5.0, 0.62), (4.8, 0.58),
)
TABLE_BASE_ROW_H = 0.0145       # шаг строки при кегле 7,5 на книжной странице
TABLE_BASE_ROW_H_WIDE = 0.0185  # шаг строки при кегле 7,5 на альбомной странице; в альбоме
                                # строк на странице больше за счёт ширины, а не мелкого текста
TABLE_LANDSCAPE_HINTS = ("landscape", "wide", "альбом", "альбомная", "горизонтальная")
TABLE_PORTRAIT_HINTS = ("portrait", "narrow", "книж", "книжная", "портрет", "вертикальная")


def _table_body(spec: Dict[str, Any]) -> "tuple[List[str], List[List[str]]]":
    """Шапка и строки таблицы: колонки дополняются, лишние ячейки отбрасываются."""
    columns = [str(c) for c in (spec.get("columns") or [])]
    rows = [[("" if c is None else str(c)) for c in row] for row in (spec.get("rows") or [])]
    ncols = max([len(columns)] + [len(row) for row in rows] or [0])
    if ncols <= 0:
        return [], []
    columns += [""] * (ncols - len(columns))
    rows = [row + [""] * (ncols - len(row)) for row in rows]
    return columns, rows


def _table_widths(columns, rows, total_width: int) -> List[int]:
    """Ширины колонок: самые широкие колонки ужимаются, пока таблица не впишется в страницу."""
    ncols = len(columns)
    gap = 2
    min_width = 8
    budget = max(ncols * min_width, total_width - gap * (ncols - 1))
    widths = [max(len(row[i]) for row in ([columns] + rows)) for i in range(ncols)]
    guard = 0
    while sum(widths) > budget and guard < 200000:
        guard += 1
        shrinkable = [i for i in range(ncols) if widths[i] > min_width]
        if not shrinkable:
            break
        widths[max(shrinkable, key=lambda i: widths[i])] -= 1
    return widths


def _table_grid_rows(spec: Dict[str, Any], total_width: int = TABLE_LINE_WIDTH):
    """Табличный блок для PDF: (шапка со разделителем, строки) как списки строк текста.

    Колонки выровнены пробелами под моноширинный шрифт. Длинные ячейки (названия тем,
    перечисления) переносятся на следующую строку внутри самой ячейки, поэтому данные не
    теряются. Важно: перенос остаётся ВНУТРИ логической строки таблицы — постраничная
    разбивка идёт по строкам целиком и не рвёт строку пополам.
    """
    columns, rows = _table_body(spec)
    if not columns:
        return [], []
    ncols = len(columns)
    gap = 2
    widths = _table_widths(columns, rows, total_width)
    # Числовая колонка — по правому краю, текстовая — по левому. Здесь именно булево значение:
    # раньше в списке лежали строки «right»/«left», условие if aligns[i] было истинным всегда,
    # и все колонки печатались по правому краю — заголовки не вставали над своими колонками.
    aligns = [bool(_is_numeric_label(columns[i]) and all(_is_numeric_cell(r[i]) for r in rows))
              for i in range(ncols)]

    def cell_lines(value: str, width: int) -> List[str]:
        text = str(value or "")
        if len(text) <= width:
            return [text]
        return textwrap.wrap(text, width=width) or [""]

    def render(row: List[str]) -> List[str]:
        cells = [cell_lines(row[i], widths[i]) for i in range(ncols)]
        height = max(len(cell) for cell in cells)
        out: List[str] = []
        for line_no in range(height):
            parts = []
            for i in range(ncols):
                text = cells[i][line_no] if line_no < len(cells[i]) else ""
                parts.append(text.rjust(widths[i]) if aligns[i] else text.ljust(widths[i]))
            out.append("  ".join(parts).rstrip())
        return out

    head = render(columns)
    head.append("─" * min(total_width, sum(widths) + gap * (ncols - 1)))
    return head, [render(row) for row in rows]


def _table_grid(spec: Dict[str, Any], total_width: int = TABLE_LINE_WIDTH) -> List[str]:
    """Плоский табличный блок (шапка + строки) — для случаев, когда разбивка не нужна."""
    head, body = _table_grid_rows(spec, total_width)
    out = list(head)
    for row in body:
        out.extend(row)
    return out


def _table_capacity(row_h: float, head_h: float, note_h: float) -> int:
    """Сколько строк текста таблицы помещается на странице при данном шаге строки."""
    return max(4, int((TABLE_TOP - head_h - note_h - TABLE_BOTTOM) / row_h))


def _readable_steps(landscape: bool):
    """Шаги подбора кегля: (кегль, шаг строки) от самого читаемого к самому мелкому."""
    base = TABLE_BASE_ROW_H_WIDE if landscape else TABLE_BASE_ROW_H
    return [(font, base * factor) for font, factor in TABLE_FIT_STEPS]


def _table_fit(spec: Dict[str, Any], landscape: bool) -> Optional[Dict[str, Any]]:
    """Раскладка таблицы на странице: шапка, строки, кегль, шаг строки, влезает ли целиком.

    Кегль берётся самый крупный из тех, при которых таблица укладывается на одну страницу.
    ``fits=False`` — не влезает даже самым мелким кеглем, таблицу придётся разбить по строкам.
    """
    width_chars = TABLE_LINE_WIDTH_WIDE if landscape else TABLE_LINE_WIDTH
    head, body = _table_grid_rows(spec, width_chars)
    if not head:
        return None
    lines = len(head) + sum(len(row) for row in body)
    caption_h = TABLE_CAPTION_H if spec.get("title") else 0.0
    note_h = TABLE_NOTE_H if spec.get("note") else 0.0
    font, row_h = _readable_steps(landscape)[-1]
    fits = False
    for candidate_font, candidate_row_h in _readable_steps(landscape):
        if lines <= _table_capacity(candidate_row_h, caption_h, note_h):
            font, row_h, fits = candidate_font, candidate_row_h, True
            break
    return {"landscape": landscape, "head": head, "body": body, "lines": lines,
            "font": font, "row_h": row_h, "fits": fits}


def _table_landscape(spec: Dict[str, Any]) -> bool:
    """Альбомная страница для конкретной таблицы.

    Явная подсказка (``layout``) важнее расчёта. Без подсказки сравниваются обе раскладки:
    альбом берётся, если только он позволяет уместить таблицу целиком либо если на альбомной
    странице кегль заметно крупнее (узкие колонки книжной страницы рвут длинные ячейки на
    много строк и текст становится мелким).
    """
    hint = str(spec.get("layout") or "").strip().lower()
    if hint in TABLE_LANDSCAPE_HINTS:
        return True
    if hint in TABLE_PORTRAIT_HINTS:
        return False
    portrait = _table_fit(spec, False)
    landscape = _table_fit(spec, True)
    if portrait is None:
        return landscape is not None
    if landscape is None:
        return False
    if landscape["fits"] != portrait["fits"]:
        return landscape["fits"]
    if not landscape["fits"]:
        return landscape["lines"] < portrait["lines"]
    return landscape["font"] > portrait["font"] + 0.2


def _text_width_in(text: str, font: float, bold: bool = False):
    """Ширина строки в дюймах. None — измерить не удалось (тогда считаем по среднему)."""
    try:
        from matplotlib.font_manager import FontProperties
        from matplotlib.textpath import TextPath

        prop = FontProperties(family="DejaVu Sans", weight="bold" if bold else "normal")
        return TextPath((0, 0), str(text), size=font, prop=prop).get_extents().width / 72.0
    except Exception:  # noqa: BLE001
        return None


def _title_font(text: str, width_in: float, max_font: float = 13.0, min_font: float = 9.0) -> float:
    """Кегль заголовка страницы: длинный заголовок раздела не должен уезжать за край листа."""
    if not text:
        return max_font
    size = max_font
    while size >= min_font:
        measured = _text_width_in(text, size, bold=True)
        if measured is None:
            measured = len(text) * 0.66 * size / 72.0
        if measured <= width_in:
            return size
        size -= 0.5
    return min_font


def _page_title_lines(title: str, width_in: float, max_font: float, min_font: float):
    """Заголовок страницы: (список строк, кегль). Сначала уменьшаем кегль, потом переносим.

    Длинный заголовок раздела («Год → месяц → основные темы, доля негатива, ключевой
    инфоповод» в верхнем регистре) иначе уезжает за правый край листа.
    """
    text = str(title or "")
    font = _title_font(text, width_in, max_font, min_font)
    measured = _text_width_in(text, font, bold=True)
    if measured is None:
        measured = len(text) * 0.66 * font / 72.0
    if measured <= width_in:
        return [text], font
    # Кегль уже минимальный, а строка всё ещё шире листа: переносим по словам, не больше двух строк.
    char_width_in = min_font * 0.62 / 72.0
    per_line = max(24, int(width_in / char_width_in))
    lines = textwrap.wrap(text, width=per_line) or [text]
    if len(lines) > 2:
        lines = textwrap.wrap(text, width=max(24, int(per_line * 1.35))) or [text]
    return lines[:2], min_font


def _pdf_table_pages(pdf, title: str, specs: List[Dict[str, Any]], meta: Dict[str, Any],
                     counter: "_PageCounter" = None) -> None:
    """Таблицы раздела отдельными страницами PDF: подпись, шапка, выровненные колонки.

    Таблица печатается ЦЕЛИКОМ на одной странице: кегль и шаг строки подбираются под её
    объём (от 7,5 к 4,8), широкая таблица уходит на альбомную страницу. Если таблица не
    влезает даже самым мелким кеглем, она разбивается по строкам целиком — шапка
    повторяется на каждой странице, а строка не рвётся пополам.
    """
    plt = _mpl()
    counter = counter or _PageCounter()

    for spec in specs:
        if not isinstance(spec, dict):
            continue
        landscape = _table_landscape(spec)
        fit = _table_fit(spec, landscape)
        if not fit:
            continue
        head, body = fit["head"], fit["body"]
        chosen_font, chosen_row_h = fit["font"], fit["row_h"]
        size = (11.69, 8.27) if landscape else (8.27, 11.69)
        usable_in = size[0] - 2 * TABLE_MARGIN * size[0]
        caption = str(spec.get("title") or "")
        note = str(spec.get("note") or "")
        caption_h = TABLE_CAPTION_H if caption else 0.0

        per_page = _table_capacity(chosen_row_h, caption_h, 0.0)
        header_h = len(head)
        chunks: List[List[str]] = []
        current = list(head)
        used = header_h
        for row in body:
            if used + len(row) > per_page and len(current) > header_h:
                chunks.append(current)          # разрыв — только между строками таблицы
                current = list(head)
                used = header_h
            current.extend(row)
            used += len(row)
        chunks.append(current)

        title_lines, title_font = _page_title_lines(title, usable_in, 13.0, 9.5)
        note_lines = textwrap.wrap(note, width=max(40, int(usable_in * 72.0 / 7.2))) if note else []
        for page_no, chunk in enumerate(chunks):
            fig = plt.figure(figsize=size, dpi=140)
            label = counter.mark()
            y = 0.955
            for line in title_lines:
                fig.text(TABLE_MARGIN, y, line, fontsize=title_font, fontweight="bold", va="top")
                y -= 0.030
            if page_no:
                fig.text(1 - TABLE_MARGIN, 0.955, "(продолжение)", fontsize=8, color="#98A2B3",
                         ha="right", va="top")
            y = TABLE_TOP
            if caption:
                fig.text(TABLE_MARGIN, y, caption, fontsize=9.5, fontweight="bold", va="top", color="#101828")
                y -= caption_h
            for line in chunk:
                fig.text(TABLE_MARGIN, y, line, fontsize=chosen_font, va="top",
                         family="DejaVu Sans Mono", color="#101828")
                y -= chosen_row_h
            if note_lines and page_no == len(chunks) - 1:
                y = max(TABLE_BOTTOM, y - 0.008)
                for line in note_lines:
                    fig.text(TABLE_MARGIN, y, line, fontsize=8, va="top", color="#667085", style="italic")
                    y -= 0.017
            fig.text(0.5, 0.03, label, fontsize=8, color="#98A2B3", ha="center")
            pdf.savefig(fig)
            plt.close(fig)


def _pdf_chart_page(pdf, chart: Dict[str, Any], counter: "_PageCounter" = None) -> None:
    plt = _mpl()
    import matplotlib.image as mpimg

    counter = counter or _PageCounter()
    fig = plt.figure(figsize=(8.27, 11.69), dpi=140)
    label = counter.mark()
    try:
        img = mpimg.imread(chart["path"])
        ax = fig.add_axes([0.06, 0.28, 0.88, 0.5])
        ax.imshow(img)
        ax.axis("off")
    except Exception:
        fig.text(0.08, 0.6, "график недоступен", fontsize=11)
    fig.text(0.08, 0.22, chart.get("title") or "", fontsize=13, fontweight="bold", va="top")
    fig.text(0.5, 0.03, label, fontsize=8, color="#98A2B3", ha="center")
    pdf.savefig(fig)
    plt.close(fig)


def _build_pdf(path: str, title: str, subtitle: str, sections: List[Dict[str, Any]], meta: Dict[str, Any]) -> None:
    """Собирает PDF за два прохода: сначала считаем страницы, потом печатаем «стр. N из M»."""
    import io as _io

    plt = _mpl()
    from matplotlib.backends.backend_pdf import PdfPages

    def render(pdf, counter: _PageCounter) -> None:
        cover_lines: List[str] = []
        if subtitle:
            cover_lines.append(subtitle)
        cover_lines.append("")
        cover_lines.append("Содержание:")
        for i, section in enumerate(sections, 1):
            cover_lines.append(f"{i}. {section.get('heading') or 'Раздел'}")
        _pdf_text_pages(pdf, title, cover_lines, meta, counter)
        for section in sections:
            head = (section.get("heading") or "Раздел").upper()
            # 1) заголовок, вводный текст и списки выводов (без двойных маркеров)
            blocks = [head]
            if section.get("text"):
                blocks.append(section["text"])
            for bullet in section.get("bullets") or []:
                line = _strip_bullet(bullet)
                if line:
                    blocks.append(f"• {line}")
            _pdf_text_pages(pdf, title, blocks, meta, counter)
            # 2) графики раздела — отдельными страницами
            for chart_id in section.get("chart_ids") or []:
                chart = meta.get("charts", {}).get(chart_id)
                if chart and os.path.isfile(chart.get("path") or ""):
                    _pdf_chart_page(pdf, chart, counter)
            # 3) таблицы раздела — табличными блоками
            tables = section.get("tables") or []
            if tables:
                _pdf_table_pages(pdf, head, tables, meta, counter)
            # 4) находки, ключевые сообщения и источники
            extra: List[str] = []
            findings = _finding_rows(section.get("findings"))
            if findings:
                extra.append("")
                extra.append("ТЕМЫ ИЗ ТЕКСТОВ СООБЩЕНИЙ")
                extra.extend(_findings_blocks(section))
            highlights = _finding_rows(section.get("highlights"))
            if highlights:
                extra.append("")
                extra.append("КЛЮЧЕВЫЕ СООБЩЕНИЯ")
                extra.extend(_highlight_blocks(section))
            for cite in (section.get("citations") or [])[:20]:
                extra.append(f"— {cite.get('title') or ''} {cite.get('url') or ''}".strip())
            if extra:
                _pdf_text_pages(pdf, title, extra, meta, counter)

    counter = _PageCounter()
    with PdfPages(_io.BytesIO()) as probe_pdf:
        render(probe_pdf, counter)
    counter.total = counter.no
    counter.reset()
    with PdfPages(path) as pdf:
        render(pdf, counter)


@tool(
    "build_report",
    title="Собрать отчёт (DOCX/PDF)",
    description=(
        "Собирает итоговый отчёт: разделы с текстом, списки выводов, ТАБЛИЦЫ (tables), ТЕМЫ С ЦИТАТАМИ "
        "(findings из analyze_texts), графики (по chart_id из make_chart) и ссылки на источники. "
        "Файлы DOCX и PDF сохраняются в папку датасета во вкладке «Отчёты» и становятся доступны пользователю для скачивания. "
        "Вызывай последним шагом, когда данные собраны и тексты прочитаны (analyze_texts)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "заголовок отчёта"},
            "subtitle": {"type": "string", "description": "подзаголовок/краткое описание"},
            "sections": {
                "type": "array",
                "description": "разделы отчёта: текст, списки выводов, графики, цитаты и текстовые находки",
                "items": {
                    "type": "object",
                    "properties": {
                        "heading": {"type": "string"},
                        "text": {"type": "string", "description": "основной текст раздела, абзацы через перевод строки"},
                        "bullets": {"type": "array", "items": {"type": "string"}, "description": "список выводов/тезисов"},
                        "chart_ids": {"type": "array", "items": {"type": "string"}, "description": "графики из make_chart"},
                        "tables": {
                            "type": "array",
                            "description": "настоящие таблицы раздела: подпись, колонки, строки",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "title": {"type": "string", "description": "подпись таблицы"},
                                    "columns": {"type": "array", "description": "шапка таблицы"},
                                    "rows": {"type": "array", "description": "строки: список списков ячеек"},
                                    "note": {"type": "string", "description": "примечание под таблицей"},
                                    "layout": {
                                        "type": "string",
                                        "enum": ["auto", "portrait", "landscape"],
                                        "description": "раскладка таблицы в PDF: auto (по содержимому), "
                                                       "portrait (книжная) или landscape (альбомная, "
                                                       "для широких таблиц)",
                                    },
                                },
                                "required": ["columns", "rows"],
                            },
                        },
                        "findings": {
                            "type": "array",
                            "description": "темы из чтения текстов (analyze_texts): тема, число сообщений, доля, цитаты",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "topic": {"type": "string"},
                                    "count": {"type": "integer"},
                                    "share": {"type": "number", "description": "доля среза: 0.46 или 46.3"},
                                    "tone": {"type": "string"},
                                    "category": {"type": "string"},
                                    "essence": {"type": "string"},
                                    "importance": {"type": "number"},
                                    "quotes": {
                                        "type": "array",
                                        "description": "цитаты с атрибуцией",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "text": {"type": "string"},
                                                "date": {"type": "string"},
                                                "hub": {"type": "string"},
                                                "author": {"type": "string"},
                                                "url": {"type": "string"},
                                            },
                                        },
                                    },
                                },
                            },
                        },
                        "highlights": {
                            "type": "array",
                            "description": "ключевые сообщения по вовлечённости",
                            "items": {"type": "object", "properties": {"text": {"type": "string"}, "why": {"type": "string"}}},
                        },
                        "citations": {
                            "type": "array",
                            "description": "ссылки на источники (сообщения, публикации)",
                            "items": {
                                "type": "object",
                                "properties": {"title": {"type": "string"}, "url": {"type": "string"}},
                            },
                        },
                    },
                    "required": ["heading"],
                },
            },
            "findings": {
                "type": "array",
                "description": "темы и цитаты из analyze_texts, если передаются отдельно от разделов",
                "items": {"type": "object"},
            },
            "highlights": {
                "type": "array",
                "description": "ключевые сообщения, если передаются отдельно от разделов",
                "items": {"type": "object"},
            },
            "folder": {"type": "string", "description": "папка во вкладке «Отчёты» (по умолчанию папка агента)"},
            "author": {"type": "string", "description": "подпись автора отчёта"},
        },
        "required": ["title", "sections"],
    },
    group="reports",
    scope="write",
    timeout=300.0,
)
async def build_report(
    ctx,
    title: str = "",
    sections: Any = None,
    subtitle: str = "",
    folder: Optional[str] = None,
    author: str = "агент Tellscope",
    findings: Any = None,
    highlights: Any = None,
    **extra: Any,
):
    sections = _coerce_list(sections, "sections")
    # Темы с цитатами и ключевые сообщения модель передаёт не только внутри раздела, но и
    # параметрами верхнего уровня. Раньше такой вызов падал с
    # TypeError: build_report() got an unexpected keyword argument 'findings' — инструмент
    # не собирал отчёт вообще, и темы с цитатами терялись. Теперь они превращаются в разделы.
    _rejected_top: List[str] = []

    def _top_level_rows(value: Any, field: str) -> List[Dict[str, Any]]:
        """Список объектов из параметра верхнего уровня; мусор не ломает сборку отчёта."""
        if value is None or value == "":
            return []
        try:
            items = _coerce_list(value, field)
        except ToolError as exc:
            _rejected_top.append(f"{field}: {exc}")
            return []
        return [item for item in items if isinstance(item, dict)]

    _top_findings = _top_level_rows(findings, "findings")
    _top_highlights = _top_level_rows(highlights, "highlights")
    if _rejected_top:
        await ctx.log(
            "build_report: параметры верхнего уровня пропущены — " + "; ".join(_rejected_top),
            level="error",
        )
    if extra:
        # Незнакомые параметры не должны ронять сборку отчёта. В журнале сервера этот же
        # TypeError случался и с другими ключами модели ('links', 'index'): вызов падал
        # целиком, и отчёт не собирался. Пишем ключи в журнал агента и продолжаем.
        await ctx.log(
            "build_report: незнакомые параметры пропущены — " + ", ".join(sorted(extra)),
            level="error",
        )
    if _top_findings:
        sections.append({
            "heading": "Темы и цитаты из текстов сообщений",
            "findings": _top_findings,
        })
    if _top_highlights:
        sections.append({
            "heading": "Ключевые сообщения",
            "highlights": _top_highlights,
        })
    if _top_findings or _top_highlights:
        await ctx.log(
            "build_report: findings/highlights переданы параметрами верхнего уровня "
            f"(темы: {len(_top_findings)}, сообщения: {len(_top_highlights)}) — добавлены разделами"
        )
    if not sections:
        raise ToolError(
            "Нужны заголовок и разделы: вызовите build_report с аргументами "
            '{"title": "название отчёта", "sections": [{"heading": "раздел", "text": "текст", '
            '"bullets": ["вывод"], "chart_ids": ["chart1"], "citations": ["https://..."]}]}'
        )
    if not title or not str(title).strip():
        raise ToolError('Укажите title — название отчёта, например "Аналитический отчёт по бренду"')
    sections = [s for s in sections if isinstance(s, dict) and (s.get("heading") or s.get("text"))]

    # -------- запрет отчёта-заглушки: без данных и графиков отчёт не считается успешным --------
    no_data_markers = ("не найдены", "нет данных", "не обнаружено", "отсутствуют данные")

    def _is_stub_section(section: Dict[str, Any]) -> bool:
        """Раздел-заглушка «данных нет»: длинный текст про отсутствие данных — это не данные.

        Без этой проверки модель обходила защиту: писала абзац «в датасете нет сообщений
        за период» на 150+ символов, отчёт считался содержательным, и запуск закрывался
        как успешный, хотя данных в срезе не было.
        """
        if section.get("findings") or section.get("highlights") or section.get("chart_ids"):
            return False
        heading = str(section.get("heading") or "").lower()
        text = str(section.get("text") or "").lower()
        if any(marker in heading for marker in no_data_markers):
            return True
        return any(marker in text for marker in no_data_markers) and len(text) < 900

    def _report_has_data() -> bool:
        for chart in (ctx.charts or {}).values():
            data = chart.get("data") if isinstance(chart, dict) else None
            series = (data or {}).get("series") if isinstance(data, dict) else (chart.get("series") if isinstance(chart, dict) else None)
            for item in (series or []):
                values = item.get("values") if isinstance(item, dict) else None
                for value in (values or []):
                    try:
                        if abs(float(value)) > 0:
                            return True
                    except (TypeError, ValueError):
                        if str(value).strip():
                            return True
        for section in sections:
            text = str(section.get("text") or "").strip()
            rows = section.get("items") or section.get("bullets") or section.get("values") or section.get("rows")
            if rows:
                return True
            # Раздел с темами и цитатами из текстов — это тоже данные, а не заглушка
            if _finding_rows(section.get("findings")) or _finding_rows(section.get("highlights")):
                return True
            if len(text) >= 150 and not _is_stub_section(section):
                return True
        return False

    if not _report_has_data():
        ctx.no_data = "данные за период не найдены"
        sections = [{
            "heading": "Данные за период не найдены",
            "text": ("В выбранном датасете нет сообщений за указанный период, поэтому отчёт не сформирован. "
                     "Запуск помечен как неуспешный. Проверьте период и тему датасета "
                     "(или выгрузите тему из Brand Analytics за нужные даты) и повторите запуск."),
        }]
    if not sections:
        raise ToolError("Ни один раздел не содержит heading или text — проверьте структуру sections")
    # Подробный разбор темы (deep_text_analysis) обязан попасть в документ — добавляем, если модель забыла
    deep = getattr(ctx, "deep_analysis", None)
    if deep and deep.get("text"):
        already = any("подробный разбор" in str(section.get("heading") or "").lower() for section in sections)
        if not already:
            sections.append(dict(deep))
    # Темы и цитаты из чтения текстов (analyze_texts) обязательны в документе целиком:
    # если модель передала только часть тем, добавляем полный раздел инструмента.
    texts = getattr(ctx, "text_analysis", None)
    if texts and (texts.get("findings") or texts.get("text")):
        tool_findings = len(_finding_rows(texts.get("findings")))
        report_findings = sum(len(_finding_rows(section.get("findings"))) for section in sections)
        if tool_findings and report_findings < tool_findings:
            sections.append(dict(texts))
    for section in sections:
        citations = section.get("citations")
        if isinstance(citations, str):
            section["citations"] = [{"title": citations, "url": citations}]
        elif isinstance(citations, list):
            normalized = []
            for item in citations:
                if isinstance(item, str):
                    normalized.append({"title": item, "url": item})
                elif isinstance(item, dict):
                    normalized.append(item)
            section["citations"] = normalized
        if isinstance(section.get("bullets"), str):
            section["bullets"] = [section["bullets"]]
        if isinstance(section.get("chart_ids"), str):
            section["chart_ids"] = [section["chart_ids"]]
        for key in ("findings", "highlights"):
            if isinstance(section.get(key), str):
                try:
                    parsed = json.loads(section[key])
                except Exception as exc:  # noqa: BLE001
                    raise ToolError(f"{key}: ожидается список или JSON-строка со списком ({exc})") from exc
                section[key] = parsed if isinstance(parsed, list) else [parsed]
        for finding in _finding_rows(section.get("findings")):
            if isinstance(finding.get("quotes"), str):
                finding["quotes"] = [{"text": finding["quotes"]}]

    # -------- правило качества: негатив без конкретной причины с цитатой недопустим --------
    def _has_text_evidence() -> bool:
        """Есть ли в отчёте тема с цитатой или ссылка на источник — то есть прочитанные тексты."""
        for section in sections:
            if section.get("citations"):
                return True
            for finding in _finding_rows(section.get("findings")):
                if [q for q in (finding.get("quotes") or []) if _quote_text(q)]:
                    return True
        return False

    def _negatives_in_slice() -> int:
        """Сколько негативных сообщений в срезе: нужно, чтобы отчёт объяснял причины, а не только счётчики."""
        known = int(getattr(ctx, "negative_in_slice", 0) or 0)
        if known:
            return known
        try:
            from .tools_data import _exact_count, _query, dates, guard

            _idx, index_name = guard(ctx)
            lo, hi = dates(ctx, None, None)
            if not (lo or hi):
                return 0
            return int(_exact_count(index_name, _query(None, lo, hi, "negative")) or 0)
        except Exception:
            return 0

    negative_count = _negatives_in_slice()
    if negative_count:
        # Запоминаем размер негатива в срезе: правило качества и статус запуска опираются на него
        ctx.negative_in_slice = max(int(getattr(ctx, "negative_in_slice", 0) or 0), negative_count)
    if negative_count and not _has_text_evidence():
        ctx.text_gap = (
            f"в срезе {negative_count} негативных сообщений, но в отчёте нет ни одной темы "
            "с цитатой — причины жалоб не раскрыты (тексты не прочитаны)"
        )
        await ctx.log("Отчёт неполный: " + ctx.text_gap, level="error")
        sections.append(
            {
                "heading": "Внимание: причины негатива не раскрыты",
                "text": (
                    f"В выборке {negative_count} негативных сообщений, однако в отчёте нет ни одной конкретной "
                    "темы или причины с цитатой из текста. Отчёт считается неполным: пояснения к графикам должны "
                    "опираться на темы и цитаты, полученные чтением текстов. Вызовите инструмент analyze_texts "
                    "за этот период и передайте его раздел (темы, доли, цитаты) в build_report, затем соберите отчёт заново."
                ),
            }
        )
    folder_name = _safe_name(folder or ctx.folder or "Агент", 40)
    out_dir = _reports_dir(ctx.user_id, folder_name)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    base = f"{_safe_name(title, 70)}_{stamp}"
    docx_path = os.path.join(out_dir, base + ".docx")
    pdf_path = os.path.join(out_dir, base + ".pdf")
    period = ""
    if ctx.min_date or ctx.max_date:
        from .tools_data import _iso

        period = f"{_iso(ctx.min_date) if ctx.min_date else '…'} — {_iso(ctx.max_date) if ctx.max_date else '…'}"
    # В шапке — тема человеческим языком и её период, а не внутреннее имя выгрузки.
    meta = {
        "dataset_label": topic_header(
            getattr(ctx, "dataset_label", "") or getattr(ctx, "dataset_name", ""),
            getattr(ctx, "dataset_name", ""),
        ),
        "period": period,
        "author": author,
        "date": datetime.now().strftime("%d.%m.%Y %H:%M"),
        "charts": ctx.charts,
    }
    # Из разделов убираем служебные технические сообщения перед сборкой документов.
    sections = _clean_sections(sections)
    _build_docx(docx_path, str(title), subtitle, sections, meta)
    pdf_ok = True
    pdf_error = ""
    try:
        _build_pdf(pdf_path, str(title), subtitle, sections, meta)
    except Exception as exc:  # PDF не должен ломать отчёт
        pdf_ok = False
        pdf_error = f"{type(exc).__name__}: {exc}"

    files = []
    for path in (docx_path, pdf_path):
        if os.path.isfile(path):
            art = ctx.add_artifact(
                "report",
                os.path.basename(path),
                path,
                url=f"/api/reports/download/{ctx.user_id}/{folder_name}/{os.path.basename(path)}",
                meta={"folder": folder_name},
            )
            files.append({"name": os.path.basename(path), "size": os.path.getsize(path), "url": art["url"]})
    # Компактный итог запуска рядом с отчётом: <ГГГГ-ММ>_summary.json одинакового формата.
    # Он нужен для последующей агрегации (годовые и межгодовые сводки) и для инструмента
    # read_reports, который читает готовые итоги вместо повторного разбора сырых данных.
    # Итог никогда не должен ломать сборку отчёта, поэтому все ошибки здесь глушим.
    summary_file = ""
    try:
        summary_file = _write_summary(
            ctx, str(title), folder_name, sections, files, negative_count,
        )
    except Exception as exc:  # noqa: BLE001
        summary_file = ""
        await ctx.log(f"Итог запуска сохранить не удалось: {type(exc).__name__}: {exc}", level="error")
    if summary_file:
        files.append({
            "name": os.path.basename(summary_file),
            "size": os.path.getsize(summary_file),
            "url": f"/api/reports/download/{ctx.user_id}/{folder_name}/{os.path.basename(summary_file)}",
        })
        await ctx.log(f"Итог запуска сохранён: {os.path.basename(summary_file)}")
    await ctx.log(f"Отчёт сохранён в папку «{folder_name}»: " + ", ".join(f["name"] for f in files))
    return {
        "folder": folder_name,
        "files": files,
        "sections": len(sections),
        "charts_used": sum(len(s.get("chart_ids") or []) for s in sections),
        "pdf_ok": pdf_ok,
        "pdf_error": pdf_error,
        "text_gap": str(getattr(ctx, "text_gap", "") or ""),
        "negative_in_slice": negative_count,
        "findings_sections": len([s for s in sections if _finding_rows(s.get("findings"))]),
        "summary_file": os.path.basename(summary_file) if summary_file else "",
    }


# ------------------------------------------------- итог запуска и чтение готовых отчётов
# Одинаковый формат итога для каждого запуска: по нему строятся сводные (годовые, межгодовые,
# «по всем месяцам») отчёты без повторного разбора сырых данных. Имя предсказуемое —
# <ГГГГ-ММ>_summary.json в папке отчётов пользователя, поэтому агрегация не требует поиска.

SUMMARY_VERSION = 1
TONE_KEYS = (("негатив", "negative"), ("нейтрал", "neutral"), ("позитив", "positive"))


def _period_key(lo: Any, hi: Any, fallback: str = "") -> str:
    """Ключ периода для имени итога: ГГГГ-ММ (месяц), ГГГГ (год), ГГГГ-ММ-ДД (один день)."""
    def parts(value: Any) -> List[int]:
        out: List[int] = []
        for chunk in re.split(r"[^0-9]+", str(value or ""))[:3]:
            if chunk.isdigit():
                out.append(int(chunk))
        return out

    start, end = parts(lo), parts(hi)
    if start and end and start[0] == end[0]:
        if start[1] == end[1]:
            if start[2] == end[2]:
                return "%04d-%02d-%02d" % (start[0], start[1], start[2])
            return "%04d-%02d" % (start[0], start[1])
        return "%04d" % start[0]
    if start and end and start[0] != end[0]:
        # Межгодовой период (например 2024–2026): ключ ГГГГ-ГГГГ. Раньше в этом случае
        # возвращался месяц начала, и межгодовой итог сохранялся как 2024-05_summary.json,
        # то есть под именем месячного итога за май 2024.
        return "%04d-%04d" % (start[0], end[0])
    if start:
        return "%04d-%02d" % (start[0], start[1])
    return fallback or datetime.now().strftime("%Y-%m")


def _slice_stats(ctx) -> Dict[str, Any]:
    """Число сообщений в срезе и распределение тональности — по ВСЕМУ срезу, а не по выборке."""
    out: Dict[str, Any] = {"total": 0, "negative": 0, "neutral": 0, "positive": 0}
    try:
        from .tools_data import _exact_count, _query, _terms, TONE_VALUES, dates, guard

        _idx, index_name = guard(ctx)
        lo, hi = dates(ctx, None, None)
        if not (lo or hi):
            return out
        query = _query(None, lo, hi, None)
        out["total"] = int(_exact_count(index_name, query) or 0)
        for row in _terms(index_name, "toneMark", size=5, query=query):
            label = TONE_VALUES.get(row.get("key"), row.get("key"))
            for ru, en in TONE_KEYS:
                if label == ru:
                    out[en] = int(row.get("count") or 0)
    except Exception:  # noqa: BLE001 — итог не должен зависеть от доступности агрегаций
        pass
    return out


# --- Спам и шум в темах -----------------------------------------------------
# Кластеры вроде «цена аккаунта» (194 дословно одинаковых сообщения «Цена - 50.000 ₽» из одного
# канала) попадали в темы месяца и выбирались «ключевым инфоповодом». Признаки считаются по
# сохранённым цитатам темы: метка Brand Analytics, рекламный шаблон и копипаста одного источника.
SPAM_CATEGORIES = ("спам", "рекламные посты", "реклама", "спам и реклама", "рекламный пост")
SPAM_AD_MARKERS = (
    # продажа и обмен аккаунтов, игровых предметов
    "цена аккаунта", "продам аккаунт", "продаю аккаунт", "куплю аккаунт", "купить аккаунт",
    "покупка аккаунта", "продажа аккаунт", "аккаунты в наличии", "обмен продажа",
    # ссылки-приглашения и реферальные предложения
    "реферальн", "рефа", "по моей ссылке", "приглашаю в",
    "больше скидок в боте", "промокод в боте", "промокоды в боте",
    # прочий накруточный мусор
    "казино", "букмекер", "ставки на спорт", "накрутка подписчиков", "бесплатные подписчики",
    "заработок в интернете", "быстрый заработок",
)
# Копипаста: одинаковая фраза во всех цитатах темы, пришедшая из одного источника.
SPAM_COPYPASTE_SHARE = 0.8
SPAM_COPYPASTE_MIN_COUNT = 5


def _noise_text(value: Any) -> str:
    """Текст для сравнения: без регистра, знаков и лишних пробелов — «почти одинаковая фраза»."""
    low = str(value or "").lower().replace("ё", "е")
    low = re.sub(r"[^0-9a-zа-я]+", " ", low)
    return re.sub(r"\s+", " ", low).strip()


def _noise_voices(topic: Dict[str, Any]) -> int:
    """Сколько разных голосов за темой: уникальные авторы и площадки среди её цитат."""
    quotes = [q for q in (topic.get("quotes") or []) if isinstance(q, dict)]
    authors = {str(q.get("author") or "").strip().lower() for q in quotes} - {""}
    hubs = {str(q.get("hub") or "").strip().lower() for q in quotes} - {""}
    return max(len(authors), len(hubs))


def topic_noise_reasons(topic: Dict[str, Any]) -> List[str]:
    """Почему тема считается спамом или шумом. Пустой список — тема нормальная.

    Проверки намеренно узкие: «цена», «акции», «скидки», «купон» и прочие слова деловой
    повестки спамом не считаются — за них отвечает копипаста одного источника и метка
    Brand Analytics. Рекламные шаблоны перечислены только те, что не относятся к теме KFC.
    """
    if not isinstance(topic, dict):
        return []
    reasons: List[str] = []
    category = _noise_text(topic.get("category"))
    if any(marker in category for marker in SPAM_CATEGORIES):
        reasons.append("метка Brand Analytics «%s»" % topic.get("category"))
    # Название темы приходит то полем name (месячный итог), то полем topic (findings чтения текстов).
    haystack = " ".join(_noise_text(x) for x in
                        (topic.get("name") or topic.get("topic"), topic.get("essence"),
                         topic.get("summary")))
    for quote in (topic.get("quotes") or []):
        if isinstance(quote, dict):
            haystack += " " + _noise_text(quote.get("text"))
    hit = next((marker for marker in SPAM_AD_MARKERS if marker in haystack), "")
    if hit:
        reasons.append("рекламный шаблон «%s»" % hit)
    quotes = [q for q in (topic.get("quotes") or [])
              if isinstance(q, dict) and str(q.get("text") or "").strip()]
    if len(quotes) >= 2 and int(topic.get("count") or 0) >= SPAM_COPYPASTE_MIN_COUNT:
        texts = [_noise_text(q.get("text")) for q in quotes]
        top = max(set(texts), key=texts.count)
        share = texts.count(top) / float(len(texts))
        authors = {str(q.get("author") or "").strip().lower() for q in quotes} - {""}
        hubs = {str(q.get("hub") or "").strip().lower() for q in quotes} - {""}
        if share >= SPAM_COPYPASTE_SHARE and (len(authors) <= 1 or len(hubs) <= 1):
            reasons.append("копипаста %.0f%% цитат из одного источника" % (share * 100))
    return reasons


def filter_spam_topics(topics: List[Dict[str, Any]]):
    """Делит темы на нормальные и спам/шум.

    Возвращает (оставшиеся темы, отфильтрованные с причинами, сколько сообщений отфильтровано).
    """
    kept: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    for topic in topics or []:
        if not isinstance(topic, dict):
            continue
        reasons = topic_noise_reasons(topic)
        if reasons:
            dropped.append({"name": str(topic.get("name") or topic.get("topic") or ""),
                            "count": int(topic.get("count") or 0),
                            "reasons": reasons})
        else:
            kept.append(topic)
    messages = sum(item["count"] for item in dropped)
    return kept, dropped, messages


def _share(value: Any, total: Any) -> float:
    try:
        total = float(total or 0)
        if total <= 0:
            return 0.0
        return round(float(value or 0) / total, 4)
    except (TypeError, ValueError):
        return 0.0


def _quote_row(quote: Any) -> Optional[Dict[str, str]]:
    """Цитата для итога: текст, ссылка, дата, площадка, автор — без служебных полей."""
    text = _quote_text(quote)
    if not text:
        return None
    row = quote if isinstance(quote, dict) else {}
    return {
        "text": text[:400],
        "url": str(row.get("url") or ""),
        "date": str(row.get("date") or ""),
        "hub": str(row.get("hub") or ""),
        "author": str(row.get("author") or ""),
    }


def _summary_payload(ctx, title: str, folder_name: str, sections: List[Dict[str, Any]],
                     files: List[Dict[str, Any]], negative_count: int) -> Dict[str, Any]:
    """Компактный итог одного запуска в одинаковом формате.

    Темы и цитаты берутся из раздела чтения текстов (ctx.text_analysis — его положил analyze_texts),
    число сообщений и тональность — по всему срезу периода. Если инструмент чтения не вызывался,
    соответствующие поля остаются пустыми: итог не выдумывает данные.
    """
    try:
        from .tools_data import _iso, dates

        lo, hi = dates(ctx, None, None)
        period_from, period_to = (_iso(lo) if lo else ""), (_iso(hi) if hi else "")
    except Exception:  # noqa: BLE001
        period_from = period_to = ""

    texts = getattr(ctx, "text_analysis", None)
    texts = texts if isinstance(texts, dict) else {}
    findings = _finding_rows(texts.get("findings"))

    topics: List[Dict[str, Any]] = []
    for row in findings:
        quotes = [q for q in (_quote_row(item) for item in (row.get("quotes") or [])) if q]
        topics.append({
            "name": row.get("topic") or row.get("name") or "",
            "count": int(row.get("count") or 0),
            "share": float(row.get("share") or 0.0),
            "share_pct": float(row.get("share_pct") or 0.0),
            "tone": row.get("tone") or "",
            "category": category,
            "essence": row.get("essence") or "",
            # Пояснение темы сохраняем в структурный итог: в годовых и межгодовых отчётах
            # пояснения берутся оттуда вместе с частотами.
            "summary": row.get("summary") or row.get("essence") or "",
            "quotes": quotes[:3],
        })

    # Спам и шум темами месяца не становятся: копипаста и рекламные шаблоны («цена аккаунта» в
    # майском отчёте 2024 года — 194 одинаковых сообщения «Цена - 50.000 ₽» из одного канала)
    # искажали и список тем, и выбор ключевого инфоповода. Сколько отфильтровано — видно пометкой.
    topics, spam_topics, spam_messages = filter_spam_topics(topics)
    topic_total = sum(int(item.get("count") or 0) for item in topics)
    for item in topics:
        item["share"] = _share(int(item.get("count") or 0), topic_total)
        item["share_pct"] = round(item["share"] * 100, 2)

    # Авторы и категории считаются уже без спама: иначе в «Активных авторах» годового отчёта
    # оказывался автор рекламного канала, а категория «цена» раздувалась спам-кластером.
    category_counts: Dict[str, int] = {}
    authors: Dict[str, int] = {}
    for item in topics:
        for quote in item["quotes"]:
            if quote.get("author"):
                authors[quote["author"]] = authors.get(quote["author"], 0) + 1
        category = str(item.get("category") or "").strip()
        if category:
            category_counts[category] = category_counts.get(category, 0) + int(item.get("count") or 0)

    highlights: List[Dict[str, Any]] = []
    for item in (texts.get("highlights") or []):
        if not isinstance(item, dict):
            continue
        highlights.append({
            "text": str(item.get("text") or "")[:300],
            "url": str(item.get("url") or ""),
            "date": str(item.get("date") or ""),
            "hub": str(item.get("hub") or ""),
            "author": str(item.get("author") or ""),
            "tone": item.get("tone"),
        })
        if item.get("author"):
            authors[str(item["author"])] = authors.get(str(item["author"]), 0) + 1

    links: List[str] = []
    for section in sections:
        for item in (section.get("citations") or []):
            url = item.get("url") if isinstance(item, dict) else str(item)
            if url and url not in links:
                links.append(str(url))
    for topic in topics:
        for quote in topic["quotes"]:
            if quote["url"] and quote["url"] not in links:
                links.append(quote["url"])

    # Инфоповоды: разделы отчёта, которые агент назвал поводами/событиями, плюс темы с датой.
    events: List[Dict[str, Any]] = []
    for section in sections:
        heading = str(section.get("heading") or "")
        if any(marker in heading.lower() for marker in ("инфоповод", "повод", "событи")):
            for bullet in (section.get("bullets") or [])[:10]:
                events.append({"name": str(bullet)[:200], "date": "", "source": heading})
            text = str(section.get("text") or "").strip()
            if text and not events:
                events.append({"name": text[:200], "date": "", "source": heading})

    stats = _slice_stats(ctx)
    total = int(stats.get("total") or 0)
    tone_abs = {key: int(stats.get(key) or 0) for _, key in TONE_KEYS}
    if negative_count and not tone_abs["negative"]:
        tone_abs["negative"] = int(negative_count)
    if not total:
        total = sum(tone_abs.values())
    # Сколько сообщений модель отнесла к темам: это честная нижняя оценка прочитанного
    # (в самом analyze_texts «прочитано» может быть больше — часть сообщений не попала ни в одну тему).
    read_in_topics = len({mid for row in findings for mid in (row.get("msg_ids") or [])})

    return {
        "version": SUMMARY_VERSION,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "report": {"title": title, "folder": folder_name, "files": [f.get("name") for f in files]},
        "dataset": {
            "index": getattr(ctx, "dataset_index", None),
            "name": getattr(ctx, "dataset_name", "") or "",
            "label": getattr(ctx, "dataset_label", "") or getattr(ctx, "dataset_name", "") or "",
        },
        "period": {
            "from": period_from,
            "to": period_to,
            "from_ts": getattr(ctx, "min_date", None),
            "to_ts": getattr(ctx, "max_date", None),
            "key": _period_key(
                period_from, period_to,
                _period_key_from_text(title, folder_name,
                                      *(str(s.get("heading") or "") for s in sections)),
            ),
        },
        "messages": {
            "in_slice": total,
            # Сколько сообщений инструмент прочитал моделью (для кластеризации — выборка
            # в тысячи сообщений), и сколько из них попало в темы отчёта.
            "read_sample": int(texts.get("read_sample") or 0),
            "read_in_topics": read_in_topics,
            "topics_total_count": sum(int(row.get("count") or 0) for row in topics),
        },
        "clusters": {
            "count": int(texts.get("clusters_count") or len(topics)),
            "kind": "темы отчёта (кластеры корпуса)" if texts.get("strategy") == "corpus" else "темы отчёта",
            "strategy": str(texts.get("strategy") or ""),
        },
        "tonality": {
            **tone_abs,
            "total": total,
            "shares": {key: _share(value, total) for key, value in tone_abs.items()},
        },
        "topics": topics,
        # Сколько сообщений ушло в спам/шум: цифра видна в отчётах, но темами месяца не считается.
        "filtered": {
            "spam_topics": spam_topics,
            "spam_messages": spam_messages,
            "note": ("отфильтровано как спам/шум: %s сообщений" % _fmt(spam_messages))
            if spam_messages else "",
        },
        "categories": [
            {"category": name, "count": count, "share": _share(count, total)}
            for name, count in sorted(category_counts.items(), key=lambda item: -item[1])
        ],
        "authors": [
            {"name": name, "count": count}
            for name, count in sorted(authors.items(), key=lambda item: -item[1])[:15]
        ],
        "events": events,
        "highlights": highlights[:10],
        "sources": {"links": links[:20], "reports": [f.get("name") for f in files]},
        "sections": [str(section.get("heading") or "") for section in sections if section.get("heading")],
        "sample_note": str(texts.get("note") or "")[:500],
        "notes": [
            "topics и quotes — из раздела чтения текстов (analyze_texts), их охват указан в sample_note",
            "tonality и messages.in_slice — по всему срезу периода",
            "authors — по процитированным и ключевым сообщениям отчёта",
        ] + ([
            "отфильтровано как спам/шум: %s сообщений (%s)"
            % (_fmt(spam_messages), "; ".join(
                "«%s» — %s" % (item["name"], _fmt(item["count"])) for item in spam_topics[:5]))
        ] if spam_messages else []),
    }


def _write_summary(ctx, title: str, folder_name: str, sections: List[Dict[str, Any]],
                   files: List[Dict[str, Any]], negative_count: int) -> str:
    """Пишет <ГГГГ-ММ>_summary.json в папку отчётов пользователя. Возвращает путь или ''."""
    payload = _summary_payload(ctx, title, folder_name, sections, files, negative_count)
    out_dir = _reports_dir(ctx.user_id, folder_name)
    path = os.path.join(out_dir, "%s_summary.json" % payload["period"]["key"])
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
    return path


def _docx_text(path: str, limit: int = 6000) -> str:
    """Текст документа DOCX: абзацы и таблицы. Нужен, чтобы читать отчёты, собранные до появления
    итоговых JSON."""
    try:
        from docx import Document

        doc = Document(path)
        parts = [str(p.text).strip() for p in doc.paragraphs if str(p.text).strip()]
        for table in doc.tables:
            for row in table.rows:
                cells = [str(cell.text).strip() for cell in row.cells if str(cell.text).strip()]
                if cells:
                    parts.append(" | ".join(cells))
        text = "\n".join(parts)
    except Exception:  # noqa: BLE001
        return ""
    return text[: max(500, int(limit))]


def _report_folders(root: str, folder: str = "") -> List[str]:
    wanted = _safe_name(folder, 40) if folder else ""
    if wanted:
        path = os.path.join(root, wanted)
        return [wanted] if os.path.isdir(path) else []
    try:
        return sorted(name for name in os.listdir(root) if os.path.isdir(os.path.join(root, name)))
    except Exception:  # noqa: BLE001
        return []


@tool(
    "read_reports",
    title="Прочитать готовые отчёты",
    description=(
        "Возвращает содержимое ранее построенных отчётов пользователя: компактные итоги запусков "
        "(<ГГГГ-ММ>_summary.json — период, число сообщений, тональность в абсолюте и долях, темы с частотами, "
        "цитаты и ссылки, ключевые сообщения) и, при include_text=true, текст самих документов DOCX. "
        "Нужен для сводных, годовых и межгодовых отчётов: сначала читай готовые итоги и строй сводку по ним, "
        "и только если итогов нет — иди в сырые данные (в этом случае скажи об этом в отчёте)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "folder": {"type": "string", "description": "папка отчётов (обычно датасет); пусто — все папки"},
            "limit": {"type": "integer", "description": "сколько последних отчётов вернуть, по умолчанию 10"},
            "include_text": {"type": "boolean", "description": "добавить текст документов DOCX (нужно для отчётов, собранных без итогового JSON)"},
            "max_chars": {"type": "integer", "description": "предел длины текста одного документа, по умолчанию 6000"},
        },
    },
    group="reports",
)
async def read_reports(ctx, folder: str = "", limit: int = 10, include_text: bool = False,
                       max_chars: int = 6000):
    root = os.path.join(BACKEND_ROOT, "data", str(ctx.user_id), REPORTS_DIR_NAME)
    limit = max(1, min(int(limit or 10), 50))
    summaries: List[Dict[str, Any]] = []
    texts: List[Dict[str, Any]] = []
    for name in _report_folders(root, folder):
        fdir = os.path.join(root, name)
        try:
            names = sorted(os.listdir(fdir))
        except Exception:  # noqa: BLE001
            continue
        summary_files = [n for n in names if n.endswith("_summary.json")]
        for file_name in summary_files[-limit:]:
            path = os.path.join(fdir, file_name)
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    payload = json.load(fh)
            except Exception:  # noqa: BLE001
                continue
            summaries.append({"folder": name, "file": file_name, "size": os.path.getsize(path),
                              "summary": payload})
        if include_text:
            covered = {item["file"].replace("_summary.json", "") for item in summaries
                       if item["folder"] == name}
            docs = [n for n in names if n.lower().endswith(".docx")]
            for file_name in docs[-limit:]:
                path = os.path.join(fdir, file_name)
                text = _docx_text(path, max_chars)
                if not text:
                    continue
                texts.append({
                    "folder": name,
                    "file": file_name,
                    "size": os.path.getsize(path),
                    "chars": len(text),
                    "has_summary": os.path.splitext(file_name)[0] in covered,
                    "text": text,
                })
    summaries.sort(key=lambda item: (str(item["summary"].get("period", {}).get("key") or ""), item["file"]))
    summaries = summaries[-limit:]
    texts = texts[-limit:]
    return {
        "root": root,
        "folders": _report_folders(root, folder),
        "summaries": summaries,
        "texts": texts,
        "summary_count": len(summaries),
        "report_count": len(texts),
        "note": (
            "Сводку строй по summaries (это итоги прошлых запусков в одинаковом формате: период, "
            "сообщения, тональность, темы с частотами, цитаты). Если у отчёта нет итогового JSON, "
            "вызови инструмент с include_text=true и прочитай текст DOCX. Если готовых отчётов нет — "
            "иди в сырые данные и напиши в отчёте, что сводка собрана по ним."
        ),
    }
