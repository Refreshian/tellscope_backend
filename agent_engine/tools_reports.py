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


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:,.2f}".replace(",", " ").replace(".", ",")
    if isinstance(value, int):
        return f"{value:,}".replace(",", " ")
    return str(value)


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
            values = [float(v or 0) for v in (s.get("values") or [])]
            color = PALETTE[i % len(PALETTE)]
            if ctype == "area":
                ax.fill_between(idx, values, alpha=0.18, color=color)
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


def _findings_blocks(section: Dict[str, Any]) -> List[str]:
    """Текстовое представление findings для PDF: тема, число/доля, цитаты с атрибуцией."""
    blocks: List[str] = []
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
        if item.get("essence"):
            blocks.append(f"   суть: {item['essence']}")
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


def _build_docx(path: str, title: str, subtitle: str, sections: List[Dict[str, Any]], meta: Dict[str, Any]) -> None:
    from docx import Document
    from docx.shared import Inches, Pt

    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)
    doc.add_heading(title, 0)
    if subtitle:
        doc.add_paragraph(subtitle)
    info = doc.add_paragraph()
    info.add_run(
        "Датасет: {dataset}\nПериод: {period}\nПодготовлено: {author}, {date}".format(
            dataset=meta.get("dataset_label") or "—",
            period=meta.get("period") or "весь период",
            author=meta.get("author") or "агент Tellscope",
            date=meta.get("date") or datetime.now().strftime("%d.%m.%Y %H:%M"),
        )
    ).italic = True
    for section in sections:
        heading = section.get("heading") or "Раздел"
        doc.add_heading(heading, level=1)
        text = section.get("text") or ""
        for para in [p.strip() for p in str(text).split("\n") if p.strip()]:
            doc.add_paragraph(para)
        for bullet in section.get("bullets") or []:
            doc.add_paragraph(str(bullet), style="List Bullet")
        # -------- текстовые находки: темы с частотами, долями и цитатами --------
        findings = _finding_rows(section.get("findings"))
        if findings:
            doc.add_heading("Темы из текстов сообщений", level=2)
            table = doc.add_table(rows=1, cols=4)
            table.style = "Light Grid Accent 1"
            header = table.rows[0].cells
            for cell, label in zip(header, ("Тема", "Сообщений", "Доля среза", "Тональность")):
                cell.text = label
            for item in findings[:12]:
                row = table.add_row().cells
                row[0].text = str(item.get("topic") or "—")
                row[1].text = _fmt(item.get("count"))
                row[2].text = _fmt_share(item.get("share")) if item.get("share") is not None else "—"
                row[3].text = str(item.get("tone") or "—")
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
        for chart_id in section.get("chart_ids") or []:
            chart = meta.get("charts", {}).get(chart_id)
            if chart and os.path.isfile(chart.get("path") or ""):
                doc.add_picture(chart["path"], width=Inches(6.3))
                cap = doc.add_paragraph()
                cap.add_run(f"Рис. {chart_id}: {chart.get('title')}").italic = True
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


def _pdf_text_pages(pdf, title: str, blocks: List[str], meta: Dict[str, Any]) -> None:
    plt = _mpl()
    from matplotlib.backends.backend_pdf import PdfPages  # noqa: F401  (тип для аннотации)

    page_lines = 52
    lines: List[str] = []
    for block in blocks:
        for raw in str(block).split("\n"):
            wrapped = textwrap.wrap(raw, width=104) or [""]
            lines.extend(wrapped)
    pages = [lines[i : i + page_lines] for i in range(0, len(lines), page_lines)] or [[""]]
    for page_no, chunk in enumerate(pages):
        fig = plt.figure(figsize=(8.27, 11.69), dpi=140)
        fig.text(0.08, 0.94, title, fontsize=15, fontweight="bold", va="top")
        if page_no == 0:
            fig.text(
                0.08,
                0.90,
                "Датасет: {d}   |   Период: {p}   |   {a}, {dt}".format(
                    d=meta.get("dataset_label") or "—",
                    p=meta.get("period") or "весь период",
                    a=meta.get("author") or "агент Tellscope",
                    dt=meta.get("date") or datetime.now().strftime("%d.%m.%Y %H:%M"),
                ),
                fontsize=8.5,
                color="#667085",
                va="top",
            )
            start_y = 0.86
        else:
            start_y = 0.90
        y = start_y
        for line in chunk:
            fig.text(0.08, y, line, fontsize=9.5, va="top", family="DejaVu Sans")
            y -= 0.0155
        fig.text(0.5, 0.03, f"стр. {page_no + 1} из {len(pages)}", fontsize=8, color="#98A2B3", ha="center")
        pdf.savefig(fig)
        plt.close(fig)


def _pdf_chart_page(pdf, chart: Dict[str, Any]) -> None:
    plt = _mpl()
    import matplotlib.image as mpimg

    fig = plt.figure(figsize=(8.27, 11.69), dpi=140)
    try:
        img = mpimg.imread(chart["path"])
        ax = fig.add_axes([0.06, 0.28, 0.88, 0.5])
        ax.imshow(img)
        ax.axis("off")
    except Exception:
        fig.text(0.08, 0.6, "график недоступен", fontsize=11)
    fig.text(0.08, 0.22, chart.get("title") or "", fontsize=13, fontweight="bold", va="top")
    pdf.savefig(fig)
    plt.close(fig)


def _build_pdf(path: str, title: str, subtitle: str, sections: List[Dict[str, Any]], meta: Dict[str, Any]) -> None:
    plt = _mpl()
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(path) as pdf:
        cover_lines: List[str] = []
        if subtitle:
            cover_lines.append(subtitle)
        cover_lines.append("")
        cover_lines.append("Содержание:")
        for i, section in enumerate(sections, 1):
            cover_lines.append(f"{i}. {section.get('heading') or 'Раздел'}")
        _pdf_text_pages(pdf, title, cover_lines, meta)
        for section in sections:
            blocks = [f"{(section.get('heading') or 'Раздел').upper()}"]
            if section.get("text"):
                blocks.append(section["text"])
            for bullet in section.get("bullets") or []:
                blocks.append(f"• {bullet}")
            findings = _finding_rows(section.get("findings"))
            if findings:
                blocks.append("")
                blocks.append("ТЕМЫ ИЗ ТЕКСТОВ СООБЩЕНИЙ")
                blocks.extend(_findings_blocks(section))
            highlights = _finding_rows(section.get("highlights"))
            if highlights:
                blocks.append("")
                blocks.append("КЛЮЧЕВЫЕ СООБЩЕНИЯ")
                blocks.extend(_highlight_blocks(section))
            for cite in (section.get("citations") or [])[:20]:
                blocks.append(f"— {cite.get('title') or ''} {cite.get('url') or ''}".strip())
            _pdf_text_pages(pdf, title, blocks, meta)
            for chart_id in section.get("chart_ids") or []:
                chart = meta.get("charts", {}).get(chart_id)
                if chart and os.path.isfile(chart.get("path") or ""):
                    _pdf_chart_page(pdf, chart)


@tool(
    "build_report",
    title="Собрать отчёт (DOCX/PDF)",
    description=(
        "Собирает итоговый отчёт: разделы с текстом, списки выводов, ТЕМЫ С ЦИТАТАМИ (findings из analyze_texts), "
        "графики (по chart_id из make_chart) и ссылки на источники. "
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
):
    sections = _coerce_list(sections, "sections")
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
    meta = {
        "dataset_label": ctx.dataset_label or ctx.dataset_name or "",
        "period": period,
        "author": author,
        "date": datetime.now().strftime("%d.%m.%Y %H:%M"),
        "charts": ctx.charts,
    }
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
    category_counts: Dict[str, int] = {}
    authors: Dict[str, int] = {}
    for row in findings:
        quotes = [q for q in (_quote_row(item) for item in (row.get("quotes") or [])) if q]
        for quote in quotes:
            if quote.get("author"):
                authors[quote["author"]] = authors.get(quote["author"], 0) + 1
        category = str(row.get("category") or "").strip()
        if category:
            category_counts[category] = category_counts.get(category, 0) + int(row.get("count") or 0)
        topics.append({
            "name": row.get("topic") or row.get("name") or "",
            "count": int(row.get("count") or 0),
            "share": float(row.get("share") or 0.0),
            "share_pct": float(row.get("share_pct") or 0.0),
            "tone": row.get("tone") or "",
            "category": category,
            "essence": row.get("essence") or "",
            "quotes": quotes[:3],
        })

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
            "key": _period_key(period_from, period_to),
        },
        "messages": {
            "in_slice": total,
            "read_in_topics": read_in_topics,
            "topics_total_count": sum(int(row.get("count") or 0) for row in topics),
        },
        "clusters": {
            "count": len(topics),
            "kind": "темы отчёта",
            "strategy": str(texts.get("strategy") or ""),
        },
        "tonality": {
            **tone_abs,
            "total": total,
            "shares": {key: _share(value, total) for key, value in tone_abs.items()},
        },
        "topics": topics,
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
        ],
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
