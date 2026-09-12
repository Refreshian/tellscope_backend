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
        "Собирает итоговый отчёт: разделы с текстом, списки выводов, графики (по chart_id из make_chart) и ссылки на источники. "
        "Файлы DOCX и PDF сохраняются в папку датасета во вкладке «Отчёты» и становятся доступны пользователю для скачивания. "
        "Вызывай последним шагом, когда данные собраны."
    ),
    parameters={
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "заголовок отчёта"},
            "subtitle": {"type": "string", "description": "подзаголовок/краткое описание"},
            "sections": {
                "type": "array",
                "description": "разделы отчёта",
                "items": {
                    "type": "object",
                    "properties": {
                        "heading": {"type": "string"},
                        "text": {"type": "string", "description": "основной текст раздела, абзацы через перевод строки"},
                        "bullets": {"type": "array", "items": {"type": "string"}, "description": "список выводов/тезисов"},
                        "chart_ids": {"type": "array", "items": {"type": "string"}, "description": "графики из make_chart"},
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
            if len(text) >= 150:
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
    await ctx.log(f"Отчёт сохранён в папку «{folder_name}»: " + ", ".join(f["name"] for f in files))
    return {
        "folder": folder_name,
        "files": files,
        "sections": len(sections),
        "charts_used": sum(len(s.get("chart_ids") or []) for s in sections),
        "pdf_ok": pdf_ok,
        "pdf_error": pdf_error,
    }
