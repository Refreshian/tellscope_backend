#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Оформление готового DOCX: акценты жирным, пометки, кликабельные ссылки.

Зачем: длинный текст плохо читается. Здесь после сборки документа:
  * фрагменты, помеченные **звёздочками**, становятся жирными;
  * строки-пометки («Важно:», «Что делать:», «Риск:», «Где смотреть:») получают
    жирное начало и цвет, чтобы их было видно при листании;
  * ссылки на сообщения превращаются в настоящие гиперссылки Word.

Запуск:  venv_py312_clean/bin/python kfc_docx_polish.py <файл.docx>
"""
from __future__ import annotations

import re
import sys

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor
from docx.opc.constants import RELATIONSHIP_TYPE as RT

MARKS = {
    "Важно:": RGBColor(0xB4, 0x23, 0x18),
    "Что делать:": RGBColor(0x06, 0x76, 0x47),
    "Риск:": RGBColor(0xB5, 0x47, 0x08),
    "Где смотреть:": RGBColor(0x17, 0x60, 0xE8),
    "Вывод:": RGBColor(0x10, 0x18, 0x28),
    "Итог:": RGBColor(0x10, 0x18, 0x28),
}
URL_RE = re.compile(r"https?://[^\s,;«»\"'<>)]+")
TOKEN_RE = re.compile(r"(\*\*.+?\*\*|https?://[^\s,;«»\"'<>)]+)", re.S)


def add_hyperlink(paragraph, url: str, text: str) -> None:
    """Настоящая гиперссылка Word: в PDF и DOCX становится кликабельной."""
    relationship_id = paragraph.part.relate_to(url, RT.HYPERLINK, is_external=True)
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), relationship_id)
    run = OxmlElement("w:r")
    properties = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "1760E8")
    underline = OxmlElement("w:u")
    underline.set(qn("w:val"), "single")
    size = OxmlElement("w:sz")
    size.set(qn("w:val"), "18")
    properties.append(color)
    properties.append(underline)
    properties.append(size)
    run.append(properties)
    node = OxmlElement("w:t")
    node.text = text
    run.append(node)
    hyperlink.append(run)
    paragraph._p.append(hyperlink)


def style_paragraph(paragraph) -> None:
    """Пересобирает абзац: жирные акценты, пометки, ссылки."""
    text = paragraph.text
    if not text.strip():
        return
    if not (TOKEN_RE.search(text) or any(text.strip().startswith(mark) for mark in MARKS)):
        return
    base_size = None
    if paragraph.runs:
        base_size = paragraph.runs[0].font.size
    mark_color = next((color for mark, color in MARKS.items() if text.strip().startswith(mark)), None)
    for run in list(paragraph.runs):
        run._element.getparent().remove(run._element)
    for chunk in TOKEN_RE.split(text):
        if not chunk:
            continue
        if chunk.startswith("**") and chunk.endswith("**") and len(chunk) > 4:
            run = paragraph.add_run(chunk[2:-2])
            run.bold = True
            if base_size:
                run.font.size = base_size
            if mark_color:
                run.font.color.rgb = mark_color
        elif chunk.startswith("http"):
            add_hyperlink(paragraph, chunk, chunk)
        else:
            run = paragraph.add_run(chunk)
            if base_size:
                run.font.size = base_size
            if mark_color and text.strip().startswith(tuple(MARKS)):
                run.font.color.rgb = mark_color
                if text.strip().startswith(("Важно:", "Что делать:", "Риск:", "Где смотреть:", "Вывод:", "Итог:")):
                    run.bold = True


def polish(path: str) -> dict:
    document = Document(path)
    paragraphs = list(document.paragraphs)
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                paragraphs.extend(cell.paragraphs)
    changed = 0
    for paragraph in paragraphs:
        before = paragraph.text
        style_paragraph(paragraph)
        if paragraph.text != before or "**" in before:
            changed += 1
    document.save(path)
    stats = {"paragraphs": len(paragraphs), "styled": changed}
    print("оформлено абзацев: %d из %d" % (changed, len(paragraphs)))
    return stats


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("укажите файл .docx")
        raise SystemExit(2)
    polish(sys.argv[1])
