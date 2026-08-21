from __future__ import annotations

import hashlib
import re
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET

import pandas as pd

from .catalog import Catalog
from .models import Message

W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
MONTHS = {
    "января": 1, "февраля": 2, "марта": 3, "апреля": 4, "мая": 5, "июня": 6,
    "июля": 7, "августа": 8, "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12,
}
SOURCE_RE = re.compile(
    r"^(?P<source>.+?),\s*(?P<city>[^,]+),\s*"
    r"(?P<day>\d{1,2})\s+(?P<month>[а-яё]+)\s+(?P<year>\d{4})"
    r"(?:\s+(?P<hour>\d{1,2}):(?P<minute>\d{2}))?\s*$",
    re.IGNORECASE,
)
URL_RE = re.compile(r"^https?://\S+$", re.IGNORECASE)


def iter_docx_paragraphs(path: Path) -> list[str]:
    with ZipFile(path) as zf:
        xml = zf.read("word/document.xml")
    root = ET.fromstring(xml)
    out: list[str] = []
    for para in root.iter(W_NS + "p"):
        text = "".join(node.text or "" for node in para.iter(W_NS + "t")).strip()
        if text:
            out.append(_strip_trailing_page(text))
    return out


def _strip_trailing_page(text: str) -> str:
    # Номер страницы в оглавлении Медиалогии часто прилипает к слову: «...Западного47».
    return re.sub(r"(?<=[А-Яа-яA-Za-z])\d{1,2}$", "", text).strip()


def parse_datetime(day: str, month: str, year: str, hour: str | None, minute: str | None) -> datetime | None:
    m = MONTHS.get(month.lower())
    if not m:
        return None
    try:
        return datetime(int(year), m, int(day), int(hour or 0), int(minute or 0))
    except ValueError:
        return None


def _header_meta(paragraphs: list[str]) -> dict:
    meta: dict = {"objects": "", "period": "", "total": None, "exported": None}
    blob = "\n".join(paragraphs[:30])
    m = re.search(r"Объекты:\s*(.+)", blob)
    if m:
        meta["objects"] = m.group(1).strip()
    m = re.search(r"Временной период:\s*(.+)", blob)
    if m:
        meta["period"] = m.group(1).strip()
    m = re.search(r"Всего сообщений:\s*(\d+)\s*\(экспортировано:\s*(\d+)\)", blob)
    if m:
        meta["total"] = int(m.group(1))
        meta["exported"] = int(m.group(2))
    return meta


def _full_text_start(paragraphs: list[str]) -> int:
    idxs = [i for i, p in enumerate(paragraphs) if p.lower().startswith("полные тексты")]
    return idxs[-1] + 1 if idxs else 0


def parse_medialogia_docx(path: Path, catalog: Catalog) -> tuple[list[Message], dict]:
    paragraphs = iter_docx_paragraphs(path)
    meta = _header_meta(paragraphs)
    start = _full_text_start(paragraphs)
    body = paragraphs[start:]
    messages: list[Message] = []
    current: list[str] = []

    def flush(block: list[str]) -> None:
        if not block:
            return
        header = SOURCE_RE.match(block[0])
        if not header:
            return
        rest = block[1:]
        title = rest[0] if rest else ""
        i = 1
        while i < len(rest) and (rest[i].startswith("Автор:") or rest[i].startswith("Фото:")):
            i += 1
        url = ""
        text_parts: list[str] = []
        for line in rest[i:]:
            if URL_RE.match(line):
                url = line
                continue
            if line.lower().startswith("к дайджесту") or line.lower().startswith("к содержанию"):
                continue
            text_parts.append(line)
        dt = parse_datetime(
            header["day"], header["month"], header["year"], header.group("hour"), header.group("minute")
        )
        source_raw = header["source"].strip()
        source, contour = catalog.normalize_source(source_raw)
        raw_id = url or f"{source}|{title}|{dt}"
        msg_id = hashlib.sha1(raw_id.encode("utf-8", errors="ignore")).hexdigest()[:16]
        messages.append(
            Message(
                id=msg_id,
                source_raw=source_raw,
                source=source,
                contour=contour,
                published_at=dt,
                title=title.strip(),
                text="\n".join(text_parts).strip(),
                url=url,
                file_name=path.name,
                city=header["city"].strip(),
            )
        )

    for line in body:
        if SOURCE_RE.match(line):
            flush(current)
            current = [line]
        elif current:
            current.append(line)
    flush(current)
    file_objects = catalog.match_objects(meta.get("objects") or "")
    # Пообъектная выгрузка: даже если алиасы задели соседний ОИВ, якорим файл.
    primary = file_objects[0] if file_objects else None
    if primary and ("Департамент" in (meta.get("objects") or "") or "проект" in (meta.get("objects") or "").lower() or "crowd" in (meta.get("objects") or "").lower() or "Узнай" in (meta.get("objects") or "") or "долголет" in (meta.get("objects") or "").lower()):
        if len(file_objects) <= 2:
            for msg in messages:
                if primary not in msg.object_ids:
                    msg.object_ids.insert(0, primary)
    meta["parsed"] = len(messages)
    meta["file_objects"] = file_objects
    return messages, meta


def parse_medialogia_xlsx(path: Path) -> list[dict]:
    """Свод Медиалогии: сообщения, главная роль, МедиаИндекс, охват, тон."""
    xl = pd.ExcelFile(path)
    rows_out: list[dict] = []
    for sheet in xl.sheet_names:
        df = pd.read_excel(xl, sheet_name=sheet, header=None)
        header_idx = None
        for i, row in df.iterrows():
            values = [str(v).strip() for v in row.tolist() if pd.notna(v)]
            if any(v == "Название объекта" for v in values):
                header_idx = int(i)
                break
        if header_idx is None:
            continue
        header = [str(v).strip() if pd.notna(v) else "" for v in df.iloc[header_idx].tolist()]
        for _, row in df.iloc[header_idx + 1 :].iterrows():
            rec = {header[j]: row.iloc[j] for j in range(len(header)) if header[j]}
            name = rec.get("Название объекта")
            if not name or not str(name).strip() or str(name) == "nan":
                continue
            rows_out.append(
                {
                    "name": str(name).strip(),
                    "messages": _num(rec.get("Количество сообщений")),
                    "main_role": _num(rec.get("Главная роль")),
                    "media_index": _num(rec.get("МедиаИндекс")),
                    "reach": _num(rec.get("Охват (из открытых источников)") or rec.get("Охват (из\xa0открытых источников)")),
                    "negative": _num(rec.get("Негативный характер упоминаний")),
                    "positive": _num(rec.get("Позитивный характер упоминаний")),
                    "quotes": _num(rec.get("Есть цитирование")),
                    "engagement": _num(rec.get("Вовлеченность")),
                    "file": path.name,
                }
            )
    return rows_out


def parse_scan_xlsx(path: Path) -> dict[str, list[tuple[str, float]]]:
    xl = pd.ExcelFile(path)
    out: dict[str, list[tuple[str, float]]] = {}
    for sheet in xl.sheet_names:
        df = pd.read_excel(xl, sheet_name=sheet, header=None)
        if df.empty:
            continue
        items: list[tuple[str, float]] = []
        for _, row in df.iloc[1:].iterrows():
            vals = [v for v in row.tolist() if pd.notna(v)]
            if len(vals) < 2:
                continue
            name = str(vals[-2] if len(vals) >= 3 else vals[0]).strip()
            try:
                value = float(vals[-1])
            except (TypeError, ValueError):
                continue
            items.append((name, value))
        out[sheet] = items
    return out


def _num(value) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_zip(path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with ZipFile(path) as zf:
        for info in zf.infolist():
            try:
                name = info.filename.encode("cp437").decode("cp866")
            except Exception:
                name = info.filename
            target = dest / name
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as out:
                out.write(src.read())


def load_folder(folder: Path, catalog: Catalog) -> tuple[list[Message], dict]:
    folder = Path(folder)
    for path in list(folder.glob("*.zip")):
        _extract_zip(path, folder / path.stem)

    messages: list[Message] = []
    notes: list[str] = []
    vendor_stats: list[dict] = []
    scan: dict[str, list] = {}
    seen: set[str] = set()

    files = sorted(folder.rglob("*"))
    for path in files:
        if path.suffix.lower() == ".docx" and not path.name.startswith("~"):
            batch, meta = parse_medialogia_docx(path, catalog)
            notes.append(f"{path.name}: текстов {meta.get('parsed')}, заявлено {meta.get('exported')}/{meta.get('total')}")
            for msg in batch:
                if msg.id in seen:
                    continue
                seen.add(msg.id)
                messages.append(msg)
        elif path.suffix.lower() in {".xlsx", ".xls"} and not path.name.startswith("~"):
            stats = parse_medialogia_xlsx(path)
            if stats:
                vendor_stats.extend(stats)
                notes.append(f"{path.name}: свод Медиалогии, объектов {len(stats)}")
            else:
                scan[path.name] = parse_scan_xlsx(path)
                notes.append(f"{path.name}: панель СКАН")

    messages.sort(key=lambda m: m.published_at or datetime.min)
    dates = [m.published_at for m in messages if m.published_at]
    return messages, {
        "notes": notes,
        "vendor_stats": vendor_stats,
        "scan": scan,
        "period_start": min(dates) if dates else None,
        "period_end": max(dates) if dates else None,
        "files": len(files),
    }
