# -*- coding: utf-8 -*-
"""Человекочитаемая подпись датасета: «Тема 01.09.2026-30.09.2026».

Имена файлов приходят из разных источников: выгрузки Brand Analytics
(``BA_Признаки_ОРВИ_20261003_175906``), загрузки Excel (``converted_…``), старые транслитные
выгрузки (``platon_31.10.2025-30.11.2025``). Показывать их в интерфейсе как есть нельзя: по
такому имени не видно ни темы, ни периода. Здесь имя приводится к виду «тема + период».

Само имя файла при этом не меняется: переименование потянуло бы за собой индекс Elasticsearch
и коллекцию векторов, а у крупных датасетов это часы работы.
"""
import re
from datetime import datetime, timedelta, timezone

MSK = timezone(timedelta(hours=3))

PERIOD_RE = re.compile(r"(\d{2}\.\d{2}\.\d{4})\s*[-\u2013\u2014]\s*(\d{2}\.\d{2}\.\d{4})")
STAMP_RE = re.compile(r"[_\s]*(?:20\d{6})[_\s]?\d{6}\b")
HASH_RE = re.compile(r"[_\s]*[0-9a-f]{8,}\b")
PREFIXES = ("ba_", "ba ", "brand_analytics_", "brand analytics ", "converted_", "converted ")

# Транслитные названия тем, которые уже встречаются в именах файлов.
ALIASES = {
    "platon": "Платон",
    "petrov platon 2025 2026": "Платон",
    "rosselhozbank": "Россельхозбанк",
    "rosbank": "Росбанк",
    "rinomaris": "Риномарис",
    "kfc": "KFC",
    "beyond taylor": "Beyond Taylor",
    "ozon": "Озон",
    "priznaki orvi": "Признаки ОРВИ",
    "признаки орви": "Признаки ОРВИ",
    # Аббревиатуры пишем заглавными: в подписи «Признаки орви» выглядит небрежно.
    "орви": "ОРВИ",
    "сми": "СМИ",
    "оив": "ОИВ",
    "pr": "PR",
    "moskovskiy transport": "Московский транспорт",
    "smirnov medicina moskvy": "Смирнов Медицина Москвы",
    "smirnov stroim dom": "Смирнов Строим дом",
    "smirnov domostroy patriotizm": "Смирнов Домострой Патриотизм",
    "smirnov domostroy sluzhenie": "Смирнов Домострой Служение",
}


def period_label(min_ts, max_ts) -> str:
    """Период данных в виде «01.09.2026-30.09.2026» (по московскому времени)."""
    def day(value):
        try:
            return datetime.fromtimestamp(float(value), MSK).strftime("%d.%m.%Y")
        except Exception:
            return ""

    lo, hi = day(min_ts), day(max_ts)
    if not (lo and hi):
        return ""
    return lo if lo == hi else "%s-%s" % (lo, hi)


def _theme(title: str) -> str:
    """Тема из «сырой» части имени: известные транслиты заменяем, остальное чистим."""
    compact = re.sub(r"\s{2,}", " ", str(title or "").strip())
    key = compact.lower()
    if key in ALIASES:
        return ALIASES[key]
    words = [ALIASES.get(word.lower(), word) for word in compact.split()]
    text = " ".join(words).strip()
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text


def _strip_technical(file_name: str) -> str:
    """Имя без префикса источника, отметки выгрузки и хеша, с пробелами вместо подчёркиваний."""
    raw = str(file_name or "").strip()
    text = raw[:-5] if raw.lower().endswith(".json") else raw
    low = text.lower()
    for prefix in PREFIXES:
        if low.startswith(prefix):
            text = text[len(prefix):]
            break
    text = HASH_RE.sub(" ", text)
    text = STAMP_RE.sub(" ", text)
    text = re.sub(r"[_]+", " ", text)
    return re.sub(r"\s{2,}", " ", text).strip(" -\u2013\u2014_")


def dataset_theme(file_name) -> str:
    """Тема датасета без периода: «Признаки ОРВИ», «KFC», «Озон отзывы»."""
    text = _strip_technical(PERIOD_RE.sub(" ", str(file_name or "")))
    return _theme(text)


def with_period(label, min_ts=None, max_ts=None, period_text: str = "") -> str:
    """Подпись «Тема 01.09.2026-30.09.2026».

    Период берётся из готовой строки, из имени файла или из данных — что есть. Если период
    в подписи уже указан, второй раз его не добавляем.
    """
    text = str(label or "").strip()
    if not text or PERIOD_RE.search(text):
        return text
    span = str(period_text or "").strip()
    if not span:
        span = period_label(min_ts, max_ts)
    if not span:
        return text
    span = span.replace("\u2014", "-").replace("\u2013", "-").replace(" ", "")
    return ("%s %s" % (text, span)).strip()


def pretty_dataset_name(file_name, min_ts=None, max_ts=None) -> str:
    """Подпись датасета для интерфейса. Исходное имя файла не меняется."""
    raw = str(file_name or "").strip()
    stem = raw[:-5] if raw.lower().endswith(".json") else raw
    period = PERIOD_RE.search(stem)
    if period:
        return ("%s %s-%s" % (dataset_theme(stem), period.group(1), period.group(2))).strip()
    theme = dataset_theme(stem)
    label = period_label(min_ts, max_ts)
    return ("%s %s" % (theme, label)).strip() if label else theme
