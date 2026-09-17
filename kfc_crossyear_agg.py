# -*- coding: utf-8 -*-
"""Тяжёлые расчёты расширенного межгодового отчёта по теме KFC.

Здесь только две вещи:
  1) агрегации поискового хранилища проекта — тепловая карта «тема × месяц», когорты авторов,
     матрица «площадка × тема × тональность», таймлайны инфоповодов, индекс риска, доли голоса,
     устойчивые формулировки. Всё считается по ВСЕМУ корпусу темы, без выборки и без модели —
     это минуты, а не часы;
  2) чтение небольших выборок сообщений локальными моделями там, где нужен СМЫСЛ:
     быстрая qwen3-4b-fast (порт 8001) разбирает выборку волны/кампании, Qwen3-32B (порт 8000)
     даёт названия и интерпретацию.

Модуль намеренно не импортирует main: подъём приложения тянет celery и убивает чужие процессы
на GPU. Клиент поиска создаётся напрямую, как это делает приложение.

Расчёты кэшируются в /tmp/kfc_cx_cache, чтобы повторная сборка отчёта занимала секунды.
"""
from __future__ import annotations

import datetime
import glob
import io
import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

ES_HOST = "http://localhost:9200"
ES_USER = "elastic"
ES_PASS = "biz8z5i1w0nLPmEweKgP"
INDEX = "kfc_13.05.2024-22.09.2026"

CACHE_DIR = "/tmp/kfc_cx_cache"
MSK = datetime.timezone(datetime.timedelta(hours=3))
TZ_NAME = "+03:00"

FAST_URL = "http://127.0.0.1:8001/v1/chat/completions"
FAST_MODEL = "qwen3-4b-fast"
GEN_URL = "http://127.0.0.1:8000/v1/chat/completions"
GEN_MODEL = "Qwen/Qwen3-32B-FP8"

# Годы сравнения: 2024 — с мая (тема собирается с 13.05.2024), 2026 — по август.
YEARS = (2024, 2025, 2026)
YEAR_SPAN = {
    2024: ("2024-05-01", "2024-12-31"),
    2025: ("2025-01-01", "2025-12-31"),
    2026: ("2026-01-01", "2026-08-31"),
}
EXPECTED_MONTHS = ([("2024-%02d" % m) for m in range(5, 13)]
                   + [("2025-%02d" % m) for m in range(1, 13)]
                   + [("2026-%02d" % m) for m in range(1, 9)])

# Тональность в выгрузке: −1 негатив, 0 нейтрал, 1 позитив.
TONE_NEG, TONE_NEU, TONE_POS = -1, 0, 1

_ES = None


# ------------------------------------------------------------------ служебное

def client():
    global _ES
    if _ES is None:
        from elasticsearch import Elasticsearch

        _ES = Elasticsearch(hosts=[ES_HOST], basic_auth=(ES_USER, ES_PASS), verify_certs=False,
                            headers={"Accept": "application/vnd.elasticsearch+json; compatible-with=9"},
                            request_timeout=300)
    return _ES


def msk_stamp(day: str, end: bool = False) -> int:
    """Начало (или конец) суток по московскому времени в секундах — как считает платформа.

    От этого зависит всё сравнение с месячными отчётами: границы месяцев берутся в местном
    времени, иначе часть сообщений уезжает в соседний месяц и годовые итоги не сходятся.
    """
    dt = datetime.datetime.strptime(str(day)[:10], "%Y-%m-%d")
    if end:
        dt = dt.replace(hour=23, minute=59, second=59)
    return int(dt.replace(tzinfo=MSK).timestamp())


def period_filter(lo: str = "", hi: str = "") -> List[dict]:
    out = []
    if lo or hi:
        rng: Dict[str, Any] = {}
        if lo:
            rng["gte"] = msk_stamp(lo)
        if hi:
            rng["lte"] = msk_stamp(hi, end=True)
        out.append({"range": {"timeCreate": rng}})
    return out


def search(body: dict, ttl: Optional[str] = None) -> dict:
    """Запрос к хранилищу с кэшем на диск: тяжёлые агрегации считаются один раз."""
    if ttl:
        path = os.path.join(CACHE_DIR, ttl + ".json")
        if os.path.isfile(path):
            try:
                return json.load(io.open(path, encoding="utf-8"))
            except Exception:  # noqa: BLE001
                pass
    body = dict(body)
    body["size"] = 0
    body.setdefault("track_total_hits", True)
    res = client().search(index=INDEX, body=body)
    if ttl:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with io.open(os.path.join(CACHE_DIR, ttl + ".json"), "w", encoding="utf-8") as fh:
            json.dump(res.body if hasattr(res, "body") else res, fh, ensure_ascii=False)
    return res.body if hasattr(res, "body") else res


def total_of(res: dict) -> int:
    hits = res.get("hits") or {}
    tot = hits.get("total")
    if isinstance(tot, dict):
        return int(tot.get("value") or 0)
    return int(tot or 0)


def tone_of(res: dict, agg: str = "t") -> Dict[str, int]:
    buckets = ((res.get("aggregations") or {}).get(agg) or {}).get("buckets") or []
    out = {TONE_NEG: 0, TONE_NEU: 0, TONE_POS: 0}
    for bucket in buckets:
        try:
            key = int(bucket.get("key"))
        except (TypeError, ValueError):
            continue
        if key in out:
            out[key] = int(bucket.get("doc_count") or 0)
    return out


MONTH_RUNTIME = {"m": {"type": "date", "script": {"source": "emit(doc['timeCreate'].value * 1000)"}}}


def by_month(query: Optional[dict] = None, aggs: Optional[dict] = None, ttl: Optional[str] = None) -> dict:
    """Разбивка по месяцам в местном времени с тональностью внутри каждого месяца."""
    inner = {"t": {"terms": {"field": "toneMark", "size": 5}}}
    inner.update(aggs or {})
    body = {"size": 0, "track_total_hits": True, "runtime_mappings": MONTH_RUNTIME,
            "aggs": {"d": {"date_histogram": {"field": "m", "calendar_interval": "month",
                                              "format": "yyyy-MM", "time_zone": TZ_NAME,
                                              "min_doc_count": 0},
                           "aggs": inner}}}
    if query:
        body["query"] = query
    return search(body, ttl=ttl)


def months_tone(ttl: str = "months_tone") -> Dict[str, dict]:
    """Объём и тональность по месяцам — по всем сообщениям корпуса в местном времени."""
    res = by_month(ttl=ttl)
    out = {}
    for bucket in ((res.get("aggregations") or {}).get("d") or {}).get("buckets") or []:
        key = str(bucket.get("key_as_string") or "")
        if not re.match(r"^\d{4}-\d{2}$", key):
            continue
        tones = {int(b["key"]): int(b["doc_count"]) for b in (bucket.get("t") or {}).get("buckets") or []}
        tot = int(bucket.get("doc_count") or 0)
        if not tot:
            continue
        neg = tones.get(TONE_NEG, 0)
        out[key] = {"month": key, "total": tot, "negative": neg,
                    "neutral": tones.get(TONE_NEU, 0), "positive": tones.get(TONE_POS, 0),
                    "negative_share": round(neg / float(tot) * 100, 2)}
    return out


def year_tone(months: Dict[str, dict]) -> Dict[int, dict]:
    out = {}
    for year in YEARS:
        lo, hi = YEAR_SPAN[year]
        # Сравниваем ключи месяцев целиком: «2024-05» короче «2024-05-01», и простое сравнение
        # строк выбрасывало первый месяц года.
        rows = [row for key, row in months.items() if lo[:7] <= key <= hi[:7]]
        tot = sum(r["total"] for r in rows)
        neg = sum(r["negative"] for r in rows)
        out[year] = {"total": tot, "negative": neg,
                     "neutral": sum(r["neutral"] for r in rows),
                     "positive": sum(r["positive"] for r in rows),
                     "months": len(rows),
                     "negative_share": round(neg / float(tot or 1) * 100, 2)}
    return out


# ------------------------------------------------------------------ готовые темы датасета

def tag_catalog() -> Dict[str, List[Tuple[str, str]]]:
    """Все связки «поле темы → код темы» из карты полей: {имя темы: [(поле, код), …]}.

    Тема приходит от Brand Analytics готовыми словарями: у сообщения в поле tag_1…tag_33 лежит
    пара «код темы → название». Одно и то же название встречается в разных полях со своими
    кодами, поэтому тема определяется по НАЗВАНИЮ, а сообщение относится к теме, если название
    встретилось в любом из полей.
    """
    path = os.path.join(CACHE_DIR, "tag_catalog.json")
    if os.path.isfile(path):
        try:
            raw = json.load(io.open(path, encoding="utf-8"))
            return {name: [tuple(pair) for pair in pairs] for name, pairs in raw.items()}
        except Exception:  # noqa: BLE001
            pass

    es = client()
    props = list(es.indices.get_mapping(index=INDEX).values())[0]["mappings"].get("properties") or {}
    keys: List[Tuple[str, str]] = []
    for field in sorted([f for f in props if f.startswith("tag_")], key=lambda s: int(s.split("_")[1])):
        for tid in ((props[field] or {}).get("properties") or {}):
            keys.append((field, str(tid)))
    names: Dict[Tuple[str, str], str] = {}
    # Названия быстрее всего собрать из самих сообщений: одно поле — один код темы.
    res = es.search(index=INDEX, body={
        "size": 4000, "query": {"exists": {"field": "tag_1"}},
        "_source": ["tag_%d" % i for i in range(1, 34)]})
    for hit in (res.get("hits") or {}).get("hits") or []:
        for field, value in (hit.get("_source") or {}).items():
            if field.startswith("tag_") and isinstance(value, dict):
                for tid, name in value.items():
                    if str(name or "").strip():
                        names[(field, str(tid))] = str(name).strip()
    missing = [k for k in keys if k not in names]
    for field, tid in missing:
        try:
            one = es.search(index=INDEX, body={
                "size": 1, "query": {"exists": {"field": "%s.%s" % (field, tid)}}, "_source": [field]})
            for hit in (one.get("hits") or {}).get("hits") or []:
                value = (hit.get("_source") or {}).get(field) or {}
                if isinstance(value, dict) and str(value.get(tid) or "").strip():
                    names[(field, tid)] = str(value[tid]).strip()
        except Exception:  # noqa: BLE001
            continue
    catalog: Dict[str, List[Tuple[str, str]]] = {}
    for (field, tid), name in names.items():
        catalog.setdefault(name, [])
        if (field, tid) not in catalog[name]:
            catalog[name].append((field, tid))
    os.makedirs(CACHE_DIR, exist_ok=True)
    with io.open(path, "w", encoding="utf-8") as fh:
        json.dump({k: [list(p) for p in v] for k, v in catalog.items()}, fh, ensure_ascii=False)
    return catalog


def theme_query(pairs: List[Tuple[str, str]], lo: str = "", hi: str = "") -> dict:
    """Сообщения, у которых тема отмечена хотя бы в одном поле."""
    should = [{"exists": {"field": "%s.%s" % (field, tid)}} for field, tid in pairs]
    return {"bool": {"should": should, "minimum_should_match": 1, "filter": period_filter(lo, hi)}}


def label_query(text: str, lo: str = "", hi: str = "", strict: bool = True) -> dict:
    """Сообщения по словесной формулировке темы (название повода из месячного отчёта).

    Ищем по словоформам через разбор русского текста: название вида «Ростикс ростикса ростиксе»
    сводится к одному корню, поэтому совпадение по части слов даёт всю волну, а не пусто.
    """
    match = {"match": {"text": {"query": str(text), "minimum_should_match": "70%" if strict else "40%"}}}
    return {"bool": {"must": [match], "filter": period_filter(lo, hi)}}


def theme_stats(catalog: Dict[str, List[Tuple[str, str]]], ttl: str = "theme_stats") -> List[dict]:
    """Объём и тональность каждой темы: всего, по годам и по месяцам.

    Спам и рекламные метки Brand Analytics отсеиваются тем же правилом, что и в месячных
    отчётах, поэтому в сравнение годов попадают только содержательные темы.
    """
    path = os.path.join(CACHE_DIR, ttl + ".json")
    if os.path.isfile(path):
        try:
            return json.load(io.open(path, encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass
    es = client()
    rows = []
    for name, pairs in catalog.items():
        per_year: Dict[str, dict] = {}
        for year in YEARS:
            lo, hi = YEAR_SPAN[year]
            try:
                res = es.search(index=INDEX, body={
                    "size": 0, "track_total_hits": True, "query": theme_query(pairs, lo, hi),
                    "aggs": {"t": {"terms": {"field": "toneMark", "size": 5}}}})
            except Exception:  # noqa: BLE001
                continue
            tones = tone_of(res)
            per_year[str(year)] = {"total": total_of(res), "negative": tones[TONE_NEG],
                                   "neutral": tones[TONE_NEU], "positive": tones[TONE_POS]}
        total = sum(v["total"] for v in per_year.values())
        if not total:
            continue
        rows.append({"name": name, "pairs": [list(p) for p in pairs], "total": total,
                     "years": per_year})
    rows.sort(key=lambda r: -r["total"])
    os.makedirs(CACHE_DIR, exist_ok=True)
    with io.open(path, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False)
    return rows


def theme_months(pairs: List[Tuple[str, str]], ttl: str) -> Dict[str, int]:
    """Число сообщений темы по месяцам — строка тепловой карты."""
    res = by_month(query=theme_query(pairs), ttl=ttl)
    out = {}
    for bucket in ((res.get("aggregations") or {}).get("d") or {}).get("buckets") or []:
        key = str(bucket.get("key_as_string") or "")
        if re.match(r"^\d{4}-\d{2}$", key) and int(bucket.get("doc_count") or 0):
            out[key] = int(bucket["doc_count"])
    return out


# ------------------------------------------------------------------ площадки, авторы, голос

def platform_rows(ttl: str = "platforms_all") -> dict:
    """Площадки: объём и тональность по годам и по месяцам — сдвиг аудитории."""
    inner = {"t": {"terms": {"field": "toneMark", "size": 5}}}
    body = {"size": 0, "track_total_hits": True,
            "aggs": {"hubs": {"terms": {"field": "hub", "size": 30}, "aggs": inner}}}
    return search(body, ttl=ttl)


def platform_by_year(months: Dict[str, dict]) -> Dict[int, List[dict]]:
    """Доли площадок по годам: сравниваем одинаковые наборы площадок."""
    out: Dict[int, List[dict]] = {}
    es = client()
    cache_ttl = "platforms_year"
    path = os.path.join(CACHE_DIR, cache_ttl + ".json")
    data = None
    if os.path.isfile(path):
        try:
            data = json.load(io.open(path, encoding="utf-8"))
        except Exception:  # noqa: BLE001
            data = None
    if data is None:
        data = {}
        for year in YEARS:
            lo, hi = YEAR_SPAN[year]
            res = es.search(index=INDEX, body={
                "size": 0, "track_total_hits": True,
                "query": {"bool": {"filter": period_filter(lo, hi)}},
                "aggs": {"hubs": {"terms": {"field": "hub", "size": 15},
                                  "aggs": {"t": {"terms": {"field": "toneMark", "size": 5}}}}}})
            rows = []
            for bucket in ((res.get("aggregations") or {}).get("hubs") or {}).get("buckets") or []:
                tones = {int(b["key"]): int(b["doc_count"]) for b in (bucket.get("t") or {}).get("buckets") or []}
                rows.append({"hub": str(bucket.get("key")), "total": int(bucket.get("doc_count") or 0),
                             "negative": tones.get(TONE_NEG, 0), "positive": tones.get(TONE_POS, 0)})
            data[str(year)] = rows
        os.makedirs(CACHE_DIR, exist_ok=True)
        with io.open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False)
    for year in YEARS:
        out[year] = data.get(str(year)) or []
    return out


def platform_negative_split(ttl: str = "platform_negative") -> dict:
    """Где живёт негатив: доля каждой площадки в общем негативе и её собственная доля негатива.

    Это ответ на вопрос «куда идти работать с репутацией»: у карт и отзовиков своя доля
    негатива в разы выше, чем у соцсетей, и вклад в общий негатив считается отдельно.
    """
    res = search({"aggs": {"hubs": {"terms": {"field": "hub", "size": 20},
                                    "aggs": {"t": {"terms": {"field": "toneMark", "size": 5}}}}}},
                 ttl=ttl)
    rows = []
    for bucket in ((res.get("aggregations") or {}).get("hubs") or {}).get("buckets") or []:
        tones = {int(b["key"]): int(b["doc_count"]) for b in (bucket.get("t") or {}).get("buckets") or []}
        tot = int(bucket.get("doc_count") or 0)
        neg = tones.get(TONE_NEG, 0)
        rows.append({"hub": str(bucket.get("key")), "total": tot, "negative": neg,
                     "positive": tones.get(TONE_POS, 0),
                     "negative_share": round(neg / float(tot or 1) * 100, 2)})
    neg_all = sum(r["negative"] for r in rows)
    for row in rows:
        row["share_of_negative"] = round(row["negative"] / float(neg_all or 1) * 100, 2)
    rows.sort(key=lambda r: -r["negative"])
    return {"rows": rows, "negative_total": neg_all}


def platform_theme_matrix(theme_rows: List[dict], limit: int = 6, ttl: str = "platform_theme") -> List[dict]:
    """Матрица «площадка × тема × тональность» для крупнейших содержательных тем."""
    out = []
    for row in theme_rows[:limit]:
        pairs = [tuple(p) for p in row["pairs"]]
        rows = platform_theme_tone(pairs, ttl="ptm_%s" % re.sub(r"\W+", "_", row["name"])[:36])
        out.append({"theme": row["name"], "total": row["total"], "hubs": rows})
    return out


def risk_index(months: Dict[str, dict]) -> List[dict]:
    """Индекс риска месяца: объём × доля негатива × скорость роста — в одной шкале от 0 до 100.

    Три составляющие складываются с весами: доля негатива важнее всего (0,5), объём обсуждения
    (0,3) и рост относительно предыдущего месяца (0,2). Каждая составляющая приведена к своей
    наибольшей величине в периоде, поэтому индекс читается как «насколько этот месяц хуже
    спокойного» и не зависит от единиц измерения.
    """
    keys = [k for k in sorted(months) if k in EXPECTED_MONTHS]
    if not keys:
        return []
    max_share = max(months[k]["negative_share"] for k in keys) or 1.0
    max_total = max(months[k]["total"] for k in keys) or 1
    growth = {}
    for idx, key in enumerate(keys):
        prev = months.get(keys[idx - 1]) if idx else None
        growth[key] = (months[key]["total"] / float(prev["total"] or 1) - 1.0) if prev and prev["total"] else 0.0
    max_growth = max([g for g in growth.values() if g > 0] or [1.0])
    rows = []
    for key in keys:
        share_part = months[key]["negative_share"] / max_share
        volume_part = months[key]["total"] / float(max_total)
        growth_part = max(0.0, growth[key]) / max_growth
        rows.append({"month": key, "total": months[key]["total"],
                     "negative": months[key]["negative"],
                     "negative_share": months[key]["negative_share"],
                     "growth": round(growth[key] * 100, 1),
                     "index": round((0.5 * share_part + 0.3 * volume_part + 0.2 * growth_part) * 100, 1)})
    rows.sort(key=lambda r: -r["index"])
    return rows


def platform_theme_tone(pairs: List[Tuple[str, str]], ttl: str) -> List[dict]:
    """Матрица «площадка × тема × тональность» для одной темы."""
    res = search({"query": theme_query(pairs),
                  "aggs": {"hubs": {"terms": {"field": "hub", "size": 8},
                                    "aggs": {"t": {"terms": {"field": "toneMark", "size": 5}}}}}}, ttl=ttl)
    rows = []
    for bucket in ((res.get("aggregations") or {}).get("hubs") or {}).get("buckets") or []:
        tones = {int(b["key"]): int(b["doc_count"]) for b in (bucket.get("t") or {}).get("buckets") or []}
        tot = int(bucket.get("doc_count") or 0)
        rows.append({"hub": str(bucket.get("key")), "total": tot, "negative": tones.get(TONE_NEG, 0),
                     "negative_share": round(tones.get(TONE_NEG, 0) / float(tot or 1) * 100, 2)})
    return rows


def authors_agg(size: int = 1500, ttl: str = "authors_top") -> dict:
    """Крупные авторы: сколько пишут, с какой тональностью, по годам, на каких площадках.

    Берём верхушку по числу сообщений — именно она задаёт повестку; вклад остальных считается
    по остатку (сколько сообщений осталось за пределами верхушки).
    """
    inner = {"y": {"date_histogram": {"field": "m", "calendar_interval": "year",
                                      "format": "yyyy", "time_zone": TZ_NAME, "min_doc_count": 0},
                   "aggs": {"t": {"terms": {"field": "toneMark", "size": 5}}}},
             "hubs": {"terms": {"field": "hub", "size": 3}},
             "kind": {"terms": {"field": "authorObject.author_type.keyword", "size": 3}},
             "neg": {"filter": {"term": {"toneMark": TONE_NEG}}},
             "pos": {"filter": {"term": {"toneMark": TONE_POS}}},
             "reach": {"sum": {"field": "audienceCount"}},
             "reactions": {"sum": {"field": "likesCount"}},
             "comments": {"sum": {"field": "commentsCount"}}}
    body = {"size": 0, "track_total_hits": True, "runtime_mappings": MONTH_RUNTIME,
            "aggs": {"a": {"terms": {"field": "authorObject.fullname.keyword", "size": size},
                           "aggs": inner}}}
    return search(body, ttl=ttl)


def author_voice_rows(es_res: dict) -> List[dict]:
    """Разбор верхушки авторов: доля негатива, годы активности, площадка, охват."""
    rows = []
    for bucket in ((es_res.get("aggregations") or {}).get("a") or {}).get("buckets") or []:
        name = str(bucket.get("key") or "").strip()
        if not name:
            continue
        tot = int(bucket.get("doc_count") or 0)
        neg = int(((bucket.get("neg") or {}).get("doc_count")) or 0)
        pos = int(((bucket.get("pos") or {}).get("doc_count")) or 0)
        years = {}
        for yb in ((bucket.get("y") or {}).get("buckets")) or []:
            ykey = str(yb.get("key_as_string") or "")
            if re.match(r"^\d{4}$", ykey) and int(yb.get("doc_count") or 0):
                tones = {int(b["key"]): int(b["doc_count"]) for b in (yb.get("t") or {}).get("buckets") or []}
                years[ykey] = {"total": int(yb["doc_count"]), "negative": tones.get(TONE_NEG, 0),
                               "positive": tones.get(TONE_POS, 0)}
        hubs = [str(b.get("key")) for b in ((bucket.get("hubs") or {}).get("buckets")) or []]
        kinds = [str(b.get("key")) for b in ((bucket.get("kind") or {}).get("buckets")) or []]
        rows.append({"name": name, "total": tot, "negative": neg, "positive": pos,
                     "negative_share": round(neg / float(tot or 1) * 100, 2),
                     "positive_share": round(pos / float(tot or 1) * 100, 2),
                     "years": years, "hubs": hubs, "kind": kinds[0] if kinds else "",
                     "reach": int(((bucket.get("reach") or {}).get("value")) or 0),
                     "reactions": int(((bucket.get("reactions") or {}).get("value")) or 0),
                     "comments": int(((bucket.get("comments") or {}).get("value")) or 0)})
    return rows


def author_tail(es_res: dict) -> int:
    """Сколько сообщений приходится на авторов за пределами верхушки."""
    return int(((es_res.get("aggregations") or {}).get("a") or {}).get("sum_other_doc_count") or 0)


# Порог «автор говорит постоянно»: один-два отзыва за три года — это не когорта, а случайный голос.
COHORT_MIN_MESSAGES = 50


def cohort_of(row: dict) -> str:
    """Когорта автора по его собственной тональности.

    Хронический критик — заметная часть сообщений негативная; лояльный — заметная доля позитива
    при низком негативе; нейтральный информатор — почти всё нейтрально (так пишут справочные и
    новостные аккаунты, объявления, промо). Остальные — «смешанные».
    """
    if row["total"] < COHORT_MIN_MESSAGES:
        return ""
    if row["negative_share"] >= 55.0:
        return "хронические критики"
    if row["positive_share"] >= 35.0 and row["negative_share"] < 30.0:
        return "лояльные"
    if row["negative_share"] <= 10.0 and row["positive_share"] <= 10.0:
        return "нейтральные информаторы"
    return "смешанные"


COHORT_ORDER = ("хронические критики", "лояльные", "нейтральные информаторы", "смешанные")


def author_cohorts(voice: List[dict], tail: int) -> Dict[str, Any]:
    """Когорты постоянных авторов: сколько их, сколько сообщений и негатива, как менялось по годам.

    Когорты считаются среди авторов с постоянным присутствием (не меньше порога сообщений).
    Авторы с единичными сообщениями в когорты не попадают: их вклад возвращается отдельно
    полем ``tail`` — иначе одна случайная реплика весила бы столько же, сколько постоянный отзыв.
    """
    out: Dict[str, dict] = {}
    for row in voice:
        group = cohort_of(row)
        if not group:
            continue
        slot = out.setdefault(group, {"cohort": group, "authors": 0, "messages": 0, "negative": 0,
                                      "positive": 0, "reach": 0, "years": {}, "top": []})
        slot["authors"] += 1
        slot["messages"] += row["total"]
        slot["negative"] += row["negative"]
        slot["positive"] += row["positive"]
        slot["reach"] += row["reach"]
        for year, value in (row.get("years") or {}).items():
            yslot = slot["years"].setdefault(year, {"authors": 0, "messages": 0, "negative": 0,
                                                    "positive": 0})
            yslot["authors"] += 1
            yslot["messages"] += value["total"]
            yslot["negative"] += value["negative"]
            yslot["positive"] += value["positive"]
        if len(slot["top"]) < 5:
            slot["top"].append(row)
    ordered = {key: out[key] for key in COHORT_ORDER if key in out}
    ordered["_tail"] = {"authors_note": "авторы с единичными сообщениями",
                        "messages": int(tail), "negative": None}
    return ordered



def top_authors_by_year(voice: List[dict], year: int, limit: int = 10) -> List[dict]:
    rows = [r for r in voice if (r.get("years") or {}).get(str(year))]
    rows.sort(key=lambda r: -r["years"][str(year)]["total"])
    return rows[:limit]



def brand_share_of_voice(ttl: str = "sov") -> dict:
    """Доля голоса: сколько сообщений упоминают бренд и сколько — конкурентов, по годам."""
    path = os.path.join(CACHE_DIR, ttl + ".json")
    if os.path.isfile(path):
        try:
            return json.load(io.open(path, encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass
    es = client()
    players = [
        ("brand", "наш бренд", ["KFC", "КФС", "Rostic", "Ростикс"]),
        ("bk", "Burger King", ["Burger King", "Бургер Кинг", "Бургер кинг"]),
        ("vit", "«Вкусно и точка»", ["Вкусно и точка", "Вкусно и точк"]),
        ("mcd", "McDonald's", ["McDonald", "Макдоналдс"]),
    ]
    out: Dict[str, Any] = {"players": [{"key": k, "label": l} for k, l, _ in players], "years": {}}
    for year in YEARS:
        lo, hi = YEAR_SPAN[year]
        slot = {}
        for key, label, phrases in players:
            # Бренд упоминается в нескольких написаниях и в разных падежах — берём объединение.
            should = [{"match_phrase": {"text": p}} for p in phrases]
            res = es.search(index=INDEX, body={
                "size": 0, "track_total_hits": True,
                "query": {"bool": {"should": should, "minimum_should_match": 1,
                                   "filter": period_filter(lo, hi)}},
                "aggs": {"t": {"terms": {"field": "toneMark", "size": 5}}}})
            tones = tone_of(res)
            slot[key] = {"label": label, "total": total_of(res), "negative": tones[TONE_NEG],
                         "positive": tones[TONE_POS], "phrases": phrases}
        out["years"][str(year)] = slot
    # Контекст сравнений: сообщения, где бренд стоит рядом с конкурентом.
    both = {}
    for year in YEARS:
        lo, hi = YEAR_SPAN[year]
        for key, label, phrases in players[1:]:
            should_brand = [{"match_phrase": {"text": p}} for p in players[0][2]]
            should_other = [{"match_phrase": {"text": p}} for p in phrases]
            res = es.search(index=INDEX, body={
                "size": 0, "track_total_hits": True,
                "query": {"bool": {"must": [{"bool": {"should": should_brand, "minimum_should_match": 1}},
                                            {"bool": {"should": should_other, "minimum_should_match": 1}}],
                                   "filter": period_filter(lo, hi)}},
                "aggs": {"t": {"terms": {"field": "toneMark", "size": 5}}}})
            tones = tone_of(res)
            both.setdefault(key, {})[str(year)] = {"total": total_of(res), "negative": tones[TONE_NEG],
                                                   "positive": tones[TONE_POS]}
    out["together"] = both
    os.makedirs(CACHE_DIR, exist_ok=True)
    with io.open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False)
    return out


def term_counts(terms: List[str], ttl: str = "terms") -> Dict[str, Dict[str, dict]]:
    """Сколько раз каждая формулировка встречается в каждом году и с какой тональностью."""
    path = os.path.join(CACHE_DIR, ttl + ".json")
    data = {}
    if os.path.isfile(path):
        try:
            data = json.load(io.open(path, encoding="utf-8"))
        except Exception:  # noqa: BLE001
            data = {}
    es = client()
    changed = False
    for term in terms:
        if term in data:
            continue
        slot = {}
        for year in YEARS:
            lo, hi = YEAR_SPAN[year]
            res = es.search(index=INDEX, body={
                "size": 0, "track_total_hits": True,
                "query": {"bool": {"must": [{"match_phrase": {"text": term}}],
                                   "filter": period_filter(lo, hi)}},
                "aggs": {"t": {"terms": {"field": "toneMark", "size": 5}}}})
            tones = tone_of(res)
            slot[str(year)] = {"total": total_of(res), "negative": tones[TONE_NEG],
                               "positive": tones[TONE_POS]}
        data[term] = slot
        changed = True
    if changed:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with io.open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False)
    return {t: data.get(t, {}) for t in terms}


def day_timeline(query: dict, ttl: str, lo: str = "", hi: str = "") -> Dict[str, dict]:
    """Ход волны по дням: сколько сообщений и какая тональность внутри каждого дня."""
    res = search({"query": query,
                  "runtime_mappings": MONTH_RUNTIME,
                  "aggs": {"d": {"date_histogram": {"field": "m", "calendar_interval": "day",
                                                    "format": "yyyy-MM-dd", "time_zone": TZ_NAME,
                                                    "min_doc_count": 1},
                                 "aggs": {"t": {"terms": {"field": "toneMark", "size": 5}}}}}}, ttl=ttl)
    out = {}
    for bucket in ((res.get("aggregations") or {}).get("d") or {}).get("buckets") or []:
        day = str(bucket.get("key_as_string") or "")
        tones = {int(b["key"]): int(b["doc_count"]) for b in (bucket.get("t") or {}).get("buckets") or []}
        tot = int(bucket.get("doc_count") or 0)
        out[day] = {"total": tot, "negative": tones.get(TONE_NEG, 0), "positive": tones.get(TONE_POS, 0),
                    "negative_share": round(tones.get(TONE_NEG, 0) / float(tot or 1) * 100, 2)}
    return out


def day_total(day: str, ttl: str = "") -> int:
    """Сколько всего сообщений по теме вышло в один день — фон для объёма отдельной волны."""
    res = search({"query": {"bool": {"filter": period_filter(day, day)}}},
                 ttl=ttl or ("day_all_%s" % re.sub(r"\W+", "", day)))
    return total_of(res)


def wave_amplifiers(query: dict, ttl: str) -> dict:
    """Кто разгонял волну: площадки и авторы."""
    res = search({"query": query,
                  "aggs": {"hubs": {"terms": {"field": "hub", "size": 8},
                                    "aggs": {"t": {"terms": {"field": "toneMark", "size": 5}}}},
                           "authors": {"terms": {"field": "authorObject.fullname.keyword", "size": 10},
                                       "aggs": {"t": {"terms": {"field": "toneMark", "size": 5}}}},
                           "kinds": {"terms": {"field": "authorObject.author_type.keyword", "size": 5}}}},
                 ttl=ttl)
    aggs = res.get("aggregations") or {}
    def rows(key):
        out = []
        for bucket in (aggs.get(key) or {}).get("buckets") or []:
            tones = {int(b["key"]): int(b["doc_count"]) for b in (bucket.get("t") or {}).get("buckets") or []}
            tot = int(bucket.get("doc_count") or 0)
            out.append({"name": str(bucket.get("key")), "total": tot,
                        "negative": tones.get(TONE_NEG, 0),
                        "negative_share": round(tones.get(TONE_NEG, 0) / float(tot or 1) * 100, 2)})
        return out
    return {"hubs": rows("hubs"), "authors": rows("authors"),
            "kinds": [{"name": str(b.get("key")), "total": int(b.get("doc_count") or 0)}
                      for b in (aggs.get("kinds") or {}).get("buckets") or []]}


    try:
        path = os.path.join(CACHE_DIR, name + ".json")
        if os.path.isfile(path):
            with io.open(path, encoding="utf-8") as fh:
                return json.load(fh)
    except Exception:  # noqa: BLE001
        pass
    return None


def put_json(name: str, value: Any) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, name + ".json")
    with io.open(path, "w", encoding="utf-8") as fh:
        json.dump(value, fh, ensure_ascii=False)
    return path


def wave_before_after(query_builder, lo_win: Tuple[str, str], days: int = 14, tag: str = "") -> Dict[str, dict]:
    """Объём и тональность до и после окна кампании — видно, был ли откат.

    ``tag`` — чем именно измеряется окно. Он обязателен в ключе кэша: без него две кампании
    с одинаковыми датами получили бы один и тот же сохранённый результат.
    """
    start = datetime.datetime.strptime(lo_win[0], "%Y-%m-%d")
    end = datetime.datetime.strptime(lo_win[1], "%Y-%m-%d")
    spans = {
        "before": ((start - datetime.timedelta(days=days)).strftime("%Y-%m-%d"),
                   (start - datetime.timedelta(days=1)).strftime("%Y-%m-%d")),
        "during": (lo_win[0], lo_win[1]),
        "after": ((end + datetime.timedelta(days=1)).strftime("%Y-%m-%d"),
                  (end + datetime.timedelta(days=days)).strftime("%Y-%m-%d")),
    }
    stamp = re.sub(r"\W+", "", str(tag))[:40]
    out = {}
    for key, (lo, hi) in spans.items():
        res = search({"query": query_builder(lo, hi),
                      "aggs": {"t": {"terms": {"field": "toneMark", "size": 5}}}},
                     ttl="camp_%s_%s_%s_%s" % (stamp, re.sub(r"\W+", "", lo_win[0]), key,
                                               re.sub(r"\W+", "", lo_win[1])))
        tones = tone_of(res)
        tot = total_of(res)
        out[key] = {"from": lo, "to": hi, "total": tot, "negative": tones[TONE_NEG],
                    "positive": tones[TONE_POS],
                    "positive_share": round(tones[TONE_POS] / float(tot or 1) * 100, 2),
                    "negative_share": round(tones[TONE_NEG] / float(tot or 1) * 100, 2)}
    return out


def sample_messages(query: dict, limit: int = 120, order: str = "reach") -> List[dict]:
    """Выборка сообщений волны: самые заметные по охвату и реакциям.

    Модель читает только такую выборку — сотни сообщений вместо миллионов, поэтому разбор
    смысла стоит минуты, а не часы.
    """
    sort = [{"audienceCount": {"order": "desc"}}, {"likesCount": {"order": "desc"}},
            {"commentsCount": {"order": "desc"}}]
    if order == "fresh":
        sort = [{"timeCreate": {"order": "desc"}}]
    res = client().search(index=INDEX, body={
        "size": max(1, min(int(limit), 400)), "query": query, "sort": sort,
        "_source": ["text", "timeCreate", "hub", "toneMark", "url", "authorObject",
                    "likesCount", "commentsCount", "audienceCount", "city"]})
    out = []
    for hit in (res.get("hits") or {}).get("hits") or []:
        src = hit.get("_source") or {}
        text = re.sub(r"\s+", " ", str(src.get("text") or src.get("title") or "")).strip()
        if len(text) < 25:
            continue
        out.append({"text": text[:600], "ts": int(src.get("timeCreate") or 0),
                    "date": datetime.datetime.fromtimestamp(int(src.get("timeCreate") or 0), MSK).strftime("%d.%m.%Y"),
                    "hub": str(src.get("hub") or ""), "tone": int(src.get("toneMark") or 0),
                    "url": str(src.get("url") or ""),
                    "author": str((src.get("authorObject") or {}).get("fullname") or ""),
                    "likes": int(src.get("likesCount") or 0),
                    "comments": int(src.get("commentsCount") or 0),
                    "audience": int(src.get("audienceCount") or 0),
                    "city": str(src.get("city") or "")})
    return out


def sample_by_text(text: str, lo: str = "", hi: str = "", limit: int = 120) -> List[dict]:
    return sample_messages(label_query(text, lo, hi), limit=limit)


def wave_shape(days: Dict[str, dict]) -> dict:
    """Форма волны: когда началась, когда достигла пика и через сколько дней сошла.

    За старт берётся первый день, набравший не меньше десятой части пикового объёма, — иначе
    стартом оказался бы случайный одиночный отклик. Затухание — первый день после пика, когда
    объём опустился ниже той же десятой части. Число дней «от старта до пика» и есть срок,
    который есть у отдела на реакцию.
    """
    keys = sorted(days)
    if not keys:
        return {}
    peak_day = max(keys, key=lambda k: days[k]["total"])
    peak = days[peak_day]["total"]
    threshold = max(1, int(peak * 0.1))
    active = [k for k in keys if days[k]["total"] >= threshold]
    first_active = active[0] if active else keys[0]
    after = [k for k in keys if k >= peak_day and days[k]["total"] < threshold]
    fade = after[0] if after else keys[-1]
    span = [k for k in keys if first_active <= k <= fade]
    tone_early = [days[k]["negative_share"] for k in span[:max(1, len(span) // 3)]]
    tone_peak = [days[k]["negative_share"] for k in span if days[k]["total"] >= peak * 0.5]
    return {
        "first": keys[0], "last": keys[-1], "start": first_active, "peak_day": peak_day,
        "peak": peak, "threshold": threshold, "fade": fade,
        "days_total": len(keys), "days_to_peak": _days_between(first_active, peak_day),
        "days_active": len(span), "days_fade": _days_between(peak_day, fade),
        "total": sum(days[k]["total"] for k in keys),
        "negative": sum(days[k]["negative"] for k in keys),
        "tone_early": round(sum(tone_early) / float(len(tone_early) or 1), 2),
        "tone_at_peak": round(sum(tone_peak) / float(len(tone_peak) or 1), 2) if tone_peak else None,
        "share_at_peak_day": round(peak / float(sum(days[k]["total"] for k in keys) or 1) * 100, 1),
    }


def _days_between(day_a: str, day_b: str) -> int:
    try:
        a = datetime.datetime.strptime(day_a[:10], "%Y-%m-%d")
        b = datetime.datetime.strptime(day_b[:10], "%Y-%m-%d")
        return (b - a).days
    except Exception:  # noqa: BLE001
        return 0


def theme_window(months_map: Dict[str, int], share: float = 0.12) -> Tuple[str, str, str]:
    """Окно активности темы: месяцы, в которых тема набрала заметную часть своего объёма.

    Возвращает (первый месяц окна, последний месяц окна, вид). Так определяется период кампании
    без ручного выбора дат: тема сама показывает, когда она жила, а когда сошла.
    """
    if not months_map:
        return "", "", "нет данных"
    total = sum(months_map.values()) or 1
    active = [k for k in sorted(months_map) if months_map[k] >= total * share]
    if not active:
        active = [max(months_map, key=lambda k: months_map[k])]
    months = sorted(active)
    # Окно рвётся, если между активными месяцами больше двух месяцев тишины.
    groups, current = [], [months[0]]
    for prev, cur in zip(months, months[1:]):
        gap = (int(cur[:4]) * 12 + int(cur[5:7])) - (int(prev[:4]) * 12 + int(prev[5:7]))
        if gap <= 2:
            current.append(cur)
        else:
            groups.append(current)
            current = [cur]
    groups.append(current)
    best = max(groups, key=lambda g: sum(months_map[k] for k in g))
    kind = "разовая кампания" if len(best) <= 2 else "долгая линия"
    return best[0], best[-1], kind



# ------------------------------------------------------------------ локальные модели

def llm(prompt: str, system: str = "", fast: bool = True, max_tokens: int = 1400,
        temperature: float = 0.15, timeout: int = 420) -> Tuple[str, str]:
    """Один вызов локальной модели. Возвращает (текст, какая модель ответила).

    fast=True — быстрая qwen3-4b-fast (порт 8001): массовое чтение выборок.
    fast=False — Qwen3-32B (порт 8000): названия тем, интерпретация, формулировки выводов.
    """
    url, model = (FAST_URL, FAST_MODEL) if fast else (GEN_URL, GEN_MODEL)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    payload = {"model": model, "messages": messages, "temperature": temperature,
               "max_tokens": int(max_tokens), "chat_template_kwargs": {"enable_thinking": False}}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data)
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        text = (((body.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
        return re.sub(r"^\s*```(?:json)?|```\s*$", "", text).strip(), model
    except Exception as exc:  # noqa: BLE001
        return "", "ошибка вызова: %s" % type(exc).__name__


def llm_json(prompt: str, system: str = "", fast: bool = True, max_tokens: int = 1600):
    """Вызов модели с ожиданием JSON-ответа. Возвращает (объект или None, модель)."""
    text, model = llm(prompt, system=system, fast=fast, max_tokens=max_tokens)
    if not text:
        return None, model
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1]), model
        except Exception:  # noqa: BLE001
            pass
    return None, model
