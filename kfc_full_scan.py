#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Полный проход по корпусу KFC: настоящие рейтинги и объёмы по каждому заведению.

Нужен потому, что при разборе только негатива рейтинг заведения считается по жалобам
и выглядит заниженным. Здесь по всем 2,93 млн сообщений собираются: объём по заведению,
распределение оценок, тональность, месяцы, охват, а также тональность по городам и месяцам.

Запуск:  nohup venv_py312_clean/bin/python -u kfc_full_scan.py > /tmp/kfc_full_scan.log 2>&1 &
"""
from __future__ import annotations

import collections
import datetime
import io
import json
import os
import re
import time

from elasticsearch import Elasticsearch

INDEX = "kfc_13.05.2024-22.09.2026"
CACHE = "/tmp/kfc_hotspots"
MSK = datetime.timezone(datetime.timedelta(hours=3))
LO_DAY, HI_DAY = "2024-05-13", "2026-08-31"


def log(message: str) -> None:
    print("[%s] %s" % (datetime.datetime.now().strftime("%H:%M:%S"), message), flush=True)


def day_stamp(day: str, end: bool = False) -> int:
    dt = datetime.datetime.strptime(day, "%Y-%m-%d")
    if end:
        dt = dt.replace(hour=23, minute=59, second=59)
    return int(dt.replace(tzinfo=MSK).timestamp())


LO_TS, HI_TS = day_stamp(LO_DAY), day_stamp(HI_DAY, end=True)


def month_of(ts) -> str:
    try:
        dt = datetime.datetime.fromtimestamp(int(float(ts)), MSK)
    except Exception:  # noqa: BLE001
        return ""
    return "%04d-%02d" % (dt.year, dt.month)


def org_id(url: str) -> str:
    if not url:
        return ""
    for pattern in (r"/org/[^/]+/(\d+)", r"/firm/(\d+)", r"maps/org/(\d+)", r"/(\d{8,})"):
        hit = re.search(pattern, url)
        if hit:
            return hit.group(1)
    return ""


def main() -> None:
    path = os.path.join(CACHE, "scan_all.json")
    if os.path.isfile(path):
        log("полный проход уже посчитан: %s" % path)
        return
    es = Elasticsearch(hosts=["http://localhost:9200"], basic_auth=("elastic", "biz8z5i1w0nLPmEweKgP"),
                       verify_certs=False, request_timeout=600)
    total = es.count(index=INDEX)["count"]
    log("всего сообщений: %d" % total)
    query = {"bool": {"filter": [{"range": {"timeCreate": {"gte": LO_TS, "lte": HI_TS}}}]}}
    source = ["timeCreate", "toneMark", "city", "url", "review_rating", "audienceCount", "hub", "hubtype"]
    started = time.time()
    scanned = 0
    restaurants = {}
    city_month_tone = collections.Counter()
    month_tone = collections.Counter()
    rating_month = collections.Counter()
    city_rating = collections.defaultdict(collections.Counter)

    resp = es.search(index=INDEX, body={"size": 5000, "query": query, "_source": source, "sort": ["_doc"]},
                     scroll="10m")
    sid = resp["_scroll_id"]
    try:
        while True:
            hits = resp["hits"]["hits"]
            if not hits:
                break
            for hit in hits:
                src = hit["_source"]
                scanned += 1
                month = month_of(src.get("timeCreate"))
                tone = int(src.get("toneMark") or 0)
                city = (src.get("city") or "").strip()
                rating = str(src.get("review_rating") or "").strip()
                reach = int(src.get("audienceCount") or 0)
                month_tone[(month, tone)] += 1
                if city:
                    city_month_tone[(city, month, tone)] += 1
                    if rating:
                        city_rating[city][rating] += 1
                if rating:
                    rating_month[(rating, month)] += 1
                rid = org_id(src.get("url") or "")
                if rid:
                    row = restaurants.get(rid)
                    if row is None:
                        row = restaurants[rid] = {
                            "id": rid, "city": city, "hub": src.get("hub") or "",
                            "count": 0, "neg": 0, "neu": 0, "pos": 0, "reach": 0,
                            "ratings": collections.Counter(), "months": collections.Counter(),
                        }
                    row["count"] += 1
                    row["reach"] += reach
                    row["months"][month] += 1
                    if tone < 0:
                        row["neg"] += 1
                    elif tone > 0:
                        row["pos"] += 1
                    else:
                        row["neu"] += 1
                    if rating:
                        row["ratings"][rating] += 1
                        if not row["city"] and city:
                            row["city"] = city
            if scanned % 200000 < 5000:
                log("прошло %d из %d (%.0f с)" % (scanned, total, time.time() - started))
            resp = es.scroll(scroll_id=sid, scroll="10m")
            sid = resp["_scroll_id"]
    finally:
        try:
            es.clear_scroll(scroll_id=sid)
        except Exception:  # noqa: BLE001
            pass

    rows = []
    for row in restaurants.values():
        ratings = dict(row["ratings"])
        votes = sum(v for k, v in ratings.items() if k.isdigit())
        avg = (sum(int(k) * v for k, v in ratings.items() if k.isdigit()) / votes) if votes else None
        rows.append({
            "id": row["id"], "city": row["city"], "hub": row["hub"], "count": row["count"],
            "neg": row["neg"], "neu": row["neu"], "pos": row["pos"],
            "negative_share": round(row["neg"] / row["count"], 4) if row["count"] else 0,
            "reach": row["reach"], "ratings": ratings, "votes": votes,
            "avg_rating": round(avg, 2) if avg else None,
            "months": dict(row["months"]),
        })
    rows.sort(key=lambda r: -r["neg"])
    data = {
        "scanned": scanned, "seconds": round(time.time() - started, 1),
        "month_tone": {"%s|%d" % k: v for k, v in month_tone.items()},
        "city_month_tone": {"%s|%s|%d" % k: v for k, v in city_month_tone.items()},
        "rating_month": {"%s|%s" % k: v for k, v in rating_month.items()},
        "city_rating": {k: dict(v) for k, v in city_rating.items()},
        "restaurants": rows[:1500],
    }
    with io.open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False)
    log("сохранено: %d сообщений, заведений %d, %.0f с" % (scanned, len(rows), time.time() - started))
    log("топ-10 заведений по негативу: " + "; ".join(
        "%s %s — %d негатива из %d, рейтинг %s" % (r["city"], r["id"], r["neg"], r["count"], r["avg_rating"])
        for r in rows[:10]))


if __name__ == "__main__":
    main()
