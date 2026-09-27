#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Нормировка по каналам: чтобы сравнивать сравнимое.

Отзывы на картах и соцсети дают разную долю негатива: в отзывах она высокая по определению,
в соцсетях низкая. Если сравнивать города «в лоб», выигрывают те, где больше отзывов.
Здесь считается ожидаемая доля негатива для каждого города при его собственном наборе каналов,
и по ней видно настоящий перевес.

Результат: /tmp/kfc_hotspots/channel_norm.json
"""
from __future__ import annotations

import datetime
import io
import json
import os

from elasticsearch import Elasticsearch

INDEX = "kfc_13.05.2024-22.09.2026"
CACHE = "/tmp/kfc_hotspots"
MSK = datetime.timezone(datetime.timedelta(hours=3))
LO_TS = int(datetime.datetime(2024, 5, 13, tzinfo=MSK).timestamp())
HI_TS = int(datetime.datetime(2026, 8, 31, 23, 59, 59, tzinfo=MSK).timestamp())


def tones(buckets) -> dict:
    return {int(b["key"]): b["doc_count"] for b in buckets}


def main() -> None:
    path = os.path.join(CACHE, "channel_norm.json")
    if os.path.isfile(path):
        print("уже посчитано:", path)
        return
    es = Elasticsearch(hosts=["http://localhost:9200"], basic_auth=("elastic", "biz8z5i1w0nLPmEweKgP"),
                       verify_certs=False, request_timeout=600)
    body = {
        "size": 0, "track_total_hits": True,
        "query": {"bool": {"filter": [{"range": {"timeCreate": {"gte": LO_TS, "lte": HI_TS}}}]}},
        "aggs": {
            "hub": {"terms": {"field": "hubtype.keyword", "size": 10},
                    "aggs": {"t": {"terms": {"field": "toneMark", "size": 3}}}},
            "city": {"terms": {"field": "city", "size": 400}, "aggs": {
                "h": {"terms": {"field": "hubtype.keyword", "size": 10},
                      "aggs": {"t": {"terms": {"field": "toneMark", "size": 3}}}}}},
        },
    }
    res = es.search(index=INDEX, body=body)
    by_hub = {}
    for bucket in res["aggregations"]["hub"]["buckets"]:
        t = tones(bucket["t"]["buckets"])
        by_hub[bucket["key"]] = {"total": bucket["doc_count"], "negative": t.get(-1, 0),
                                 "neutral": t.get(0, 0), "positive": t.get(1, 0),
                                 "share": round(t.get(-1, 0) / float(bucket["doc_count"] or 1), 4)}
    cities = {}
    for bucket in res["aggregations"]["city"]["buckets"]:
        hubs = {}
        for hb in bucket["h"]["buckets"]:
            t = tones(hb["t"]["buckets"])
            hubs[hb["key"]] = {"total": hb["doc_count"], "negative": t.get(-1, 0),
                               "neutral": t.get(0, 0), "positive": t.get(1, 0)}
        cities[bucket["key"]] = hubs
    data = {"built_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "by_hubtype": by_hub, "cities": cities}
    with io.open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False)
    print("каналы:", json.dumps({k: {"total": v["total"], "negative": v["negative"], "share": v["share"]}
                                 for k, v in by_hub.items()}, ensure_ascii=False, indent=1))
    print("городов: %d" % len(cities))
    for city in ("Новосибирск", "Москва", "Шушары", "Кольцово", "Санкт-Петербург"):
        hubs = cities.get(city) or {}
        total = sum(v["total"] for v in hubs.values())
        neg = sum(v["negative"] for v in hubs.values())
        expected = sum(v["total"] * by_hub.get(k, {}).get("share", 0) for k, v in hubs.items())
        print("  %-18s всего %7d негатив %6d (%.1f%%) ожидалось %8.0f → перевес x%.2f" % (
            city, total, neg, 100.0 * neg / max(1, total), expected, neg / max(1.0, expected)))


if __name__ == "__main__":
    main()
