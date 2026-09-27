#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Агрегаты для блока «Очаги напряжения» межгодового отчёта KFC 2024–2026.

Считает по всему корпусу темы (2,96 млн сообщений) то, чего нет в месячных отчётах:

  * проход по всем негативным сообщениям: месяц × город × площадка × рейтинг × вес охвата;
  * рестораны: сколько негатива у конкретной точки, её рейтинг, охват, примеры;
  * темы (готовые метки Brand Analytics) в разрезе город × месяц;
  * поводы-инциденты с собственными названиями;
  * группы жалоб по формулировкам (цена, скорость, чистота, персонал, еда, доставка и т.д.).

Всё кэшируется в /tmp/kfc_hotspots и переиспользуется сборщиком отчёта.
Запуск:  nohup venv_py312_clean/bin/python -u kfc_hotspots_agg.py > /tmp/kfc_hotspots.log 2>&1 &
"""
from __future__ import annotations

import collections
import datetime
import io
import json
import os
import re
import sys
import time

from elasticsearch import Elasticsearch

INDEX = "kfc_13.05.2024-22.09.2026"
CACHE = "/tmp/kfc_hotspots"
LEGACY_CACHE = "/tmp/kfc_cx_cache"

ES_HOST = "http://localhost:9200"
ES_USER = "elastic"
ES_PASS = "biz8z5i1w0nLPmEweKgP"

MSK = datetime.timezone(datetime.timedelta(hours=3))

# Границы отчёта: тема собирается с 13.05.2024, отчёт считается по 31.08.2026.
LO_DAY = "2024-05-13"
HI_DAY = "2026-08-31"

MONTH_KEYS = (["2024-%02d" % m for m in range(5, 13)]
              + ["2025-%02d" % m for m in range(1, 13)]
              + ["2026-%02d" % m for m in range(1, 9)])

# Группы жалоб: формулировки, по которым видно, на что именно жалуются.
DRIVERS = {
    "Цена и размер порции": ["дорого", "подорожал", "цена выросла", "переплата", "цены"],
    "Скорость и ожидание": ["ждали", "долго ждать", "очередь", "медленно", "задержали заказ"],
    "Чистота зала и туалетов": ["грязно", "грязь", "туалет", "немытые столы"],
    "Грубость персонала": ["грубо", "хамство", "хамит", "нагрубил"],
    "Доставка и курьер": ["курьер", "доставка опоздала", "заказ не привезли"],
    "Холодная еда": ["холодные", "остывший", "остыло", "холодный бургер"],
    "Качество и свежесть продукта": ["сырой", "прожарен", "просрочка", "невкусно"],
    "Отравления и санитария": ["отравился", "отравление", "тошнило", "сальмонелла"],
    "Нехватка курицы и ассортимент": ["нет курицы", "закончилась", "не было в наличии"],
    "Техсбой: кассы и приложение": ["не работает приложение", "касса не работает", "сбой"],
    "Закрытие ресторанов": ["закрыли", "закрытие", "закрылся"],
}


def log(message: str) -> None:
    print("[%s] %s" % (datetime.datetime.now().strftime("%H:%M:%S"), message), flush=True)


def client():
    return Elasticsearch(hosts=[ES_HOST], basic_auth=(ES_USER, ES_PASS), verify_certs=False,
                         headers={"Accept": "application/vnd.elasticsearch+json; compatible-with=9"},
                         request_timeout=600)


def cache_path(name: str) -> str:
    os.makedirs(CACHE, exist_ok=True)
    return os.path.join(CACHE, name + ".json")


def save(name: str, value) -> None:
    with io.open(cache_path(name), "w", encoding="utf-8") as fh:
        json.dump(value, fh, ensure_ascii=False)
    log("сохранено %s (%.1f КБ)" % (name, os.path.getsize(cache_path(name)) / 1024.0))


def load(name: str):
    path = cache_path(name)
    if os.path.isfile(path):
        try:
            return json.load(io.open(path, encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return None
    return None


def day_stamp(day: str, end: bool = False) -> int:
    dt = datetime.datetime.strptime(day, "%Y-%m-%d")
    if end:
        dt = dt.replace(hour=23, minute=59, second=59)
    return int(dt.replace(tzinfo=MSK).timestamp())


LO_TS = day_stamp(LO_DAY)
HI_TS = day_stamp(HI_DAY, end=True)


def _month_start(month: str) -> int:
    year, mon = int(month[:4]), int(month[5:7])
    return int(datetime.datetime(year, mon, 1, tzinfo=MSK).timestamp())


def _month_ranges() -> dict:
    """Границы месяцев в московском времени: field timeCreate хранит секунды, а не дату."""
    out = {}
    keys = MONTH_KEYS
    for index, key in enumerate(keys):
        start = max(_month_start(key), LO_TS)
        nxt = _month_start(keys[index + 1]) if index + 1 < len(keys) else day_stamp("2026-09-01")
        out[key] = {"range": {"timeCreate": {"gte": start, "lt": nxt}}}
    return out


MONTH_RANGES = _month_ranges()


def month_buckets(agg_result: dict) -> dict:
    """doc_count по месяцам из агрегации filters."""
    buckets = ((agg_result or {}).get("buckets") or {})
    out = {}
    for key, value in buckets.items():
        if key in MONTH_RANGES and isinstance(value, dict) and value.get("doc_count"):
            out[key] = value["doc_count"]
    return out


def period_filter():
    return [{"range": {"timeCreate": {"gte": LO_TS, "lte": HI_TS}}}]


def month_of(ts) -> str:
    try:
        dt = datetime.datetime.fromtimestamp(int(float(ts)), MSK)
    except Exception:  # noqa: BLE001
        return ""
    return "%04d-%02d" % (dt.year, dt.month)


def org_id(url: str) -> str:
    """Номер заведения на картах — по нему видно, о каком именно ресторане речь."""
    if not url:
        return ""
    for pattern in (r"/org/[^/]+/(\d+)", r"/firm/(\d+)", r"maps/org/(\d+)", r"/(\d{8,})"):
        hit = re.search(pattern, url)
        if hit:
            return hit.group(1)
    return ""


# --------------------------------------------------------------- проход по негативу

def scroll_negatives(es) -> dict:
    """Один проход по всем негативным сообщениям: город, ресторан, рейтинг, охват, месяц."""
    cached = load("negatives_scan")
    if cached:
        log("проход по негативу уже посчитан: %d сообщений" % cached.get("scanned", 0))
        return cached

    query = {"bool": {"must": [{"term": {"toneMark": -1}}], "filter": period_filter()}}
    source = ["timeCreate", "city", "region", "hub", "hubtype", "type", "url", "review_rating",
              "audienceCount", "massMediaAudience", "likesCount", "commentsCount", "repostsCount",
              "story", "text", "title", "wom"]
    started = time.time()
    scanned = 0
    city_month = collections.Counter()
    city_reach = collections.Counter()
    city_media = collections.Counter()
    city_rating = collections.defaultdict(collections.Counter)
    city_region = {}
    hub_month = collections.Counter()
    type_month = collections.Counter()
    rating_month = collections.Counter()
    story_month = collections.Counter()
    story_city = collections.defaultdict(collections.Counter)
    story_reach = collections.Counter()
    restaurant = {}
    samples = collections.defaultdict(list)

    resp = es.search(index=INDEX, body={"size": 4000, "query": query, "_source": source,
                                        "sort": ["_doc"]}, scroll="10m")
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
                city = (src.get("city") or "").strip()
                hub = src.get("hub") or ""
                story = (src.get("story") or "").strip()
                reach = int(src.get("audienceCount") or 0)
                media = int(src.get("massMediaAudience") or 0)
                rating = str(src.get("review_rating") or "").strip()
                if city:
                    city_month[(city, month)] += 1
                    city_reach[city] += reach
                    city_media[city] += media
                    if rating:
                        city_rating[city][rating] += 1
                    city_region.setdefault(city, (src.get("region") or "").strip())
                hub_month[((src.get("hubtype") or "").strip(), month)] += 1
                type_month[((src.get("type") or "").strip(), month)] += 1
                if rating:
                    rating_month[(rating, month)] += 1
                if story:
                    story_month[(story, month)] += 1
                    story_reach[story] += reach
                    if city:
                        story_city[story][city] += 1
                rid = org_id(src.get("url") or "")
                if rid:
                    row = restaurant.get(rid)
                    if row is None:
                        row = restaurant[rid] = {
                            "id": rid, "hub": hub, "city": city,
                            "region": (src.get("region") or "").strip(),
                            "count": 0, "first": month, "last": month,
                            "reach": 0, "media": 0, "likes": 0,
                            "ratings": collections.Counter(), "months": collections.Counter(),
                            "url": src.get("url") or "",
                        }
                    row["count"] += 1
                    row["months"][month] += 1
                    row["reach"] += reach
                    row["media"] += media
                    row["likes"] += int(src.get("likesCount") or 0)
                    if rating:
                        row["ratings"][rating] += 1
                    if month and (not row["first"] or month < row["first"]):
                        row["first"] = month
                    if month and month > row["last"]:
                        row["last"] = month
                    if len(samples[rid]) < 6:
                        text = (src.get("text") or src.get("title") or "").strip().replace("\n", " ")
                        if len(text) > 60:
                            samples[rid].append({"date": month, "hub": hub, "rating": rating,
                                                 "text": text[:400], "url": src.get("url") or "",
                                                 "reach": reach})
            if scanned % 40000 < 4000:
                log("прошло %d сообщений (%.0f с)" % (scanned, time.time() - started))
            resp = es.scroll(scroll_id=sid, scroll="10m")
            sid = resp["_scroll_id"]
    finally:
        try:
            es.clear_scroll(scroll_id=sid)
        except Exception:  # noqa: BLE001
            pass

    restaurants = []
    for row in restaurant.values():
        row = dict(row)
        row["ratings"] = dict(row["ratings"])
        row["months"] = dict(row["months"])
        row["samples"] = samples.get(row["id"], [])
        row["avg_rating"] = (sum(int(k) * v for k, v in row["ratings"].items() if k.isdigit())
                             / max(1, sum(v for k, v in row["ratings"].items() if k.isdigit())))
        restaurants.append(row)
    restaurants.sort(key=lambda r: -r["count"])

    data = {
        "scanned": scanned,
        "seconds": round(time.time() - started, 1),
        "city_month": {"%s|%s" % k: v for k, v in city_month.items()},
        "city_reach": dict(city_reach),
        "city_media": dict(city_media),
        "city_region": city_region,
        "city_rating": {k: dict(v) for k, v in city_rating.items()},
        "hub_month": {"%s|%s" % k: v for k, v in hub_month.items()},
        "type_month": {"%s|%s" % k: v for k, v in type_month.items()},
        "rating_month": {"%s|%s" % k: v for k, v in rating_month.items()},
        "story_month": {"%s|%s" % k: v for k, v in story_month.items()},
        "story_reach": dict(story_reach),
        "story_city": {k: dict(v) for k, v in story_city.items()},
        "restaurants": restaurants[:800],
    }
    save("negatives_scan", data)
    return data


# ------------------------------------------------------------------ база для сравнения

def baseline(es) -> dict:
    """Сколько всего сообщений и негатива по городам — знаменатель для доли негатива."""
    cached = load("baseline")
    if cached:
        return cached
    body = {
        "size": 0, "track_total_hits": True,
        "query": {"bool": {"filter": period_filter()}},
        "aggs": {
            "city": {"terms": {"field": "city", "size": 300},
                     "aggs": {"t": {"terms": {"field": "toneMark", "size": 3}}}},
            "months": {"filters": {"filters": MONTH_RANGES},
                       "aggs": {"t": {"terms": {"field": "toneMark", "size": 3}}}},
            "reach_neg": {"filter": {"term": {"toneMark": -1}},
                          "aggs": {"a": {"sum": {"field": "audienceCount"}},
                                   "m": {"sum": {"field": "massMediaAudience"}}}},
        },
    }
    res = es.search(index=INDEX, body=body)
    cities = {}
    for bucket in res["aggregations"]["city"]["buckets"]:
        tones = {b["key"]: b["doc_count"] for b in bucket["t"]["buckets"]}
        cities[bucket["key"]] = {"total": bucket["doc_count"],
                                 "negative": tones.get(-1, 0), "neutral": tones.get(0, 0),
                                 "positive": tones.get(1, 0)}
    months = {}
    for key, bucket in (res["aggregations"]["months"]["buckets"] or {}).items():
        if key not in MONTH_RANGES:
            continue
        tones = {b["key"]: b["doc_count"] for b in bucket["t"]["buckets"]}
        months[key] = {"total": bucket["doc_count"], "negative": tones.get(-1, 0),
                       "neutral": tones.get(0, 0), "positive": tones.get(1, 0)}
    data = {"total": res["hits"]["total"]["value"], "cities": cities, "months": months,
            "negative_reach": res["aggregations"]["reach_neg"]["a"]["value"],
            "negative_media": res["aggregations"]["reach_neg"]["m"]["value"]}
    save("baseline", data)
    return data


# ------------------------------------------------------------------------ темы и поводы

def theme_catalog():
    """Готовые темы: имя → пары (поле, номер) из кэша прошлой сборки."""
    for name in ("tag_catalog.json", "theme_stats.json"):
        path = os.path.join(LEGACY_CACHE, name)
        if os.path.isfile(path):
            raw = json.load(io.open(path, encoding="utf-8"))
            if isinstance(raw, dict):
                return {k: [tuple(p) for p in v] for k, v in raw.items()}
            return {row["name"]: [tuple(p) for p in row["pairs"]] for row in raw}
    return {}


def themes_geo(es) -> dict:
    """Тема × месяц × город × площадка — по негативу, плюс охват."""
    cached = load("themes_geo")
    if cached:
        return cached
    catalog = theme_catalog()
    if not catalog:
        log("нет справочника тем — блок тем пропущен")
        return {}
    # берём темы, у которых больше всего негатива, но не больше 40 запросов
    stats = load("theme_negative_stats") or {}
    if not stats:
        stats = {}
        for name, pairs in list(catalog.items()):
            should = [{"exists": {"field": "%s.%s" % (field, tid)}} for field, tid in pairs]
            q = {"bool": {"should": should, "minimum_should_match": 1,
                          "filter": [{"term": {"toneMark": -1}}] + period_filter()}}
            try:
                res = es.search(index=INDEX, size=0, track_total_hits=True, query=q)
                stats[name] = res["hits"]["total"]["value"]
            except Exception as exc:  # noqa: BLE001
                log("тема %s: ошибка %s" % (name, str(exc)[:80]))
                stats[name] = 0
        save("theme_negative_stats", stats)
    top = sorted(stats.items(), key=lambda kv: -kv[1])[:40]
    out = {}
    for name, count in top:
        if not count:
            continue
        pairs = catalog.get(name) or []
        should = [{"exists": {"field": "%s.%s" % (field, tid)}} for field, tid in pairs]
        q = {"bool": {"should": should, "minimum_should_match": 1,
                      "filter": [{"term": {"toneMark": -1}}] + period_filter()}}
        body = {"size": 0, "query": q, "aggs": {
            "m": {"filters": {"filters": MONTH_RANGES}},
            "c": {"terms": {"field": "city", "size": 30}},
            "h": {"terms": {"field": "hubtype.keyword", "size": 8}},
            "reach": {"sum": {"field": "audienceCount"}},
        }}
        try:
            res = es.search(index=INDEX, body=body)
        except Exception as exc:  # noqa: BLE001
            log("тема %s: пропуск (%s)" % (name, str(exc)[:60]))
            continue
        months = month_buckets(res["aggregations"]["m"])
        cities = {b["key"]: b["doc_count"] for b in res["aggregations"]["c"]["buckets"]}
        hubs = {b["key"]: b["doc_count"] for b in res["aggregations"]["h"]["buckets"]}
        out[name] = {"negative": count, "months": months, "cities": cities, "hubs": hubs,
                     "reach": res["aggregations"]["reach"]["value"]}
        log("тема: %s — негатива %d" % (name, count))
    save("themes_geo", out)
    return out


def stories(es) -> dict:
    """Поводы с собственными названиями: объём, месяцы, города, охват."""
    cached = load("stories")
    if cached:
        return cached
    q = {"bool": {"must": [{"term": {"toneMark": -1}}], "filter": period_filter()}}
    body = {"size": 0, "query": q, "aggs": {"s": {"terms": {"field": "story.keyword", "size": 250}}}}
    res = es.search(index=INDEX, body=body)
    out = {}
    for bucket in res["aggregations"]["s"]["buckets"]:
        story = (bucket["key"] or "").strip()
        if not story:
            continue
        q2 = {"bool": {"must": [{"term": {"toneMark": -1}}, {"term": {"story.keyword": story}}],
                       "filter": period_filter()}}
        body2 = {"size": 0, "query": q2, "aggs": {
            "m": {"filters": {"filters": MONTH_RANGES}},
            "c": {"terms": {"field": "city", "size": 15}},
            "h": {"terms": {"field": "hub.keyword", "size": 10}},
            "reach": {"sum": {"field": "audienceCount"}},
            "media": {"sum": {"field": "massMediaAudience"}},
        }}
        try:
            res2 = es.search(index=INDEX, body=body2)
        except Exception:  # noqa: BLE001
            continue
        out[story] = {
            "negative": bucket["doc_count"],
            "months": month_buckets(res2["aggregations"]["m"]),
            "cities": {b["key"]: b["doc_count"] for b in res2["aggregations"]["c"]["buckets"]},
            "hubs": {b["key"]: b["doc_count"] for b in res2["aggregations"]["h"]["buckets"]},
            "reach": res2["aggregations"]["reach"]["value"],
            "media": res2["aggregations"]["media"]["value"],
        }
    save("stories", out)
    return out


def drivers(es) -> dict:
    """Группы жалоб по формулировкам: месяц, город, площадка, охват."""
    cached = load("drivers")
    if cached:
        return cached
    out = {}
    for name, terms in DRIVERS.items():
        should = [{"match_phrase": {"text": term}} for term in terms]
        q = {"bool": {"should": should, "minimum_should_match": 1,
                      "filter": [{"term": {"toneMark": -1}}] + period_filter()}}
        body = {"size": 0, "track_total_hits": True, "query": q, "aggs": {
            "m": {"filters": {"filters": MONTH_RANGES}},
            "c": {"terms": {"field": "city", "size": 25}},
            "h": {"terms": {"field": "hubtype.keyword", "size": 8}},
            "reach": {"sum": {"field": "audienceCount"}},
        }}
        try:
            res = es.search(index=INDEX, body=body)
        except Exception as exc:  # noqa: BLE001
            log("группа %s: ошибка %s" % (name, str(exc)[:80]))
            continue
        out[name] = {
            "terms": terms,
            "negative": res["hits"]["total"]["value"],
            "months": month_buckets(res["aggregations"]["m"]),
            "cities": {b["key"]: b["doc_count"] for b in res["aggregations"]["c"]["buckets"]},
            "hubs": {b["key"]: b["doc_count"] for b in res["aggregations"]["h"]["buckets"]},
            "reach": res["aggregations"]["reach"]["value"],
        }
        log("группа: %s — негатива %d" % (name, out[name]["negative"]))
    save("drivers", out)
    return out


def main() -> None:
    started = time.time()
    es = client()
    log("индекс: %s, всего сообщений %d" % (INDEX, es.count(index=INDEX)["count"]))
    base = baseline(es)
    log("база: %d сообщений за период, городов %d" % (base["total"], len(base["cities"])))
    scan = scroll_negatives(es)
    log("негатив: %d сообщений, ресторанов %d" % (scan["scanned"], len(scan["restaurants"])))
    themes = themes_geo(es)
    log("тем разобрано: %d" % len(themes))
    storms = stories(es)
    log("поводов с названиями: %d" % len(storms))
    groups = drivers(es)
    log("групп жалоб: %d" % len(groups))
    summary = {
        "built_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "seconds": round(time.time() - started, 1),
        "totals": {"messages": base["total"], "negative": scan["scanned"],
                   "negative_reach": base["negative_reach"],
                   "negative_media": base["negative_media"],
                   "cities": len(base["cities"]), "restaurants": len(scan["restaurants"]),
                   "themes": len(themes), "stories": len(storms)},
        "top_cities": sorted(((c, v["negative"]) for c, v in base["cities"].items()),
                             key=lambda kv: -kv[1])[:20],
        "top_restaurants": [(r["city"], r["id"], r["count"], round(r["avg_rating"], 2))
                            for r in scan["restaurants"][:20]],
        "top_stories": sorted(((k, v["negative"]) for k, v in storms.items()),
                              key=lambda kv: -kv[1])[:15],
        "top_drivers": sorted(((k, v["negative"]) for k, v in groups.items()),
                              key=lambda kv: -kv[1]),
    }
    save("summary", summary)
    log("готово за %.0f с" % (time.time() - started))
    print(json.dumps(summary, ensure_ascii=False, indent=1)[:4000], flush=True)


if __name__ == "__main__":
    main()
