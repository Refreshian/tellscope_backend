#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Пакет фактов для управленческой части отчёта: только числа, без формулировок.

Собирает из агрегатов всё, на что опираются инсайты, карта очагов и цена бездействия.
Формулировки пишут модели отдельно и не имеют права добавлять числа.

Результат: /tmp/kfc_hotspots/facts.json
"""
from __future__ import annotations

import datetime
import io
import json
import os

CACHE = "/tmp/kfc_hotspots"
MONTHS = (["2024-%02d" % m for m in range(5, 13)]
          + ["2025-%02d" % m for m in range(1, 13)]
          + ["2026-%02d" % m for m in range(1, 9)])
YEAR_MONTHS = {2024: [m for m in MONTHS if m.startswith("2024")],
               2025: [m for m in MONTHS if m.startswith("2025")],
               2026: [m for m in MONTHS if m.startswith("2026")]}


def load(name: str):
    path = os.path.join(CACHE, name + ".json")
    if not os.path.isfile(path):
        return {}
    with io.open(path, encoding="utf-8") as fh:
        return json.load(fh)


def main() -> None:
    baseline = load("baseline")
    ranked = load("ranked")
    norm = load("channel_norm")
    full = load("scan_all")
    llm = load("llm_hotspots")
    months = baseline.get("months") or {}

    years = {}
    for year, keys in YEAR_MONTHS.items():
        total = sum(months[k]["total"] for k in keys if k in months)
        negative = sum(months[k]["negative"] for k in keys if k in months)
        positive = sum(months[k]["positive"] for k in keys if k in months)
        years[str(year)] = {"total": total, "negative": negative, "positive": positive,
                            "negative_share": round(negative / float(total or 1), 4),
                            "positive_share": round(positive / float(total or 1), 4)}

    ratings = {}
    for key, count in (full.get("rating_month") or {}).items():
        rating, _month = key.split("|") if "|" in key else (key, "")
        if rating.strip():
            ratings[rating] = ratings.get(rating, 0) + int(count)

    hubtype = {k: {"total": v["total"], "negative": v["negative"], "share": v["share"]}
               for k, v in (norm.get("by_hubtype") or {}).items()}

    cities = []
    for row in (ranked.get("city_ranked") or [])[:15]:
        if not row.get("city", "").strip():
            continue
        entry = {
            "city": row["city"], "index": row["index"], "negative": row["negative"],
            "total": row["total"], "share": round(row["share"], 4),
            "expected": row.get("expected"), "excess_norm": row.get("excess_norm"),
            "growth": row["growth"], "reach": row["reach"], "peak_month": row.get("peak_month"),
            "top_themes": [list(t) for t in (row.get("top_themes") or [])[:4]],
            "top_drivers": [list(t) for t in (row.get("top_drivers") or [])[:4]],
            "top_restaurants": [{"id": r["id"], "negative": r["neg"], "count": r["count"],
                                 "share": round(r["negative_share"], 4), "rating": r.get("avg_rating"),
                                 "growth": r.get("growth")}
                                for r in (row.get("top_restaurants") or [])[:3]],
        }
        verdict = ((llm.get("cities") or {}).get(row["city"]) or {})
        if verdict.get("reading"):
            entry["what_people_say"] = verdict["reading"].get("complaints") or []
            entry["causes"] = verdict["reading"].get("causes") or []
        if (verdict.get("verdict") or {}).get("meaning"):
            entry["meaning"] = verdict["verdict"]["meaning"]
            entry["actions"] = verdict["verdict"].get("actions") or []
            entry["if_inaction"] = verdict["verdict"].get("if_inaction") or ""
        cities.append(entry)

    restaurants = []
    for row in (ranked.get("restaurants_ranked") or [])[:30]:
        verdict = ((llm.get("restaurants") or {}).get(row["id"]) or {})
        entry = {"id": row["id"], "city": row["city"], "negative": row["negative"],
                 "count": row["count"], "share": round(row["share"], 4),
                 "rating": row.get("avg_rating"), "growth": row["growth"],
                 "reach": row["reach"], "peak_month": row.get("peak_month"),
                 "place": row.get("place")}
        if verdict.get("reading"):
            entry["what_people_say"] = (verdict["reading"].get("complaints") or [])[:4]
        if (verdict.get("verdict") or {}).get("meaning"):
            entry["meaning"] = verdict["verdict"]["meaning"]
            entry["actions"] = (verdict["verdict"].get("actions") or [])[:2]
        restaurants.append(entry)

    facts = {
        "built_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "period": {"from": "2024-05-13", "to": "2026-08-31", "months": len(MONTHS)},
        "totals": {"messages": baseline.get("total"), "negative": sum(v["negative"] for v in months.values()),
                   "positive": sum(v["positive"] for v in months.values()),
                   "negative_share": round(sum(v["negative"] for v in months.values())
                                           / float(baseline.get("total") or 1), 4),
                   "negative_reach": baseline.get("negative_reach"),
                   "negative_media_reach": baseline.get("negative_media")},
        "years": years,
        "hubtype": hubtype,
        "ratings": ratings,
        "concentration": ranked.get("concentration") or {},
        "scenarios": ranked.get("scenarios") or {},
        "cities": cities,
        "restaurants": restaurants,
        "drivers": [{"name": d["name"], "negative": d["negative"], "growth": d["growth"],
                     "last_q": d["last_q"], "prev_q": d["prev_q"], "reach": d["reach"],
                     "hubs": d.get("hubs") or []} for d in (ranked.get("drivers_ranked") or [])],
        "themes": [{"name": t["name"], "negative": t["negative"], "reach": t["reach"],
                    "cities": (t.get("cities") or [])[:4], "hubs": (t.get("hubs") or [])[:3]}
                   for t in (ranked.get("themes_ranked") or [])[:12]],
        "stories": [{"name": s["name"], "negative": s["negative"], "reach": s["reach"],
                     "media": s["media"], "peak_month": s.get("peak_month"),
                     "cities": (s.get("cities") or [])[:4]}
                    for s in (ranked.get("stories_ranked") or [])[:15]],
    }
    with io.open(os.path.join(CACHE, "facts.json"), "w", encoding="utf-8") as fh:
        json.dump(facts, fh, ensure_ascii=False)

    print("период: %s — %s" % (facts["period"]["from"], facts["period"]["to"]))
    print("сообщений %d, негатив %d (%.2f%%)" % (facts["totals"]["messages"], facts["totals"]["negative"],
                                                 facts["totals"]["negative_share"] * 100))
    print("годы:", json.dumps(years, ensure_ascii=False))
    print("оценки:", json.dumps(ratings, ensure_ascii=False))
    print("каналы:", json.dumps({k: v["negative"] for k, v in hubtype.items()}, ensure_ascii=False))
    print("городов в пакете: %d (из них с разбором моделями: %d)"
          % (len(cities), sum(1 for c in cities if c.get("meaning"))))
    print("заведений в пакете: %d (с разбором: %d)"
          % (len(restaurants), sum(1 for r in restaurants if r.get("meaning"))))
    print("жалобы с ростом:", json.dumps([(d["name"], d["negative"], d["growth"])
                                          for d in facts["drivers"] if d["growth"] > 1.1],
                                         ensure_ascii=False))


if __name__ == "__main__":
    main()
