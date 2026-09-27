#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ранжирование очагов напряжения и расчёт «цены бездействия» по KFC 2024–2026.

Читает агрегаты из /tmp/kfc_hotspots и собирает:

  * рейтинг очагов по городам: объём негатива, доля негатива против среднего по сети,
    рост за последний квартал, вес охвата, главные темы и жалобы, примеры сообщений;
  * рейтинг заведений: адресный список точек, где негатив копится;
  * рейтинг поводов и групп жалоб: что разгоняет негатив и где;
  * концентрацию негатива: какие очаги дают основную часть веса;
  * сценарии на два квартала: тренд + сезонность, три варианта развития.

Результат: /tmp/kfc_hotspots/ranked.json
Запуск:  venv_py312_clean/bin/python kfc_hotspots_rank.py
"""
from __future__ import annotations

import datetime
import io
import json
import math
import os

CACHE = "/tmp/kfc_hotspots"
NATIONAL_SHARE = None  # считается из базы

MONTHS = (["2024-%02d" % m for m in range(5, 13)]
          + ["2025-%02d" % m for m in range(1, 13)]
          + ["2026-%02d" % m for m in range(1, 9)])
LAST_Q = MONTHS[-3:]
PREV_Q = MONTHS[-6:-3]
MONTH_NAMES = ["январь", "февраль", "март", "апрель", "май", "июнь", "июль",
               "август", "сентябрь", "октябрь", "ноябрь", "декабрь"]


def load(name: str):
    path = os.path.join(CACHE, name + ".json")
    if not os.path.isfile(path):
        return {}
    with io.open(path, encoding="utf-8") as fh:
        return json.load(fh)


def save(name: str, value) -> None:
    with io.open(os.path.join(CACHE, name + ".json"), "w", encoding="utf-8") as fh:
        json.dump(value, fh, ensure_ascii=False)


def quarter(values: dict, months=None) -> int:
    """Сумма по указанным месяцам: без списка — по всем."""
    if months is None:
        return sum(int(v or 0) for v in values.values())
    return sum(int(values.get(month) or 0) for month in months)


def month_title(key: str) -> str:
    if not key or len(key) < 7:
        return key
    return "%s %s" % (MONTH_NAMES[int(key[5:7]) - 1], key[:4])


def main() -> None:
    baseline = load("baseline")
    scan = load("negatives_scan")
    full = load("scan_all")
    themes = load("themes_geo")
    storms = load("stories")
    groups = load("drivers")
    norm = load("channel_norm")
    hub_share = {k: v.get("share", 0) for k, v in (norm.get("by_hubtype") or {}).items()}
    city_hubs = norm.get("cities") or {}

    total = baseline["total"]
    national_share = baseline["months"]
    neg_total_all = sum(v["negative"] for v in national_share.values()) or 1
    nat_share = neg_total_all / float(total)

    # --- негатив и рост по городам
    city_neg = {}
    reach_by_city = scan.get("city_reach") or {}
    media_by_city = scan.get("city_media") or {}
    for key, count in (scan.get("city_month") or {}).items():
        city, month = key.split("|") if "|" in key else (key, "")
        row = city_neg.setdefault(city, {"months": {}, "negative": 0, "reach": 0})
        row["months"][month] = row["months"].get(month, 0) + int(count)
        row["negative"] += int(count)
    for city, row in city_neg.items():
        row["city"] = city
        row["reach"] = int(reach_by_city.get(city) or 0)
        row["media"] = int(media_by_city.get(city) or 0)
        base = (baseline.get("cities") or {}).get(city) or {}
        row["total"] = int(base.get("total") or 0)
        row["share"] = (row["negative"] / row["total"]) if row["total"] else 0.0
        row["last_q"] = quarter(row["months"], LAST_Q)
        row["prev_q"] = quarter(row["months"], PREV_Q)
        row["growth"] = ((row["last_q"] + 1) / float(row["prev_q"] + 1)) if row["prev_q"] else 1.0
        row["excess"] = (row["share"] / nat_share) if nat_share else 0.0
        # Нормировка по каналам: сколько негатива ждали при таком же наборе площадок.
        hubs = city_hubs.get(city) or {}
        expected = sum((h.get("total") or 0) * hub_share.get(kind, 0) for kind, h in hubs.items())
        row["expected"] = round(expected)
        row["excess_norm"] = round(row["negative"] / expected, 2) if expected else 0.0
        reviews = hubs.get("Отзывы") or {}
        row["reviews_total"] = reviews.get("total") or 0
        row["reviews_negative"] = reviews.get("negative") or 0
        row["reviews_share"] = (round(row["reviews_negative"] / float(row["reviews_total"]), 4)
                                if row["reviews_total"] else 0.0)

    # --- профиль города: темы, жалобы, заведения
    theme_by_city = {}
    for theme, data in themes.items():
        for city, count in (data.get("cities") or {}).items():
            theme_by_city.setdefault(city, []).append((theme, int(count)))
    driver_by_city = {}
    for name, data in groups.items():
        for city, count in (data.get("cities") or {}).items():
            driver_by_city.setdefault(city, []).append((name, int(count)))
    restaurant_by_city = {}
    for row in full.get("restaurants") or []:
        if row.get("city"):
            restaurant_by_city.setdefault(row["city"], []).append(row)

    scan_rest = {r["id"]: r for r in (scan.get("restaurants") or [])}

    reach_max = max([r["reach"] for r in city_neg.values()] or [1])
    neg_max = max([r["negative"] for r in city_neg.values()] or [1])

    for city, row in city_neg.items():
        if not city.strip():
            continue
        row["top_themes"] = sorted(theme_by_city.get(city, []), key=lambda kv: -kv[1])[:5]
        row["top_drivers"] = sorted(driver_by_city.get(city, []), key=lambda kv: -kv[1])[:5]
        row["top_restaurants"] = sorted(restaurant_by_city.get(city, []),
                                        key=lambda r: -r["neg"])[:5]
        row["peak_month"] = max(row["months"].items(), key=lambda kv: kv[1])[0] if row["months"] else ""
        # индекс напряжения: объём × превышение над средним × рост × вес охвата
        volume = math.sqrt(row["negative"] / float(neg_max))
        growth = min(3.0, max(0.5, row["growth"]))
        reach = 1.0 + math.log10(1 + row["reach"]) / 12.0
        row["index"] = round(100.0 * volume * max(0.3, row["excess_norm"] or row["excess"])
                             * growth * reach, 2)

    cities_ranked = sorted((r for c, r in city_neg.items() if c.strip()),
                           key=lambda r: -r["index"])
    for place, row in enumerate(cities_ranked, 1):
        row["place"] = place

    # --- заведения: где копится негатив
    restaurants = []
    for row in full.get("restaurants") or []:
        if not row.get("count") or row["count"] < 30:
            continue
        neg = row["neg"]
        share = row["negative_share"]
        avg = row.get("avg_rating")
        months = row.get("months") or {}
        last_q = sum(int(months.get(m) or 0) for m in LAST_Q)
        prev_q = sum(int(months.get(m) or 0) for m in PREV_Q)
        growth = ((last_q + 1) / float(prev_q + 1)) if prev_q else 1.0
        severity = (5.0 - avg) / 4.0 if avg else 0.5
        score = (math.sqrt(neg) * max(0.3, share / nat_share) * min(2.5, max(0.5, growth))
                 * (0.5 + severity))
        sample = scan_rest.get(row["id"]) or {}
        restaurants.append({
            "id": row["id"], "city": row["city"], "hub": row.get("hub") or "",
            "count": row["count"], "negative": neg, "positive": row["pos"], "neutral": row["neu"],
            "share": round(share, 4), "avg_rating": avg, "votes": row.get("votes") or 0,
            "reach": row["reach"], "growth": round(growth, 2), "last_q": last_q, "prev_q": prev_q,
            "score": round(score, 2), "months": months,
            "samples": (sample.get("samples") or [])[:4],
            "peak_month": max(months.items(), key=lambda kv: kv[1])[0] if months else "",
        })
    restaurants.sort(key=lambda r: -r["score"])
    for place, row in enumerate(restaurants, 1):
        row["place"] = place

    # --- поводы и жалобы
    stories_ranked = sorted(
        ({"name": name, "negative": d["negative"], "reach": d.get("reach") or 0,
          "media": d.get("media") or 0, "months": d.get("months") or {},
          "cities": list((d.get("cities") or {}).items())[:5],
          "peak_month": max((d.get("months") or {"": 0}).items(), key=lambda kv: kv[1])[0]}
         for name, d in storms.items() if name.strip()),
        key=lambda r: -r["negative"])[:40]
    drivers_ranked = sorted(
        ({"name": name, "negative": d["negative"], "reach": d.get("reach") or 0,
          "months": d.get("months") or {},
          "cities": list((d.get("cities") or {}).items())[:8],
          "hubs": list((d.get("hubs") or {}).items())[:5],
          "last_q": quarter(d.get("months") or {}, LAST_Q),
          "prev_q": quarter(d.get("months") or {}, PREV_Q)}
         for name, d in groups.items()),
        key=lambda r: -r["negative"])
    for row in drivers_ranked:
        row["growth"] = round(((row["last_q"] + 1) / float(row["prev_q"] + 1)) if row["prev_q"] else 1.0, 2)

    themes_ranked = sorted(
        ({"name": name, "negative": d["negative"], "reach": d.get("reach") or 0,
          "months": d.get("months") or {}, "cities": list((d.get("cities") or {}).items())[:8],
          "hubs": list((d.get("hubs") or {}).items())[:5]}
         for name, d in themes.items()),
        key=lambda r: -r["negative"])

    # --- концентрация негатива: сколько веса дают немногие
    neg_reach = baseline.get("negative_reach") or 0
    neg_media = baseline.get("negative_media") or 0
    top10_cities = sum(r["negative"] for r in cities_ranked[:10])
    top30_rest = sum(r["negative"] for r in restaurants[:30])
    top50_rest = sum(r["negative"] for r in restaurants[:50])
    top5_themes = sum(r["negative"] for r in themes_ranked[:5])
    theme_total = sum(r["negative"] for r in themes_ranked) or 1
    concentration = {
        "negative_total": neg_total_all,
        "negative_reach": neg_reach,
        "negative_media": neg_media,
        "top10_cities": top10_cities,
        "top10_cities_share": round(top10_cities / float(neg_total_all), 4),
        "top30_restaurants": top30_rest,
        "top30_restaurants_share": round(top30_rest / float(neg_total_all), 4),
        "top50_restaurants": top50_rest,
        "top5_themes": top5_themes,
        "top5_themes_share": round(top5_themes / float(theme_total), 4),
        "cities_with_growth": sum(1 for r in cities_ranked if r["growth"] > 1.15 and r["negative"] > 500),
        "restaurants_with_growth": sum(1 for r in restaurants if r["growth"] > 1.3 and r["negative"] > 100),
    }

    # --- сценарии: тренд + сезонность по 28 месяцам
    series = [(month, national_share[month]["negative"] / float(national_share[month]["total"]))
              for month in MONTHS if month in national_share and national_share[month]["total"]]
    shares = [value for _, value in series]
    mean_share = sum(shares) / len(shares)
    xs = list(range(len(shares)))
    mx = sum(xs) / len(xs)
    my = mean_share
    slope = (sum((x - mx) * (y - my) for x, y in zip(xs, shares))
             / max(1e-9, sum((x - mx) ** 2 for x in xs)))
    seasonal = {}
    for index, (month, value) in enumerate(series):
        seasonal.setdefault(int(month[5:7]), []).append(value - (my + slope * (index - mx)))
    seasonal_index = {key: sum(values) / len(values) for key, values in seasonal.items()}

    forecast = []
    for step, (year, month) in enumerate([(2026, 9), (2026, 10), (2026, 11), (2026, 12),
                                          (2027, 1), (2027, 2)], start=1):
        trend = my + slope * (len(series) - 1 + step - mx)
        base = max(0.02, trend + seasonal_index.get(month, 0.0))
        forecast.append({"month": "%04d-%02d" % (year, month), "base": round(base, 4),
                         "no_action": round(base * 1.06, 4),
                         "targeted": round(base * 0.93, 4),
                         "systemic": round(base * 0.85, 4)})
    scenarios = {
        "mean_share": round(mean_share, 4),
        "slope_per_month": round(slope, 5),
        "last_months": [(month, round(value, 4)) for month, value in series[-6:]],
        "seasonal": {str(k): round(v, 4) for k, v in sorted(seasonal_index.items())},
        "forecast": forecast,
        "note": ("Расчёт по наблюдённому тренду и средней сезонности за 28 месяцев. "
                 "Вариант «без действий» — продолжение тренда с поправкой на накопление очагов, "
                 "«точечные» — работа с адресными очагами, «системные» — работа с причинами "
                 "по всей сети. Это расчёт по прошлым данным, а не гарантия."),
    }

    result = {
        "built_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "national_share": round(nat_share, 4),
        "city_ranked": cities_ranked[:30],
        "restaurants_ranked": restaurants[:60],
        "stories_ranked": stories_ranked,
        "drivers_ranked": drivers_ranked,
        "themes_ranked": themes_ranked,
        "concentration": concentration,
        "scenarios": scenarios,
    }
    save("ranked", result)

    print("=== очаги по городам (индекс | негатив | доля | перевес с поправкой на каналы | рост | охват):")
    for row in cities_ranked[:15]:
        print("  %2d. %-20s %6.2f | %6d | %5.2f%% | x%.2f (без поправки x%.2f) | рост %.2f | охват %s" % (
            row["place"], row["city"], row["index"], row["negative"], row["share"] * 100,
            row["excess_norm"], row["excess"], row["growth"],
            "{:,}".format(row["reach"]).replace(",", " ")))
    print("=== заведения:")
    for row in restaurants[:15]:
        print("  %2d. %-18s %-16s негатив %4d из %4d (%4.1f%%), рейтинг %s, рост %.2f" % (
            row["place"], row["city"], row["id"], row["negative"], row["count"],
            row["share"] * 100, row["avg_rating"], row["growth"]))
    print("=== концентрация:", json.dumps(concentration, ensure_ascii=False))
    print("=== сценарии:", json.dumps({k: v for k, v in scenarios.items() if k != "seasonal"}, ensure_ascii=False)[:900])
    print("=== группы жалоб с ростом:")
    for row in drivers_ranked:
        print("  %-34s %6d | последний квартал %5d | рост %.2f" % (
            row["name"], row["negative"], row["last_q"], row["growth"]))


if __name__ == "__main__":
    main()
