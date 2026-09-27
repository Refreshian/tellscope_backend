#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Цепочки распространения: как инфоповод расходится по источникам и площадкам.

Для крупнейших поводов считается: когда началось, кто опубликовал первым, через сколько
дней наступил пик, какие площадки разгоняли волну, сколько сообщений и охвата она собрала.
Это и есть «цепочка»: от первого сообщения к пересказам и обсуждениям.

Результат: /tmp/kfc_topics/propagation.json + /tmp/kfc_topics/propagation_*.png
"""
from __future__ import annotations

import collections
import datetime
import io
import json
import os
import time

from elasticsearch import Elasticsearch

INDEX = "kfc_13.05.2024-22.09.2026"
OUT = "/tmp/kfc_topics"
MSK = datetime.timezone(datetime.timedelta(hours=3))
LO_TS = int(datetime.datetime(2024, 5, 13, tzinfo=MSK).timestamp())
HI_TS = int(datetime.datetime(2026, 8, 31, 23, 59, 59, tzinfo=MSK).timestamp())
TOP_STORIES = 6


def log(message: str) -> None:
    print("[%s] %s" % (datetime.datetime.now().strftime("%H:%M:%S"), message), flush=True)


def day(ts) -> str:
    try:
        return datetime.datetime.fromtimestamp(int(float(ts)), MSK).strftime("%Y-%m-%d")
    except Exception:  # noqa: BLE001
        return ""


def main() -> None:
    path = os.path.join(OUT, "propagation.json")
    if os.path.isfile(path):
        log("уже посчитано")
        return
    es = Elasticsearch(hosts=["http://localhost:9200"], basic_auth=("elastic", "biz8z5i1w0nLPmEweKgP"),
                       verify_certs=False, request_timeout=600)
    base = {"bool": {"must": [{"term": {"toneMark": -1}}],
                     "filter": [{"range": {"timeCreate": {"gte": LO_TS, "lte": HI_TS}}}]}}
    stories = json.load(io.open("/tmp/kfc_hotspots/stories.json", encoding="utf-8"))
    top = sorted(stories.items(), key=lambda kv: -kv[1]["negative"])[:TOP_STORIES]

    result = []
    for story, info in top:
        query = {"bool": {"must": [{"term": {"toneMark": -1}}, {"term": {"story.keyword": story}}],
                          "filter": [{"range": {"timeCreate": {"gte": LO_TS, "lte": HI_TS}}}]}}
        rows = []
        started = time.time()
        resp = es.search(index=INDEX, body={"size": 2000, "query": query, "sort": ["_doc"],
                                            "_source": ["timeCreate", "hub", "hubtype", "url", "text",
                                                        "audienceCount", "city", "authorObject"]},
                         scroll="10m")
        sid = resp["_scroll_id"]
        try:
            while True:
                hits = resp["hits"]["hits"]
                if not hits:
                    break
                for hit in hits:
                    src = hit["_source"]
                    rows.append({
                        "day": day(src.get("timeCreate")), "hub": src.get("hub") or "",
                        "hubtype": (src.get("hubtype") or "").strip(),
                        "reach": int(src.get("audienceCount") or 0), "url": src.get("url") or "",
                        "text": (src.get("text") or "").strip().replace("\n", " ")[:300],
                        "city": (src.get("city") or "").strip(),
                    })
                resp = es.scroll(scroll_id=sid, scroll="10m")
                sid = resp["_scroll_id"]
        finally:
            try:
                es.clear_scroll(scroll_id=sid)
            except Exception:  # noqa: BLE001
                pass
        by_day = collections.Counter()
        by_hub = collections.Counter()
        reach_by_hub = collections.Counter()
        for row in rows:
            if row["day"]:
                by_day[row["day"]] += 1
            if row["hub"]:
                by_hub[row["hub"]] += 1
                reach_by_hub[row["hub"]] += row["reach"]
        days = sorted(by_day)
        peak = max(by_day.items(), key=lambda kv: kv[1]) if by_day else ("", 0)
        first_rows = sorted(rows, key=lambda row: row["day"])[:3]
        hub_days = collections.defaultdict(list)
        for row in rows:
            if row["hub"] and row["day"]:
                hub_days[row["hub"]].append(row["day"])
        hub_first = {hub: min(values) for hub, values in hub_days.items() if len(values) >= 3}
        top_hubs = [hub for hub, _ in by_hub.most_common(6)]
        timeline = {hub: {day_key: 0 for day_key in days} for hub in top_hubs}
        for row in rows:
            if row["hub"] in timeline and row["day"]:
                timeline[row["hub"]][row["day"]] += 1
        days_to_peak = 0
        if days and peak[0]:
            days_to_peak = (datetime.datetime.strptime(peak[0], "%Y-%m-%d")
                            - datetime.datetime.strptime(days[0], "%Y-%m-%d")).days
        result.append({
            "story": story, "messages": len(rows), "reach": sum(row["reach"] for row in rows),
            "days": days, "by_day": dict(by_day), "first_day": days[0] if days else "",
            "last_day": days[-1] if days else "", "peak_day": peak[0], "peak_messages": peak[1],
            "days_to_peak": days_to_peak,
            "hubs": by_hub.most_common(10),
            "hub_reach": reach_by_hub.most_common(10),
            "hub_first": dict(sorted(hub_first.items(), key=lambda kv: kv[1])[:8]),
            "timeline": timeline,
            "cities": collections.Counter(row["city"] for row in rows if row["city"]).most_common(6),
            "examples": [{"text": row["text"], "hub": row["hub"], "url": row["url"]}
                         for row in first_rows],
        })
        log("повод «%s»: %d сообщений, пик %s (+%d дней), %.0f с"
            % (story[:50], len(rows), peak[0], days_to_peak, time.time() - started))

    with io.open(path, "w", encoding="utf-8") as fh:
        json.dump({"stories": result,
                   "built_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}, fh,
                  ensure_ascii=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for story in result[:3]:
        days = story["days"]
        if len(days) < 3:
            continue
        fig, ax = plt.subplots(figsize=(10.5, 4.6), dpi=170)
        bottom = [0] * len(days)
        for hub, values in story["timeline"].items():
            series = [values.get(day_key, 0) for day_key in days]
            ax.bar(range(len(days)), series, bottom=bottom, label=hub, width=0.85)
            bottom = [left + right for left, right in zip(bottom, series)]
        ax.set_xticks(range(0, len(days), max(1, len(days) // 12)))
        ax.set_xticklabels([days[index] for index in range(0, len(days), max(1, len(days) // 12))],
                           rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("негативных сообщений")
        ax.set_title("Как расходилась волна: %s" % story["story"][:70], fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        name = "propagation_%d.png" % (result.index(story) + 1)
        fig.savefig(os.path.join(OUT, name), dpi=170)
        plt.close(fig)
        log("график цепочки: %s" % name)
    log("готово")


if __name__ == "__main__":
    main()
