from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime

from .catalog import Catalog
from .models import Message, ObjectStats, ReportBundle


def build_report(
    messages: list[Message],
    catalog: Catalog,
    ingest_meta: dict,
    period_label: str = "",
) -> ReportBundle:
    vendor = ingest_meta.get("vendor_stats") or []
    vendor_by_id: dict[str, dict] = {}
    for row in vendor:
        hits = catalog.match_objects(row["name"])
        if hits:
            vendor_by_id[hits[0]] = row

    grouped: dict[str, list[Message]] = defaultdict(list)
    for msg in messages:
        for oid in msg.object_ids:
            grouped[oid].append(msg)

    stats: list[ObjectStats] = []
    for oid, items in grouped.items():
        st = ObjectStats(object_id=oid, messages=len(items))
        media = Counter(m.source for m in items)
        st.media = dict(media)
        st.unique_media = len(media)
        top2 = sum(c for _, c in media.most_common(2))
        st.top2_share = top2 / len(items) if items else 0.0
        speakers: Counter[str] = Counter()
        daily: Counter[str] = Counter()
        for msg in items:
            speakers.update(msg.speaker_ids)
            if msg.published_at:
                daily[msg.published_at.strftime("%Y-%m-%d")] += 1
            role = msg.role_by_object.get(oid) or "episodic"
            if role == "main":
                st.main_role += 1
            elif role == "background":
                st.background_role += 1
            else:
                st.episodic_role += 1
            sent = msg.sentiment or "neutral"
            if sent == "positive":
                st.positive += 1
            elif sent == "negative":
                st.negative += 1
            else:
                st.neutral += 1
            if msg.initiated is not None:
                st.initiated_known += 1
                if msg.initiated:
                    st.initiated += 1
        st.speakers = dict(speakers)
        st.daily = dict(daily)
        vrow = vendor_by_id.get(oid)
        if vrow:
            st.from_vendor = True
            st.messages = int(vrow["messages"] or st.messages)
            st.main_role = int(vrow["main_role"] or st.main_role)
            st.media_index = vrow.get("media_index")
            st.reach = vrow.get("reach")
            if vrow.get("positive") is not None:
                st.positive = int(vrow["positive"])
            if vrow.get("negative") is not None:
                st.negative = int(vrow["negative"])
            rest = max(st.messages - st.positive - st.negative, 0)
            st.neutral = rest
        stats.append(st)

    stats.sort(key=lambda s: s.messages, reverse=True)

    missing: list[str] = []
    if not any(s.reach for s in stats):
        missing.append("охват (нет Excel Медиалогии)")
    if not any(s.media_index for s in stats):
        missing.append("МедиаИндекс (нет Excel Медиалогии)")
    if all(s.negative == 0 and s.positive == 0 for s in stats):
        missing.append("тональность Медиалогии — считаем эвристикой/Tellscope")

    start = ingest_meta.get("period_start")
    end = ingest_meta.get("period_end")
    if not period_label and start and end:
        period_label = f"{start:%d.%m.%Y} — {end:%d.%m.%Y}"

    vendor_totals = {oid: int(row["messages"] or 0) for oid, row in vendor_by_id.items()}
    return ReportBundle(
        period_label=period_label,
        period_start=start if isinstance(start, datetime) else None,
        period_end=end if isinstance(end, datetime) else None,
        messages=messages,
        object_stats=stats,
        vendor_totals=vendor_totals,
        missing_metrics=missing,
        notes=list(ingest_meta.get("notes") or []),
    )


def cooccurrence_media(stats: list[ObjectStats], limit: int = 12) -> list[list[int]]:
    top = stats[:limit]
    sets = [{name for name, _ in Counter(s.media).most_common()} for s in top]
    matrix: list[list[int]] = []
    for i, a in enumerate(sets):
        row = []
        for j, b in enumerate(sets):
            row.append(len(a & b) if i != j else 0)
        matrix.append(row)
    return matrix


def daily_series(messages: list[Message]) -> list[tuple[str, int]]:
    c: Counter[str] = Counter()
    for msg in messages:
        if msg.published_at:
            c[msg.published_at.strftime("%Y-%m-%d")] += 1
    return sorted(c.items())
