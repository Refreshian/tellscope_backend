from __future__ import annotations

from collections import Counter

from .catalog import Catalog
from .models import Message, ObjectStats, ReportBundle


def _fmt(n: float | int | None) -> str:
    if n is None:
        return "н/д"
    if isinstance(n, float) and n >= 1_000_000:
        return f"{n / 1_000_000:.1f} млн"
    if isinstance(n, (int, float)) and n >= 1000:
        return f"{n:,.0f}".replace(",", " ")
    return str(int(n) if isinstance(n, float) and n.is_integer() else n)


def four_volume(bundle: ReportBundle, catalog: Catalog) -> list[str]:
    stats = bundle.object_stats
    if not stats:
        return ["В выборке нет размеченных объектов."]
    top = stats[0]
    second = stats[1] if len(stats) > 1 else None
    rest = sum(s.messages for s in stats[1:])
    projects = [s for s in stats if catalog.by_id.get(s.object_id) and catalog.by_id[s.object_id].kind == "project"]
    proj_n = sum(s.messages for s in projects)
    bullets = [
        f"{catalog.label(top.object_id)} формирует ядро инфополя: {top.messages} материалов"
        + (f" — больше, чем остальные объекты вместе ({rest})." if rest and top.messages > rest else "."),
    ]
    if second:
        share = 100 * second.messages / max(sum(s.messages for s in stats), 1)
        bullets.append(
            f"Второй по объёму — {catalog.label(second.object_id)}: {second.messages} материалов ({share:.0f}% размеченного потока)."
        )
    if len(stats) >= 3:
        tail = ", ".join(f"{catalog.label(s.object_id)} — {s.messages}" for s in stats[2:6])
        bullets.append(f"Следующий эшелон: {tail}.")
    if projects:
        bullets.append(
            f"Городские проекты вместе дают {proj_n} материалов — присутствие {'эпизодическое' if proj_n < 30 else 'заметное'}."
        )
    return bullets[:4]


def four_speakers(bundle: ReportBundle, catalog: Catalog) -> list[str]:
    total: Counter[str] = Counter()
    by_obj: dict[str, Counter[str]] = {}
    for st in bundle.object_stats:
        by_obj[st.object_id] = Counter(st.speakers)
        total.update(st.speakers)
    if not total:
        return ["Узнаваемые спикеры в выборке почти не размечены — имеет смысл прогнать тексты через Tellscope."]
    bullets = []
    for oid, cnt in list(by_obj.items())[:4]:
        if not cnt:
            continue
        top = ", ".join(
            f"{catalog.person_by_id[p].short if p in catalog.person_by_id else p}: {n}"
            for p, n in cnt.most_common(3)
        )
        bullets.append(f"{catalog.label(oid)} — {top}.")
    none = [catalog.label(s.object_id) for s in bundle.object_stats[:8] if not s.speakers]
    if none:
        bullets.append("Без узнаваемого ньюсмейкера: " + ", ".join(none[:5]) + ".")
    return bullets[:4]


def four_reach(bundle: ReportBundle, catalog: Catalog) -> list[str]:
    ranked = [s for s in bundle.object_stats if s.reach]
    if not ranked:
        return ["Охват в Word-выгрузке отсутствует. Нужен Excel Медиалогии со столбцом «Охват» либо оценка по справочнику СМИ."]
    ranked.sort(key=lambda s: s.reach or 0, reverse=True)
    bullets = []
    for st in ranked[:3]:
        per = (st.reach or 0) / max(st.messages, 1)
        bullets.append(
            f"{catalog.label(st.object_id)} — {_fmt(st.reach)} контактов, {_fmt(per)} на материал."
        )
    if len(ranked) >= 2:
        best_eff = max(ranked, key=lambda s: (s.reach or 0) / max(s.messages, 1))
        bullets.append(
            f"Максимальная отдача охвата на материал — {catalog.label(best_eff.object_id)}."
        )
    return bullets[:4]


def four_media(bundle: ReportBundle, catalog: Catalog) -> list[str]:
    bullets = []
    for st in bundle.object_stats[:4]:
        top = Counter(st.media).most_common(3)
        if not top:
            continue
        desc = ", ".join(f"{n} ({c})" for n, c in top)
        bullets.append(f"{catalog.label(st.object_id)} опирается на {desc}.")
    return bullets[:4] or ["Распределение по СМИ не собралось — проверьте парсер источников."]


def four_tone(bundle: ReportBundle, catalog: Catalog) -> list[str]:
    neg = [s for s in bundle.object_stats if s.negative]
    pos = sorted(bundle.object_stats, key=lambda s: s.positive, reverse=True)
    bullets = []
    if not neg:
        bullets.append("Негативных материалов в размеченной выборке нет: основной фон нейтральный.")
    else:
        bullets.append(
            "Негатив точечный: "
            + ", ".join(f"{catalog.label(s.object_id)} — {s.negative}" for s in neg[:4])
            + "."
        )
    for st in pos[:3]:
        if st.positive:
            bullets.append(f"Позитив {catalog.label(st.object_id)}: {st.positive} материалов.")
    return bullets[:4]


def four_role(bundle: ReportBundle, catalog: Catalog) -> list[str]:
    bullets = []
    for st in bundle.object_stats[:4]:
        total = max(st.messages, 1)
        bullets.append(
            f"{catalog.label(st.object_id)}: главная роль {100 * st.main_role / total:.0f}% "
            f"({st.main_role} из {st.messages}), эпизодическая {100 * st.episodic_role / total:.0f}%, "
            f"фоновая {100 * st.background_role / total:.0f}%."
        )
    return bullets[:4]


def four_concentration(bundle: ReportBundle, catalog: Catalog) -> list[str]:
    ranked = sorted(bundle.object_stats, key=lambda s: s.top2_share, reverse=True)
    bullets = [
        "Концентрация — доля топ-2 СМИ в материалах объекта. Высокая доля означает зависимость от узкого пула площадок."
    ]
    for st in ranked[:3]:
        bullets.append(f"{catalog.label(st.object_id)}: {100 * st.top2_share:.0f}% на двух площадках ({st.unique_media} СМИ всего).")
    return bullets[:4]


def four_initiated(bundle: ReportBundle, catalog: Catalog) -> list[str]:
    ranked = sorted(
        [s for s in bundle.object_stats if s.initiated_known],
        key=lambda s: s.initiated / max(s.initiated_known, 1),
        reverse=True,
    )
    bullets = ["К инициированным отнесены публикации с признаками работы пресс-службы / Информационного центра."]
    for st in ranked[:3]:
        share = 100 * st.initiated / max(st.initiated_known, 1)
        bullets.append(f"{catalog.label(st.object_id)} — {share:.0f}% инициированных выходов ({st.initiated} из {st.initiated_known}).")
    return bullets[:4]


def four_index(bundle: ReportBundle, catalog: Catalog) -> list[str]:
    ranked = [s for s in bundle.object_stats if s.media_index]
    if not ranked:
        return ["МедиаИндекс в этой поставке не пришёл. Это поле Excel Медиалогии, его нельзя восстановить из Word."]
    ranked.sort(key=lambda s: s.media_index or 0, reverse=True)
    return [
        f"{catalog.label(s.object_id)} — {s.media_index:.0f} пунктов при {s.messages} материалах."
        for s in ranked[:4]
    ]


def four_overlap(bundle: ReportBundle, catalog: Catalog) -> list[str]:
    top = bundle.object_stats[:6]
    if len(top) < 2:
        return ["Недостаточно объектов для матрицы пересечений."]
    sets = {s.object_id: set(s.media) for s in top}
    pairs = []
    ids = [s.object_id for s in top]
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            pairs.append((len(sets[a] & sets[b]), a, b))
    pairs.sort(reverse=True)
    bullets = ["Пересечения по охватам — сколько СМИ два объекта используют одновременно."]
    for n, a, b in pairs[:3]:
        bullets.append(f"{catalog.label(a)} и {catalog.label(b)} делят {n} площадок.")
    return bullets[:4]


def three_observations(bundle: ReportBundle, catalog: Catalog) -> list[dict]:
    if not bundle.object_stats:
        return []
    vol = bundle.object_stats[0]
    conc = max(bundle.object_stats, key=lambda s: s.top2_share)
    role = min(bundle.object_stats[:6], key=lambda s: s.main_role / max(s.messages, 1))
    return [
        {
            "title": f"{catalog.label(vol.object_id)} — объём",
            "fact": f"{vol.messages} материалов, концентрация топ-2 СМИ {100 * vol.top2_share:.0f}%.",
            "meaning": "регулярное присутствие зависит от ограниченного пула площадок" if vol.top2_share > 0.6 else "поток распределён по нескольким каналам",
            "full": "Позволит оценить стоимость охвата и целесообразность расширения медиамикса.",
        },
        {
            "title": f"{catalog.label(role.object_id)} — заметность",
            "fact": f"главная роль {100 * role.main_role / max(role.messages, 1):.0f}% при {role.messages} материалах.",
            "meaning": "объём не равен качеству присутствия: объект часто идёт фоном чужой повестки",
            "full": "Определить, какие активности дают ключевую роль, а не только упоминание.",
        },
        {
            "title": f"{catalog.label(conc.object_id)} — концентрация",
            "fact": f"{100 * conc.top2_share:.0f}% выходов на двух СМИ, всего {conc.unique_media} площадок.",
            "meaning": "риск зависимости от одного-двух каналов",
            "full": "Сопоставить результат с затратами и выделить наиболее результативные площадки.",
        },
    ]


def top_headlines(messages: list[Message], object_id: str, n: int = 8) -> list[str]:
    items = [m.title for m in messages if object_id in m.object_ids and m.title]
    return items[:n]
