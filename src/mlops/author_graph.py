"""Author-graph enrichment: message metadata, structural edges, cluster summaries."""
from __future__ import annotations

import re
import threading
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from html import unescape
from typing import Any
from urllib.parse import urlparse

try:
    from fastapi.responses import JSONResponse
except ImportError:  # pragma: no cover

    class JSONResponse:
        def __init__(self, content=None, status_code=200, **_kwargs):
            self.status_code = status_code
            self.content = content or {}


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S | re.I)
_THINK_OPEN = re.compile(r"<think>.*", re.S | re.I)
_THINK_TAG = re.compile(r"</?think>", re.I)
_TOKEN = re.compile(r"[A-Za-zА-Яа-яЁё0-9]{4,}")

LINK_TYPES = ("exact", "similar", "same_hub", "reprint", "co_time")
MAX_TOPICS = 8
MAX_EXTRA_DEGREE = 6
PRODUCT = "graph-cluster"


def _clean_llm(text: str) -> str:
    cleaned = _THINK_BLOCK.sub("", text or "")
    cleaned = _THINK_OPEN.sub("", cleaned)
    cleaned = _THINK_TAG.sub("", cleaned)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def _safe(value: Any, default: str = "") -> str:
    if value is None:
        return default
    try:
        import pandas as pd

        if pd.isna(value):
            return default
    except Exception:
        pass
    text = str(value).strip()
    return text if text and text.lower() not in {"nan", "none", "null"} else default


def _num(value: Any, default: float = 0.0) -> float:
    try:
        import pandas as pd

        if value is None or pd.isna(value):
            return default
    except Exception:
        if value is None:
            return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _hub_from_url(url: str) -> str:
    if not url:
        return ""
    try:
        host = (urlparse(url).netloc or "").lower()
    except Exception:
        return ""
    if host.startswith("www."):
        host = host[4:]
    if "telegram" in host or host in {"t.me", "telegram.me"}:
        return "telegram"
    if "vk." in host:
        return "vk.com"
    if "ok.ru" in host or "odnoklassniki" in host:
        return "ok.ru"
    return host


def _as_dt(value: Any):
    if value is None:
        return None
    try:
        import pandas as pd

        if pd.isna(value):
            return None
        if isinstance(value, datetime):
            return value
        if hasattr(value, "to_pydatetime"):
            return value.to_pydatetime()
        if isinstance(value, (int, float)):
            ts = float(value)
            if ts > 10_000_000_000:
                ts /= 1000.0
            return datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None)
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            return None
        return parsed.to_pydatetime()
    except Exception:
        return None


def _iso(value) -> str:
    dt = _as_dt(value)
    if not dt:
        return ""
    try:
        return dt.isoformat(sep=" ", timespec="minutes")
    except Exception:
        return str(dt)


def _tokens(text: str) -> set[str]:
    return {tok.lower() for tok in _TOKEN.findall(text or "")}


def _mode(values: list[str]) -> str:
    cleaned = [item for item in values if item]
    if not cleaned:
        return ""
    return Counter(cleaned).most_common(1)[0][0]


def collect_author_meta(df) -> dict[str, dict]:
    """One pass over the CSV: metadata per author used on nodes and extra edges."""
    grouped = defaultdict(lambda: {
        "hubtypes": [],
        "hubs": [],
        "labels": set(),
        "tokens": set(),
        "posts": [],
        "author_urls": [],
        "regions": [],
        "cities": [],
        "types": [],
        "sex": "",
        "age": "",
        "dup": 0.0,
        "likes": 0.0,
        "comments": 0.0,
        "reposts": 0.0,
        "views": 0.0,
        "er": [],
        "times": [],
    })
    records = df.to_dict("records") if hasattr(df, "to_dict") else list(df or [])
    for mapping in records:
        if not isinstance(mapping, dict):
            continue
        author = _safe(mapping.get("fullname"))
        if not author:
            continue
        bucket = grouped[author]
        url = _safe(mapping.get("url"))
        labels = _safe(mapping.get("labels"))
        hubtype = _safe(mapping.get("hubtype"))
        audience = _num(mapping.get("audienceCount"))
        dt = _as_dt(mapping.get("timeCreate"))
        bucket["hubtypes"].append(hubtype)
        bucket["hubs"].append(_hub_from_url(url))
        if labels:
            bucket["labels"].add(labels)
            bucket["tokens"].update(_tokens(labels))
        bucket["author_urls"].append(_safe(mapping.get("author_url")))
        bucket["regions"].append(_safe(mapping.get("region")))
        bucket["cities"].append(_safe(mapping.get("city")))
        bucket["types"].append(_safe(mapping.get("author_type") or mapping.get("type")))
        if not bucket["sex"]:
            bucket["sex"] = _safe(mapping.get("sex"))
        if not bucket["age"]:
            bucket["age"] = _safe(mapping.get("age"))
        bucket["dup"] = max(bucket["dup"], _num(mapping.get("duplicateCount")))
        bucket["likes"] += _num(mapping.get("likesCount"))
        bucket["comments"] += _num(mapping.get("commentsCount"))
        bucket["reposts"] += _num(mapping.get("repostsCount"))
        bucket["views"] += _num(mapping.get("viewsCount"))
        er = _num(mapping.get("er"))
        if er:
            bucket["er"].append(er)
        if dt:
            bucket["times"].append(dt)
        bucket["posts"].append({
            "text": labels,
            "url": url,
            "hubtype": hubtype,
            "hub": _hub_from_url(url),
            "audience": int(audience),
            "time": _iso(dt),
            "likes": int(_num(mapping.get("likesCount"))),
            "views": int(_num(mapping.get("viewsCount"))),
            "duplicates": int(_num(mapping.get("duplicateCount"))),
        })
    out = {}
    for author, bucket in grouped.items():
        posts = sorted(bucket["posts"], key=lambda item: item.get("audience") or 0, reverse=True)
        times = sorted(bucket["times"])
        primary = posts[0] if posts else {}
        out[author] = {
            "hubtype": _mode(bucket["hubtypes"]),
            "hubtypes": [name for name, _count in Counter(bucket["hubtypes"]).most_common(4) if name],
            "hub": _mode(bucket["hubs"]) or _hub_from_url(primary.get("url") or ""),
            "hubs": [name for name, _count in Counter(bucket["hubs"]).most_common(4) if name],
            "labels": bucket["labels"],
            "tokens": bucket["tokens"],
            "posts": posts[:MAX_TOPICS],
            "primary_url": primary.get("url") or "",
            "author_url": _mode(bucket["author_urls"]),
            "region": _mode(bucket["regions"]),
            "city": _mode(bucket["cities"]),
            "type": _mode(bucket["types"]),
            "sex": bucket["sex"],
            "age": bucket["age"],
            "duplicate_count": int(bucket["dup"]),
            "likes": int(bucket["likes"]),
            "comments": int(bucket["comments"]),
            "reposts": int(bucket["reposts"]),
            "views": int(bucket["views"]),
            "er": round(sum(bucket["er"]) / len(bucket["er"]), 4) if bucket["er"] else 0,
            "period_start": _iso(times[0]) if times else "",
            "period_end": _iso(times[-1]) if times else "",
            "tmin": times[0] if times else None,
            "tmax": times[-1] if times else None,
        }
    return out


def _edge_key(source: str, target: str) -> tuple[str, str]:
    return (source, target) if source <= target else (target, source)


def _index_pairs(index: dict[str, list[str]], max_bucket: int = 60) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for ids in index.values():
        uniq = list(dict.fromkeys(ids))
        if len(uniq) < 2 or len(uniq) > max_bucket:
            continue
        for i, left in enumerate(uniq):
            for right in uniq[i + 1 :]:
                pairs.add(_edge_key(left, right))
    return pairs


def _time_close(left: dict, right: dict, hours: int = 36) -> bool:
    if not left.get("tmin") or not right.get("tmin"):
        return False
    gap = min(abs((left["tmax"] - right["tmin"]).total_seconds()), abs((right["tmax"] - left["tmin"]).total_seconds()))
    overlap = not (left["tmax"] < right["tmin"] or right["tmax"] < left["tmin"])
    return overlap or gap <= hours * 3600


def add_structural_links(graph_data: dict, meta: dict[str, dict]) -> list[dict]:
    nodes = graph_data.get("nodes") or []
    existing = {_edge_key(str(link.get("source")), str(link.get("target"))) for link in graph_data.get("links") or []}
    by_label: dict[str, list[str]] = defaultdict(list)
    by_hubtype: dict[str, list[str]] = defaultdict(list)
    by_hub: dict[str, list[str]] = defaultdict(list)
    by_day: dict[str, list[str]] = defaultdict(list)
    for node in nodes:
        nid = str(node.get("id") or "")
        info = meta.get(nid) or {}
        for label in info.get("labels") or []:
            by_label[label].append(nid)
        for hubtype in info.get("hubtypes") or []:
            by_hubtype[hubtype].append(nid)
        for hub in info.get("hubs") or []:
            by_hub[hub].append(nid)
        tmin = info.get("tmin")
        if tmin:
            by_day[tmin.strftime("%Y-%m-%d")].append(nid)

    candidates = set()
    candidates |= _index_pairs(by_label, max_bucket=120)
    candidates |= _index_pairs(by_hubtype, max_bucket=40)
    candidates |= _index_pairs(by_hub, max_bucket=40)
    candidates |= _index_pairs(by_day, max_bucket=50)

    extras: list[dict] = []
    degree: dict[str, int] = defaultdict(int)
    for source, target in candidates:
        if source == target or (source, target) in existing:
            continue
        if degree[source] >= MAX_EXTRA_DEGREE or degree[target] >= MAX_EXTRA_DEGREE:
            continue
        left = meta.get(source) or {}
        right = meta.get(target) or {}
        shared_labels = (left.get("labels") or set()) & (right.get("labels") or set())
        shared_hubtypes = set(left.get("hubtypes") or []) & set(right.get("hubtypes") or [])
        shared_hubs = set(left.get("hubs") or []) & set(right.get("hubs") or [])
        shared_tokens = (left.get("tokens") or set()) & (right.get("tokens") or set())
        link_type = ""
        weight = 1.0
        reason = ""
        if shared_labels and (left.get("duplicate_count") or 0) >= 1 and (right.get("duplicate_count") or 0) >= 1:
            link_type = "reprint"
            weight = float(min(left.get("duplicate_count") or 1, right.get("duplicate_count") or 1))
            reason = next(iter(shared_labels))[:120]
        elif (shared_hubtypes or shared_hubs) and (shared_labels or len(shared_tokens) >= 2):
            link_type = "same_hub"
            weight = 2.0 + len(shared_labels)
            reason = (next(iter(shared_hubs), "") or next(iter(shared_hubtypes), ""))[:80]
        elif _time_close(left, right) and (shared_hubtypes or shared_labels or len(shared_tokens) >= 1):
            link_type = "co_time"
            weight = 1.5
            reason = "публикации в одном окне времени"
        if not link_type:
            continue
        extras.append({
            "source": source,
            "target": target,
            "weight": weight,
            "type": link_type,
            "reason": reason,
        })
        existing.add((source, target))
        degree[source] += 1
        degree[target] += 1
    return extras


def _link_ends(link: dict) -> tuple[str, str]:
    source = link.get("source")
    target = link.get("target")
    if isinstance(source, dict):
        source = source.get("id")
    if isinstance(target, dict):
        target = target.get("id")
    return str(source or ""), str(target or "")


def connected_groups(graph_data: dict) -> list[set[str]]:
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: str, right: str) -> None:
        root_l, root_r = find(left), find(right)
        if root_l != root_r:
            parent[root_r] = root_l

    for node in graph_data.get("nodes") or []:
        nid = str(node.get("id") or "")
        if nid:
            find(nid)
    for link in graph_data.get("links") or []:
        source, target = _link_ends(link)
        if source and target:
            union(source, target)
    groups: dict[str, set[str]] = defaultdict(set)
    for nid in parent:
        groups[find(nid)].add(nid)
    return list(groups.values())


def graph_to_nx(graph_data: dict):
    import networkx as nx

    graph = nx.Graph()
    for node in graph_data.get("nodes") or []:
        nid = str(node.get("id") or "")
        if nid:
            graph.add_node(nid, **{k: v for k, v in node.items() if k != "id"})
    for link in graph_data.get("links") or []:
        source, target = _link_ends(link)
        if source and target and source != target:
            graph.add_edge(source, target, weight=link.get("weight") or 1, type=link.get("type") or "")
    return graph


def detect_clusters(graph_data: dict) -> list[dict]:
    communities = []
    try:
        from networkx.algorithms.community import greedy_modularity_communities

        graph = graph_to_nx(graph_data)
        if graph.number_of_nodes():
            communities = [set(group) for group in greedy_modularity_communities(graph)]
    except Exception:
        communities = []
    if not communities:
        communities = connected_groups(graph_data)
    communities.sort(key=len, reverse=True)
    node_by_id = {str(node.get("id")): node for node in graph_data.get("nodes") or []}
    clusters = []
    for idx, members in enumerate(communities, 1):
        if len(members) < 2:
            continue
        authors = [node_by_id[nid] for nid in members if nid in node_by_id]
        authors.sort(key=lambda node: node.get("audience") or 0, reverse=True)
        hubtypes = Counter(str(node.get("hubtype") or "") for node in authors if node.get("hubtype"))
        topics = Counter()
        for node in authors:
            for topic in node.get("topics") or []:
                text = topic if isinstance(topic, str) else topic.get("text")
                if text:
                    topics[str(text)] += 1
        starts = [node.get("period_start") for node in authors if node.get("period_start")]
        ends = [node.get("period_end") for node in authors if node.get("period_end")]
        edge_types = Counter()
        for link in graph_data.get("links") or []:
            source, target = _link_ends(link)
            if source in members and target in members:
                edge_types[str(link.get("type") or "other")] += 1
        clusters.append({
            "id": idx,
            "size": len(authors),
            "audience": int(sum(node.get("audience") or 0 for node in authors)),
            "authors": [node.get("label") or node.get("id") for node in authors[:8]],
            "author_ids": [str(node.get("id")) for node in authors],
            "hubtypes": [{"name": name, "count": count} for name, count in hubtypes.most_common(4)],
            "topics": [text for text, _count in topics.most_common(5)],
            "period_start": min(starts) if starts else "",
            "period_end": max(ends) if ends else "",
            "edge_types": dict(edge_types),
        })
        for node in authors:
            node["cluster_id"] = idx
    return clusters


def _enrich_node(node: dict, info: dict) -> dict:
    topics = info.get("posts") or node.get("topics") or []
    node.update({
        "hubtype": info.get("hubtype") or node.get("hubtype") or "",
        "hubtypes": info.get("hubtypes") or [],
        "hub": info.get("hub") or "",
        "primary_url": info.get("primary_url") or "",
        "url": info.get("primary_url") or node.get("url") or "",
        "author_url": info.get("author_url") or "",
        "region": info.get("region") or "",
        "city": info.get("city") or "",
        "sex": info.get("sex") or "",
        "age": info.get("age") or "",
        "duplicate_count": info.get("duplicate_count") or 0,
        "likes": info.get("likes") or 0,
        "comments": info.get("comments") or 0,
        "reposts": info.get("reposts") or 0,
        "views": info.get("views") or 0,
        "er": info.get("er") or 0,
        "period_start": info.get("period_start") or "",
        "period_end": info.get("period_end") or "",
        "topics": topics[:MAX_TOPICS],
        "cluster_id": node.get("cluster_id") or 0,
    })
    if info.get("type") and (not node.get("type") or node.get("type") == "unknown"):
        node["type"] = info["type"]
    return node


def enhance_author_graph(df, graph_data: dict) -> dict:
    graph_data = graph_data or {"nodes": [], "links": []}
    meta = collect_author_meta(df) if df is not None and getattr(df, "empty", True) is False else {}
    for node in graph_data.get("nodes") or []:
        _enrich_node(node, meta.get(str(node.get("id") or "")) or {})
    extras = add_structural_links(graph_data, meta) if meta else []
    graph_data.setdefault("links", [])
    graph_data["links"].extend(extras)
    clusters = detect_clusters(graph_data)
    counts = Counter(str(link.get("type") or "other") for link in graph_data.get("links") or [])
    graph_data["clusters"] = clusters
    graph_data["link_counts"] = dict(counts)
    return graph_data


def patch_graph_builder(cls) -> None:
    orig_build = cls.build_author_graph
    orig_stats = cls.get_graph_statistics

    def build_author_graph(self):
        data = orig_build(self)
        try:
            data = enhance_author_graph(self.df, data)
            self._last_graph = data
            self.G = graph_to_nx(data)
        except Exception as exc:
            print(f"author_graph enhance skipped: {exc}")
            try:
                import networkx as nx

                self.G = nx.Graph()
            except Exception:
                pass
        return data

    def get_graph_statistics(self):
        last = getattr(self, "_last_graph", None)
        if last:
            try:
                if getattr(self, "G", None) is None or self.G.number_of_nodes() == 0:
                    self.G = graph_to_nx(last)
            except Exception:
                pass
        stats = orig_stats(self)
        if last:
            stats["link_counts"] = last.get("link_counts") or {}
            stats["clusters_count"] = len(last.get("clusters") or [])
        return stats

    cls.build_author_graph = build_author_graph
    cls.get_graph_statistics = get_graph_statistics


def _progress(job_id: str, message: str, step: str, percent: int, status: str = "running", **extra) -> None:
    try:
        from mlops.jobs import register

        register(
            job_id,
            product=PRODUCT,
            status=status,
            message=message,
            progress_step=step,
            progress_percent=percent,
            **extra,
        )
    except Exception:
        pass


def _cluster_prompt(cluster: dict, nodes: list[dict], links: list[dict]) -> str:
    authors = []
    for node in nodes[:16]:
        topics = []
        for topic in (node.get("topics") or [])[:3]:
            text = topic if isinstance(topic, str) else topic.get("text")
            if text:
                topics.append(str(text)[:140])
        authors.append(
            f"- {node.get('label') or node.get('id')}: {node.get('hubtype') or 'площадка?'} · "
            f"охват {node.get('audience') or 0} · постов {node.get('posts_count') or 0}"
            + (f" · {'; '.join(topics)}" if topics else "")
        )
    edge_line = ", ".join(f"{name} {count}" for name, count in (cluster.get("edge_types") or {}).items())
    hubs = ", ".join(f"{item.get('name')} {item.get('count')}" for item in cluster.get("hubtypes") or [])
    return (
        f"Кластер {cluster.get('id')}: {cluster.get('size')} авторов, суммарный охват {cluster.get('audience')}.\n"
        f"Период: {cluster.get('period_start') or '—'} — {cluster.get('period_end') or '—'}\n"
        f"Площадки: {hubs or 'нет'}\n"
        f"Связи внутри: {edge_line or 'нет'}\n"
        f"Темы: {'; '.join(cluster.get('topics') or []) or 'нет'}\n\n"
        f"Авторы:\n" + "\n".join(authors) + "\n\n"
        f"Число связей в выборке: {len(links)}"
    )


def run_cluster_summary(body: dict, job_id: str) -> str:
    from mlops.gateway import chat
    from mlops.prompts import load_prompt

    cluster = body.get("cluster") if isinstance(body.get("cluster"), dict) else {}
    nodes = [node for node in (body.get("nodes") or []) if isinstance(node, dict)]
    links = [link for link in (body.get("links") or []) if isinstance(link, dict)]
    if not nodes:
        raise RuntimeError("В кластере нет авторов")
    _progress(job_id, "Собираем сводку кластера…", "summary", 20)
    try:
        system = "/no_think\n" + load_prompt("dashboard_qa_cluster_v1")
    except Exception:
        system = "/no_think\nТы аналитик соцмедиа. Кратко опиши кластер авторов по данным. Не выдумывай цифры."
    user = "/no_think\n" + _cluster_prompt(cluster, nodes, links)
    result = chat(
        provider="vllm",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.1,
        max_tokens=700,
        timeout=90,
        extra={"chat_template_kwargs": {"enable_thinking": False}},
        profile="dashboard_qa_graph",
    )
    memo = _clean_llm(unescape(str(result.content or "")))
    if not memo:
        raise RuntimeError("Модель вернула пустую сводку")
    _progress(job_id, "Сводка готова", "done", 100, status="done", memo=memo)
    return memo


async def handle_cluster_summary(request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception as exc:
        return JSONResponse(status_code=400, content={"error": "Неверный формат данных", "details": str(exc)})
    nodes = body.get("nodes") or []
    if not nodes:
        return JSONResponse(status_code=400, content={"error": "Нужны авторы кластера"})
    from mlops.ai_bot_rag import gpu_ready

    ready, holders = gpu_ready()
    if not ready:
        first = holders[0] if holders else {}
        return JSONResponse(
            status_code=503,
            content={
                "error": "Сейчас выполняется другая задача. Попробуйте через несколько минут.",
                "jobs": holders,
                "busy": first.get("product") or "",
            },
        )
    job_id = uuid.uuid4().hex[:16]
    _progress(job_id, "Запускаем сводку кластера…", "start", 6)

    def worker():
        try:
            run_cluster_summary(body, job_id)
        except Exception as exc:
            _progress(job_id, str(exc)[:400], "error", 0, status="error")

    threading.Thread(target=worker, daemon=True).start()
    return JSONResponse(
        content={
            "job_id": job_id,
            "status": "running",
            "message": "Запускаем сводку кластера…",
            "progress": {"step": "start", "percent": 6, "message": "Запускаем сводку кластера…"},
        }
    )


async def handle_cluster_summary_status(request) -> JSONResponse:
    job_id = str(request.query_params.get("job_id") or "").strip()
    if not job_id:
        return JSONResponse(status_code=400, content={"error": "Нужен job_id"})
    try:
        from mlops.jobs import get

        job = get(job_id) or {}
    except Exception:
        job = {}
    if not job:
        return JSONResponse(status_code=404, content={"error": "Задача не найдена", "job_id": job_id})
    status = str(job.get("status") or "running")
    percent = 6
    try:
        percent = int(float(job.get("progress_percent") or 6))
    except (TypeError, ValueError):
        percent = 6
    return JSONResponse(
        content={
            "job_id": job_id,
            "status": status,
            "message": job.get("message") or "",
            "progress": {
                "step": job.get("progress_step") or "",
                "percent": percent,
                "message": job.get("message") or "",
            },
            "memo": (job.get("memo") or "") if status == "done" else "",
            "error": job.get("message") if status == "error" else "",
        }
    )
