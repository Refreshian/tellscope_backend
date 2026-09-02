from datetime import datetime

import pandas as pd

from mlops.author_graph import (
    add_structural_links,
    collect_author_meta,
    detect_clusters,
    enhance_author_graph,
)


def _df():
    return pd.DataFrame(
        [
            {
                "timeCreate": datetime(2025, 12, 27, 7, 0),
                "title": "",
                "url": "https://telegram.me/alpha/1",
                "hubtype": "Мессенджеры каналы",
                "commentsCount": 1,
                "audienceCount": 1000,
                "repostsCount": 0,
                "likesCount": 10,
                "er": 0.1,
                "viewsCount": 50,
                "duplicateCount": 3,
                "region": "Москва",
                "city": "Москва",
                "labels": "ипотека банк ставка",
                "fullname": "Альфа",
                "author_url": "https://telegram.me/alpha",
                "author_type": "Сообщество",
                "sex": "",
                "age": "",
            },
            {
                "timeCreate": datetime(2025, 12, 27, 8, 0),
                "title": "",
                "url": "https://telegram.me/beta/2",
                "hubtype": "Мессенджеры каналы",
                "commentsCount": 0,
                "audienceCount": 800,
                "repostsCount": 2,
                "likesCount": 4,
                "er": 0.05,
                "viewsCount": 20,
                "duplicateCount": 3,
                "region": "Москва",
                "city": "Москва",
                "labels": "ипотека банк ставка",
                "fullname": "Бета",
                "author_url": "https://telegram.me/beta",
                "author_type": "Сообщество",
                "sex": "",
                "age": "",
            },
            {
                "timeCreate": datetime(2025, 12, 20, 10, 0),
                "title": "",
                "url": "https://vk.com/wall-1_1",
                "hubtype": "Соцсети",
                "commentsCount": 0,
                "audienceCount": 50,
                "repostsCount": 0,
                "likesCount": 1,
                "er": 0,
                "viewsCount": 5,
                "duplicateCount": 0,
                "region": "",
                "city": "",
                "labels": "погода в городе",
                "fullname": "Гамма",
                "author_url": "https://vk.com/gamma",
                "author_type": "Личный профиль",
                "sex": "",
                "age": "",
            },
        ]
    )


def test_collect_meta_and_primary_url():
    meta = collect_author_meta(_df())
    assert meta["Альфа"]["hubtype"] == "Мессенджеры каналы"
    assert meta["Альфа"]["hub"] == "telegram"
    assert meta["Альфа"]["primary_url"].endswith("/alpha/1")
    assert meta["Альфа"]["author_url"] == "https://telegram.me/alpha"
    assert meta["Альфа"]["duplicate_count"] == 3
    assert meta["Альфа"]["likes"] == 10


def test_structural_reprint_and_enhance():
    graph = {
        "nodes": [
            {"id": "Альфа", "label": "Альфа", "type": "Сообщество", "audience": 1000, "posts_count": 1, "topics": []},
            {"id": "Бета", "label": "Бета", "type": "Сообщество", "audience": 800, "posts_count": 1, "topics": []},
            {"id": "Гамма", "label": "Гамма", "type": "Личный профиль", "audience": 50, "posts_count": 1, "topics": []},
        ],
        "links": [],
    }
    out = enhance_author_graph(_df(), graph)
    types = {link["type"] for link in out["links"]}
    assert "reprint" in types
    alpha = next(node for node in out["nodes"] if node["id"] == "Альфа")
    assert alpha["primary_url"]
    assert alpha["hubtype"] == "Мессенджеры каналы"
    assert alpha["cluster_id"] >= 1
    assert out["clusters"]
    assert out["link_counts"].get("reprint", 0) >= 1
    assert any(node.get("cluster_id") for node in out["nodes"])


def test_no_duplicate_existing_edges():
    graph = {
        "nodes": [
            {"id": "Альфа", "label": "Альфа", "topics": []},
            {"id": "Бета", "label": "Бета", "topics": []},
        ],
        "links": [{"source": "Альфа", "target": "Бета", "type": "exact", "weight": 1}],
    }
    extras = add_structural_links(graph, collect_author_meta(_df()))
    assert extras == []
    clusters = detect_clusters({**graph, "links": graph["links"]})
    assert clusters[0]["size"] == 2
