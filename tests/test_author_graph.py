from datetime import datetime

import pandas as pd

from mlops.author_graph import (
    add_structural_links,
    collect_author_meta,
    compose_cluster_about,
    detect_clusters,
    enhance_author_graph,
    link_type_label,
    polish_cluster_memo,
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


def test_link_type_labels_are_russian():
    assert link_type_label("exact") == "точные темы"
    assert link_type_label("same_hub") == "одна площадка"
    assert link_type_label("co_time") == "близко по времени"


def test_polish_cluster_memo_headings_and_authors():
    memo = (
        "👥 Кто в кластере\n"
        "**Artem** входит в ядро кластера.\n\n"
        "🔗 Чем связаны\n"
        "Связи внутри кластера: exact 101, same_hub 285, co_time 61.\n\n"
        "⚠️ На что обратить внимание\n"
        "- ничего важного"
    )
    out = polish_cluster_memo(
        memo,
        [{"label": "Artem", "primary_url": "https://t.me/artem/12"}],
        cluster={"edge_types": {"exact": 101, "same_hub": 285, "co_time": 61}},
    )
    assert "##" not in out
    assert "На что обратить внимание" not in out
    assert "одинаковый тип источника" in out
    assert "одном и том же" in out
    assert "🔗 Чем связаны" in out
    assert "[Artem](https://t.me/artem/12)" in out


def test_polish_links_topic_phrases_to_sources():
    memo = (
        "💬 О чём пишут\n"
        "Основная тема — введение платного проезда на обходе Нижнекамска и Челнов.\n\n"
        "🔗 Чем связаны\n"
        "Связи: exact 2."
    )
    out = polish_cluster_memo(
        memo,
        [{
            "label": "ТАСС",
            "primary_url": "https://t.me/tass/1",
            "topics": [{
                "text": "Тематика текста: введение платного проезда на обходе Нижнекамска и Челнов",
                "url": "https://t.me/tass/10",
            }],
        }],
        cluster={"edge_types": {"exact": 2}},
    )
    assert "https://t.me/tass/10" in out
    assert out.count("https://t.me/tass/10") == 1
    assert "[[[" not in out
    assert "На что обратить внимание" not in out
    assert "похож" in out


def test_polish_does_not_nest_same_phrase_links():
    memo = (
        "💬 О чём пишут\n"
        "Основная тема — введение платного проезда на обходе Нижнекамска и Челнов.\n"
    )
    nodes = [
        {
            "label": f"A{i}",
            "primary_url": f"https://t.me/a/{i}",
            "topics": [{
                "text": "введение платного проезда на обходе Нижнекамска и Челнов",
                "url": f"https://t.me/post/{i}",
            }],
        }
        for i in range(1, 6)
    ]
    out = polish_cluster_memo(memo, nodes, cluster={"edge_types": {"exact": 1}})
    assert out.count("](https://t.me/post/") == 1
    assert "[[" not in out


def test_compose_cluster_about_joins_and_puts_jobs_last():
    text = compose_cluster_about(
        [
            "Тематика текста: введение платного проезда на обходе Нижнекамска и Челнов",
            "Тематика текста — это вакансия на работу сотрудника транспортной безопасности с указанием условий, адресов объектов и оплаты",
            "Тематика текста — оказание услуг по оплате штрафов, проездных сборов и оформлению разрешений для транспорта на границах стран, включая Казахстан, Узбекистан, Россию, Китай и Европу",
        ]
    )
    assert "Тематика текста" not in text
    assert "платного проезда" in text
    assert text.index("платного проезда") < text.index("штрафов")
    assert text.index("штрафов") < text.index("вакансия")
    assert "; Тематика" not in text


def test_compose_cluster_about_drops_near_duplicates():
    text = compose_cluster_about(
        [
            "Тематика текста: введение платного проезда на обходе Нижнекамска и Челнов",
            "Тематика текста — введение платного обхода Нижнекамска и Челнов с автоматизированной системой оплаты",
            "Тематика текста — реклама автошколы, предлагающей обучение водителей",
        ]
    )
    assert text.lower().count("нижнекамска") == 1
    assert "реклама автошколы" in text.lower()
