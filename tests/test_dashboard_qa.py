from mlops.dashboard_qa import compact_graph, compact_information, compact_media, compact_tonality


def test_compact_tonality_uses_filtered_hubs_and_ranges():
    text = compact_tonality(
        {
            "current_tab": "Негативные упоминания",
            "filters": {
                "commentsRange": [0, 10],
                "likesRange": [0, 20],
                "viewsRange": [0, 100],
                "audienceRange": [0, 5000],
                "original_mentions": 800,
            },
            "data": {
                "tonality_values": {"negative_count": 12, "positive_count": 3},
                "tonality_hubs_values": {
                    "negative_hubs": [
                        {"name": "ТАСС", "values": 10, "audience_sum": 4000},
                        {"name": "Мелкий блог", "values": 2, "audience_sum": 40},
                    ],
                    "positive_hubs": [{"name": "Другой", "values": 3, "audience_sum": 10}],
                },
            },
        }
    )
    assert "15 упоминаний из 800" in text
    assert "ТАСС" in text
    assert "Другой" not in text
    assert "аудитория 0–5000" in text


def test_compact_media_prefers_filtered_graphs():
    text = compact_media(
        {
            "filters": {"indexRange": [10, 50]},
            "data": {
                "first_graph": {
                    "negative_smi": [{"name": "Старый", "index": 90, "message_count": 9}],
                    "positive_smi": [],
                },
                "second_graph": [{"name": "Старый", "index": 90, "url": "https://old.example"}],
                "filtered_first_graph": {
                    "negative_smi": [{"name": "Отфильтрованный", "index": 20, "message_count": 4}],
                    "positive_smi": [],
                },
                "filtered_second_graph": [{"name": "Отфильтрованный", "index": 20, "url": "https://new.example"}],
            },
        }
    )
    assert "Отфильтрованный" in text
    assert "Старый" not in text
    assert "10–50" in text
    assert "https://new.example" in text


def test_compact_information_counts_filtered_values():
    text = compact_information(
        {
            "filters": {
                "audienceRange": [0, 1000],
                "repostsRange": [0, 5],
                "erRange": [0, 1],
                "viewsCountRange": [0, 100],
                "original_messages": 40,
            },
            "data": {
                "num_messages": 40,
                "values": [
                    {
                        "author": {
                            "fullname": "Альфа",
                            "hub": "telegram.org",
                            "author_type": "Сообщество",
                            "audienceCount": 800,
                            "viewsCount": 20,
                            "url": "https://t.me/a/1",
                        },
                        "reposts": [{}, {}],
                    }
                ],
            },
        }
    )
    assert "1 сообщений из 40" in text
    assert "Альфа" in text
    assert "telegram.org" in text


def test_compact_graph_uses_filtered_counts_and_cluster():
    text = compact_graph(
        {
            "graph": {
                "nodes": [
                    {"id": "a", "label": "Альфа", "audience": 1000, "posts_count": 4, "cluster_id": 2, "type": "Сообщество"},
                    {"id": "b", "label": "Бета", "audience": 10, "posts_count": 1, "cluster_id": 2},
                ],
                "links": [{"type": "exact"}, {"type": "exact"}, {"type": "same_hub"}],
                "clusters": [{"id": 2, "size": 2, "about": "Платный проезд"}],
            },
            "statistics": {
                "nodes_count": 2,
                "edges_count": 3,
                "original_nodes_count": 50,
                "original_edges_count": 400,
                "edge_types": {"exact": 2, "same_hub": 1},
            },
            "metadata": {
                "focused_cluster_id": 2,
                "enabled_link_types": ["exact", "same_hub"],
                "author_search": "альф",
            },
        }
    )
    assert "2 узлов, 3 связей (исходно 50 узлов, 400 связей)" in text
    assert "кластер 2" in text
    assert "Платный проезд" in text
    assert "Альфа" in text
    assert "альф" in text
