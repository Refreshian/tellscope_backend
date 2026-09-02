from mlops.ai_bot_rag import (
    coverage_payload,
    docs_markdown,
    es_bool_query,
    es_filter_clauses,
    evidence_markdown,
    evidence_pack,
    normalize_instructions,
    public_sources,
    rrf_merge,
    source_from_es,
    stratify_citations,
    tone_label,
    user_prompt,
)


def _agg_payload(count, hub="Соцсети", hub_count=6810):
    return {
        "hits": {"total": {"value": count}},
        "aggregations": {
            "tone": {
                "buckets": [
                    {"key": -1, "doc_count": 409},
                    {"key": 0, "doc_count": 5132},
                    {"key": 1, "doc_count": 1269},
                ]
            },
            "hubtype": {
                "buckets": [
                    {
                        "key": hub,
                        "doc_count": hub_count,
                        "tone": {"buckets": [{"key": -1, "doc_count": 409}]},
                    }
                ]
            },
            "period": {"min": 1765542540000, "max": 1767038460000},
            "audience": {"value": 1000},
            "hubs": {"buckets": [{"key": "vk.com", "doc_count": 10}]},
        },
    }


class FakeES:
    def __init__(self, payload=None):
        self.payload = payload or _agg_payload(26844)
        self.queries = []

    def search(self, **kwargs):
        self.queries.append(kwargs)
        return self.payload


def test_tone_and_filters():
    assert tone_label(-1) == "негатив"
    assert tone_label(1) == "позитив"
    assert es_filter_clauses({"tone": "neg"}) == [{"term": {"toneMark": -1}}]
    assert "hubtype.keyword" in es_filter_clauses({"channel": "smi"})[0]["terms"]
    names = es_filter_clauses({"hubtypes": ["Соцсети", "Блоги"], "channel": "smi"})[0]["terms"][
        "hubtype.keyword"
    ]
    assert names == ["Соцсети", "Блоги"]


def test_rrf_and_sources():
    merged = rrf_merge([["a", "b", "c"], ["c", "a"]])
    assert merged[0] in {"a", "c"}
    item = source_from_es(
        {
            "text": "x" * 400,
            "title": 'Отзыв <span class="highlight">Банк</span>',
            "hub": "telegram.org",
            "hubtype": "Мессенджеры каналы",
            "url": "https://t.me/x",
            "hash": "h1",
            "authorObject": {"fullname": "Канал"},
            "timeCreate": 1767038515,
            "audienceCount": 49,
            "duplicateCount": 6,
            "toneMark": -1,
        },
        "theme_a",
        0.42,
    )
    cards = public_sources([item])
    assert cards[0]["tone"] == "негатив"
    assert cards[0]["title"] == "Отзыв Банк"
    assert "[1]" in docs_markdown([item], "что пишут")


def test_instructions_go_to_prompt_as_guidance():
    packed = normalize_instructions(
        [{"name": "brief.md", "text": "Смотри только отзывы и не выдумывай KPI."}]
    )
    prompt = user_prompt("что пишут", "PR и репутация", ["theme_a"], "docs", instructions=packed)
    assert "brief.md" in prompt
    assert "не выдумывай KPI" in prompt
    assert "не факты корпуса" in prompt.lower() or "не факты" in prompt.lower()


def test_evidence_pack_query_and_filters():
    es = FakeES()
    pack = evidence_pack(es, {1: "theme_a"}, ["theme_a"], "кредит", {"hubtypes": ["Соцсети"]})
    assert pack["corpus"]["count"] == 26844
    assert pack["query_hits"]["count"] == 26844
    assert pack["applied_filters"] is True
    queries = [item.get("query") for item in es.queries]
    assert {"match_all": {}} in queries
    assert any("multi_match" in str(query) for query in queries)
    assert any("hubtype.keyword" in str(query) for query in queries)
    md = evidence_markdown(pack)
    assert "Elasticsearch" in md
    assert "26844" in md.replace("\u00a0", "").replace(" ", "")
    assert "Соцсети" in md
    prompt = user_prompt(
        "что пишут",
        "Обзор",
        ["theme_a"],
        "docs",
        evidence_md=md,
    )
    assert "Elasticsearch" in prompt
    assert "не всю тему" in prompt.lower() or "не равны всей теме" in prompt.lower()


def test_es_bool_query_shapes():
    assert es_bool_query() == {"match_all": {}}
    q = es_bool_query({"tone": "neg"}, "банк")
    assert q["bool"]["must"][0]["multi_match"]["query"] == "банк"
    assert {"term": {"toneMark": -1}} in q["bool"]["filter"]


def test_stratify_keeps_hubtypes():
    items = []
    for i, hub in enumerate(["Соцсети", "Блоги", "Форумы", "Соцсети", "Блоги"]):
        items.append(
            {
                "hash": f"h{i}",
                "text": f"t{i}",
                "score": 0.9 - i * 0.1,
                "source": {"hubtype": hub, "audienceCount": 1000 - i * 10, "duplicateCount": i},
            }
        )
    picked = stratify_citations(items, limit=4, per_hubtype=1)
    types = {item["source"]["hubtype"] for item in picked}
    assert types == {"Соцсети", "Блоги", "Форумы"}


def test_coverage_label_has_corpus_not_percent():
    pack = {
        "corpus": {"count": 26844},
        "query_hits": {"count": 4321},
    }
    cov = coverage_payload(pack, 24, 12, 50)
    assert "26844" in cov["label"].replace("\u00a0", "").replace(" ", "")
    assert "4321" in cov["label"].replace("\u00a0", "").replace(" ", "")
    assert "12/24" in cov["label"]
    assert "%" not in cov["label"]
    assert "Elasticsearch" in cov["note"]
