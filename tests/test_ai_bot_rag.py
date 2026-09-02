from mlops.ai_bot_rag import (
    docs_markdown,
    es_filter_clauses,
    public_sources,
    rrf_merge,
    source_from_es,
    tone_label,
    user_prompt,
    normalize_instructions,
)


def test_tone_and_filters():
    assert tone_label(-1) == "негатив"
    assert tone_label(1) == "позитив"
    assert es_filter_clauses({"tone": "neg"}) == [{"term": {"toneMark": -1}}]
    assert "hubtype.keyword" in es_filter_clauses({"channel": "smi"})[0]["terms"]


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
