from mlops.eval_grounding import invented_numbers
from mlops.eval_plan import validate_plan
from mlops.lock import load_lock
from mlops.prompts import list_prompts, render_prompt
from mlops.timeutil import is_placeholder_ts


def test_default_plan_shape():
    plan = {
        "filters": {},
        "analysis_needed": True,
        "report_structure": [
            {"type": "title"},
            {"type": "section", "title": "Введение", "content_source": "user_query"},
            {"type": "overall_stats", "title": "Статистика"},
        ],
        "charts": [{"id": "sex", "type": "pie", "group_by": "sex"}],
    }
    assert validate_plan(plan) == []


def test_plan_rejects_empty_structure():
    assert validate_plan({"report_structure": []})
    assert validate_plan({})
    assert validate_plan({"report_structure": [{"type": "unknown"}]})


def test_prompt_registry_has_smart_agent_and_dashboards():
    ids = {item["id"] for item in list_prompts()}
    for needed in (
        "smart_agent_plan_v1",
        "topic_label_v1",
        "dashboard_qa_raw_v1",
        "dashboard_qa_tonality_v1",
        "dashboard_qa_graph_v1",
        "dashboard_qa_media_v1",
        "dashboard_qa_voice_v1",
        "dashboard_qa_bot_v1",
    ):
        assert needed in ids
    text = render_prompt("smart_agent_plan_v1", columns_info=" (cols)", target_language="русском")
    assert "русском" in text
    assert "{target_language}" not in text
    prompts = load_lock().get("prompts") or {}
    assert prompts.get("smart_agent_plan") == "smart_agent_plan_v1"
    assert prompts.get("dashboard_qa_raw") == "dashboard_qa_raw_v1"
    raw = render_prompt("dashboard_qa_raw_v1")
    assert "Не выдумывай цифры" in raw
    assert "{текст}" in raw


def test_dashboard_answers_must_not_invent_numbers():
    context = "Всего 1200 сообщений. Негатив 30%. Источник telegram.org — 400 постов."
    ok = "По срезу 1200 сообщений, негатив 30%, telegram.org даёт 400 постов."
    assert invented_numbers(ok, context) == []
    invented = "По срезу 5000 сообщений негатив вырос до 80%."
    extra = invented_numbers(invented, context)
    assert "5000" in extra
    assert "80%" in extra or "80" in extra
    assert invented_numbers("Данных недостаточно для оценки.", context) == []
    assert invented_numbers("Индекс цитирования 99.", context)
    assert invented_numbers("Около 30% негатива.", "негатив 30%") == []
    assert invented_numbers("", "1200") == []


def test_placeholder_timestamp_is_not_a_live_heartbeat():
    assert is_placeholder_ts("2020-01-01T00:00:00")
    assert is_placeholder_ts("")
    assert not is_placeholder_ts("2026-08-31T17:00:00")
    assert not is_placeholder_ts("2026-04-04T20:07:40")
