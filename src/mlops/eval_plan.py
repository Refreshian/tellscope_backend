"""CPU-only checks for smart-agent JSON plans. No GPU."""
from __future__ import annotations

ALLOWED_SECTION_TYPES = {
    "title",
    "section",
    "overall_stats",
    "thematic_breakdown",
    "sentiment_overview",
    "custom_chart_section",
}
ALLOWED_CHART_TYPES = {"bar", "pie"}


def validate_plan(plan: dict) -> list[str]:
    errors: list[str] = []
    if not isinstance(plan, dict):
        return ["plan is not an object"]
    structure = plan.get("report_structure")
    if not isinstance(structure, list) or not structure:
        errors.append("missing report_structure")
        return errors
    for i, section in enumerate(structure):
        if not isinstance(section, dict):
            errors.append(f"report_structure[{i}] is not an object")
            continue
        kind = section.get("type")
        if kind not in ALLOWED_SECTION_TYPES:
            errors.append(f"report_structure[{i}] unknown type: {kind}")
    charts = plan.get("charts")
    if charts is not None:
        if not isinstance(charts, list):
            errors.append("charts is not a list")
        else:
            for i, chart in enumerate(charts):
                if not isinstance(chart, dict):
                    errors.append(f"charts[{i}] is not an object")
                    continue
                if chart.get("type") not in ALLOWED_CHART_TYPES:
                    errors.append(f"charts[{i}] unknown type: {chart.get('type')}")
    return errors
