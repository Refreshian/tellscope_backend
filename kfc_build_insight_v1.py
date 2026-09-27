#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Сборка управленческой версии межгодового отчёта KFC 2024–2026.

Порядок разделов отвечает на вопрос заказчика «зачем этот отчёт»:

  1. Зачем этот отчёт — что решается по нему;
  2. Главное: семь инсайтов — цифра, смысл, действие, цена бездействия;
  3. Карта очагов напряжения — где именно болит, по городам, ресторанам, жалобам, поводам;
  4. Цена бездействия — вес негатива и три сценария;
  5. Что делать: план на 30, 90 дней и год;
  6. Доказательная база — подробные расчёты по темам, тональности, авторам, кампаниям,
     конкурентам, рискам и языку бренда;
  7. Приложение — методика и примеры сообщений.

Запуск:  venv_py312_clean/bin/python -u kfc_build_insight_v1.py [--apply]
"""
from __future__ import annotations

import io
import json
import os
import sys
import time

BACKEND = "/home/dev/tellscope_app/tellscope_backend"
sys.path.insert(0, BACKEND)

import kfc_build_crossyear_v2 as X  # noqa: E402
import kfc_insight_blocks as B  # noqa: E402

TITLE = "Межгодовой отчёт по теме KFC 2024–2026: очаги напряжения"
SUBTITLE = ("Зачем отчёт, семь инсайтов, карта очагов напряжения, цена бездействия "
            "и план на 30, 90 дней и год")
BRIEF_TITLE = "KFC 2024–2026: краткая версия для конференции"
BRIEF_SUBTITLE = "Зачем отчёт, семь выводов, очаги напряжения и план действий"
OLD_SUMMARY = "2024-2026_расширенный_summary.json"
OLD_SUMMARY_BACKUP = ("_черновики_расширенного_20260917_2140/"
                      "2024-2026_расширенный_summary.json")


def keep_summary(new_name: str) -> None:
    """Сборщик платформы пишет сводку под одним именем: сохраняем её и возвращаем прежнюю."""
    produced = os.path.join(X.OUT_DIR, OLD_SUMMARY)
    if not os.path.isfile(produced):
        return
    with io.open(produced, encoding="utf-8") as fh:
        payload = json.load(fh)
    with io.open(os.path.join(X.OUT_DIR, new_name), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
    backup = os.path.join(X.OUT_DIR, OLD_SUMMARY_BACKUP)
    if os.path.isfile(backup):
        with io.open(backup, encoding="utf-8") as src:
            text = src.read()
        with io.open(produced, "w", encoding="utf-8") as dst:
            dst.write(text)
    print("   сводка сохранена как %s, прежняя сводка возвращена на место" % new_name)


def main() -> None:
    apply = "--apply" in sys.argv
    started = time.time()
    ctx = X.Ctx()
    os.makedirs(X.CHART_DIR, exist_ok=True)

    print("=== данные отчёта")
    data = X.Data()
    print("   месячных отчётов: %d | тем: %d" % (len(data.summaries), len(data.themes)))
    print("=== разбор поводов, кампаний, сравнений")
    data.events = X.collect_events(data)
    data.campaigns = X.collect_campaigns(data)
    data.comparison_read = X.collect_comparison(data)
    data.appendix = X.collect_appendix(data)
    print("   поводов %d, кампаний %d, примеров %d"
          % (len(data.events), len(data.campaigns), len(data.appendix)))

    facts = B.load("facts")
    insights = B.load("insights")
    print("=== управленческая часть: городов %d, заведений %d, инсайтов %d, план: %s"
          % (len(facts.get("cities") or []), len(facts.get("restaurants") or []),
             len(insights.get("insights") or []),
             ", ".join("%s — %d" % (k, len(v or []))
                       for k, v in ((insights.get("plan") or {}).get("план") or {}).items()) or "нет"))

    sections = [
        B.polish(B.build_why_block(facts, insights)),
        B.polish(B.build_insights_block(facts, insights, ctx)),
        B.polish(B.build_hotspots_block(facts, ctx)),
        B.polish(B.build_cost_block(facts, ctx)),
        B.polish(B.build_plan_block(insights)),
        X.build_brand_talk_block(data, ctx),
        X.build_tonality_block(data, ctx),
        X.build_events_block(data, ctx),
        X.build_authors_block(data, ctx),
        X.build_campaigns_block(data, ctx),
        X.build_competitors_block(data, ctx),
        X.build_risk_block(data, ctx),
        X.build_language_block(data, ctx),
        X.build_appendix_block(data, ctx),
    ]
    problems = X.audit_sections(sections)
    sections = X.polish_sections(sections)
    problems += ["после косметики: " + p for p in X.audit_sections(sections)]
    print("=== проверка текстов: замечаний %d" % len(problems))
    for problem in problems[:20]:
        print("   !! %s" % problem)
    used = {cid for section in sections for cid in (section.get("chart_ids") or [])}
    print("   графиков построено %d, в разделах %d" % (len(ctx.charts), len(used)))
    print("   разделы: %s" % " | ".join(str(s.get("heading")) for s in sections))

    if not apply:
        print("(сухой прогон: документы не собираются, %.0f с)" % (time.time() - started))
        return
    print("=== краткая версия для конференции")
    brief = X.build_one(BRIEF_TITLE, BRIEF_SUBTITLE,
                        [B.polish(section) for section in B.build_brief_sections(facts, insights, ctx)],
                        data, ctx)
    keep_summary("KFC_2024-2026_краткая_версия_summary.json")
    print("=== полный отчёт")
    built = X.build_one(TITLE, SUBTITLE, sections, data, ctx)
    keep_summary("KFC_2024-2026_очаги_напряжения_summary.json")
    json.dump({"files": [built[0], built[1], built[2], brief[0], brief[1], brief[2]],
               "counts": built[3], "brief_counts": brief[3], "problems": problems},
              io.open("/tmp/kfc_insight_v1_built.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("готово за %.0f с" % (time.time() - started))


if __name__ == "__main__":
    main()
