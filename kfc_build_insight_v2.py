#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Сборка усиленного отчёта: инсайты, смысловые группы, авторы, волны, контекст, экраны.

Порядок разделов отвечает на вопрос «что с брендом происходит»:
  1. Главное в числах;
  2. Инсайты (16 карточек по приоритету);
  3. Карта боли: смысловые группы жалоб;
  4. Кто говорит: авторы и площадки;
  5. Как расходятся волны;
  6. Цена бездействия;
  7. Что делать: план на 30, 90 дней и год;
  8. Что происходило вокруг бренда (внешний контекст);
  9. Как это выглядит в Tellscope (экраны системы);
 10. Доказательная база (подробные расчёты) и приложение с примерами и ссылками.

После сборки документ оформляется: числа и выводы выделяются жирным, ссылки становятся
кликабельными.

Запуск:  venv_py312_clean/bin/python -u kfc_build_insight_v2.py [--apply]
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
import kfc_blocks2 as B2  # noqa: E402
import kfc_insight_blocks as B  # noqa: E402
import kfc_brief  # noqa: E402
import kfc_docx_polish as P  # noqa: E402

TITLE = "Межгодовой отчёт по теме KFC 2024–2026: карта боли и очаги напряжения"
SUBTITLE = ("Смысловые группы жалоб, авторы, волны инцидентов, внешний контекст "
            "и план действий")
BRIEF_TITLE = "KFC 2024–2026: краткая версия для конференции"
BRIEF_SUBTITLE = ("Главное в числах, карта боли, авторы, волны и план действий")


def main() -> None:
    apply = "--apply" in sys.argv
    started = time.time()
    ctx = X.Ctx()
    os.makedirs(X.CHART_DIR, exist_ok=True)

    print("=== данные отчёта")
    data = X.Data()
    print("   месячных отчётов: %d | тем: %d" % (len(data.summaries), len(data.themes)))
    data.events = X.collect_events(data)
    data.campaigns = X.collect_campaigns(data)
    data.comparison_read = X.collect_comparison(data)
    data.appendix = X.collect_appendix(data)

    facts = B.load("facts")
    insights_old = B.load("insights")
    insights_new = B2.load("insights3")
    clusters = B2.load("clusters_final")
    authors = B2.load("author_core")
    print("=== новая аналитика: инсайтов %d, смысловых групп %d, авторов %s"
          % (len(insights_new.get("insights") or []), len(clusters.get("clusters") or []),
             (authors.get("global") or {}).get("authors_total")))

    sections = [
        B2.polish(B2.build_headline_block(facts, insights_new)),
        B2.polish(B2.build_insights_block(insights_new)),
        B2.polish(B2.build_pain_map_block(facts, ctx)),
        B2.polish(B2.build_geo_block(ctx)),
        B2.polish(B2.build_platform_block(ctx)),
        B2.polish(B2.build_campaigns_block(ctx)),
        B2.polish(B2.build_authors_block(facts, ctx)),
        B2.polish(B2.build_waves_block(ctx)),
        B.polish(B.build_cost_block(facts, ctx)),
        B.polish(B.build_plan_block(insights_old)),
        B2.polish(B2.build_external_block()),
        B2.polish(B2.build_screens_block(ctx)),
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
    for problem in problems[:15]:
        print("   !! %s" % problem)
    used = {cid for section in sections for cid in (section.get("chart_ids") or [])}
    print("   рисунков построено %d, в разделах %d" % (len(ctx.charts), len(used)))
    print("   разделы: %s" % " | ".join(str(s.get("heading")) for s in sections))
    print("   таблиц: %d" % sum(len(s.get("tables") or []) for s in sections))

    if not apply:
        print("(сухой прогон, %.0f с)" % (time.time() - started))
        return

    built = X.build_one(TITLE, SUBTITLE, sections, data, ctx)
    docx_path = built[0]
    print("=== оформление документа")
    P.polish(docx_path)
    print("=== краткая версия для конференции")
    brief_sections = kfc_brief.build_brief_sections(facts, insights_new, ctx)
    brief = X.build_one(BRIEF_TITLE, BRIEF_SUBTITLE, brief_sections, data, ctx)
    P.polish(brief[0])
    print("   краткая версия: разделов %d, таблиц %d, рисунков %d"
          % (len(brief_sections), brief[3]["tables"], brief[3]["charts"]))
    pdf_note = ""
    if os.path.exists("/usr/bin/soffice"):
        import subprocess
        for path in (docx_path, brief[0]):
            if not os.path.isfile(path):
                continue
            try:
                subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir",
                                os.path.dirname(path), path],
                               check=False, timeout=900, capture_output=True)
                pdf_note = " PDF пересобран из оформленных документов"
            except Exception as exc:  # noqa: BLE001
                pdf_note = " PDF из документа не собрался: %s" % str(exc)[:80]
    with io.open("/tmp/kfc_insight_v2_built.json", "w", encoding="utf-8") as fh:
        json.dump({"files": [built[0], built[1], built[2], brief[0], brief[1], brief[2]],
                   "counts": built[3], "brief_counts": brief[3],
                   "problems": problems, "sections": [s.get("heading") for s in sections],
                   "brief_sections": [s.get("heading") for s in brief_sections],
                   "tables": sum(len(s.get("tables") or []) for s in sections)},
                  fh, ensure_ascii=False, indent=1)
    print("готово за %.0f с%s" % (time.time() - started, pdf_note))


if __name__ == "__main__":
    main()
