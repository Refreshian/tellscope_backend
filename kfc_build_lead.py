#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Сборка сжатого аналитического документа по бренду KFC / Rostic's.

Несколько страниц: главное в числах, выводы, из чего состоит негатив, где именно болит,
кто создаёт негатив, скорость волн и кампании, что делать.

Запуск:  venv_py312_clean/bin/python -u kfc_build_lead.py [--apply]
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
import kfc_lead_review as R  # noqa: E402
import kfc_docx_polish as P  # noqa: E402

TITLE = "KFC / Rostic's: карта боли бренда по 2,9 млн сообщений"
SUBTITLE = "Что видно в данных за 2024–2026 годы: боли, города, авторы, волны и действия"


def main() -> None:
    apply = "--apply" in sys.argv
    started = time.time()
    ctx = X.Ctx()
    os.makedirs(X.CHART_DIR, exist_ok=True)

    print("=== данные")
    data = X.Data()
    facts = B.load("facts")
    insights = B2.load("insights3")
    sections = R.build_sections(facts, insights, ctx)
    problems = X.audit_sections(sections)
    sections = X.polish_sections(sections)
    problems += ["после косметики: " + p for p in X.audit_sections(sections)]
    print("   разделов: %d, таблиц: %d, рисунков: %d"
          % (len(sections), sum(len(s.get("tables") or []) for s in sections),
             sum(len(s.get("chart_ids") or []) for s in sections)))
    print("=== проверка текстов: замечаний %d" % len(problems))
    for problem in problems[:10]:
        print("   !! %s" % problem)

    if not apply:
        print("(сухой прогон, %.0f с)" % (time.time() - started))
        return
    built = X.build_one(TITLE, SUBTITLE, sections, data, ctx)
    P.polish(built[0])
    print("собран: %s" % os.path.basename(built[0]))
    print("        %s" % os.path.basename(built[1]))
    if os.path.exists("/usr/bin/soffice"):
        import subprocess
        try:
            subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir",
                            os.path.dirname(built[0]), built[0]], check=False, timeout=900,
                           capture_output=True)
            print("        PDF пересобран из оформленного документа")
        except Exception as exc:  # noqa: BLE001
            print("        PDF не собрался: %s" % str(exc)[:80])
    with io.open("/tmp/kfc_lead_built.json", "w", encoding="utf-8") as fh:
        json.dump({"files": [built[0], built[1], built[2]], "counts": built[3],
                   "problems": problems, "sections": [s.get("heading") for s in sections]},
                  fh, ensure_ascii=False, indent=1)
    print("готово за %.0f с" % (time.time() - started))


if __name__ == "__main__":
    main()
