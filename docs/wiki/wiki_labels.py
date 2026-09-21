#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Выгрузка подписей интерфейса из исходников фронтенда.

Нужна для актуализации документации: подписи в интерфейсе меняются — этот скрипт
показывает, что сейчас написано на каждом экране, чтобы страницы wiki совпадали
с реальным интерфейсом.

Результат: /opt/tellscope-wiki/labels.txt (плюс печать в консоль)
"""
from __future__ import annotations

import io
import os
import re
import subprocess

FRONTEND = "/home/dev/tellscope_app/tellscope_frontend"
OUT = "/opt/tellscope-wiki/labels.txt"
SCREENS = {
    "Тональный ландшафт": "src/components/screens/user-tonality/UserTonality.jsx",
    "Граф информации": "src/components/screens/information/Information.jsx",
    "СМИ": "src/components/screens/media-rating/MediaRating.jsx",
    "Голос клиента": "src/components/screens/voice-of-customer/VoiceOfCustomer.jsx",
    "Наборы данных": "src/components/screens/data-set-page/DataSetPage.jsx",
    "Проверка тональности": "src/components/screens/data-set-page/DataSetPage.jsx",
    "Связи авторов": "src/components/GraphVisualization/GraphAnalysis.jsx",
    "Кластеризация": "src/components/screens/clustering/Clustering.jsx",
    "Smart Agent": "src/components/screens/smart-agent/SmartAgent.jsx",
    "Agent Mode": "src/components/screens/agent-mode/AgentMode.jsx",
    "ИИ анализ": "src/components/screens/tables/ai-analytics-page/AiAnalyticsPage.jsx",
    "Анализ тем": "src/components/screens/tables/ai-analytics-page/analysis-of-themes/AnalysisOfThemesPage.jsx",
    "ОИВ рейтинг": "src/components/screens/mosinform-rating/MosinformRating.jsx",
    "Конкуренты": "src/components/screens/competitive/Competitive.jsx",
    "Центр ИИ-задач": "src/components/screens/harness/Harness.jsx",
    "Dify": "src/components/screens/dify-constructor/DifyConstructor.jsx",
    "PR-кампании": "src/components/screen/pr-campaign/PrCampaign.jsx",
    "Админ": "src/components/screens/admin/AdminPage.jsx",
    "Меню": "src/data/menuPage.data.js",
}


def labels_of(path: str) -> list:
    full = os.path.join(FRONTEND, path)
    if not os.path.isfile(full):
        return []
    text = io.open(full, encoding="utf-8", errors="replace").read()
    found = re.findall(r"'([^']{6,80})'|\"([^\"]{6,80})\"", text)
    out = []
    for left, right in found:
        value = (left or right).strip()
        if re.search(r"[А-Яа-яЁё]", value) and value not in out:
            out.append(value)
    return out[:28]


def main() -> None:
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    lines = ["Подписи интерфейса Tellscope (для сверки с документацией)", ""]
    for title, path in SCREENS.items():
        items = labels_of(path)
        lines.append("== %s (%s)" % (title, path))
        if not items:
            lines.append("   файл не найден")
        for item in items:
            lines.append("   - " + item)
        lines.append("")
    text = "\n".join(lines)
    with io.open(OUT, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text[:1500])
    print("... всего знаков: %d, файл: %s" % (len(text), OUT))


if __name__ == "__main__":
    main()
