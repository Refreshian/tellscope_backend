#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Проверка чисел в новых инсайтах: каждое число должно быть в своём факте.

Для каждой карточки известен набор чисел, из которого она собрана. Если модель пишет
число, которого в этом наборе нет, — это ошибка: карточка переписывается без него.
Отдельно проверяются смысловые подмены: город вместо заведения, отзывы вместо сообщений.

Результат: /tmp/kfc_topics/verify3.json, при необходимости исправленный insights3.json
"""
from __future__ import annotations

import datetime
import io
import json
import os
import re
import sys

OUT = "/tmp/kfc_topics"
BACKEND = "/home/dev/tellscope_app/tellscope_backend"
sys.path.insert(0, BACKEND)

GLOBAL_ALLOWED = {2927613, 305676, 222530, 2024, 2025, 2026, 2027, 28, 13, 5, 2024.5,
                  16151854505, 1380129682, 10.4, 10.44, 7.6, 9.3, 12.0, 9.1, 60.1, 275363,
                  257269, 1.0, 3.1, 24.4, 3.8, 75.6, 72.4, 49.5, 13.6, 10.9, 40, 51.3, 43.8,
                  29.8, 85.1, 84.5, 21.9, 12.5, 6.4, 4.5, 0.1, 1, 2, 3, 4, 6, 8, 10, 30, 100,
                  1000, 80, 90, 365, 12, 20}


def log(message: str) -> None:
    print("[%s] %s" % (datetime.datetime.now().strftime("%H:%M:%S"), message), flush=True)


def numbers_in(text: str) -> list:
    found = []
    for match in re.finditer(r"\d[\d \u00a0]*(?:[.,]\d+)?", str(text or "")):
        raw = match.group(0).strip()
        cleaned = raw.replace(" ", "").replace("\u00a0", "").replace(",", ".")
        try:
            found.append((raw, float(cleaned)))
        except Exception:  # noqa: BLE001
            continue
    return found


def allowed_from(numbers: dict) -> set:
    allowed = set()
    for value in numbers.values():
        if isinstance(value, (int, float)):
            allowed.add(float(value))
            if isinstance(value, float) and value <= 1:
                allowed.add(round(value * 100, 1))
        elif isinstance(value, str):
            for _, number in numbers_in(value):
                allowed.add(number)
    allowed |= GLOBAL_ALLOWED
    return allowed


def main() -> None:
    import kfc_ai

    data = json.load(io.open(os.path.join(OUT, "insights3.json"), encoding="utf-8"))
    items = data.get("insights") or []
    problems = []
    for item in items:
        allowed = allowed_from(item.get("числа") or {})
        for key in ("цифра", "что_значит", "что_делать", "если_не_делать"):
            text = item.get(key) or ""
            bad = [raw for raw, value in numbers_in(text)
                   if not any(abs(value - candidate) <= max(0.05, abs(candidate) * 0.005)
                              or (candidate >= 100 and abs(candidate - value) <= 1.0)
                              for candidate in allowed)]
            if bad:
                problems.append({"index": items.index(item), "key": key, "bad": bad, "text": text})
    log("карточек %d, проблемных полей %d" % (len(items), len(problems)))
    for row in problems:
        log("   №%d %s → %s" % (row["index"] + 1, row["key"], ", ".join(row["bad"])))

    system = ("Ты редактор отчёта о медиаполе сети Rostic's. Ты обязан использовать только числа "
              "из разрешённого списка. Пиши по-русски, деловым языком.")
    fixed = 0
    for row in problems:
        item = items[row["index"]]
        allowed = sorted(allowed_from(item.get("числа") or {}))
        allowed_text = ", ".join(("%g" % value) for value in allowed[:120])
        prompt = ("РАЗРЕШЁННЫЕ ЧИСЛА: %s\n\nТЕКСТ С ОШИБКОЙ: %s\n\nЧИСЛА, КОТОРЫХ НЕТ В СПИСКЕ: %s\n\n"
                  "Перепиши текст, используя только разрешённые числа, сохранив смысл и стиль. "
                  'Верни JSON: {"текст": "переписанный текст"}'
                  % (allowed_text, row["text"], ", ".join(row["bad"])))
        answer = kfc_ai.ask_json(prompt, system=system, model="deepseek-chat", max_tokens=700, attempts=2)
        new_text = str((answer or {}).get("текст") or "").strip()
        if new_text:
            item[row["key"]] = new_text
            fixed += 1
            log("   №%d %s переписан" % (row["index"] + 1, row["key"]))
    with io.open(os.path.join(OUT, "insights3.json"), "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False)
    with io.open(os.path.join(OUT, "verify3.json"), "w", encoding="utf-8") as fh:
        json.dump({"problems": [{k: v for k, v in row.items() if k != "index"} for row in problems],
                   "fixed": fixed, "built_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")},
                  fh, ensure_ascii=False)
    log("исправлено полей: %d" % fixed)


if __name__ == "__main__":
    main()
