#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Управленческая часть отчёта: «зачем», 7 инсайтов, план 30/90/365.

Модель получает только посчитанные факты и обязана опираться на их числа.
Ни одного нового числа она добавить не может: если числа нет в фактах — его нельзя писать.

Результат: /tmp/kfc_hotspots/insights.json
"""
from __future__ import annotations

import datetime
import io
import json
import os
import re
import time
import urllib.error
import urllib.request

CACHE = "/tmp/kfc_hotspots"
GEN_URL = "http://127.0.0.1:8000/v1/chat/completions"
GEN_MODEL = "Qwen/Qwen3-32B-FP8"


def log(message: str) -> None:
    print("[%s] %s" % (datetime.datetime.now().strftime("%H:%M:%S"), message), flush=True)


def load(name: str):
    with io.open(os.path.join(CACHE, name + ".json"), encoding="utf-8") as fh:
        return json.load(fh)


def save(name: str, value) -> None:
    with io.open(os.path.join(CACHE, name + ".json"), "w", encoding="utf-8") as fh:
        json.dump(value, fh, ensure_ascii=False)


def ask(prompt: str, system: str = "", max_tokens: int = 3000, temperature: float = 0.35) -> str:
    body = json.dumps({
        "model": GEN_MODEL,
        "messages": ([{"role": "system", "content": system}] if system else [])
                    + [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": temperature,
    }).encode("utf-8")
    request = urllib.request.Request(GEN_URL, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:400]
        log("модель отказала: %s, запрос %d знаков, ответ: %s" % (exc.code, len(prompt), detail))
        # окно модели 8192 токена: если не влезло, пробуем с урезанным запросом
        short = prompt[:max(1200, len(prompt) // 2)]
        request = urllib.request.Request(GEN_URL,
                                         data=json.dumps({
                                             "model": GEN_MODEL,
                                             "messages": ([{"role": "system", "content": system}] if system else [])
                                                         + [{"role": "user", "content": short}],
                                             "max_tokens": min(max_tokens, 1500), "temperature": temperature,
                                         }).encode("utf-8"),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=900) as response:
            payload = json.loads(response.read().decode("utf-8"))
    return (payload["choices"][0]["message"]["content"] or "").strip()


def json_from(text: str):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        return None
    try:
        return json.loads(text[start:end + 1])
    except Exception:  # noqa: BLE001
        return None


def digest(facts: dict) -> str:
    """Короткая выжимка фактов: у модели окно 8 тысяч токенов, лишнее не влезает."""
    totals = facts["totals"]
    lines = []
    lines.append("Период: %s — %s (%d месяцев)." % (facts["period"]["from"], facts["period"]["to"],
                                                   facts["period"]["months"]))
    lines.append("Всего сообщений %d, негативных %d (%s%%), позитивных %d (%s%%)."
                 % (totals["messages"], totals["negative"], round(totals["negative_share"] * 100, 1),
                    totals["positive"], round(totals["positive"] / float(totals["messages"]) * 100, 1)))
    lines.append("Аудиторные контакты негативных сообщений: %d, из них медийная аудитория %d."
                 % (totals["negative_reach"], totals["negative_media_reach"]))
    lines.append("По годам: " + "; ".join(
        "%s — %d сообщений, доля негатива %s%%" % (year, row["total"], round(row["negative_share"] * 100, 1))
        for year, row in sorted(facts["years"].items())))
    lines.append("Площадки (негатив): " + "; ".join(
        "%s — %d (%s%% внутри площадки)" % (name, row["negative"], round(row["share"] * 100, 1))
        for name, row in sorted(facts["hubtype"].items(), key=lambda kv: -kv[1]["negative"])[:5]))
    ratings = {k: v for k, v in facts["ratings"].items() if k in ("1", "2", "3", "4", "5")}
    lines.append("Оценки в отзывах: " + "; ".join("%s — %d" % (k, ratings[k]) for k in sorted(ratings)))
    conc = facts["concentration"]
    lines.append("Концентрация: 10 городов дают %s%% негатива, 30 заведений — %s%%, "
                 "5 тем — %s%% негатива по темам; городов с ростом %d, заведений с ростом %d."
                 % (round(conc.get("top10_cities_share", 0) * 100, 1),
                    round(conc.get("top30_restaurants_share", 0) * 100, 1),
                    round(conc.get("top5_themes_share", 0) * 100, 1),
                    conc.get("cities_with_growth", 0), conc.get("restaurants_with_growth", 0)))
    scen = facts["scenarios"]
    forecast = scen.get("forecast") or []
    if forecast:
        first = forecast[0]
        lines.append("Сценарии на %s: без действий %s%%, адресная работа %s%%, системная работа %s%% "
                     "(средняя доля негатива за период %s%%)."
                     % (first["month"], round(first["no_action"] * 100, 1), round(first["targeted"] * 100, 1),
                        round(first["systemic"] * 100, 1), round(scen.get("mean_share", 0) * 100, 1)))
    lines.append("")
    lines.append("Города-очаги (город | негатив | доля негатива | перевес над ожидаемым | рост за квартал):")
    for row in facts["cities"][:8]:
        lines.append("- %s | %d | %s%% | x%s | x%s%s"
                     % (row["city"], row["negative"], round(row["share"] * 100, 1),
                        row.get("excess_norm"), row["growth"],
                        (" | пишут: " + "; ".join((row.get("what_people_say") or [])[:2]))
                        if row.get("what_people_say") else ""))
    lines.append("")
    lines.append("Заведения-очаги (город | негативных из всего | доля | рейтинг | рост | пишут):")
    for row in facts["restaurants"][:8]:
        lines.append("- %s | %d из %d | %s%% | %s | x%s%s"
                     % (row["city"], row["negative"], row["count"], round(row["share"] * 100, 1),
                        row.get("rating"), row["growth"],
                        (" | " + "; ".join((row.get("what_people_say") or [])[:2]))
                        if row.get("what_people_say") else ""))
    lines.append("")
    lines.append("Группы жалоб (негатив за период | рост за квартал):")
    for row in facts["drivers"]:
        lines.append("- %s | %d | x%s" % (row["name"], row["negative"], row["growth"]))
    lines.append("")
    lines.append("Крупнейшие поводы (негатив | пик):")
    for row in facts["stories"][:6]:
        lines.append("- %s | %d | %s" % (row["name"][:70], row["negative"], row.get("peak_month")))
    text = "\n".join(lines)
    return text[:9000]


def normalize_insights(items) -> list:
    """Модель может назвать поле иначе — приводим к одному виду."""
    out = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        def pick(*keys):
            for key in keys:
                value = item.get(key)
                if value:
                    return str(value).strip()
            return ""
        row = {
            "заголовок": pick("заголовок", "title", "heading"),
            "цифра": pick("цифра", "числа", "number"),
            "что_значит": pick("что_значит", "что_значает", "значит", "смысл"),
            "что_делать": pick("что_делать", "действие", "действия"),
            "если_не_делать": pick("если_не_делать", "если_не_действовать", "риск"),
        }
        if row["заголовок"]:
            out.append(row)
    return out


def main() -> None:
    facts = load("facts")
    packed = digest(facts)
    log("выжимка фактов: %d знаков" % len(packed))
    result = load("insights") or {}
    result["digest_len"] = len(packed)
    result["facts_used"] = {"cities": len(facts["cities"]), "restaurants": len(facts["restaurants"])}

    system = ("Ты аналитик, который пишет отчёт для руководства сети фастфуда "
              "(Rostic's, бывший KFC). Пишешь по-русски, простым деловым языком, без жаргона, "
              "без технических терминов. Опираешься только на переданные числа: "
              "новые цифры, даты, названия и факты придумывать запрещено.")

    if not (result.get("why") or {}).get("решения"):
        prompt_why = (
            "Ниже факты по бренду за период. Напиши вводную часть отчёта: зачем он нужен.\n\n"
            "ФАКТЫ:\n%s\n\n"
            "Верни JSON:\n"
            '{"зачем": "3-5 предложений: что показывает отчёт и почему это важно именно сейчас, '
            'с опорой на числа", '
            '"решения": ["решение 1 — что руководство может решить по этому отчёту", "решение 2", "решение 3"], '
            '"в_одном_экране": ["3-4 самых важных числа периода, каждое с коротким пояснением"]}\n'
            "Решения должны быть управленческими (что делать с ресторанами, с сервисом, с кампаниями), "
            "а не «прочитать отчёт»."
            % packed)
        raw = ask(prompt_why, system=system, max_tokens=1400)
        why = json_from(raw) or {}
        log("вводная часть: решений %d" % len(why.get("решения") or []))
        result["why"] = why
        save("insights", result)

    if len(result.get("insights") or []) < 7:
        prompt_insights = (
            "Ниже факты о бренде. Напиши ровно 7 инсайтов — то, что руководство должно понять "
            "и что можно использовать на встрече с командой.\n\n"
            "ФАКТЫ:\n%s\n\n"
            "Требования: каждый инсайт опирается на конкретное число из фактов; никаких новых цифр; "
            "никаких общих слов вроде «важно следить за ситуацией».\n"
            "Верни JSON (имена полей менять нельзя):\n"
            '{"инсайты": [{"заголовок": "короткий заголовок до 9 слов", '
            '"цифра": "главное число инсайта с коротким пояснением", '
            '"что_значит": "2-3 предложения: что это означает для бизнеса", '
            '"что_делать": "1-2 предложения: конкретное действие", '
            '"если_не_делать": "1 предложение: что будет, если не реагировать"}]}\n'
            "Инсайты не должны повторять друг друга: пусть это будут разные стороны картины — "
            "где болит, кто именно болит, что повторяется, что растёт, что компания делает хорошо, "
            "чего стоит ждать в ближайшие месяцы, где скрытый резерв."
            % packed)
        raw = ask(prompt_insights, system=system, max_tokens=2600)
        data = json_from(raw) or {}
        insights = normalize_insights(data.get("инсайты") or data.get("insights") or [])
        log("инсайтов: %d" % len(insights))
        result["insights"] = insights[:7]
        save("insights", result)

    horizons = (result.get("plan") or {}).get("план") or {}
    if not (horizons.get("30_дней") or horizons.get("90_дней")):
        prompt_plan = (
            "Ниже факты и уже сформулированные инсайты. Составь план действий на три горизонта.\n\n"
            "ФАКТЫ:\n%s\n\nИНСАЙТЫ:\n%s\n\n"
            "Верни JSON (имена полей менять нельзя):\n"
            '{"план": {"30_дней": [{"действие": "...", "владелец": "кто именно (маркетинг, PR, '
            'операционный блок, служба качества, клиентский сервис, IT)", "срок": "например, 2 недели", '
            '"kpi": "по какому числу видно, что получилось"}], "90_дней": [...], "год": [...]}, '
            '"как_мерить": ["2-3 показателя, по которым видно, что работа идёт"]}\n'
            "В каждом горизонте 3-5 действий. Действия конкретные и выполнимые, "
            "с привязкой к очагам из фактов (города, заведения, типы жалоб). "
            "KPI — из уже упомянутых чисел или их долей, без выдуманных значений."
            % (packed,
               json.dumps(result.get("insights") or [], ensure_ascii=False)[:3000]))
        raw = ask(prompt_plan, system=system, max_tokens=2600)
        plan = json_from(raw) or {}
        log("план: 30 дней %d, 90 дней %d, год %d" % (
            len((plan.get("план") or {}).get("30_дней") or []),
            len((plan.get("план") or {}).get("90_дней") or []),
            len((plan.get("план") or {}).get("год") or [])))
        result["plan"] = plan
        save("insights", result)
    result["built_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    save("insights", result)
    log("готово")


if __name__ == "__main__":
    main()
