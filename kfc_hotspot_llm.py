#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Разбор очагов напряжения локальными моделями.

По каждому очагу (город, заведение, тема, повод) собирается выборка самых заметных сообщений,
быстрая модель читает, на что именно жалуются и что называют причиной, большая модель формулирует,
что это значит для бизнеса и что делать — строго по посчитанным числам, без новых цифр.

Результат: /tmp/kfc_hotspots/llm_hotspots.json (кэш по каждому очагу — можно перезапускать)
Запуск:  nohup venv_py312_clean/bin/python -u kfc_hotspot_llm.py > /tmp/kfc_hotspot_llm.log 2>&1 &
"""
from __future__ import annotations

import datetime
import io
import json
import os
import re
import time
import urllib.request

from elasticsearch import Elasticsearch

INDEX = "kfc_13.05.2024-22.09.2026"
CACHE = "/tmp/kfc_hotspots"
MSK = datetime.timezone(datetime.timedelta(hours=3))
LO_TS = int(datetime.datetime(2024, 5, 13, tzinfo=MSK).timestamp())
HI_TS = int(datetime.datetime(2026, 8, 31, 23, 59, 59, tzinfo=MSK).timestamp())

FAST_URL = "http://127.0.0.1:8001/v1/chat/completions"
FAST_MODEL = "qwen3-4b-fast"
GEN_URL = "http://127.0.0.1:8000/v1/chat/completions"
GEN_MODEL = "Qwen/Qwen3-32B-FP8"


def log(message: str) -> None:
    print("[%s] %s" % (datetime.datetime.now().strftime("%H:%M:%S"), message), flush=True)


def load(name: str):
    path = os.path.join(CACHE, name + ".json")
    if not os.path.isfile(path):
        return {}
    with io.open(path, encoding="utf-8") as fh:
        return json.load(fh)


def save(name: str, value) -> None:
    with io.open(os.path.join(CACHE, name + ".json"), "w", encoding="utf-8") as fh:
        json.dump(value, fh, ensure_ascii=False)


def ask(url: str, model: str, prompt: str, system: str = "", max_tokens: int = 1200,
        temperature: float = 0.2) -> str:
    body = json.dumps({
        "model": model,
        "messages": ([{"role": "system", "content": system}] if system else [])
                    + [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": temperature,
    }).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
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


def sample_messages(es, must, limit: int = 30) -> list:
    query = {"bool": {"must": must, "filter": [{"range": {"timeCreate": {"gte": LO_TS, "lte": HI_TS}}}]}}
    body = {"size": limit, "query": query, "_source": ["text", "title", "city", "hub", "url",
                                                       "timeCreate", "review_rating", "audienceCount"],
            "sort": [{"audienceCount": {"order": "desc"}}]}
    try:
        res = es.search(index=INDEX, body=body)
    except Exception as exc:  # noqa: BLE001
        log("выборка не собралась: %s" % str(exc)[:100])
        return []
    out = []
    for hit in res["hits"]["hits"]:
        src = hit["_source"]
        text = (src.get("text") or src.get("title") or "").strip().replace("\n", " ")
        if len(text) < 40:
            continue
        day = ""
        try:
            day = datetime.datetime.fromtimestamp(int(float(src.get("timeCreate") or 0)), MSK).strftime("%d.%m.%Y")
        except Exception:  # noqa: BLE001
            pass
        out.append({"text": text[:400], "city": src.get("city") or "", "hub": src.get("hub") or "",
                    "date": day, "rating": src.get("review_rating") or "",
                    "url": src.get("url") or "", "reach": int(src.get("audienceCount") or 0)})
    return out


def read_what(messages: list, where: str, fast_calls: dict) -> dict:
    """Быстрая модель: на что жалуются и что называют причиной."""
    if not messages:
        return {}
    chunks = []
    for index in range(0, len(messages), 10):
        part = messages[index:index + 10]
        lines = []
        for number, item in enumerate(part, 1):
            lines.append("%d) %s" % (number, item["text"][:260]))
        chunks.append("\n".join(lines))
    complaints = []
    triggers = []
    for chunk in chunks[:4]:
        prompt = (
            "Ниже сообщения клиентов сети фастфуда (Rostic's, бывший KFC) — %s.\n\n%s\n\n"
            "Ответь строго по этим сообщениям, без выдумок. Формат JSON:\n"
            '{"жалобы": ["короткая формулировка", ...], "причина": "что чаще всего называют причиной", '
            '"повтор": "повторяющийся сюжет, если он есть, иначе пустая строка"}\n'
            "Жалобы — 3-6 пунктов, каждое до 12 слов, своими словами, без номеров и без ссылок."
            % (where, chunk))
        raw = ask(FAST_URL, FAST_MODEL, prompt, max_tokens=700, temperature=0.1)
        fast_calls["fast"] = fast_calls.get("fast", 0) + 1
        data = json_from(raw)
        if not data:
            continue
        for item in (data.get("жалобы") or []):
            text = str(item).strip().rstrip(".")
            if text and text not in complaints:
                complaints.append(text)
        if data.get("причина"):
            triggers.append(str(data["причина"]).strip())
        if data.get("повтор"):
            triggers.append(str(data["повтор"]).strip())
    return {"complaints": complaints[:8], "causes": triggers[:4]}


def interpret(fact: dict, reading: dict, fast_calls: dict) -> dict:
    """Большая модель: что это значит и что делать — по уже посчитанным числам."""
    prompt = (
        "Ты пишешь раздел отчёта для руководства сети фастфуда о «очаге напряжения».\n"
        "Данные (используй только эти числа, новых цифр не придумывай):\n"
        "%s\n\n"
        "Что пишут клиенты (разбор выборки сообщений):\n"
        "жалобы: %s\nпричины: %s\n\n"
        "Верни JSON строго такого вида:\n"
        '{"что_значит": "2-3 предложения: почему это важно для бизнеса, с опорой на числа выше", '
        '"что_делать": ["действие 1", "действие 2", "действие 3"], '
        '"если_бездействовать": "1-2 предложения: что будет, если не реагировать"}'
        "\nПиши по-русски, без технических терминов, без выдуманных фактов и цифр, "
        "каждое действие — конкретное и выполнимое."
        % (json.dumps(fact, ensure_ascii=False, indent=1),
           "; ".join(reading.get("complaints") or []) or "нет данных",
           "; ".join(reading.get("causes") or []) or "нет данных"))
    raw = ask(GEN_URL, GEN_MODEL, prompt, max_tokens=1200, temperature=0.3)
    fast_calls["gen"] = fast_calls.get("gen", 0) + 1
    data = json_from(raw)
    if not data:
        return {}
    return {"meaning": str(data.get("что_значит") or "").strip(),
            "actions": [str(a).strip() for a in (data.get("что_делать") or []) if str(a).strip()],
            "if_inaction": str(data.get("если_бездействовать") or "").strip()}


def city_fact(row: dict) -> dict:
    return {
        "тип": "город",
        "город": row["city"],
        "негативных_сообщений": row["negative"],
        "всего_сообщений": row["total"],
        "доля_негатива": "%s%%" % round(row["share"] * 100, 1),
        "ожидалось_при_таких_площадках": row.get("expected"),
        "перевес": "x%s" % row.get("excess_norm"),
        "рост_за_последний_квартал": "x%s" % row["growth"],
        "охват_негатива": row["reach"],
        "топ_темы": [t for t, _ in (row.get("top_themes") or [])[:4]],
        "частые_жалобы": [d for d, _ in (row.get("top_drivers") or [])[:4]],
        "пик": row.get("peak_month"),
        "заведения_с_худшей_картиной": [
            "%s: %d негативных из %d, рейтинг %s" % (r["city"], r["neg"], r["count"], r.get("avg_rating"))
            for r in (row.get("top_restaurants") or [])[:3]],
    }


def restaurant_fact(row: dict) -> dict:
    return {
        "тип": "заведение",
        "город": row["city"],
        "номер_заведения_на_картах": row["id"],
        "негативных_отзывов": row["negative"],
        "всего_отзывов": row["count"],
        "доля_негатива": "%s%%" % round(row["share"] * 100, 1),
        "средний_рейтинг": row.get("avg_rating"),
        "рост_за_последний_квартал": "x%s" % row["growth"],
        "охват": row["reach"],
        "пик": row.get("peak_month"),
    }


def main() -> None:
    ranked = load("ranked")
    if not ranked:
        log("нет ранжирования — сначала kfc_hotspots_rank.py")
        return
    es = Elasticsearch(hosts=["http://localhost:9200"], basic_auth=("elastic", "biz8z5i1w0nLPmEweKgP"),
                       verify_certs=False, request_timeout=600)
    result = load("llm_hotspots") or {"cities": {}, "restaurants": {}, "built_at": ""}
    facts = {"fast": 0, "gen": 0}
    started = time.time()

    cities = [row for row in (ranked.get("city_ranked") or []) if row.get("city", "").strip()]
    for row in cities[:12]:
        city = row["city"]
        if city in result["cities"] and result["cities"][city].get("meaning"):
            continue
        messages = sample_messages(es, [{"term": {"toneMark": -1}}, {"term": {"city": city}}], limit=30)
        reading = read_what(messages, "город %s" % city, facts)
        fact = city_fact(row)
        verdict = interpret(fact, reading, facts)
        result["cities"][city] = {"fact": fact, "reading": reading, "verdict": verdict,
                                  "samples": messages[:3], "index": row["index"]}
        log("город %s: жалоб %d, действий %d" % (city, len(reading.get("complaints") or []),
                                                 len(verdict.get("actions") or [])))
        save("llm_hotspots", result)

    for row in (ranked.get("restaurants_ranked") or [])[:20]:
        key = row["id"]
        if key in result["restaurants"] and result["restaurants"][key].get("meaning"):
            continue
        samples = [{"text": s.get("text", ""), "hub": s.get("hub", ""), "date": s.get("date", ""),
                    "rating": s.get("rating", ""), "url": s.get("url", ""),
                    "reach": s.get("reach", 0)} for s in (row.get("samples") or [])]
        if not samples:
            messages = sample_messages(es, [{"term": {"toneMark": -1}},
                                            {"match_phrase": {"url": row["id"]}}], limit=20)
            samples = messages
        reading = read_what(samples, "заведение в городе %s" % row["city"], facts)
        fact = restaurant_fact(row)
        verdict = interpret(fact, reading, facts)
        result["restaurants"][key] = {"fact": fact, "reading": reading, "verdict": verdict,
                                      "samples": samples[:3], "score": row["score"]}
        log("заведение %s (%s): жалоб %d" % (key, row["city"], len(reading.get("complaints") or [])))
        save("llm_hotspots", result)

    result["built_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    result["stats"] = {"fast_calls": facts.get("fast", 0), "gen_calls": facts.get("gen", 0),
                       "seconds": round(time.time() - started, 1),
                       "cities": len(result["cities"]), "restaurants": len(result["restaurants"])}
    save("llm_hotspots", result)
    log("готово: городов %d, заведений %d, вызовов быстрой %d, большой %d, %.0f с"
        % (len(result["cities"]), len(result["restaurants"]), facts.get("fast", 0),
           facts.get("gen", 0), time.time() - started))


if __name__ == "__main__":
    main()
