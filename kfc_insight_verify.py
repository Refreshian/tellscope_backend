#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Проверка чисел в текстах моделей: каждое число должно быть в фактах.

Модель могла перепутать, к чему относится число (например, назвать долю негатива заведения
долей негатива города). Здесь каждое число из «зачем», инсайтов и плана сверяется с посчитанными
фактами; неподтверждённые числа собираются в список, и текст переписывается моделью заново
с разрешённым списком чисел.

Результат: /tmp/kfc_hotspots/verify.json (и исправленный insights.json)
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

SKIP = {2024, 2025, 2026, 2027, 28, 12, 24, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 30, 90, 365, 100}


def log(message: str) -> None:
    print("[%s] %s" % (datetime.datetime.now().strftime("%H:%M:%S"), message), flush=True)


def load(name: str):
    with io.open(os.path.join(CACHE, name + ".json"), encoding="utf-8") as fh:
        return json.load(fh)


def save(name: str, value) -> None:
    with io.open(os.path.join(CACHE, name + ".json"), "w", encoding="utf-8") as fh:
        json.dump(value, fh, ensure_ascii=False)


def ask(prompt: str, system: str = "", max_tokens: int = 1600, temperature: float = 0.2) -> str:
    body = json.dumps({"model": GEN_MODEL,
                       "messages": ([{"role": "system", "content": system}] if system else [])
                                   + [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": temperature}).encode("utf-8")
    request = urllib.request.Request(GEN_URL, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            return (json.loads(response.read().decode("utf-8"))["choices"][0]["message"]["content"] or "").strip()
    except urllib.error.HTTPError as exc:
        log("модель отказала: %s %s" % (exc.code, exc.read().decode("utf-8", "replace")[:200]))
        return ""


def json_from(text: str):
    text = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        return None
    try:
        return json.loads(text[start:end + 1])
    except Exception:  # noqa: BLE001
        return None


def allowed_numbers(facts: dict) -> dict:
    """Все числа, которые есть в фактах: значение → на что оно опирается."""
    out = {}

    def add(value, context):
        try:
            number = float(value)
        except Exception:  # noqa: BLE001
            return
        key = round(number, 2)
        out.setdefault(key, set()).add(context)

    totals = facts["totals"]
    add(totals["messages"], "всего сообщений")
    add(totals["negative"], "всего негативных")
    add(totals["positive"], "всего позитивных")
    add(round(totals["negative_share"] * 100, 1), "доля негатива за период, %")
    add(round(totals["positive"] / float(totals["messages"]) * 100, 1), "доля позитива за период, %")
    for year, row in facts["years"].items():
        add(row["total"], "сообщений в %s" % year)
        add(row["negative"], "негатив в %s" % year)
        add(round(row["negative_share"] * 100, 1), "доля негатива %s, %%" % year)
        add(round(row["positive_share"] * 100, 1), "доля позитива %s, %%" % year)
    for name, row in (facts["hubtype"] or {}).items():
        add(row["total"], "сообщений на площадке %s" % name)
        add(row["negative"], "негатив площадки %s" % name)
        add(round(row["share"] * 100, 1), "доля негатива площадки %s, %%" % name)
    for rating, count in (facts["ratings"] or {}).items():
        if rating in ("1", "2", "3", "4", "5"):
            add(count, "оценка %s, отзывов" % rating)
    conc = facts["concentration"]
    for key, value in conc.items():
        context = "концентрация: " + key
        if isinstance(value, float) and value <= 1:
            add(round(value * 100, 1), context + ", %")
        else:
            add(value, context)
    for row in facts["cities"]:
        add(row["negative"], "негатив города %s" % row["city"])
        add(row["total"], "сообщений города %s" % row["city"])
        add(round(row["share"] * 100, 1), "доля негатива города %s, %%" % row["city"])
        add(row.get("expected"), "ожидаемый негатив города %s" % row["city"])
        add(round(row.get("excess_norm") or 0, 2), "перевес города %s" % row["city"])
        add(round(row["growth"], 2), "рост города %s" % row["city"])
        add(row["reach"], "охват негатива города %s" % row["city"])
    for row in facts["restaurants"]:
        add(row["negative"], "негативных отзывов заведения %s" % row["city"])
        add(row["count"], "всего отзывов заведения %s" % row["city"])
        add(round(row["share"] * 100, 1), "доля негатива заведения %s, %%" % row["city"])
        add(row.get("rating"), "рейтинг заведения %s" % row["city"])
        add(round(row["growth"], 2), "рост заведения %s" % row["city"])
    for row in facts["drivers"]:
        add(row["negative"], "негатив группы «%s»" % row["name"])
        add(row["last_q"], "негатив группы «%s» за квартал" % row["name"])
        add(round(row["growth"] * 100 - 100, 1), "рост группы «%s», %%" % row["name"])
        add(round(row["growth"], 2), "рост группы «%s», разы" % row["name"])
    for row in facts["stories"]:
        add(row["negative"], "негатив повода «%s»" % row["name"][:50])
        add(row["reach"], "охват повода «%s»" % row["name"][:50])
    scen = facts["scenarios"] or {}
    for row in scen.get("forecast") or []:
        add(round(row["no_action"] * 100, 1), "сценарий без действий, %")
        add(round(row["targeted"] * 100, 1), "сценарий адресной работы, %")
        add(round(row["systemic"] * 100, 1), "сценарий системной работы, %")
    add(round((scen.get("mean_share") or 0) * 100, 1), "средняя доля негатива, %")
    add(facts["period"]["months"], "число месяцев в периоде")

    # производные разницы: только те, что прямо следуют из долей по годам
    year_shares = sorted(round(row["negative_share"] * 100, 1) for row in facts["years"].values())
    for index, left in enumerate(year_shares):
        for right in year_shares[index + 1:]:
            out.setdefault(round(abs(right - left), 1), set()).add("разница долей негатива между годами")
    # производные доли, которые считаются прямо из фактов
    ratings = facts.get("ratings") or {}
    negatives = float(facts["totals"]["negative"] or 1)
    if ratings.get("1"):
        out.setdefault(round(int(ratings["1"]) / negatives * 100, 1),
                       set()).add("доля отзывов с оценкой 1 во всём негативе, %")
    if ratings.get("2") and ratings.get("3"):
        out.setdefault(int(ratings["2"]) + int(ratings["3"]), set()).add("отзывы с оценкой 2 и 3")
    for row in facts["cities"]:
        out.setdefault(round(row["negative"] / negatives * 100, 2),
                       set()).add("вес города %s в общем негативе, %%" % row["city"])
    for row in facts["restaurants"]:
        out.setdefault(round(row["negative"] / negatives * 100, 2),
                       set()).add("вес заведения в общем негативе, %%")
    return out


def semantic_checks(text: str, facts: dict) -> list:
    """Смысловые подмены, которые по одному числу не видны."""
    problems = []
    negatives = float(facts["totals"]["negative"] or 1)
    lowered = str(text or "").lower()
    for match in re.finditer(r"(\d+[.,]?\d*)\s*%\s*от всех негативных", lowered):
        claim = float(match.group(1).replace(",", "."))
        window = lowered[max(0, match.start() - 90):match.start()]
        counts = [float(item.replace(" ", "")) for item in re.findall(r"\d[\d ]{4,}", window)]
        if not counts:
            continue
        real = max(counts) / negatives * 100
        if abs(real - claim) > 0.6:
            problems.append("%s%% названо долей от всего негатива, а на деле это %s%%"
                            % (match.group(1), ("%.1f" % real).replace(".", ",")))
    for match in re.finditer(r"приходится на (\d+) завед", lowered):
        if int(match.group(1)) != len(facts["restaurants"]):
            problems.append("«приходится на %s заведений» — число не подтверждено" % match.group(1))
    allowed_counts = {facts["concentration"].get("restaurants_with_growth"),
                      facts["concentration"].get("cities_with_growth"), len(facts["restaurants"])}
    for match in re.finditer(r"\bв (\d+) заведениях\b", lowered):
        if int(match.group(1)) not in {value for value in allowed_counts if value}:
            problems.append("«в %s заведениях» — число не подтверждено" % match.group(1))
    # рост за квартал: у города и у заведения разные числа, их легко перепутать
    city = next((row for row in facts["cities"] if row["city"].lower() in lowered), None)
    for match in re.finditer(r"рост\s*x?\s*([\d]+[.,]?\d*)\s*за квартал", lowered):
        value = float(match.group(1).replace(",", "."))
        if not city:
            continue
        if abs(value - city["growth"]) <= 0.05:
            continue
        if any(abs(value - item["growth"]) <= 0.05 for item in facts["restaurants"]):
            problems.append("рост x%s назван ростом города %s, а на деле это рост заведения "
                            "(у города x%s)" % (match.group(1), city["city"],
                                                ("%.2f" % city["growth"]).replace(".", ",")))
    return problems


def numbers_in(text: str) -> list:
    found = []
    for match in re.finditer(r"\d[\d \u00a0]*(?:[.,]\d+)?", str(text or "")):
        raw = match.group(0).strip()
        cleaned = raw.replace(" ", "").replace("\u00a0", "").replace(",", ".")
        try:
            value = float(cleaned)
        except Exception:  # noqa: BLE001
            continue
        found.append((raw, value))
    return found


def check(text: str, allowed: dict) -> list:
    bad = []
    for raw, value in numbers_in(text):
        if value in SKIP or (value.is_integer() and int(value) in SKIP):
            continue
        if value < 6 and float(value).is_integer():
            continue
        hit = False
        for candidate, _contexts in allowed.items():
            if abs(candidate - value) <= max(0.05, abs(candidate) * 0.005):
                hit = True
                break
            if candidate >= 100 and abs(candidate - value) <= 1.0:
                hit = True
                break
        if not hit:
            bad.append(raw)
    return bad


def city_context_mismatch(text: str, facts: dict) -> list:
    """Ловит подмену: доля негатива заведения выдана за долю негатива города."""
    problems = []
    lowered = str(text or "").lower()
    for row in facts["cities"]:
        city = row["city"]
        if len(city) < 4 or city.lower() not in lowered:
            continue
        city_share = round(row["share"] * 100, 1)
        restaurant_shares = {round(item["share"] * 100, 1)
                             for item in facts["restaurants"] if item["city"] == city}
        for raw, value in numbers_in(text):
            if "%" not in text[text.find(raw) + len(raw):text.find(raw) + len(raw) + 3]:
                continue
            if abs(value - city_share) <= 0.6:
                continue
            if any(abs(value - candidate) <= 0.6 for candidate in restaurant_shares):
                problems.append("%s%% — это доля негатива заведения, а не города %s (у города %s%%)"
                                % (raw, city, city_share))
    return problems


def main() -> None:
    facts = load("facts")
    insights = load("insights")
    allowed = allowed_numbers(facts)
    log("разрешённых чисел: %d" % len(allowed))

    report = {"built_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
              "allowed": len(allowed), "fields": [], "repaired": 0}
    system = ("Ты редактор отчёта для руководства. Ты обязан использовать только числа из "
              "разрешённого списка: любое другое число — ошибка. Пиши по-русски, простым деловым "
              "языком, без технических терминов.")

    allowed_list = "; ".join("%s (%s)" % (("%g" % value), sorted(contexts)[0])
                             for value, contexts in sorted(allowed.items())[:260])

    def repair(field: str, text: str, bad: list, question: str, keys: dict) -> str:
        prompt = (
            "РАЗРЕШЁННЫЕ ЧИСЛА (только они допустимы):\n%s\n\n"
            "ТЕКСТ:\n%s\n\n"
            "В тексте есть числа, которых нет в разрешённом списке: %s. "
            "Причина ошибки обычно в том, что число относится к другому объекту "
            "(например, доля негатива заведения названа долей негатива города).\n"
            "Перепиши текст так, чтобы: числа были только из разрешённого списка и относились "
            "к правильному объекту; смысл сохранился; стиль остался деловым.\n"
            "Верни JSON: %s"
            % (allowed_list, text, ", ".join(bad), json.dumps(keys, ensure_ascii=False)))
        raw = ask(prompt, system=system)
        return raw

    # --- зачем
    why = insights.get("why") or {}
    entries = []          # (куда писать, индекс, ключ, подпись, текст)
    if why.get("зачем"):
        entries.append(("why", None, "зачем", "зачем", why["зачем"]))
    for index, text in enumerate(why.get("решения") or []):
        entries.append(("why_list", index, "решения", "решение", text))
    for index, text in enumerate(why.get("в_одном_экране") or []):
        entries.append(("why_list", index, "в_одном_экране", "экран", text))
    # --- инсайты
    for index, item in enumerate(insights.get("insights") or []):
        for key in ("цифра", "что_значит", "что_значает", "что_делать", "если_не_делать"):
            if item.get(key):
                entries.append(("insight", index, key,
                                "инсайт «%s» / %s" % (item.get("заголовок", "")[:40], key), item[key]))
    # --- план: действия проверяем, цели (kpi) — это плановые значения, а не факты
    plan = (insights.get("plan") or {}).get("план") or {}
    targets = []
    for horizon, items in plan.items():
        for index, item in enumerate(items or []):
            for key, value in (item or {}).items():
                if not isinstance(value, str) or not value:
                    continue
                if key == "kpi":
                    targets.append({"field": "план:%s:kpi" % horizon, "text": value})
                    continue
                entries.append(("plan", (horizon, index), key, "план:%s/%s" % (horizon, key), value))

    flagged = []
    for kind, index, key, label, text in entries:
        bad = check(text, allowed)
        bad += city_context_mismatch(text, facts)
        bad += semantic_checks(text, facts)
        if bad:
            flagged.append({"kind": kind, "index": index, "key": key, "field": label,
                            "numbers": bad, "text": text})
    report["fields"] = [{k: v for k, v in row.items() if k != "kind" and k != "index"} for row in flagged]
    report["targets"] = targets
    log("полей проверено: %d, с сомнительными числами: %d, плановых целей: %d"
        % (len(entries), len(flagged), len(targets)))
    for row in flagged:
        log("   %s → %s" % (row["field"][:70], "; ".join(row["numbers"])))

    # --- переписываем проблемные места и проверяем заново
    for row in flagged:
        hints = ["%s — доля негатива города %s%%" % (city["city"], ("%.1f" % (city["share"] * 100)).replace(".", ","))
                 for city in facts["cities"] if city["city"].lower() in row["text"].lower()]
        prompt = (
            "РАЗРЕШЁННЫЕ ЧИСЛА (только они допустимы, каждое со своим смыслом):\n%s\n\n"
            "ТЕКСТ, ГДЕ ЕСТЬ ОШИБКА:\n%s\n\n"
            "ЧТО НЕ ТАК: %s\n\n"
            "%s\n"
            "Перепиши текст: числа только из разрешённого списка и только в правильном смысле, "
            "смысл фразы сохрани, стиль деловой, длина та же.\n"
            'Верни JSON: {"текст": "переписанный текст"}'
            % (allowed_list, row["text"], "; ".join(row["numbers"]),
               ("Подсказка (если речь о городе, используй именно эти числа): " + "; ".join(hints) + "\n")
               if hints else ""))
        raw = ask(prompt, system=system)
        fixed = str((json_from(raw) or {}).get("текст") or "").strip()
        if fixed:
            still = check(fixed, allowed) + city_context_mismatch(fixed, facts) + semantic_checks(fixed, facts)
            if not still:
                row["fixed"] = fixed
                report["repaired"] += 1
                log("   переписано: %s" % row["field"][:60])
                continue
            log("   переписать не удалось (%s), убираю предложение с ошибкой" % ", ".join(still))
        # запасной путь: выбрасываем предложения с неподтверждёнными числами
        kept = []
        for sentence in re.split(r"(?<=[.!?])\s+", row["text"]):
            if check(sentence, allowed) or city_context_mismatch(sentence, facts) or semantic_checks(sentence, facts):
                continue
            kept.append(sentence)
        fallback = " ".join(kept).strip()
        if not fallback:
            # предложение одно и целиком с ошибкой: собираем факт заново из чисел
            city_row = next((row_c for row_c in facts["cities"]
                             if row_c["city"].lower() in row["text"].lower()), None)
            if row["key"] == "цифра" and "оценк" in row["text"].lower():
                ratings = facts.get("ratings") or {}
                first = int(ratings.get("1") or 0)
                middle = int(ratings.get("2") or 0) + int(ratings.get("3") or 0)
                fallback = ("%d отзывов с оценкой 1 — это %s%% всего негатива; ещё %d отзывов "
                            "с оценками 2 и 3, их тоже можно вернуть."
                            % (first, ("%.1f" % (first / negatives * 100)).replace(".", ","), middle))
                log("   собрано заново из фактов: %s" % row["field"][:60])
            elif row["key"] == "цифра" and city_row:
                worst = next((item for item in facts["restaurants"] if item["city"] == city_row["city"]), None)
                parts = ["%s — %d негативных сообщений, доля негатива %s%%, перевес над ожидаемым x%s, "
                         "рост за квартал x%s"
                         % (city_row["city"], city_row["negative"],
                            ("%.1f" % (city_row["share"] * 100)).replace(".", ","),
                            ("%.2f" % (city_row.get("excess_norm") or 0)).replace(".", ","),
                            ("%.2f" % city_row["growth"]).replace(".", ","))]
                if worst:
                    parts.append("худшее заведение города: %d негативных отзывов из %d, рейтинг %s"
                                 % (worst["negative"], worst["count"],
                                    ("%.2f" % (worst.get("rating") or 0)).replace(".", ",")))
                fallback = "; ".join(parts) + "."
                log("   собрано заново из фактов: %s" % row["field"][:60])
            else:
                fallback = row["text"]
                for city_candidate in facts["cities"]:
                    city = city_candidate["city"]
                    if city.lower() not in fallback.lower():
                        continue
                    correct = ("%s%%" % round(city_candidate["share"] * 100, 1)).replace(".", ",")
                    for match in re.finditer(r"\d+[.,]?\d*\s*%", fallback):
                        value = float(match.group(0).replace("%", "").replace(",", ".").strip())
                        if abs(value - round(city_candidate["share"] * 100, 1)) > 0.6:
                            fallback = fallback.replace(match.group(0), correct + " ", 1)
                    # рост за квартал у города свой, у заведения свой
                    fallback = re.sub(r"рост\s*x?\s*[\d]+[.,]?\d*\s*за квартал",
                                      "рост x%s за квартал" % ("%.2f" % city_candidate["growth"]).replace(".", ","),
                                      fallback)
                    fallback = re.sub(r"\bв \d+ заведениях\b", "в заведениях города", fallback)
                    fallback = re.sub(r"\bна \d+ заведений\b", "на заведения города", fallback)
                    fallback = re.sub(r"\b\d+ заведений\b", "заведения", fallback)
                fallback = re.sub(r"\s{2,}", " ", fallback).strip()
                log("   подставлена правильная доля: %s" % row["field"][:60])
        row["fixed"] = fallback

    for row in flagged:
        fixed = row.get("fixed")
        if not fixed:
            continue
        if row["kind"] == "why":
            insights["why"][row["key"]] = fixed
        elif row["kind"] == "why_list":
            insights["why"][row["key"]][row["index"]] = fixed
        elif row["kind"] == "insight":
            insights["insights"][row["index"]][row["key"]] = fixed
        elif row["kind"] == "plan":
            horizon, index = row["index"]
            insights["plan"]["план"][horizon][index][row["key"]] = fixed
    save("insights", insights)
    save("verify", report)
    log("исправлено полей: %d, отчёт проверки сохранён" % report["repaired"])


if __name__ == "__main__":
    main()
