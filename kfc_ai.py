#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Единый доступ к моделям для отчёта: локальные Qwen и внешний DeepSeek через aitunnel.

Правило: числа считаем сами, формулировки — модели. Для выводов и названий используем
внешний DeepSeek (он лучше держит русский деловой язык), для чтения больших выборок —
быстрые локальные модели на своих видеокартах.
"""
from __future__ import annotations

import io
import json
import re
import time
import urllib.error
import urllib.request

ENV_PATH = "/home/dev/tellscope_app/tellscope_backend/.env"

FAST_URL = "http://127.0.0.1:8001/v1/chat/completions"
FAST_MODEL = "qwen3-4b-fast"
LOCAL_URL = "http://127.0.0.1:8000/v1/chat/completions"
LOCAL_MODEL = "Qwen/Qwen3-32B-FP8"

_ENV = None


def env() -> dict:
    global _ENV
    if _ENV is None:
        values = {}
        try:
            for line in io.open(ENV_PATH, encoding="utf-8"):
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    values[key.strip()] = value.strip().strip('"').strip("'")
        except Exception:  # noqa: BLE001
            pass
        _ENV = values
    return _ENV


def _post(url: str, payload: dict, headers: dict, timeout: int = 900) -> dict:
    request = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def ask_remote(prompt: str, system: str = "", model: str = "deepseek-chat",
               max_tokens: int = 2000, temperature: float = 0.3, attempts: int = 3) -> str:
    """Внешняя модель через aitunnel: сильный русский язык для выводов."""
    values = env()
    key = values.get("AITUNNEL_API_KEY", "")
    base = values.get("AITUNNEL_BASE_URL", "https://api.aitunnel.ru/v1").rstrip("/")
    if not key:
        raise RuntimeError("нет ключа aitunnel")
    payload = {"model": model, "max_tokens": max_tokens, "temperature": temperature,
               "messages": ([{"role": "system", "content": system}] if system else [])
                           + [{"role": "user", "content": prompt}]}
    last = ""
    for attempt in range(1, attempts + 1):
        try:
            data = _post(base + "/chat/completions", payload,
                         {"Content-Type": "application/json", "Authorization": "Bearer " + key})
            return (data["choices"][0]["message"].get("content") or "").strip()
        except urllib.error.HTTPError as exc:
            last = "HTTP %s: %s" % (exc.code, exc.read().decode("utf-8", "replace")[:200])
        except Exception as exc:  # noqa: BLE001
            last = "%s: %s" % (type(exc).__name__, str(exc)[:200])
        time.sleep(4 * attempt)
    raise RuntimeError(last or "внешняя модель не ответила")


def ask_local(prompt: str, system: str = "", max_tokens: int = 2000, temperature: float = 0.3,
              model: str = LOCAL_MODEL, url: str = LOCAL_URL) -> str:
    payload = {"model": model, "max_tokens": max_tokens, "temperature": temperature,
               "chat_template_kwargs": {"enable_thinking": False},
               "messages": ([{"role": "system", "content": system}] if system else [])
                           + [{"role": "user", "content": prompt}]}
    data = _post(url, payload, {"Content-Type": "application/json"})
    message = data["choices"][0]["message"]
    return (message.get("content") or message.get("reasoning_content") or "").strip()


def ask_fast(prompt: str, system: str = "", max_tokens: int = 800, temperature: float = 0.1) -> str:
    return ask_local(prompt, system=system, max_tokens=max_tokens, temperature=temperature,
                     model=FAST_MODEL, url=FAST_URL)


def json_from(text: str):
    """Достаёт JSON из ответа модели, даже если он в тройных кавычках или с пояснением."""
    text = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        return None
    try:
        return json.loads(text[start:end + 1])
    except Exception:  # noqa: BLE001
        try:
            return json.loads(text[start:end + 1].replace("\n", " "))
        except Exception:  # noqa: BLE001
            return None


def ask_json(prompt: str, system: str = "", model: str = "deepseek-chat", max_tokens: int = 2000,
             local: bool = False, fast: bool = False, attempts: int = 2):
    """Ответ модели в виде словаря: с повтором, если формат не сложился."""
    for attempt in range(1, attempts + 1):
        try:
            if fast:
                text = ask_fast(prompt, system=system, max_tokens=max_tokens)
            elif local:
                text = ask_local(prompt, system=system, max_tokens=max_tokens)
            else:
                text = ask_remote(prompt, system=system, model=model, max_tokens=max_tokens)
        except Exception as exc:  # noqa: BLE001
            print("модель %s не ответила: %s" % (model, str(exc)[:120]))
            continue
        parsed = json_from(text)
        if parsed:
            return parsed
        print("ответ без JSON (попытка %d): %s" % (attempt, text[:150].replace("\n", " ")))
    return None
