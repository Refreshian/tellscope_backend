# -*- coding: utf-8 -*-
"""Чтение текстов датасета локальной моделью Qwen (vLLM): темы, цитаты, важные сообщения.

Зачем это нужно: статистика (графики, счётчики, распределение тональности) показывает,
СКОЛЬКО сообщений и какой тональности, но не объясняет, ПОЧЕМУ люди недовольны.
`analyze_texts` читает сами тексты сообщений за период и возвращает:

  * topics     — темы с числом сообщений, долей, средней тональностью и 2–3 цитатами
                 (цитата всегда дословная, с датой, площадкой, автором и ссылкой);
  * highlights — ключевые сообщения по вовлечённости: комментарии, репосты, лайки, просмотры,
                 аудитория, ER, аудитория СМИ (если счётчиков в источнике нет — откат на оценку
                 отзыва, объём текста и число юридических/эмоциональных маркеров, и это прямо
                 помечается в отчёте);
  * categories — сопоставление тем с фиксированным каркасом категорий;
  * summary    — аналитический текст по темам и цитатам;
  * stats      — сколько сообщений прочитано, сколько батчей, время, модель, токены.

Разбор идёт пачками по 15–20 сообщений с повторами при ошибке; ответ модели — строго JSON.
Число сообщений по теме считается по идентификаторам, которые вернула модель, а не по её
арифметике, поэтому частоты и доли всегда сходятся с выборкой.

Вызов LLM идёт через существующий путь mlops.gateway (через _qwen из tools_llm) — свой
HTTP-клиент не создаётся.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import math
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from .context import RunCancelled, compact
from .registry import ToolError, tool
from .tools_llm import _extract_json, _qwen

# ---------------------------------------------------------------- параметры разбора
# Замеры на датасете 1105 (41 сообщение): пачка из 15–20 сообщений — это один вызов Qwen
# на ~90–110 с, и время почти не зависит от размера пачки (доминирует генерация JSON),
# поэтому по умолчанию пачки крупнее, а чтение идёт параллельно: 41 сообщение = 3 пачки
# одной волной ≈ столько же, сколько одна пачка. Пачки и параллелизм можно переопределить
# переменными окружения (TELLSCOPE_TEXTS_BATCH_SIZE / TELLSCOPE_TEXTS_PARALLEL) без правки кода.
BATCH_SIZE = int(os.environ.get("TELLSCOPE_TEXTS_BATCH_SIZE") or 20)   # сообщений в пачке (15–25)
MIN_BATCH_SIZE = 15
MAX_BATCH_SIZE = 25
# Пределы чтения и пороги стратегий — дефолты; рабочая настройка берётся из _texts_config()
# (mlops/lock.yaml, секция texts, и переменные окружения TELLSCOPE_TEXTS_*).
# Пороги и лимиты чтения — дефолты; единственный источник настройки — mlops/lock.yaml
# (секция texts) и переменные окружения TELLSCOPE_TEXTS_*. Значения совпадают с заданием:
#   до 5 000 сообщений          — читаем весь срез целиком;
#   5 000–20 000                — тоже целиком, но предупреждаем, что это долго;
#   больше 20 000               — кластеры по всему корпусу + чтение представителей кластеров
#                                 и самых значимых сообщений.
WARN_READ_LIMIT = 5000        # с этого размера срез читается целиком, но это долго
FULL_READ_LIMIT = 20000       # до этого размера читаем ВЕСЬ срез целиком
CLUSTER_MIN = 20000           # свыше — кластеры по всему корпусу и чтение представителей
CLUSTER_READ_LIMIT = 600      # сколько сообщений всего читаем в режиме кластеризации
CLUSTER_PER_CLUSTER = 8       # представителей каждого кластера
CLUSTER_TOP_MESSAGES = 60     # плюс самые значимые сообщения корпуса
CLUSTER_MIN_SIZE = 30         # минимальный размер кластера HDBSCAN
CLUSTER_MAX = 40              # сколько кластеров максимум попадает в отчёт
CLUSTER_EMBED_BATCH = 64      # пачка эмбеддингов корпуса
CLUSTER_SVD_DIMS = 50         # SVD перед UMAP (быстрее и без потери структуры)
CLUSTER_UMAP_NEIGHBORS = 30   # соседей UMAP — как в кластеризации датасетов проекта
CLUSTER_CORPUS_LIMIT = 200000  # страховочный предел корпуса для эмбеддингов
CLUSTER_TEXT_CHARS = 300      # сколько символов текста уходит в эмбеддинги кластеризации
MAX_PARALLEL_BATCHES = 16     # сколько пачек читать одновременно (страховка от перегрузки vLLM)
# Сколько сообщений читать по умолчанию, если предел не задан явно.
DEFAULT_LIMIT = 60
# Меньше одной пачки просить бессмысленно: модель не увидит срез целиком, а пользователь —
# «прочитано 15 из 41». Такой предел инструмент поднимает до DEFAULT_LIMIT сам.
MIN_USEFUL_LIMIT = 20

MESSAGE_CHARS = 480      # обрезка текста сообщения в промпте (max_model_len vLLM = 8192)
BATCH_CHAR_BUDGET = 9000  # страховка по длине промпта пачки
# Лимит генерации одной пачки. 1500 токенов пачке из 20 сообщений не хватало: ответ обрывался
# на finish_reason="length", JSON оставался незакрытым, а _extract_json находил первый ВЛОЖЕННЫЙ
# объект — и инструмент отдавал пустые темы и цитаты при непустом «прочитано N». 3000 — запас
# и для Qwen3-32B, и для быстрой Qwen3-4B-Instruct-2507; при обрыве пачка делится пополам
# (см. _run_batch) и шаг честно сообщает об ошибке, а не отдаёт пустой разбор.
BATCH_MAX_TOKENS = 3000
SUMMARY_MAX_TOKENS = 1100
# Пачки читаются параллельно: с профилем быстрого чтения это 8 запросов к 4B, без него — 4 к 32B;
# и то и другое далеко от max_model_len и max-num-seqs, память vLLM не упирается.
PARALLEL_BATCHES = int(os.environ.get("TELLSCOPE_TEXTS_PARALLEL") or 4)

MODEL_LABEL = "Qwen3-32B-FP8 (vLLM, локально, без оплаты)"

# Каркас категорий: темы модели приводятся к нему, чтобы отчёты были сопоставимы между датасетами.
CATEGORY_FRAMEWORK = [
    "доставка",
    "качество товара",
    "цена",
    "возврат/деньги",
    "поддержка",
    "упаковка",
    "сроки",
    "другое",
]

# Ключевые слова для приведения темы к каркасу, если модель не назвала категорию точно.
CATEGORY_KEYWORDS: List[Tuple[str, Tuple[str, ...]]] = [
    ("доставка", ("доставк", "курьер", "пункт выдачи", "пвз", "привез", "привоз", "достав", "почт", "курьерск")),
    ("качество товара", ("качеств", "брак", "поврежд", "сломан", "некачеств", "дефект", "порван", "разбит", "грязн")),
    ("цена", ("цен", "дорог", "переплат", "стоимост", "скидк", "наценк", "дешевл")),
    ("возврат/деньги", ("возврат", "вернул", "деньг", "денеж", "refund", "компенсац", "оплат", "списан", "не вернул", "возмещ")),
    ("поддержка", ("поддержк", "оператор", "служб", "консультант", "ответил", "чат", "горяч", "обращени")),
    ("упаковка", ("упаковк", "коробк", "пакет", "запечатан", "тара", "плёнк")),
    ("сроки", ("срок", "задержк", "опозда", "вовремя", "просроч", "долго ждал", "неделю", "месяц ждал")),
]

# Поля вовлечённости Brand Analytics. В выгрузке BA («полнотекстовые сообщения», JSON) они есть
# всегда: audienceCount, commentsCount, likesCount, repostsCount, viewsCount, er, massMediaAudience,
# duplicateCount, citeIndex. Отзовики и сайты-рекомендации (otzovik.com, irecommend.ru) счётчики не
# публикуют — там BA отдаёт нули, и инструмент честно уходит в fallback (см. NO_ENGAGEMENT_NOTE).
# Веса по умолчанию — из кода: без настройки поведение остаётся предсказуемым и воспроизводимым.
# Переопределяются в mlops/lock.yaml (секция engagement) и переменными окружения
# TELLSCOPE_ENGAGEMENT_WEIGHTS / TELLSCOPE_ENGAGEMENT_SCALE / TELLSCOPE_ENGAGEMENT_ENABLED;
# читает их mlops.lock.engagement_cfg, применяет _engagement_config ниже.
#
# Почему значения именно такие:
#  * er ИСКЛЮЧЁН (вес 0). В выгрузке Brand Analytics ER = commentsCount + likesCount + repostsCount,
#    то есть те же самые реакции: раньше он прибавлялся к сумме и удваивал их вклад. Если ER нужен
#    как «плотность», включайте его вес осознанно — это снова двойной счёт.
#  * аудитория и просмотры — это потенциал охвата, а не реакция людей, поэтому их вес ниже, чем у
#    комментариев и репостов: сообщество на 2 млн подписчиков с нулём реакций не должно обгонять
#    сообщение с 260 комментариями и 839 лайками.
#  * duplicateCount — мягкий сигнал перепечатки, низкий вес.
ENGAGEMENT_WEIGHTS = {
    "commentsCount": 3.0,
    "repostsCount": 4.0,
    "likesCount": 2.0,
    "viewsCount": 0.5,
    "audienceCount": 0.5,
    "massMediaAudience": 0.5,
    "er": 0.0,
    "duplicateCount": 0.5,
}

# Шкала сжатия метрик — вместо линейной суммы, чтобы один выброс не забивал топ.
# Порядок значений внутри поля сохраняется, меняется только размах:
#   log    — f(v) = log1p(v)  (по умолчанию: аудитория 2 184 064 даёт 14.6, а не 2 184 064)
#   sqrt   — f(v) = sqrt(v)
#   linear — f(v) = v         (прежнее поведение: для сравнения и совместимости)
ENGAGEMENT_SCALES = ("log", "sqrt", "linear")
ENGAGEMENT_SCALE = "log"
ENGAGEMENT_SCALE_LABELS = {
    "log": "логарифмическая шкала",
    "sqrt": "корневая шкала",
    "linear": "линейная шкала",
}
# Порядок показа метрик в основании ранжирования (stats.importance_basis) и в отчёте.
ENGAGEMENT_ORDER = [
    "commentsCount", "repostsCount", "likesCount", "viewsCount",
    "audienceCount", "er", "massMediaAudience", "duplicateCount",
]
ENGAGEMENT_LABELS = {
    "commentsCount": "комментарии",
    "repostsCount": "репосты",
    "likesCount": "лайки",
    "viewsCount": "просмотры",
    "audienceCount": "аудитория",
    "er": "ER",
    "massMediaAudience": "аудитория СМИ",
    "duplicateCount": "дубли",
}
# Подписи метрик для промпта модели (естественные «комментариев 12, лайков 3»).
PROMPT_METRICS = [
    ("commentsCount", "комментариев"),
    ("likesCount", "лайков"),
    ("repostsCount", "репостов"),
    ("viewsCount", "просмотров"),
    ("audienceCount", "аудитория"),
    ("massMediaAudience", "аудитория СМИ"),
]
# Поля, по которым считается max-агрегация в Elasticsearch и определяется, заполнена ли
# вовлечённость в срезе. Агрегируем только по числовым полям индекса — см. _numeric_engagement_fields.
ENGAGEMENT_AGG_FIELDS = list(ENGAGEMENT_ORDER)
# Порядок сортировки в Elasticsearch: от «трудного» действия к «лёгкому».
ENGAGEMENT_SORT_ORDER = ["commentsCount", "repostsCount", "likesCount", "viewsCount", "audienceCount"]
# Запас кандидатов, когда вовлечённость заполнена: итоговый топ считает _importance по ВСЕМ метрикам,
# поэтому одной сортировки Elasticsearch по комментариям мало.
ENGAGEMENT_POOL_FACTOR = 4
ENGAGEMENT_POOL_MAX = 400
# Числовые типы Elasticsearch, по которым безопасно считать max и сортировать.
ES_NUMERIC_TYPES = {
    "long", "integer", "short", "byte", "double", "float", "half_float",
    "scaled_float", "unsigned_long",
}

# Откат, когда счётчиков в источнике нет: юридический и эмоциональный накал сообщения.
LEGAL_MARKERS = (
    "суд", "иск", "претензи", "роспотребнадзор", "прокурат", "юрист", "адвокат", "закон",
    "штраф", "жалоб", "нарушен", "обман", "фальсифи", "мошенн", "компенсац", "незаконн",
)
EMOTION_MARKERS = (
    "ужас", "кошмар", "отврат", "возмут", "наглост", "хамств", "развод", "позор",
    "катастроф", "никогда больше", "не советую", "испортил", "потерял", "обманул",
    "беспредел", "отвратительн",
)
MARKER_WEIGHT = 6.0
NO_ENGAGEMENT_NOTE = "в источнике нет данных о вовлечённости (площадка не публикует счётчики)"

TONE_LABELS = {-1: "негатив", 0: "нейтрал", 1: "позитив"}

SOURCE_FIELDS = [
    "text", "title", "timeCreate", "hub", "hubtype", "url", "toneMark", "city",
    "likesCount", "commentsCount", "repostsCount", "viewsCount", "audienceCount",
    "er", "massMediaAudience", "duplicateCount", "review_rating", "authorObject", "idExternal",
]

BATCH_SYSTEM = (
    "Ты аналитик отзывов и сообщений соцмедиа и СМИ. Ты читаешь реальные тексты сообщений "
    "и группируешь их по темам строго по фактам из текста, ничего не додумывая. "
    "Отвечай ТОЛЬКО JSON, без пояснений вокруг."
)

BATCH_INSTRUCTION = """Разбери сообщения{scope}{focus}.
Сгруппируй их в темы, которые реально видны в текстах.

Верни JSON строго такого вида:
{{
  "topics": [
    {{"name": "название темы, 3-6 слов",
      "category": "категория из списка ниже",
      "msg_ids": ["m1", "m4"],
      "essence": "суть темы одним предложением",
      "tone": "негатив | нейтрал | позитив | смешанно",
      "quotes": [{{"msg_id": "m1", "quote": "дословная выдержка из сообщения"}}]}}
  ],
  "highlights": [{{"msg_id": "m3", "why": "почему это сообщение важно"}}],
  "summary": "2-3 предложения: что происходит в этих сообщениях"
}}

Категории (ровно одна на тему): {categories}
Жёсткие правила:
- msg_ids — только идентификаторы из квадратных скобок; каждый id попадает ровно в одну тему;
- quote — дословная выдержка до 200 символов, без пересказа и без правок;
- в каждой теме 1-2 цитаты, у каждой цитаты указан msg_id;
- highlights — 2-4 самых значимых сообщения (сильные претензии, массовая проблема, показательный позитив);
- тем не меньше 2 и не больше 8.

Сообщения:
{messages}"""

SUMMARY_SYSTEM = (
    "Ты аналитик соцмедиа и СМИ. Пиши деловым русским языком, только текст, без вводных фраз "
    "вида «в данном отчёте». Опирайся строго на переданные темы, цитаты и числа."
)

SUMMARY_INSTRUCTION = """Ниже результат чтения текстов датасета «{dataset}» за период {period}.
Прочитано сообщений: {count}. Негативных в срезе: {negative}. Тональность среза: {tone}.

Темы (тема — сообщений — доля — тональность — суть):
{topics}

Цитаты из сообщений:
{quotes}

Ключевые сообщения (по вовлечённости):
{highlights}

Напиши аналитический текст 3-5 абзацами: 1) какие темы доминируют, сколько это сообщений и долей;
2) на что конкретно жалуются — с опорой на приведённые цитаты; 3) что хвалят; 4) какие темы
требуют действий в первую очередь и почему. Числа бери только из данных выше. Без JSON и без заголовков."""


# --------------------------------------------------------------------- утилиты

def _num(value: Any) -> Optional[float]:
    """Приводит значение из Elasticsearch к числу (viewsCount приходит строкой)."""
    if value in (None, "", [], {}):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        match = re.search(r"-?\d+(?:[.,]\d+)?", str(value))
        if not match:
            return None
        try:
            number = float(match.group(0).replace(",", "."))
        except ValueError:
            return None
    return number


def _flat(text: Any) -> str:
    return " ".join(str(text or "").split())


def _snippet(text: Any, limit: int = 220) -> str:
    """Дословная выдержка из текста сообщения: без пробельных «дыр», обрезка по границе слова."""
    clean = _flat(text)
    if len(clean) <= limit:
        return clean
    cut = clean[:limit]
    space = cut.rfind(" ")
    return (cut[:space] if space > limit * 0.6 else cut).rstrip(" ,;:.-") + "…"


def _norm_key(text: Any) -> str:
    """Ключ для склейки одинаковых тем между пачками (регистр, пробелы, пунктуация)."""
    return re.sub(r"[^0-9a-zа-яё]+", "", _flat(text).lower())


def _bulk_profile() -> Dict[str, Any]:
    """Профиль быстрого чтения пачек (mlops/lock.yaml, секция texts_bulk). Пусто — читаем на 32B.

    Отдельный профиль нужен потому, что VLLM_BASE_URL/VLLM_MODEL переопределять нельзя: их
    используют оркестрация, harness и остальные инструменты. Профиль включается правкой
    lock.yaml (enabled: true) или TELLSCOPE_TEXTS_BULK=true; mlops.lock читает секцию через
    read_lock_fresh, поэтому переключатель работает без перезапуска приложения.
    """
    try:
        from mlops.lock import texts_bulk_cfg

        cfg = texts_bulk_cfg() or {}
    except Exception:
        return {}
    if not cfg.get("enabled"):
        return {}

    def _int(key: str, default: int) -> int:
        try:
            return int(cfg.get(key) or default)
        except (TypeError, ValueError):
            return default

    return {
        "vllm_cfg": cfg,
        "model": str(cfg.get("model") or ""),
        "base_url": str(cfg.get("base_url") or ""),
        "max_tokens": _int("max_tokens", BATCH_MAX_TOKENS),
        "batch_size": _int("batch_size", BATCH_SIZE),
        "parallel": _int("parallel_batches", PARALLEL_BATCHES),
        "fallback": bool(cfg.get("fallback_to_generate", True)),
    }


def _gateway_model_label(bulk: Optional[Dict[str, Any]] = None) -> str:
    """Честная подпись модели чтения: быстрая 4B, когда её профиль включён, иначе общий 32B."""
    try:
        if bulk and bulk.get("model"):
            return f"{bulk['model']} (vLLM, локально, без оплаты)"
        from mlops.lock import generate_cfg

        model = str((generate_cfg() or {}).get("model") or "").strip()
        if model:
            return f"{model} (vLLM, локально, без оплаты)"
    except Exception:
        pass
    return MODEL_LABEL


# ------------------------------------------------------------------ выборка сообщений

def _markers(text: str) -> Dict[str, int]:
    """Юридические и эмоциональные маркеры текста — сигнал важности, когда счётчиков нет."""
    low = _flat(text).lower()
    legal = sum(1 for marker in LEGAL_MARKERS if marker in low)
    emotion = sum(1 for marker in EMOTION_MARKERS if marker in low)
    return {"legal": legal, "emotion": emotion, "total": legal + emotion}


def _markers_basis(markers: Dict[str, Any]) -> str:
    return "юридических %s, эмоциональных %s" % (markers.get("legal") or 0, markers.get("emotion") or 0)


def _fmt_num(value: Any) -> str:
    """Число по-русски: 24 672 вместо 24672.0."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number.is_integer():
        return "{:,.0f}".format(number).replace(",", " ")
    return ("%.2f" % number).rstrip("0").rstrip(".")


def _engagement_config() -> Tuple[Dict[str, float], str, bool]:
    """Активные веса, шкала и признак «включено»: дефолты из кода + mlops (lock.yaml и окружение).

    Настройку читает mlops.lock.engagement_cfg. Если настройки нет или она битая, работают дефолтные
    веса из кода: поведение без конфигурации предсказуемо, а опечатка в YAML не должна ронять инструмент.
    """
    weights = dict(ENGAGEMENT_WEIGHTS)
    scale = ENGAGEMENT_SCALE
    enabled = True
    try:
        from mlops.lock import engagement_cfg

        cfg = engagement_cfg() or {}
    except Exception:
        cfg = {}
    raw_weights = cfg.get("weights")
    if isinstance(raw_weights, dict):
        for field, value in raw_weights.items():
            number = _num(value)
            if number is not None and number >= 0:
                weights[str(field)] = float(number)
    raw_scale = str(cfg.get("scale") or "").strip().lower()
    if raw_scale in ENGAGEMENT_SCALES:
        scale = raw_scale
    if cfg.get("enabled") is not None:
        enabled = bool(cfg.get("enabled"))
    if not any(weight > 0 for weight in weights.values()):
        enabled = False  # все веса обнулили настройкой — это и есть выключение вовлечённости
    return weights, scale, enabled


def _texts_config() -> Dict[str, Any]:
    """Настройка адаптивного чтения: дефолты из кода + mlops/lock.yaml (секция texts) + окружение.

    Единственное место, где живут пороги чтения: и порог полного чтения, и порог кластеризации,
    и лимиты чтения представителей кластеров. Настройка читается на каждом запуске инструмента,
    поэтому подкручивается без правки кода и без рестарта. Битая настройка не ломает инструмент —
    остаётся дефолт.
    """
    defaults: Dict[str, Any] = {
        "warn_read_limit": WARN_READ_LIMIT,
        "full_read_limit": FULL_READ_LIMIT,
        "cluster_min_messages": CLUSTER_MIN,
        "cluster_read_limit": CLUSTER_READ_LIMIT,
        "cluster_per_cluster": CLUSTER_PER_CLUSTER,
        "cluster_top_messages": CLUSTER_TOP_MESSAGES,
        "cluster_min_size": CLUSTER_MIN_SIZE,
        "cluster_max": CLUSTER_MAX,
        "cluster_embed_batch": CLUSTER_EMBED_BATCH,
        "cluster_svd_dims": CLUSTER_SVD_DIMS,
        "cluster_umap_neighbors": CLUSTER_UMAP_NEIGHBORS,
        "cluster_corpus_limit": CLUSTER_CORPUS_LIMIT,
        "cluster_text_chars": CLUSTER_TEXT_CHARS,
        "theme_field_prefix": "tag_",
    }
    raw: Dict[str, Any] = {}
    try:
        from mlops.lock import texts_cfg

        value = texts_cfg() or {}
        if isinstance(value, dict):
            raw = value
    except Exception:
        raw = {}
    cfg: Dict[str, Any] = dict(defaults)
    for key, default in defaults.items():
        if key == "theme_field_prefix":
            # Префикс полей готовых тем датасета (подсказка для названий кластеров).
            cfg[key] = str(raw.get(key) or default).strip() or str(default)
            continue
        number = _num(raw.get(key))
        if number is not None and number > 0:
            cfg[key] = int(number)
    # Пороги не должны противоречить друг другу: читать целиком — только до порога полного
    # чтения, кластеризация — только после него.
    cfg["full_read_limit"] = max(MIN_USEFUL_LIMIT, int(cfg["full_read_limit"]))
    cfg["warn_read_limit"] = max(
        MIN_USEFUL_LIMIT, min(int(cfg["warn_read_limit"]), int(cfg["full_read_limit"]))
    )
    cfg["cluster_min_messages"] = max(int(cfg["cluster_min_messages"]), int(cfg["full_read_limit"]))
    return cfg


def _scale_value(value: Any, scale: str) -> float:
    """Сжимающее преобразование одной метрики: sum(weight * f(value)) вместо weight * value."""
    number = _num(value)
    if number is None or number <= 0:
        return 0.0
    if scale == "linear":
        return float(number)
    if scale == "sqrt":
        return math.sqrt(float(number))
    return math.log1p(float(number))


def _scale_painless(source: str, scale: str) -> str:
    """Та же формула на Painless — для runtime-поля Elasticsearch в _engagement_script.

    Без сжатия отбор кандидатов в ES разошёлся бы с итоговым ранжированием в _importance,
    и в топ попадали бы не те сообщения, что посчитал инструмент.
    """
    safe = "Math.max(0.0, %s)" % source
    if scale == "linear":
        return safe
    if scale == "sqrt":
        return "Math.sqrt(%s)" % safe
    return "Math.log(1.0 + %s)" % safe


def _engagement_weights_brief(weights: Dict[str, float]) -> str:
    """Краткая запись применённых весов: «репосты 4, комментарии 3, лайки 2»."""
    rows = sorted(
        ((field, float(weight)) for field, weight in weights.items() if weight),
        key=lambda item: (-item[1], item[0]),
    )
    return ", ".join(
        "%s %s" % (ENGAGEMENT_LABELS.get(field, field), _fmt_num(weight)) for field, weight in rows[:6]
    ) or "нет"


def _engagement_basis(doc: Dict[str, Any]) -> str:
    """Основание сортировки: применённая шкала, веса и ФАКТИЧЕСКИЕ значения метрик.

    В отчёт идут реальные числа из выгрузки (2 184 064), а не логарифмы: пользователь должен видеть,
    на чём основан порядок. Сжатие — деталь расчёта, поэтому в строке указана только сама шкала.
    """
    weights, scale, _enabled = _engagement_config()
    parts = [
        "%s %s" % (ENGAGEMENT_LABELS.get(field, field), _fmt_num(doc.get(field)))
        for field in ENGAGEMENT_ORDER
        if doc.get(field) and weights.get(field)
    ]
    tail = ", ".join(parts) or "счётчики нулевые"
    return "вовлечённость (%s; веса: %s): %s" % (
        ENGAGEMENT_SCALE_LABELS.get(scale, scale), _engagement_weights_brief(weights), tail)


def _doc_from_hit(hit: Dict[str, Any]) -> Dict[str, Any]:
    """Нормализует документ Elasticsearch: текст, атрибуция и все метрики вовлечённости."""
    src = hit.get("_source") or {}
    author = src.get("authorObject") or {}
    raw_text = _flat(src.get("text") or src.get("title") or "")
    doc: Dict[str, Any] = {
        "es_id": hit.get("_id"),
        "text": raw_text,
        "title": _flat(src.get("title")),
        "time": src.get("timeCreate"),
        "hub": _flat(src.get("hub")),
        "hubtype": _flat(src.get("hubtype")),
        "url": _flat(src.get("url")),
        "city": _flat(src.get("city")),
        "author": _flat(author.get("fullname")),
        "author_type": _flat(author.get("author_type")),
        "tone_mark": src.get("toneMark"),
        "rating": _num(src.get("review_rating")),
        "chars": len(raw_text),
    }
    doc["markers"] = _markers(raw_text)
    weights, scale, _enabled = _engagement_config()
    engagement = 0.0
    for field in sorted(set(weights) | set(ENGAGEMENT_WEIGHTS)):
        value = _num(src.get(field))
        doc[field] = value
        weight = weights.get(field)
        if value and weight:
            engagement += _scale_value(value, scale) * float(weight)
    doc["engagement"] = round(engagement, 2)
    return doc


def _importance(doc: Dict[str, Any], engagement_available: bool) -> Tuple[float, str]:
    """Важность сообщения.

    Основной сигнал — вовлечённость: комментарии, репосты, лайки, просмотры, аудитория,
    аудитория СМИ. Метрики берутся из самой выгрузки Brand Analytics, а не оцениваются моделью.
    Считается сумма weight_field * f(value_field), где f — сжимающая шкала (по умолчанию
    логарифмическая), поэтому крупное сообщество с нулём реакций не вытесняет обсуждение.
    ER в сумму не входит: в BA это сумма тех же реакций (см. ENGAGEMENT_WEIGHTS).

    Если в срезе счётчиков нет (типично для отзовиков и сайтов-рекомендаций, которые их вообще
    не публикуют), работает честный и прозрачный откат: важнее то сообщение, у которого ниже
    оценка отзыва, длиннее текст и больше юридических и эмоциональных маркеров. Основание расчёта
    всегда возвращается текстом и попадает в stats.importance_basis и в отчёт.
    """
    if engagement_available and doc.get("engagement"):
        return float(doc["engagement"]), _engagement_basis(doc)
    markers = doc.get("markers") or {}
    marker_points = float(markers.get("total") or 0) * MARKER_WEIGHT
    rating = doc.get("rating")
    if rating is not None:
        score = round((5.0 - rating) * 20.0 + min(doc.get("chars") or 0, 3000) / 300.0 + marker_points, 2)
        return score, "оценка отзыва + объём текста + маркеры (%s) — %s" % (
            _markers_basis(markers), NO_ENGAGEMENT_NOTE)
    score = round((doc.get("chars") or 0) / 100.0 + marker_points, 2)
    return score, "объём текста + маркеры (%s) — %s" % (_markers_basis(markers), NO_ENGAGEMENT_NOTE)


def _numeric_engagement_fields(index_name: str) -> List[str]:
    """Поля вовлечённости, которые в индексе реально числовые.

    Старые датасеты создавались до явного маппинга, поэтому viewsCount и citeIndex попадали в
    Elasticsearch как text (BA присылает пустую строку). max-агрегация по text-полю падает целиком,
    и раньше это выключало вовлечённость во всём срезе, хотя счётчики в данных были.
    """
    from .tools_data import _es

    try:
        mappings = _es().indices.get_mapping(index=index_name)
    except Exception:
        return []
    props = (list(mappings.values())[0].get("mappings") or {}).get("properties") or {}
    candidates = list(ENGAGEMENT_ORDER) + [f for f in ENGAGEMENT_WEIGHTS if f not in ENGAGEMENT_ORDER]
    return [
        field for field in candidates
        if (props.get(field) or {}).get("type") in ES_NUMERIC_TYPES
    ]


def _engagement_agg(index_name: str, query: Dict[str, Any], fields: List[str]) -> Dict[str, float]:
    """Max по полям вовлечённости. Одно «плохое» поле не должно выключать вовлечённость целиком."""
    from .tools_data import _es

    if not fields:
        return {}
    body = {"size": 0, "query": query, "aggs": {field: {"max": {"field": field}} for field in fields}}
    try:
        agg = _es().search(index=index_name, body=body).get("aggregations") or {}
        return {field: float((agg.get(field) or {}).get("value") or 0) for field in fields}
    except Exception:
        pass
    out: Dict[str, float] = {}
    for field in fields:
        try:
            one = _es().search(
                index=index_name,
                body={"size": 0, "query": query, "aggs": {"v": {"max": {"field": field}}}},
            ).get("aggregations") or {}
            out[field] = float((one.get("v") or {}).get("value") or 0)
        except Exception:
            continue
    return out


def _engagement_script(fields: List[str]) -> Optional[Dict[str, Any]]:
    """Runtime-поле __engagement — та же взвешенная сумма метрик, что считает _importance.

    Нужно, чтобы Elasticsearch отдавал именно верхушку по вовлечённости, а не верхушку по
    комментариям: сообщение с большой аудиторией и малым числом реакций иначе вытеснялось бы
    из выборки. Имена полей, веса и шкала берутся из активной настройки вовлечённости,
    скрипт собирается на сервере.
    """
    weights, scale, enabled = _engagement_config()
    if not enabled:
        return None
    parts = []
    for field in fields:
        weight = weights.get(field)
        if not weight:
            continue
        raw = "(doc['%s'].size()==0 ? 0.0 : doc['%s'].value)" % (field, field)
        parts.append("s += (%s) * %s;" % (_scale_painless(raw, scale), float(weight)))
    if not parts:
        return None
    return {"type": "double", "script": {"source": "double s = 0.0; " + " ".join(parts) + " emit(s);"}}


def _fetch_slice(index_name: str, lo, hi, tone: str, limit: int) -> Dict[str, Any]:
    """Берёт сообщения среза из Elasticsearch и определяет, заполнена ли вовлечённость."""
    from .tools_data import _es, _exact_count, _iso, _query

    query = _query(None, lo, hi, tone)
    total = _exact_count(index_name, query)

    numeric = _numeric_engagement_fields(index_name)
    agg_fields = [field for field in ENGAGEMENT_AGG_FIELDS if field in numeric]
    engagement_max = _engagement_agg(index_name, query, agg_fields)
    engagement_enabled = _engagement_config()[2]
    engagement_available = engagement_enabled and any(value > 0 for value in engagement_max.values())
    engagement_fields = [f for f in ENGAGEMENT_ORDER if engagement_max.get(f, 0) > 0]

    # Отбор кандидатов. Если вовлечённость заполнена, Elasticsearch считает по runtime-полю
    # __engagement ту же взвешенную величину, что и _importance, и мы забираем с запасом именно
    # верхушку по ней. Если счётчиков нет (отзовики и сайты-рекомендации в Brand Analytics) —
    # берём самые свежие сообщения среза. Способы отсортированы от лучшего к запасному: текстовый
    # или отсутствующий в старом индексе столбец не должен ломать выборку.
    runtime = None
    size = int(limit)
    candidate_specs: List[Optional[Dict[str, Any]]] = []
    if engagement_available:
        size = min(max(int(limit) * ENGAGEMENT_POOL_FACTOR, int(limit)), ENGAGEMENT_POOL_MAX)
        runtime = _engagement_script(numeric)
        if runtime:
            candidate_specs.append([{"__engagement": {"order": "desc"}}, {"timeCreate": {"order": "desc"}}])
        candidate_specs.append(
            [{field: {"order": "desc"}} for field in ENGAGEMENT_SORT_ORDER
             if field in numeric and engagement_max.get(field, 0) > 0]
            + [{"timeCreate": {"order": "desc"}}]
        )
    candidate_specs.append([{"timeCreate": {"order": "desc"}}])

    res = None
    last_exc: Optional[Exception] = None
    for sort_spec in candidate_specs:
        search_body: Dict[str, Any] = {
            "size": size,
            "_source": SOURCE_FIELDS,
            "query": query,
            "sort": sort_spec,
        }
        if runtime and sort_spec and "__engagement" in sort_spec[0]:
            search_body["runtime_mappings"] = {"__engagement": runtime}
        try:
            res = _es().search(index=index_name, body=search_body)
            break
        except Exception as exc:  # noqa: BLE001 — пробуем следующий способ отбора
            last_exc = exc
            res = None
    if res is None:
        raise ToolError(f"Ошибка выборки сообщений: {last_exc}") from last_exc

    hits = (res.get("hits") or {}).get("hits") or []
    docs: List[Dict[str, Any]] = []
    for hit in hits:
        doc = _doc_from_hit(hit)
        doc["date"] = _iso(doc.get("time"))
        doc["tone"] = TONE_LABELS.get(doc.get("tone_mark"), doc.get("tone_mark"))
        score, basis = _importance(doc, engagement_available)
        doc["importance"] = score
        doc["importance_basis"] = basis
        docs.append(doc)

    # m1 — самое важное сообщение среза: модель видит важное первым, ключевые сообщения
    # отчёта отбираются по этой же величине.
    docs.sort(key=lambda item: -float(item.get("importance") or 0))
    docs = docs[: int(limit)]
    lines: List[str] = []
    docs_by_id: Dict[str, Dict[str, Any]] = {}
    for pos, doc in enumerate(docs, start=1):
        msg_id = f"m{pos}"
        doc["msg_id"] = msg_id
        docs_by_id[msg_id] = doc
        lines.append(_message_line(msg_id, doc))
    return {
        "docs": docs,
        "docs_by_id": docs_by_id,
        "lines": lines,
        "messages_in_slice": total,
        "engagement_available": engagement_available,
        "engagement_fields": engagement_fields,
        "engagement_max": engagement_max,
        "engagement_numeric_fields": numeric,
        "engagement_enabled": engagement_enabled,
    }


# ------------------------------------------------------------------ кластеризация корпуса

async def _cluster_slice(ctx, index_name: str, query: Dict[str, Any],
                         cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Кластеры по всему корпусу среза с прогрессом. None — не получилось (откат на выборку).

    Эмбеддинги, UMAP и HDBSCAN считаются в отдельном потоке, поэтому прогресс оттуда отправляем
    в цикл событий через run_coroutine_threadsafe: запуск не должен выглядеть зависшим, пока
    считаются эмбеддинги корпуса.
    """
    from . import text_cluster

    tracker = getattr(ctx, "progress", None)
    loop = asyncio.get_running_loop()

    def _sub(done: int, total: int, detail: str) -> None:
        if tracker is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(
                tracker.sub(max(0, int(done)), max(1, int(total)),
                            stage="кластеризация корпуса", detail=detail),
                loop,
            )
        except Exception:  # noqa: BLE001 — прогресс не должен ломать расчёт
            pass

    def _progress(stage: str, info: Dict[str, Any]) -> None:
        tail = ", ".join("%s: %s" % (key, value) for key, value in (info or {}).items())
        _sub(0, 1, stage + ((" (" + tail + ")") if tail else ""))

    def _run():
        return text_cluster.cluster_corpus(
            index_name, query, cfg,
            progress=_progress,
            on_page=lambda done, total: _sub(done, min(total, max(done, 1)),
                                             "собираю корпус: %d сообщений" % done),
            on_chunk=lambda done, total: _sub(done, total,
                                              "эмбеддинги корпуса: %d из %d" % (done, total)),
        )

    try:
        if tracker is not None:
            async with tracker.heartbeat("кластеризация корпуса"):
                return await loop.run_in_executor(None, _run)
        return await loop.run_in_executor(None, _run)
    except Exception as exc:  # noqa: BLE001 — честный откат на выборку лучше падения запуска
        await ctx.log(
            "Кластеризация корпуса не удалась (%s: %s) — читаю выборку значимых сообщений, "
            "в отчёте это помечено" % (type(exc).__name__, exc),
            level="error",
        )
        return None


def _corpus_docs(corpus: Dict[str, Any], engagement_available: bool) -> List[Dict[str, Any]]:
    """Нормализует документы корпуса тем же _doc_from_hit, что и выборку среза.

    Важность считается теми же весами вовлечённости, поэтому «ключевые сообщения» в обоих
    режимах сопоставимы, а настройка весов действует одинаково.
    """
    from .tools_data import _iso

    out: List[Dict[str, Any]] = []
    for item in corpus.get("docs") or []:
        doc = _doc_from_hit({"es_id": item.get("es_id"), "_source": item.get("source") or {}})
        doc["date"] = _iso(doc.get("time"))
        doc["tone"] = TONE_LABELS.get(doc.get("tone_mark"), doc.get("tone_mark"))
        score, basis = _importance(doc, engagement_available)
        doc["importance"] = score
        doc["importance_basis"] = basis
        out.append(doc)
    return out


def _corpus_picks(corpus: Dict[str, Any], cfg: Dict[str, Any],
                  engagement_available: bool) -> List[Dict[str, Any]]:
    """Кого читать в режиме кластеризации: представители кластеров + самые значимые сообщения.

    Частоты тем берутся из размеров кластеров по всему корпусу (их считает text_cluster),
    а модель читает нескольких значимых представителей каждого кластера — на них строятся
    цитаты и пояснения — и дополнительно самые значимые сообщения корпуса целиком.
    """
    docs = _corpus_docs(corpus, engagement_available)
    labels = corpus.get("labels") or []
    per_cluster = max(1, int(cfg["cluster_per_cluster"]))
    top_messages = max(0, int(cfg["cluster_top_messages"]))
    read_limit = max(MIN_USEFUL_LIMIT, int(cfg["cluster_read_limit"]))

    picks: List[Dict[str, Any]] = []
    taken: set = set()

    def _take(position: int, cluster_id: int, role: str) -> None:
        taken.add(position)
        docs[position]["cluster_id"] = int(cluster_id)
        docs[position]["corpus_role"] = role
        picks.append(docs[position])

    for cluster in corpus.get("clusters") or []:
        members = [pos for pos in (cluster.get("members") or []) if 0 <= pos < len(docs)]
        members.sort(key=lambda pos: -float(docs[pos].get("importance") or 0))
        taken_in_cluster = 0
        for pos in members:
            if taken_in_cluster >= per_cluster or len(picks) >= read_limit:
                break
            if pos in taken:
                continue
            _take(pos, int(cluster.get("id", -1)), "cluster")
            taken_in_cluster += 1

    if top_messages:
        order = sorted(range(len(docs)), key=lambda pos: -float(docs[pos].get("importance") or 0))
        for pos in order:
            if len(picks) >= read_limit or top_messages <= 0:
                break
            if pos in taken:
                continue
            label = int(labels[pos]) if pos < len(labels) else -1
            _take(pos, label, "top")
            top_messages -= 1
    return picks


def _cluster_rows(corpus: Dict[str, Any], merged_topics: List[Dict[str, Any]],
                  docs_by_id: Dict[str, Dict[str, Any]], corpus_total: int) -> List[Dict[str, Any]]:
    """Темы отчёта при кластеризации: частоты по всему корпусу + цитаты из прочитанных представителей.

    Название кластера берём у модели (её тема пересекается по msg_id с прочитанными
    представителями), иначе — подсказка из готовых тем датасета (Brand Analytics), иначе —
    ключевые слова кластера. Тональность кластера посчитана по всем его сообщениям в корпусе.
    """
    by_cluster: Dict[int, List[Dict[str, Any]]] = {}
    for doc in docs_by_id.values():
        cluster_id = doc.get("cluster_id")
        if cluster_id is None:
            continue
        by_cluster.setdefault(int(cluster_id), []).append(doc)

    rows: List[Dict[str, Any]] = []
    for cluster in corpus.get("clusters") or []:
        cluster_id = int(cluster.get("id", -1))
        members = sorted(by_cluster.get(cluster_id) or [],
                         key=lambda item: -float(item.get("importance") or 0))
        quotes = [entry for entry in (_quote_entry(doc, "") for doc in members[:3]) if entry]
        msg_ids = [str(doc.get("msg_id")) for doc in members if doc.get("msg_id")]
        name, essence = "", ""
        for row in merged_topics:
            if msg_ids and set(msg_ids) & set(row.get("msg_ids") or []):
                if not name:
                    name = _flat(row.get("topic")) or _flat(row.get("name"))
                if not essence:
                    essence = _flat(row.get("essence"))
                if name and essence:
                    break
        if not name:
            name = (str(cluster.get("tag_hint") or "").strip()
                    or ", ".join(str(word) for word in (cluster.get("keywords") or [])[:4])
                    or "Кластер %d" % cluster_id)
        count = int(cluster.get("size") or 0)
        rows.append({
            "topic": name,
            "count": count,
            "share": round(count / float(corpus_total or 1), 4),
            "share_pct": round(100.0 * count / float(corpus_total or 1), 1),
            "tone": str(cluster.get("tone_label") or "—"),
            "tone_avg": cluster.get("tone_avg"),
            "model_tone": "",
            "essence": essence,
            "category": _match_category("", name, [entry["text"] for entry in quotes]),
            "msg_ids": msg_ids,
            "quotes": quotes,
            "importance": round(
                sum(float(doc.get("importance") or 0) for doc in members) / len(members), 2
            ) if members else 0.0,
            "corpus": True,
            "cluster_id": cluster_id,
            "tag_hint": str(cluster.get("tag_hint") or ""),
            "keywords": list(cluster.get("keywords") or []),
        })
    rows.sort(key=lambda item: (-item["count"], -item["importance"]))
    return rows


def _categories_from_rows(rows: List[Dict[str, Any]], total: int) -> List[Dict[str, Any]]:
    """Категории каркаса по готовым темам: те же правила, что и в _merge."""
    buckets: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        bucket = buckets.setdefault(row["category"], {"category": row["category"], "count": 0, "topics": [], "share": 0.0})
        bucket["count"] += int(row.get("count") or 0)
        bucket["topics"].append(row["topic"])
    for bucket in buckets.values():
        bucket["share"] = round(bucket["count"] / float(total or 1), 4)
        bucket["share_pct"] = round(100.0 * bucket["count"] / float(total or 1), 1)
    return sorted(buckets.values(), key=lambda item: -item["count"])


def _message_line(msg_id: str, doc: Dict[str, Any]) -> str:
    """Строка сообщения для промпта: id, площадка, дата, метрики и текст."""
    metrics = []
    if doc.get("rating") is not None:
        metrics.append(f"рейтинг {doc['rating']:g}")
    for field, label in PROMPT_METRICS:
        if doc.get(field):
            metrics.append(f"{label} {doc[field]:g}")
    tail = (", " + ", ".join(metrics)) if metrics else ""
    return "[%s] (%s, %s, %s%s) %s" % (
        msg_id, doc.get("hub") or "источник", doc.get("date") or "", doc.get("tone") or "—", tail,
        (doc.get("text") or "")[:MESSAGE_CHARS],
    )


def _chunks(docs: List[Dict[str, Any]], batch_size: int) -> List[List[Dict[str, Any]]]:
    """Режет сообщения на пачки: по количеству и по бюджету символов промпта."""
    batches: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    size = 0
    for doc in docs:
        length = min(len(doc.get("text") or ""), MESSAGE_CHARS) + 60
        if current and (len(current) >= batch_size or size + length > BATCH_CHAR_BUDGET):
            batches.append(current)
            current, size = [], 0
        current.append(doc)
        size += length
    if current:
        batches.append(current)
    return batches


async def _cancel_watcher(ctx, tasks: List[Any]) -> None:
    """Гасит незавершённые пачки, как только пользователь остановил запуск.

    Отмена asyncio-задачи закрывает HTTP-запрос к vLLM: сервер видит обрыв клиента и
    освобождает слот генерации, поэтому очередь модели не забивается.
    """
    try:
        while True:
            await asyncio.sleep(1.0)
            if ctx.cancelled():
                for task in tasks:
                    if not task.done():
                        task.cancel()
                return
    except asyncio.CancelledError:
        raise


# ------------------------------------------------------------- разбор пачек моделью

async def _run_batch(ctx, number: int, total: int, scope: str, focus: str, batch: List[str],
                     bulk: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Один вызов локальной модели по пачке сообщений; ответ — строго JSON.

    Обрыв по лимиту генерации раньше проходил незамеченным: JSON оставался незакрытым,
    _extract_json возвращал первый вложенный объект, и пачка «читалась успешно» без единой темы.
    Теперь finish_reason виден (finish_reason="length"), обрезанная пачка делится пополам и
    повторяется, а если не разобралась и после дробления — шаг честно сообщает об ошибке.
    """
    cfg = (bulk or {}).get("vllm_cfg")
    max_tokens = int((bulk or {}).get("max_tokens") or BATCH_MAX_TOKENS)
    max_depth = 2 if len(batch) > 4 else 1
    tokens = 0
    errors: List[str] = []
    splits: List[str] = []
    state: Dict[str, Any] = {"model": "", "fallback": 0}

    async def _read(lines: List[str], depth: int) -> List[Dict[str, Any]]:
        nonlocal tokens
        prompt = BATCH_INSTRUCTION.format(
            scope=scope,
            focus=focus,
            categories=", ".join(CATEGORY_FRAMEWORK),
            messages="\n".join(lines),
        )
        # Сначала профиль быстрого чтения (если включён), затем общий vLLM: недоступный
        # порт 8001 не должен останавливать разбор — откат на 32B обязателен.
        candidates: List[Any] = [cfg, None] if cfg else [None]
        last_error = ""
        for attempt in range(2 if depth else 1):
            for vllm_cfg in candidates:
                meta: Dict[str, Any] = {}
                try:
                    text, used = await _qwen(
                        ctx, prompt, system=BATCH_SYSTEM, max_tokens=max_tokens,
                        vllm_cfg=vllm_cfg, meta=meta,
                    )
                    tokens += used
                except Exception as exc:  # noqa: BLE001 — пачка не должна ломать весь разбор
                    last_error = f"{type(exc).__name__}: {exc}"
                    if vllm_cfg is not None:
                        state["fallback"] += 1
                        await ctx.log(
                            f"Быстрая модель чтения не ответила ({last_error[:140]}) — "
                            f"читаю пачку {number} на общей модели",
                            level="error",
                        )
                    continue
                if vllm_cfg is not None and meta.get("model"):
                    state["model"] = str(meta["model"])
                finish = str(meta.get("finish_reason") or "")
                parsed = _extract_json(text)
                if parsed and isinstance(parsed.get("topics"), list) and finish != "length":
                    return [parsed]
                if finish == "length":
                    last_error = (
                        f"ответ обрезан по лимиту {max_tokens} токенов (finish_reason=length)"
                    )
                    break
                last_error = "в ответе нет списка topics — повторили с напоминанием про JSON"
                prompt = prompt + "\n\nНапоминаю: ответ — только JSON, без текста вокруг."
        raise ToolError(last_error or "нет ответа модели")

    async def _read_split(lines: List[str], depth: int) -> "Tuple[List[Dict[str, Any]], int]":
        """Читает пачку; при обрыве делит её пополам. Возвращает (разборы, сколько прочитано)."""
        try:
            return await _read(lines, depth), len(lines)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{len(lines)} сообщений: {exc}")
            if len(lines) <= 2 or depth >= max_depth:
                return [], 0
            half = len(lines) // 2
            splits.append(f"{len(lines)}\u2192{half}+{len(lines) - half}")
            parsed: List[Dict[str, Any]] = []
            read = 0
            for part in (lines[:half], lines[half:]):
                got, done = await _read_split(part, depth + 1)
                parsed.extend(got)
                read += done
            return parsed, read

    parsed_list, read = await _read_split(batch, 0)
    if not parsed_list:
        message = "; ".join(errors[:2]) or "нет ответа модели"
        await ctx.log(f"Пачка {number}/{total} не разобрана: {message}", level="error")
        return {"ok": False, "error": message, "tokens": tokens, "read": 0,
                "model": state["model"], "fallback": state["fallback"]}
    topics_count = sum(len(item.get("topics") or []) for item in parsed_list)
    detail = f"Пачка {number}/{total}: разобрано тем {topics_count}"
    if splits:
        detail += f" (пачка делилась: {', '.join(splits)})"
    if errors:
        detail += f"; не разобрано: {'; '.join(errors[:2])}"
    await ctx.log(detail)
    return {"ok": True, "parsed": parsed_list, "tokens": tokens, "read": read,
            "partial": bool(errors), "model": state["model"], "fallback": state["fallback"]}


# ------------------------------------------------------------------ склейка результата

def _match_category(raw: Any, topic_name: str, quotes: List[str]) -> str:
    """Приводит категорию темы к фиксированному каркасу."""
    candidate = _flat(raw).lower().strip()
    for known in CATEGORY_FRAMEWORK:
        if candidate == known:
            return known
    if candidate:
        for known in CATEGORY_FRAMEWORK:
            head = known.split("/")[0]
            if head and (head in candidate or candidate in known):
                return known
    haystack = (_flat(topic_name) + " " + " ".join(_flat(q) for q in quotes)).lower()
    best, best_score = "другое", 0
    for known, keywords in CATEGORY_KEYWORDS:
        score = sum(1 for word in keywords if word in haystack)
        if score > best_score:
            best, best_score = known, score
    return best


def _quote_entry(doc: Optional[Dict[str, Any]], raw_quote: str) -> Optional[Dict[str, Any]]:
    """Собирает цитату с атрибуцией. Цитата обязана быть дословной — иначе берём выдержку из текста."""
    if doc is None:
        return None
    text = _flat(doc.get("text"))
    quote = _flat(raw_quote)
    verified = False
    if quote and text:
        probe = _flat(quote).strip("«»\"' ").lower()
        if len(probe) > 12:
            flat_text = text.lower()
            if probe in flat_text:
                verified = True
            else:  # модель могла склеить несколько предложений — проверяем по началу цитаты
                head = probe[:60]
                verified = bool(head) and head in flat_text
    if not verified:
        quote = _snippet(text)  # гарантированно дословная выдержка из сообщения
    return {
        "text": quote,
        "verified": verified,
        "date": doc.get("date") or "",
        "hub": doc.get("hub") or "",
        "author": doc.get("author") or "",
        "url": doc.get("url") or "",
        "tone": doc.get("tone"),
        "rating": doc.get("rating"),
        "msg_id": doc.get("msg_id"),
    }


def _merge(parsed_batches: List[Dict[str, Any]], docs_by_id: Dict[str, Dict[str, Any]], total_read: int) -> Dict[str, Any]:
    """Склеивает темы из пачек. Частоты считаются по msg_id, а не по арифметике модели."""
    topics: Dict[str, Dict[str, Any]] = {}
    highlight_votes: Dict[str, str] = {}
    for parsed in parsed_batches:
        for item in parsed.get("topics") or []:
            if not isinstance(item, dict):
                continue
            name = _flat(item.get("name"))
            if not name:
                continue
            key = _norm_key(name) or name.lower()
            slot = topics.setdefault(
                key,
                {"name": name, "msg_ids": set(), "raw_quotes": {}, "essence": "", "category_raw": "", "tone_raw": ""},
            )
            if len(name) > len(slot["name"]) and slot["name"].lower() in name.lower():
                slot["name"] = name  # предпочитаем более развёрнутое название темы
            for msg_id in item.get("msg_ids") or []:
                msg_id = _flat(msg_id)
                if msg_id in docs_by_id:
                    slot["msg_ids"].add(msg_id)
            for quote in item.get("quotes") or []:
                if not isinstance(quote, dict):
                    continue
                msg_id = _flat(quote.get("msg_id"))
                if msg_id in docs_by_id and _flat(quote.get("quote")):
                    slot["raw_quotes"].setdefault(msg_id, _flat(quote.get("quote")))
            if not slot["essence"] and _flat(item.get("essence")):
                slot["essence"] = _flat(item.get("essence"))
            if not slot["category_raw"] and _flat(item.get("category")):
                slot["category_raw"] = _flat(item.get("category"))
            if not slot["tone_raw"] and _flat(item.get("tone")):
                slot["tone_raw"] = _flat(item.get("tone"))
        for item in parsed.get("highlights") or []:
            if isinstance(item, dict):
                msg_id = _flat(item.get("msg_id"))
                if msg_id in docs_by_id:
                    highlight_votes.setdefault(msg_id, _flat(item.get("why")))

    rows: List[Dict[str, Any]] = []
    for slot in topics.values():
        msg_ids = sorted(slot["msg_ids"], key=lambda mid: -float(docs_by_id[mid].get("importance") or 0))
        if not msg_ids:
            continue
        tones = [docs_by_id[mid].get("tone_mark") for mid in msg_ids if isinstance(docs_by_id[mid].get("tone_mark"), (int, float))]
        tone_avg = round(sum(tones) / len(tones), 2) if tones else None
        # Тема с негативом и позитивом одновременно — «смешанная», иначе средняя тональность темы
        if tones and min(tones) < 0 < max(tones):
            tone_label = "смешанная"
        elif tone_avg is not None:
            tone_label = TONE_LABELS.get(int(round(tone_avg)), "смешанная")
        else:
            tone_label = "—"
        # 2–3 цитаты на тему: сначала те, что назвала модель, затем — из самых важных сообщений темы
        picked: List[str] = [mid for mid in msg_ids if mid in slot["raw_quotes"]][:3]
        for msg_id in msg_ids:
            if len(picked) >= 3:
                break
            if msg_id not in picked:
                picked.append(msg_id)
        quotes = [q for q in (_quote_entry(docs_by_id[mid], slot["raw_quotes"].get(mid, "")) for mid in picked[:3]) if q]
        rows.append(
            {
                "topic": slot["name"],
                "count": len(msg_ids),
                "share": round(len(msg_ids) / total_read, 4) if total_read else 0.0,
                "share_pct": round(100.0 * len(msg_ids) / total_read, 1) if total_read else 0.0,
                "tone": tone_label,
                "tone_avg": tone_avg,
                "model_tone": slot["tone_raw"],
                "essence": slot["essence"],
                "category": _match_category(slot["category_raw"], slot["name"], [q["text"] for q in quotes]),
                "msg_ids": msg_ids,
                "quotes": quotes,
                "importance": round(sum(float(docs_by_id[mid].get("importance") or 0) for mid in msg_ids) / len(msg_ids), 2),
            }
        )
    rows.sort(key=lambda item: (-item["count"], -item["importance"]))

    categories: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        bucket = categories.setdefault(row["category"], {"category": row["category"], "count": 0, "topics": [], "share": 0.0})
        bucket["count"] += row["count"]
        bucket["topics"].append(row["topic"])
    for bucket in categories.values():
        bucket["share"] = round(bucket["count"] / total_read, 4) if total_read else 0.0
        bucket["share_pct"] = round(100.0 * bucket["count"] / total_read, 1) if total_read else 0.0
    ordered_categories = sorted(categories.values(), key=lambda item: -item["count"])

    highlights: List[Dict[str, Any]] = []
    for msg_id, why in highlight_votes.items():
        doc = docs_by_id[msg_id]
        highlights.append(_highlight_entry(doc, why, model_picked=True))
    listed = {item["msg_id"] for item in highlights}
    for doc in sorted(docs_by_id.values(), key=lambda item: -float(item.get("importance") or 0)):
        if len(highlights) >= 8:
            break
        if doc.get("msg_id") in listed:
            continue
        highlights.append(_highlight_entry(doc, "", model_picked=False))
        listed.add(doc.get("msg_id"))
    highlights.sort(key=lambda item: (-float(item.get("importance") or 0)))
    return {"topics": rows, "categories": ordered_categories, "highlights": highlights[:8]}


def _highlight_entry(doc: Dict[str, Any], why: str, *, model_picked: bool) -> Dict[str, Any]:
    return {
        "msg_id": doc.get("msg_id"),
        "text": _snippet(doc.get("text"), 320),
        "why": why,
        "model_picked": model_picked,
        "date": doc.get("date") or "",
        "hub": doc.get("hub") or "",
        "author": doc.get("author") or "",
        "url": doc.get("url") or "",
        "tone": doc.get("tone"),
        "rating": doc.get("rating"),
        "importance": doc.get("importance"),
        "importance_basis": doc.get("importance_basis"),
        "engagement": doc.get("engagement"),
        "metrics": {
            field: doc.get(field) for field in ENGAGEMENT_ORDER if doc.get(field)
        },
        "markers": doc.get("markers") or {},
    }


def _fallback_summary(topics: List[Dict[str, Any]], total_read: int, negative: int, highlight: List[Dict[str, Any]]) -> str:
    """Текст раздела без модели: темы, доли и цитаты всё равно должны попасть в отчёт."""
    if not topics:
        return f"Прочитано {total_read} сообщений, устойчивых тем из текстов выделить не удалось."
    top = topics[0]
    lines = [
        f"Прочитано {total_read} сообщений среза, негативных среди них {negative}. "
        f"Самая частая тема — «{top['topic']}»: {top['count']} сообщений ({top['share_pct']}% выборки).",
        "Распределение по темам: " + "; ".join(f"«{row['topic']}» — {row['count']} ({row['share_pct']}%)" for row in topics[:6]) + ".",
    ]
    for row in topics[:3]:
        if row.get("quotes"):
            quote = row["quotes"][0]
            lines.append(f"Тема «{row['topic']}»: «{quote['text']}» ({quote['hub']}, {quote['date']}, {quote['author']}).")
    if highlight:
        lines.append(f"Самое вовлекающее сообщение среза ({highlight[0].get('importance_basis')}): «{highlight[0]['text']}».")
    return "\n\n".join(lines)


# --------------------------------------------------- компактные формы для ответа
# Списки отдаём обычными массивами объектов: результат читает и модель, и build_report,
# поэтому служебных элементов внутри списков быть не должно.

def _public_quote(quote: Dict[str, Any], limit: int = 300) -> Dict[str, Any]:
    return {
        "text": _snippet(quote.get("text"), limit),
        "date": quote.get("date") or "",
        "hub": quote.get("hub") or "",
        "author": quote.get("author") or "",
        "url": quote.get("url") or "",
        "tone": quote.get("tone"),
        "rating": quote.get("rating"),
        "verified": bool(quote.get("verified")),
    }


def _public_topics(topics: List[Dict[str, Any]], limit: int = 12) -> List[Dict[str, Any]]:
    rows = []
    for row in topics[:limit]:
        rows.append(
            {
                "topic": row["topic"],
                "count": row["count"],
                "share": row["share"],
                "share_pct": row["share_pct"],
                "tone": row["tone"],
                "category": row["category"],
                "essence": row["essence"],
                "importance": row["importance"],
                "msg_ids": row["msg_ids"][:20],
                "quotes": [_public_quote(q) for q in (row.get("quotes") or [])[:2]],
            }
        )
    return rows


def _public_categories(categories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "category": row["category"],
            "count": row["count"],
            "share": row["share"],
            "share_pct": row["share_pct"],
            "topics": row["topics"][:8],
        }
        for row in categories
    ]


def _public_highlights(highlights: List[Dict[str, Any]], limit: int = 8) -> List[Dict[str, Any]]:
    return [
        {
            "text": item["text"],
            "why": item.get("why") or "",
            "date": item.get("date") or "",
            "hub": item.get("hub") or "",
            "author": item.get("author") or "",
            "url": item.get("url") or "",
            "tone": item.get("tone"),
            "rating": item.get("rating"),
            "importance": item.get("importance"),
            "importance_basis": item.get("importance_basis") or "",
            "engagement": item.get("engagement"),
            "metrics": item.get("metrics") or {},
        }
        for item in highlights[:limit]
    ]


# ------------------------------------------------------------------------- инструмент

@tool(
    "analyze_texts",
    title="Читать тексты датасета (локальная модель)",
    description=(
        "Читает САМИ ТЕКСТЫ сообщений датасета за период локальной моделью Qwen и отдаёт темы с числом "
        "сообщений, долями, средней тональностью и цитатами (с датой, площадкой, автором и ссылкой), "
        "сопоставление тем с каркасом категорий (доставка, качество товара, цена, возврат/деньги, поддержка, "
        "упаковка, сроки, другое), ключевые сообщения по вовлечённости и аналитический текст. "
        "Вызывай этот инструмент ПЕРЕД build_report, чтобы выводы и пояснения к графикам опирались на темы "
        "и цитаты из текстов, а не только на статистику. Обязателен, когда в срезе есть негатив: "
        "причины жалоб нужно подтверждать цитатами. Срез до 20 000 сообщений читается ЦЕЛИКОМ "
        "(5 000–20 000 — дольше, но тоже целиком), и тогда темы и цитаты построены по всем сообщениям "
        "среза; в срезе больше 20 000 сообщений кластеры считаются по всему корпусу, а модель читает "
        "представителей кластеров и самые значимые сообщения: частоты тем берутся по всему корпусу, "
        "цитаты — из прочитанного, и это прямо помечается в отчёте. Пачки читаются параллельно, "
        "limit и batch_size указывать не нужно."
    ),
    parameters={
        "type": "object",
        "properties": {
            "index": {"type": "string", "description": "тема: название датасета или её номер; по умолчанию — выбранный в интерфейсе"},
            "min_date": {"type": "string", "description": "начало периода: YYYY-MM-DD, ISO или unix-секунды"},
            "max_date": {"type": "string", "description": "конец периода: YYYY-MM-DD, ISO или unix-секунды"},
            "tone": {"type": "string", "enum": ["all", "negative", "positive", "neutral"], "description": "какие сообщения читать: all — весь срез периода (по умолчанию), negative только сужает срез до негатива"},
            "limit": {"type": "integer", "description": "сколько сообщений прочитать: по умолчанию весь срез (до 20 000) или выборка представителей кластеров при кластеризации. Занижать не нужно — меньше 20 инструмент поднимает сам"},
            "batch_size": {"type": "integer", "description": "сообщений в одной пачке для модели, 15–25 (по умолчанию 20); указывать не обязательно"},
            "focus": {"type": "string", "description": "на что смотреть в первую очередь (например «причины возвратов»)"},
        },
    },
    group="analytics",
    timeout=7200.0,
)
async def analyze_texts(
    ctx,
    index: Optional[int] = None,
    min_date: Any = None,
    max_date: Any = None,
    tone: str = "all",
    limit: int = DEFAULT_LIMIT,
    batch_size: int = BATCH_SIZE,
    focus: Optional[str] = None,
    parallel: Optional[int] = None,
):
    from .tools_data import _iso, _require_period, dates, guard

    started = time.time()
    idx, index_name = guard(ctx, index)
    lo, hi = dates(ctx, min_date, max_date)
    lo, hi = _require_period(index_name, lo, hi)

    # Служебный параметр (нет в схеме инструмента): сколько пачек читать одновременно.
    # Нужен для замеров и тонкой настройки под нагрузку vLLM.
    # Профиль быстрого чтения: своя модель, своя пачка и свой параллелизм (по умолчанию 8).
    bulk = _bulk_profile()
    parallel_batches = max(1, min(int(parallel or bulk.get("parallel") or PARALLEL_BATCHES),
                                  MAX_PARALLEL_BATCHES))

    if bulk.get("batch_size") and int(batch_size or 0) == BATCH_SIZE:
        # Профиль может задать свой размер пачки; явно переданный аргумент важнее профиля.
        batch_size = int(bulk["batch_size"])
    batch_size = max(MIN_BATCH_SIZE, min(int(batch_size or BATCH_SIZE), MAX_BATCH_SIZE))
    texts_cfg = _texts_config()
    full_read = max(MIN_USEFUL_LIMIT, int(texts_cfg["full_read_limit"]))
    requested_limit = max(1, min(int(limit or DEFAULT_LIMIT), full_read))
    limit = requested_limit
    if limit < MIN_USEFUL_LIMIT:
        # Планировщик (или внешний вызов) мог поставить предел «на одну пачку» — например 15,
        # как размер пачки по умолчанию. Тогда срез оставался непрочитанным: пользователь видел
        # «прочитано 15 из 41» и тишину в журнале. Читаем весь срез до DEFAULT_LIMIT.
        limit = DEFAULT_LIMIT
        await ctx.log(
            f"Предел чтения {requested_limit} меньше одной пачки — читаю весь срез "
            f"(до {DEFAULT_LIMIT} сообщений, пачек по {batch_size})"
        )
    focus_text = f" Особое внимание: {_flat(focus)}." if _flat(focus) else ""
    scope = f" за период {_iso(lo)} — {_iso(hi)}" + (f" (тональность: {tone})" if str(tone).lower() not in ("all", "any", "") else "")

    # ------------------------------------------------------- стратегия чтения среза
    # Пороги (mlops/lock.yaml, секция texts): до 5 000 сообщений читаем срез ЦЕЛИКОМ; 5 000–20 000 —
    # тоже целиком, но предупреждаем, что это долго; больше 20 000 — локальной моделью весь корпус
    # не прочитать, поэтому кластеры считаются по всему корпусу (эмбеддинги проекта + UMAP/HDBSCAN),
    # а модель читает представителей кластеров и самые значимые сообщения. Частоты тем в отчёте —
    # по всему корпусу, цитаты и пояснения — из прочитанных представителей.
    from .tools_data import _exact_count, _query

    slice_query = _query(None, lo, hi, tone)
    try:
        slice_total = int(_exact_count(index_name, slice_query))
    except Exception:  # noqa: BLE001 — без точного размера работаем по выборке
        slice_total = 0
    warn_read = int(texts_cfg["warn_read_limit"])
    cluster_min = int(texts_cfg["cluster_min_messages"])

    corpus = None
    strategy = "full"
    if slice_total > cluster_min:
        strategy = "corpus"
        await ctx.log(
            f"В срезе {slice_total} сообщений — больше {cluster_min}: считаю кластеры по всему "
            "корпусу, а модель прочитает представителей кластеров и самые значимые сообщения. "
            "Частоты тем в отчёте будут посчитаны по всему корпусу"
        )
        corpus = await _cluster_slice(ctx, index_name, slice_query, texts_cfg)
        if corpus is None:
            strategy = "sample"
    elif slice_total > warn_read:
        strategy = "full_long"
        minutes = max(1, int(round(slice_total / 450.0)))
        await ctx.log(
            f"В срезе {slice_total} сообщений: читаю ВЕСЬ срез целиком. Это долго — обычно около "
            f"{minutes} мин (пачек по {batch_size}, {parallel_batches} параллельно); прогресс виден в журнале"
        )

    if strategy in ("full", "full_long"):
        limit = max(MIN_USEFUL_LIMIT, min(slice_total or requested_limit, full_read))
    elif strategy == "corpus":
        clusters_n = len(corpus.get("clusters") or [])
        limit = max(MIN_USEFUL_LIMIT, min(
            int(texts_cfg["cluster_read_limit"]),
            clusters_n * max(1, int(texts_cfg["cluster_per_cluster"]))
            + max(0, int(texts_cfg["cluster_top_messages"])),
        ))
    else:
        limit = max(MIN_USEFUL_LIMIT, min(
            int(texts_cfg["cluster_read_limit"]),
            max(requested_limit, int(texts_cfg["cluster_top_messages"])),
        ))
        await ctx.log("Срез большой: читаю выборку значимых сообщений, в отчёте это помечено")

    found = _fetch_slice(index_name, lo, hi, tone, limit)
    if not found["docs"] and (ctx.min_date or ctx.max_date) and (lo, hi) != (ctx.min_date, ctx.max_date):
        # Модель могла указать период «от себя». Если по периоду задачи сообщения есть,
        # читаем период задачи: иначе запуск заканчивается впустую («прочитано 0 сообщений»),
        # а отчёт помечается неполным из-за отсутствия тем и цитат.
        retry = _fetch_slice(index_name, ctx.min_date, ctx.max_date, tone, limit)
        if retry["docs"]:
            await ctx.log(
                f"По периоду {_iso(lo)} — {_iso(hi)} сообщений нет — читаю период задачи "
                f"{_iso(ctx.min_date)} — {_iso(ctx.max_date)}"
            )
            lo, hi = ctx.min_date, ctx.max_date
            found = retry
    docs, docs_by_id, lines = found["docs"], found["docs_by_id"], found["lines"]
    if not docs:
        if not int(found.get("messages_in_slice") or 0):
            # За выбранный период сообщений нет вообще: помечаем запуск как «данные не найдены»,
            # иначе модель напишет отчёт-заглушку и запуск закроется как успешный.
            ctx.no_data = "данные за период не найдены"
            await ctx.log("За указанный период сообщений нет — данные не найдены", level="error")
        return {
            "index": idx,
            "index_name": index_name,
            "messages_analyzed": 0,
            "messages_in_slice": int(found.get("messages_in_slice") or 0),
            "topics": [],
            "highlights": [],
            "categories": [],
            "summary": "",
            "stats": {"messages_processed": 0, "messages_in_slice": int(found.get("messages_in_slice") or 0), "batches": 0, "seconds": round(time.time() - started, 1)},
            "note": "За указанный период сообщений нет — читать нечего. Проверьте период и тональность.",
        }

    if corpus is not None:
        # В режиме кластеризации читаем не срез, а представителей кластеров и самые значимые
        # сообщения корпуса: по ним модель даёт цитаты и пояснения к темам, частоты которых
        # посчитаны по всему корпусу.
        picks = _corpus_picks(corpus, texts_cfg, bool(found.get("engagement_available")))[:limit]
        docs_by_id = {}
        lines = []
        for position, doc in enumerate(picks, start=1):
            msg_id = "m%d" % position
            doc["msg_id"] = msg_id
            docs_by_id[msg_id] = doc
            lines.append(_message_line(msg_id, doc))
        found["docs"], found["docs_by_id"], found["lines"] = picks, docs_by_id, lines
        await ctx.log(
            f"Читаю представителей кластеров и самые значимые сообщения: кластеров "
            f"{len(corpus.get('clusters') or [])}, сообщений к прочтению {len(picks)}"
        )

    # Предел числа пачек — по фактическому размеру чтения: в режиме «целиком» это сотни пачек,
    # и обрезать их нельзя, иначе срез останется недочитанным.
    max_batches = max(1, (int(limit) // max(1, MIN_BATCH_SIZE)) + 2)
    batches = _chunks(docs, batch_size)[:max_batches]
    # Дальше работают только те сообщения, которые реально ушли в модель.
    batches = [batch for batch in batches if batch]
    read_ids = {doc["msg_id"] for batch in batches for doc in batch}
    docs_by_id = {mid: doc for mid, doc in found["docs_by_id"].items() if mid in read_ids}
    selected = len(read_ids)
    slice_total = int(found.get("messages_in_slice") or 0) or len(docs)
    if selected < len(docs):
        await ctx.log(
            f"Читаю {selected} сообщений из {len(docs)} отобранных "
            f"(пачек не больше {max_batches} по {batch_size} сообщений)"
        )
    if selected < slice_total:
        # Не скрываем от пользователя, что срез прочитан не целиком, и объясняем, как это
        # отражено в отчёте: при большом срезе темы считаются по всему корпусу, а модель
        # читает значимую выборку — это помечено в разделе и в scope_note.
        if strategy == "corpus":
            await ctx.log(
                f"Срез {slice_total} сообщений: частоты тем считаю по всему корпусу (кластеров "
                f"{len(corpus.get('clusters') or [])}), читаю {selected} представителей и значимых сообщений"
            )
        elif strategy != "full":
            await ctx.log(
                f"Срез {slice_total} сообщений: читаю выборку значимых сообщений ({selected}), "
                "в отчёте это помечено"
            )
        else:
            await ctx.log(
                f"В срезе {slice_total} сообщений, читаю {selected}: предел чтения {limit}. "
                f"Для полного среза вызывайте инструмент без limit (по умолчанию {DEFAULT_LIMIT}, "
                f"максимум {full_read})"
            )

    if not docs_by_id:
        raise ToolError("Не удалось подготовить сообщения для чтения — повторите запуск")

    negative = sum(1 for doc in docs_by_id.values() if doc.get("tone_mark") == -1)
    positive = sum(1 for doc in docs_by_id.values() if doc.get("tone_mark") == 1)
    neutral = sum(1 for doc in docs_by_id.values() if doc.get("tone_mark") == 0)
    model_label = _gateway_model_label(bulk)
    if bulk:
        # Человеческая строка без технических деталей (адрес, порт, лимиты): пользователю важно
        # понимать, что чтение идёт быстрой моделью, а не читать параметры инстанса.
        await ctx.log("Читаю тексты быстрой моделью — это в несколько раз быстрее обычного")
    await ctx.log(
        f"Чтение текстов: {selected} сообщений, {len(batches)} пачек, модель {model_label}. "
        f"Тональность среза: негатив {negative}, нейтрал {neutral}, позитив {positive}"
    )

    semaphore = asyncio.Semaphore(parallel_batches)
    results: List[Dict[str, Any]] = []
    tokens_used = 0
    # Прогресс по пачкам: это самый долгий шаг запуска, и раньше здесь была «тишина» на 2–3 минуты —
    # пользователь не понимал, работает задача или зависла. Теперь на старте сообщаем, сколько
    # сообщений и пачек читаем и сколько это обычно занимает, после каждой пачки обновляем прогресс,
    # а между пачками идёт heartbeat (не реже раза в 8 секунд).
    tracker = getattr(ctx, "progress", None)
    read_done = 0
    batches_done = 0
    reading_started = time.time()
    if tracker is not None:
        waves = -(-len(batches) // max(1, parallel_batches))
        await tracker.sub(
            0,
            selected,
            stage="чтение текстов",
            detail=(
                (f"читаю выборку значимых сообщений: пачек {len(batches)} по {batch_size}, "
                 if strategy != "full" else
                 f"читаю {selected} сообщений среза: пачек {len(batches)} по {batch_size}, ")
                + f"{min(parallel_batches, len(batches))} параллельно (обычно ~{waves * 2} мин)"
            ),
            units_done=0,
            units_total=len(batches),
            units_parallel=parallel_batches,
        )

    async def worker(number: int, batch: List[Dict[str, Any]]) -> None:
        nonlocal tokens_used, read_done, batches_done
        batch_started = time.time()
        lines = [_message_line(doc["msg_id"], doc) for doc in batch]
        # Остановка и пауза проверяются ДО семафора: на паузе пачка не занимает место
        # в очереди vLLM, а после продолжения не перечитывает уже прочитанное.
        ctx.check_cancelled()
        await ctx.wait_if_paused()
        async with semaphore:
            ctx.check_cancelled()
            await ctx.wait_if_paused()
            outcome = await _run_batch(ctx, number, len(batches), scope, focus_text, lines, bulk)
        tokens_used += int(outcome.get("tokens") or 0)
        results.append(outcome)
        done_ok = bool(outcome.get("ok"))
        batches_done += 1
        if done_ok:
            # Считаем прочитанным только то, что модель действительно разобрала: у обрезанной
            # пачки после дробления это меньше её исходного размера.
            read_done = min(selected, read_done + int(outcome.get("read") or len(batch)))
        if tracker is not None:
            batches_ok = len([item for item in results if item.get("ok")])
            detail = (
                f"прочитано {read_done} из {selected} сообщений "
                f"(пачек: {batches_ok} из {len(batches)}, последняя за {time.time() - batch_started:.0f} с)"
                if done_ok
                else f"пачка {number} из {len(batches)} не разобрана: {str(outcome.get('error'))[:120]}"
            )
            await tracker.sub(
                read_done,
                selected,
                stage="чтение текстов",
                detail=detail,
                units_done=batches_ok,
                units_total=len(batches),
                units_parallel=parallel_batches,
            )

    tasks = [asyncio.ensure_future(worker(n, batch)) for n, batch in enumerate(batches, start=1)]
    watcher = None
    if getattr(ctx, "cancel_check", None) is not None:
        # Пока пачки читаются, следим за остановкой: незавершённые пачки гасим сразу,
        # чтобы не держать очередь vLLM и не оставлять висящих задач.
        watcher = asyncio.ensure_future(_cancel_watcher(ctx, tasks))
    try:
        if tracker is not None:
            # Свой heartbeat внутри чтения: даже если инструмент вызван в обход registry,
            # в интерфейсе не будет «мёртвой» тишины дольше 8–15 секунд.
            async with tracker.heartbeat("чтение текстов"):
                await asyncio.gather(*tasks, return_exceptions=True)
        else:
            await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        if watcher is not None:
            watcher.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await watcher
    if ctx.cancelled():
        # Пользователь остановил запуск: выходим наверх, execute_run поставит статус cancelled.
        raise RunCancelled("остановлено пользователем")
    if tracker is not None:
        await tracker.sub(
            read_done,
            selected,
            stage="чтение текстов",
            detail=(
                f"чтение завершено: {read_done} из {selected} сообщений, "
                f"пачек {batches_done} из {len(batches)}, {time.time() - reading_started:.0f} с"
            ),
            units_done=batches_done,
            units_total=len(batches),
            units_parallel=parallel_batches,
        )
    parsed_batches = [parsed for item in results if item.get("ok") for parsed in (item.get("parsed") or [])]
    failed = len([item for item in results if not item.get("ok")])
    reading_models = sorted({str(item.get("model") or "").strip() for item in results
                             if str(item.get("model") or "").strip()})
    fallback_batches = sum(int(item.get("fallback") or 0) for item in results)
    if not parsed_batches:
        raise ToolError(
            "Локальная модель не разобрала ни одну пачку сообщений — повторите запуск позже "
            "или уменьшите limit"
        )

    # «Прочитано» — ровно то, что модель разобрала, а не размер выборки.
    total_read = min(selected, read_done) or selected
    merged = _merge(parsed_batches, docs_by_id, total_read)
    topics, categories, highlights = merged["topics"], merged["categories"], merged["highlights"]
    corpus_total = int(found.get("messages_in_slice") or 0) or total_read
    if corpus is not None:
        # Частоты тем — по всему корпусу (размеры кластеров), цитаты и пояснения — из прочитанных
        # представителей. Иначе доли считались бы от прочитанной выборки, и по отчёту нельзя было
        # бы понять, что тема частая во всём корпусе.
        cluster_rows = _cluster_rows(corpus, merged["topics"], docs_by_id, corpus_total)
        if cluster_rows:
            topics = cluster_rows
            categories = _categories_from_rows(topics, corpus_total)

    if total_read and not topics:
        # Непустое «прочитано N» при пустых темах — провал шага, а не успешный разбор: раньше
        # такой запуск закрывался отчётом без тем и цитат (JSON обрезался по лимиту генерации,
        # разборщик возвращал вложенный объект вместо корня).
        raise ToolError(
            f"Прочитано {total_read} сообщений, но модель не выделила ни одной темы с цитатами — "
            "разбор недействителен. Повторите запуск; если повторяется, уменьшите batch_size "
            f"(сейчас {batch_size}) или поднимите лимит генерации (сейчас {BATCH_MAX_TOKENS})."
        )
    if total_read and not any(row.get("quotes") for row in topics):
        raise ToolError(
            f"Прочитано {total_read} сообщений и выделено тем: {len(topics)}, но ни одной цитаты — "
            "разбор недействителен, повторите запуск."
        )

    tone_note = f"негатив {negative}, нейтрал {neutral}, позитив {positive}"
    digest_topics = "\n".join(
        f"- «{row['topic']}» — {row['count']} сообщений, {row['share_pct']}%, тональность: {row['tone']}"
        + (f", суть: {row['essence']}" if row.get("essence") else "")
        for row in topics[:10]
    ) or "— темы не выделены"
    digest_quotes = "\n".join(
        f"- «{quote['text']}» ({row['topic']}; {quote['hub']}, {quote['date']}, {quote['author']})"
        for row in topics[:8]
        for quote in (row.get("quotes") or [])[:2]
    ) or "— цитат нет"
    digest_highlights = "\n".join(
        f"- ({item['date']}, {item['hub']}, {item['tone']}) {item['text'][:200]}" for item in highlights[:5]
    ) or "— нет"

    summary = ""
    try:
        summary, used = await _qwen(
            ctx,
            SUMMARY_INSTRUCTION.format(
                dataset=index_name,
                period=f"{_iso(lo)} — {_iso(hi)}",
                count=total_read,
                negative=negative,
                tone=tone_note,
                topics=digest_topics,
                quotes=digest_quotes,
                highlights=digest_highlights,
            ),
            system=SUMMARY_SYSTEM,
            max_tokens=SUMMARY_MAX_TOKENS,
            temperature=0.25,
        )
        tokens_used += used
        if summary.lstrip().startswith("{"):
            parsed_summary = _extract_json(summary) or {}
            for key in ("summary", "анализ", "analysis", "text", "разбор"):
                value = parsed_summary.get(key)
                if isinstance(value, str) and value.strip():
                    summary = value.strip()
                    break
    except Exception as exc:  # noqa: BLE001 — без итогового текста раздел всё равно собирается
        await ctx.log(f"Итоговый текст по темам собрать не удалось: {exc}", level="error")
    if not summary.strip():
        summary = _fallback_summary(topics, total_read, negative, highlights)

    # ---------------------------------------------------------- раздел для отчёта
    findings = [
        {
            "topic": row["topic"],
            "count": row["count"],
            "share": row["share"],
            "share_pct": row["share_pct"],
            "tone": row["tone"],
            "category": row["category"],
            "essence": row["essence"],
            "importance": row["importance"],
            "quotes": row["quotes"],
            "msg_ids": row["msg_ids"][:20],
        }
        for row in topics
    ]
    # Пометка о способе чтения — без конкретных чисел: она объясняет смысл, а не размеры.
    # У среза, прочитанного целиком, пометки нет: там темы и цитаты построены по всем сообщениям.
    if strategy in ("full", "full_long"):
        reading_strategy = "прочитано целиком"
        reading_note = ""
    elif strategy == "corpus":
        reading_strategy = "кластеризация по всему корпусу + чтение представителей"
        reading_note = (
            "Темы, цитаты и ключевые сообщения построены по выборке наиболее значимых сообщений периода; "
            "тональность, площадки и динамика — по всей совокупности сообщений среза. "
            "Частоты тем посчитаны по всему корпусу среза."
        )
    else:
        reading_strategy = "выборка значимых сообщений"
        reading_note = (
            "Темы, цитаты и ключевые сообщения построены по выборке наиболее значимых сообщений периода; "
            "тональность, площадки и динамика — по всей совокупности сообщений среза."
        )
    if slice_total and total_read >= slice_total:
        heading = f"Темы и цитаты из текстов ({total_read} сообщений)"
    elif strategy in ("full", "full_long"):
        heading = f"Темы и цитаты из текстов ({total_read} сообщений)"
    else:
        # Прочитана выборка представителей: число прочитанного в заголовке только запутало бы —
        # размер выборки не равен размеру среза, а сам срез с пометкой объяснён в тексте раздела.
        heading = "Темы и цитаты из текстов"
    citations = [
        {"title": f"{quote['hub']} · {quote['date']} · {quote['author']}".strip(" ·"), "url": quote["url"]}
        for row in topics
        for quote in (row.get("quotes") or [])
        if quote.get("url")
    ][:10]
    section = {
        "heading": heading,
        "text": summary,
        "findings": findings,
        "citations": citations,
        "chart_ids": [],
    }
    if reading_note:
        section["note"] = reading_note
    # build_report подставит раздел сам, если модель про него забудет
    ctx.text_analysis = section
    ctx.negative_in_slice = int(negative)

    scope_note = (
        f"Прочитаны тексты {total_read} сообщений датасета «{index_name}» за период {_iso(lo)} — {_iso(hi)} "
        f"локальной моделью {model_label} (пачек: {len(batches)}). "
        f"Тональность среза: {tone_note}. Отбор важных сообщений: "
        f"{docs[0].get('importance_basis') if docs else '—'}."
    )
    if reading_note:
        scope_note += " " + reading_note
    if failed:
        scope_note += f" Не разобрано пачек: {failed}."
    if fallback_batches:
        scope_note += (
            f" Часть пачек ({fallback_batches}) прочитана на общей модели: быстрая модель чтения "
            "не отвечала."
        )

    engagement_available = bool(found.get("engagement_available"))
    engagement_fields = found.get("engagement_fields") or []
    engagement_weights, engagement_scale, engagement_enabled = _engagement_config()
    if engagement_available:
        engagement_note = (
            "Ранжирование ключевых сообщений: вовлечённость (" + ", ".join(
                ENGAGEMENT_LABELS.get(field, field) for field in engagement_fields
                if engagement_weights.get(field)) + "), "
            + ENGAGEMENT_SCALE_LABELS.get(engagement_scale, engagement_scale) + ", веса: "
            + _engagement_weights_brief(engagement_weights) + "."
        )
    elif not engagement_enabled:
        engagement_note = (
            "Ранжирование по вовлечённости отключено настройкой (mlops/lock.yaml: "
            "engagement.enabled=false) — важные сообщения отобраны по оценке отзыва, "
            "объёму текста и маркерам."
        )
        scope_note += " Метрики вовлечённости не учитывались: ранжирование отключено настройкой."
    else:
        engagement_note = NO_ENGAGEMENT_NOTE.capitalize() + " — важные сообщения отобраны по оценке отзыва, объёму текста и маркерам."
        scope_note += " Метрики вовлечённости: " + NO_ENGAGEMENT_NOTE + "."

    seconds = round(time.time() - started, 1)
    return {
        "index": idx,
        "index_name": index_name,
        "period": {"from": _iso(lo), "to": _iso(hi)},
        "tone_filter": tone,
        "messages_in_slice": int(found.get("messages_in_slice") or 0),
        "messages_analyzed": total_read,
        "engagement_available": bool(found.get("engagement_available")),
        "engagement_fields": found.get("engagement_fields") or [],
        "engagement_max": compact(found.get("engagement_max") or {}, max_items=10, max_str=40),
        "tone_counts": {"negative": negative, "neutral": neutral, "positive": positive},
        # topics/categories/highlights отдаём как есть (без compact): compact добавляет в список
        # служебный элемент {"_total_items": ...} и ломает перебор, а он нужен и модели, и отчёту.
        "topics": _public_topics(topics),
        "categories": _public_categories(categories),
        "highlights": _public_highlights(highlights),
        "summary": summary,
        "scope_note": scope_note,
        "report_section": section,
        "stats": {
            "messages_processed": total_read,
            "messages_selected": selected,
            "messages_in_slice": int(found.get("messages_in_slice") or 0),
            "batches": len(batches),
            "batches_failed": failed,
            "batch_size": batch_size,
            "topics": len(topics),
            "highlights": len(highlights),
            "seconds": seconds,
            "model": model_label,
            "models": reading_models,
            "fallback_batches": fallback_batches,
            "local_model_tokens": tokens_used,
            "cost_usd": 0.0,
            "importance_basis": docs[0].get("importance_basis") if docs else "",
            "reading_strategy": reading_strategy,
            "reading_strategy_key": strategy,
            "reading_note": reading_note,
            "slice_total": int(found.get("messages_in_slice") or 0),
            "clusters": len(corpus.get("clusters") or []) if corpus else 0,
            "clusters_total": int(corpus.get("clusters_total") or 0) if corpus else 0,
            "cluster_noise": int(corpus.get("noise") or 0) if corpus else 0,
            "cluster_tagged_docs": int(corpus.get("tagged") or 0) if corpus else 0,
            "cluster_stages": dict(corpus.get("stages") or {}) if corpus else {},
            "cluster_truncated": bool(corpus.get("truncated")) if corpus else False,
            "cluster_skipped_no_text": int(corpus.get("skipped_no_text") or 0) if corpus else 0,
            "corpus_tagged_docs": int(corpus.get("tagged_docs") or 0) if corpus else 0,
            "engagement_available": engagement_available,
            "engagement_fields": engagement_fields,
            "engagement_max": found.get("engagement_max") or {},
            "engagement_note": engagement_note,
            "engagement_scale": engagement_scale,
            "engagement_weights": {f: w for f, w in engagement_weights.items() if w},
        },
        "note": (
            "Раздел report_section обязательно передай в build_report (findings с темами, долями и цитатами "
            "попадут в DOCX/PDF отдельным блоком). Пояснения к каждому графику строй на этих темах и цитатах, "
            "а не только на статистике. Цитаты уже сверены с текстами сообщений. "
            + (reading_note + " " if reading_note else "")
            + engagement_note
        ),
    }
