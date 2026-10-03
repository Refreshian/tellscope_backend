# -*- coding: utf-8 -*-
"""Выбор коллекции и модели эмбеддингов для одной операции.

Зачем отдельный модуль. До сих пор модель и имя коллекции были разбросаны и
**согласованы случайно**: платформенные вкладки кодировали запрос моделью
`deepvk/USER2-base` (768) и искали в коллекции `<индекс>` (768), а PR кодировал
`deepvk/USER-bge-m3` (1024) и искал в `<индекс>__bge` (1024). Стоило поменять модель
в одном месте — и вектор одной размерности уходил в коллекцию другой, а поиск
возвращал **пустой результат без ошибки**.

Здесь эта связка собирается в одном месте: для корпуса выбирается коллекция, и тут же
сообщается, **какой моделью** кодировать запрос к ней. Читатели больше не решают это
сами.

Правило выбора: если рядом есть коллекция `<индекс>__bge` и в ней есть точки — берём её
и модель bge-m3; иначе работаем с базовой коллекцией старой моделью. Поэтому переход
безопасен: корпус без `__bge` продолжает работать как раньше, а смешанный корпус
обрабатывается покорпусно.

Роли префиксов обязательны для bge-m3: `query: ` для запроса, `passage: ` для документа.
Без них векторы запроса и документа оказываются в разных подобластях.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

BGE_SUFFIX = "__bge"
BGE_MODEL = "deepvk/USER-bge-m3"
LEGACY_MODEL = "deepvk/USER2-base"

EMBED_SERVICE_URL = os.environ.get("EMBED_SERVICE_URL", "http://127.0.0.1:5055")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")

# Переключатель миграции. Замер на 561 документе показал: на `__bge` поиск по теме
# теряет 18–22% качества (MRR 0,194 → 0,151), но кодирование запросов ускоряется в
# 3,7 раза, а поиск в агентном режиме — в 60 раз. Поэтому выбор оставлен переменной
# окружения: откат делается без правки кода и перезапуска сборки.
#   EMBED_PREFER_BGE=0 — вернуться на базовые коллекции старой моделью
PREFER_BGE = os.environ.get("EMBED_PREFER_BGE", "1").strip().lower() not in ("0", "false", "no")

# Кэш «есть ли коллекция и сколько в ней точек»: спрашивается на каждый запрос, а
# ходить в Qdrant каждый раз незачем. Обновляется по имени коллекции.
_COLLECTION_CACHE: Dict[str, Optional[int]] = {}


def _collection_points(collection: str) -> Optional[int]:
    if collection in _COLLECTION_CACHE:
        return _COLLECTION_CACHE[collection]
    import json
    import urllib.parse
    import urllib.request

    try:
        url = "%s/collections/%s" % (QDRANT_URL.rstrip("/"),
                                     urllib.parse.quote(collection, safe=""))
        with urllib.request.urlopen(url, timeout=10) as resp:
            info = json.loads(resp.read().decode("utf-8")).get("result") or {}
        points = int(info.get("points_count") or 0)
    except Exception:
        points = None
    _COLLECTION_CACHE[collection] = points
    return points


def forget_cache(collection: str = "") -> None:
    """Сбросить кэш: после загрузки файла коллекция могла появиться или измениться."""
    if collection:
        _COLLECTION_CACHE.pop(collection, None)
    else:
        _COLLECTION_CACHE.clear()


def collection_and_encoder(index_name: str) -> Tuple[str, str]:
    """Возвращает `(имя коллекции, кодировщик)` для корпуса.

    Кодировщик: `bge` — сервис эмбеддингов bge-m3 (1024), `legacy` — модель менеджера
    (768). Отдельная строка вместо булева флага: читателю нужно не «какая коллекция»,
    а «чем кодировать запрос к ней».
    """
    name = str(index_name or "").strip()
    if not name:
        return "", "legacy"
    if not PREFER_BGE:
        return name, "legacy"
    candidate = name + BGE_SUFFIX
    points = _collection_points(candidate)
    if points:
        return candidate, "bge"
    return name, "legacy"


def encoder_for_collection(collection: str) -> str:
    """Кодировщик по имени коллекции — для тех, кто получает коллекцию уже готовой.

    Определяем по суффиксу, а не по наличию в Qdrant: имя `__bge` однозначно означает
    модель bge-m3, и лишний запрос к Qdrant здесь ни к чему.
    """
    return "bge" if str(collection or "").endswith(BGE_SUFFIX) else "legacy"


def encode_query(text: str, encoder: str = "bge") -> Optional[List[float]]:
    """Вектор запроса. None — если получить не удалось (вызывающий решает, что делать)."""
    text = str(text or "").strip()
    if not text:
        return None
    if encoder == "bge":
        try:
            from pr import embed_client
        except ImportError:  # pragma: no cover
            return None
        return embed_client.embed_one(text, role="query")
    try:
        from embedding_model_manager import model_manager
    except ImportError:  # pragma: no cover
        return None
    import numpy as np

    vectors = model_manager.encode_texts([text], batch_size=1, normalize_embeddings=True)
    array = np.asarray(vectors)
    vector = array[0] if array.ndim == 2 else array
    return [float(x) for x in vector]


def encode_query_for(index_name: str, text: str) -> Tuple[str, Optional[List[float]], str]:
    """Сразу всё: `(коллекция, вектор, кодировщик)`.

    Удобно вызывающим, которые раньше брали имя коллекции и вектор по отдельности и
    могли рассогласовать их.
    """
    collection, encoder = collection_and_encoder(index_name)
    return collection, encode_query(text, encoder), encoder


def encode_passages(texts: Sequence[str], max_chars: int = 0,
                    batch: int = 32,
                    timeout: float = 0.0) -> Optional[List[List[float]]]:
    """Векторы документов для записи в `__bge` — моделью bge-m3 через сервис.

    Батчи режем сами. Сервис обрезает вход по своему `max_batch` **молча**
    (`texts[:MAX_BATCH]`), и если отправить больше, вернётся меньше векторов, чем
    текстов, — а рассинхрон векторов и документов испортит загрузку незаметно.

    `max_chars` ограничивает длину: сервис считает по всему тексту, но очень длинные
    документы дороже, чем дают прироста (замер: 1200 → 4000 символов дали +18%
    относительно, 4000 → полный текст — меньше). 0 — без ограничения.
    """
    prepared = [str(t or "")[:max_chars] if max_chars else str(t or "") for t in texts]
    if not prepared:
        return []
    try:
        from pr import embed_client
    except ImportError:  # pragma: no cover
        return None
    step = max(1, int(batch))
    wait = float(timeout or os.environ.get("EMBED_PASSAGE_TIMEOUT") or 600)
    out: List[List[float]] = []
    for start in range(0, len(prepared), step):
        piece = prepared[start:start + step]
        vectors = _embed_piece(piece, embed_client, wait)
        if vectors is None:
            # Лучше честная ошибка, чем молча укороченный список векторов.
            return None
        out.extend(vectors)
    return out


def _embed_piece(piece: Sequence[str], embed_client: Any,
                 timeout: float) -> Optional[List[List[float]]]:
    """Векторы куска текстов; если не уложились в таймаут — делим кусок пополам.

    Сервис считает на CPU, а клиент ждёт ответ ограниченное время и при таймауте молча
    возвращает None. Запрос из 128 длинных отзывов в 60 с не укладывался, загрузка шла
    «без векторов», и датасет оставался без семантического поиска — при этом статус
    загрузки выглядел рабочим. Поэтому уменьшаем кусок до того, который успевает
    посчитаться, вместо того чтобы терять векторы всего датасета.
    """
    vectors = embed_client.embed_texts(piece, role="passage", timeout=timeout)
    if vectors and len(vectors) == len(piece):
        return [list(row) for row in vectors]
    if len(piece) > 1:
        mid = len(piece) // 2
        left = _embed_piece(piece[:mid], embed_client, timeout)
        right = _embed_piece(piece[mid:], embed_client, timeout)
        if left is not None and right is not None:
            return left + right
    return None


def health() -> Dict[str, Any]:
    """Состояние обоих кодировщиков — для диагностики."""
    out: Dict[str, Any] = {"service": EMBED_SERVICE_URL, "qdrant": QDRANT_URL}
    try:
        from pr import embed_client
        out["bge"] = embed_client.health()
    except Exception as exc:
        out["bge"] = {"status": "error", "error": str(exc)[:200]}
    try:
        from embedding_model_manager import model_manager
        out["legacy_model"] = getattr(model_manager, "model_name", "")
    except Exception as exc:
        out["legacy_model"] = "недоступна: %s" % str(exc)[:120]
    return out
