# -*- coding: utf-8 -*-
"""Семантическое расширение поискового запроса для агентного режима.

Проблема: буквальный поиск по фразе «отравлений» не находит «траванулся», «отравился»,
«пищевое отравление». Решение — три слоя:

1. морфология (pymorphy3): все формы слов темы («отравлений» → «отравление», «отравления»…);
2. синонимы от локальной модели Qwen: разговорные и смысловые варианты («траванулся», «тошнит»);
3. словарь частых тем — страховка, если модель недоступна;
4. векторный поиск в Qdrant (если для датасета построены эмбеддинги) — ловит всё остальное.

Точные счётчики по каждой формулировке считаются одним запросом (filters-агрегация),
чтобы в отчёте было видно, по каким именно формулировкам шёл поиск.
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Dict, List, Optional, Tuple

MAX_VARIANTS = 20

# Доменные подсказки: частые темы соцмедиа и их разговорные варианты
SEMANTIC_HINTS: Dict[str, List[str]] = {
    "отравлен": ["отравился", "отравилась", "отравились", "траванулся", "траванулась", "тошнит", "рвало",
                 "пищевое отравление", "кишечная инфекция", "плохо стало после еды", "отравление едой"],
    "качеств": ["некачественная еда", "испорченная еда", "просрочка", "просроченный", "несвежее", "тухлое",
                "залежалый", "холодная еда", "сырое", "невкусно", "отвратительно"],
    "сервис": ["обслуживание", "хамство", "грубость", "персонал", "нахамили", "ждали долго", "очередь", "очереди"],
    "цен": ["дорого", "подорожало", "цены выросли", "переплатил", "дешевле", "скидка", "акция", "надбавка"],
    "доставк": ["курьер", "заказ не привезли", "опаздывает доставка", "холодный заказ", "перепутали заказ"],
    "гряз": ["антисанитария", "таракан", "грязь", "немытые руки", "волос в еде", "мусор", "воняет"],
    "закрыт": ["закрытие", "закрыли", "уходит из России", "прекратил работу", "уход бренда"],
    "конкурент": ["сравнение с", "лучше чем", "хуже чем", "перешли в", "альтернатива"],
    "коллаборац": ["коллаб", "коллаборация", "совместная акция", "спецпроект"],
}

_LLM_SYSTEM = (
    "Ты помогаешь аналитику соцмедиа строить поисковые запросы. Ты знаешь разговорную речь, "
    "сленг и опечатки, которые люди используют в соцсетях и отзывах. Отвечай только JSON."
)

_LLM_PROMPT = (
    "Тема анализа: «{topic}». Подбери до 10 коротких поисковых формулировок на русском, которые реально "
    "встречаются в соцсетях, отзывах и новостях и передают тот же смысл: синонимы, разговорные и жаргонные "
    "варианты, устойчивые сочетания, названия явления. Не добавляй общие слова («проблема», «ситуация») и не "
    "повторяй саму тему. Ответ — JSON вида {{\"variants\": [\"...\", \"...\"]}}."
)

_MORPH = None
_MORPH_TRIED = False


def _morph():
    """Морфологический анализатор (pymorphy3 → pymorphy2 как запасной вариант)."""
    global _MORPH, _MORPH_TRIED
    if _MORPH_TRIED:
        return _MORPH
    _MORPH_TRIED = True
    for module_name in ("pymorphy3", "pymorphy2"):
        try:
            module = __import__(module_name)
            _MORPH = module.MorphAnalyzer()
            break
        except Exception:
            continue
    return _MORPH


def _norm(text: str) -> str:
    return " ".join(str(text or "").lower().split())


def morphology_variants(topic: str, per_word: int = 3) -> List[str]:
    """Формы слов темы: «отравлений» → «отравление», «отравления», «отравлением»."""
    morph = _morph()
    words = _norm(topic).split()
    if not morph or not words:
        return []
    out: List[str] = []
    lemma_words: List[str] = []
    word_forms: List[List[str]] = []
    for word in words:
        try:
            parsed = morph.parse(word)[0]
            lemma = parsed.normal_form
            lemma_words.append(lemma)
            forms = [lemma]
            for item in parsed.lexeme:
                candidate = item.word
                if candidate not in forms and len(forms) < per_word + 1:
                    forms.append(candidate)
            word_forms.append(forms)
        except Exception:
            lemma_words.append(word)
            word_forms.append([word])

    lemma_phrase = " ".join(lemma_words)
    if lemma_phrase and lemma_phrase != _norm(topic):
        out.append(lemma_phrase)

    # меняем по одному слову, остальные приводим к нормальной форме — так не растёт комбинаторика
    for position, forms in enumerate(word_forms):
        for form in forms:
            if form == lemma_words[position]:
                continue
            phrase = list(lemma_words)
            phrase[position] = form
            value = " ".join(phrase)
            if value not in out:
                out.append(value)
    return [v for v in out if len(v) >= 4]


def dictionary_variants(topic: str) -> List[str]:
    """Подсказки из словаря частых тем."""
    low = _norm(topic)
    stems = {word[:8] for word in low.split()}
    out: List[str] = []
    for key, hints in SEMANTIC_HINTS.items():
        if key in low or any(stem.startswith(key[:6]) for stem in stems):
            out.extend(hints)
    return out


async def llm_variants(ctx, topic: str, limit: int = 10) -> List[str]:
    """Синонимы и разговорные формулировки от локальной модели Qwen (бесплатно)."""
    try:
        from mlops import gateway

        result = await gateway.achat(
            provider="vllm",
            messages=[
                {"role": "system", "content": _LLM_SYSTEM},
                {"role": "user", "content": _LLM_PROMPT.format(topic=topic)},
            ],
            temperature=0.3,
            max_tokens=400,
            timeout=180,
            extra={"chat_template_kwargs": {"enable_thinking": False}},
            profile="agent",
        )
        text = re.sub(r"<think>.*?</think>", "", result.content or "", flags=re.S | re.I)
        decoder = json.JSONDecoder()
        start = text.find("{")
        variants: List[str] = []
        while start != -1 and not variants:
            try:
                obj, _ = decoder.raw_decode(text[start:])
                if isinstance(obj, dict):
                    raw = obj.get("variants") or obj.get("synonyms") or []
                    if isinstance(raw, list):
                        variants = [str(v) for v in raw]
            except Exception:
                pass
            start = text.find("{", start + 1)
        cleaned = []
        for item in variants:
            value = _norm(item)
            if 4 <= len(value) <= 60 and value not in cleaned:
                cleaned.append(value)
        return cleaned[:limit]
    except Exception as exc:
        if ctx is not None:
            await ctx.log(f"Синонимы от модели не получены ({exc}) — работаем по морфологии и словарю", level="error")
        return []


async def expand_terms(ctx, topic: str, *, use_llm: bool = True, max_variants: int = MAX_VARIANTS) -> Dict[str, Any]:
    """Собирает варианты поиска по теме: оригинал + морфология + синонимы модели + словарь."""
    original = _norm(topic)
    groups: Dict[str, List[str]] = {"original": [original] if original else []}
    groups["morphology"] = morphology_variants(topic)
    groups["dictionary"] = dictionary_variants(topic)
    groups["llm"] = (await llm_variants(ctx, topic) if use_llm else [])

    ordered: List[str] = []
    for key in ("original", "llm", "dictionary", "morphology"):
        for value in groups.get(key) or []:
            if value and value not in ordered:
                ordered.append(value)
    # если синонимы есть, морфологию подрезаем: она самая «шумная»
    result = ordered[:max_variants]
    return {
        "topic": topic,
        "variants": result,
        "groups": {key: [v for v in values if v in result] for key, values in groups.items()},
        "sources": [key for key in ("original", "llm", "dictionary", "morphology") if groups.get(key)],
    }


def build_query(variants: List[str], filters: List[dict]) -> dict:
    """ES-запрос: любая из формулировок (phrase), плюс нечёткое совпадение по основной."""
    should: List[dict] = []
    for index, variant in enumerate(variants):
        should.append({"match_phrase": {"text": {"query": variant, "slop": 1, "boost": 4.0 if index == 0 else 1.0}}})
    if variants:
        should.append({"match": {"text": {"query": variants[0], "fuzziness": "AUTO", "boost": 0.5}}})
    query: Dict[str, Any] = {"bool": {"should": should, "minimum_should_match": 1}}
    if filters:
        query["bool"]["filter"] = filters
    return query


def count_terms(index_name: str, variants: List[str], filters: List[dict]) -> Dict[str, int]:
    """Счётчики по каждой формулировке одним запросом (filters-агрегация)."""
    from .tools_data import _es

    names = [f"v{i}" for i in range(len(variants))]
    aggs = {
        name: {"bool": {"must": [{"match_phrase": {"text": variant}}], "filter": filters}}
        for name, variant in zip(names, variants)
    }
    body = {"size": 0, "aggs": {"terms": {"filters": {"filters": aggs}}}}
    try:
        res = _es().search(index=index_name, body=body)
        buckets = ((res.get("aggregations") or {}).get("terms") or {}).get("buckets") or {}
    except Exception:
        return {}
    out: Dict[str, int] = {}
    for name, variant in zip(names, variants):
        bucket = buckets.get(name) or {}
        out[variant] = int(bucket.get("doc_count") or 0)
    return out


def collection_name_for(index_name: str) -> str:
    return str(index_name or "")


def vector_search_ids(collection: str, topic: str, limit: int = 200, score_threshold: float = 0.3) -> Tuple[List[str], int, str]:
    """Векторный поиск в Qdrant. Возвращает (id записей в ES, сколько найдено, причина отказа)."""
    try:
        from embedding_model_manager import model_manager
        from qdrant_client import QdrantClient
    except Exception as exc:
        return [], 0, f"нет зависимостей: {exc}"

    client = None
    try:
        client = QdrantClient("localhost", port=6333, timeout=30)
        info = client.get_collection(collection_name=collection)
        if not getattr(info, "points_count", 0):
            return [], 0, "коллекция пуста"
    except Exception as exc:
        return [], 0, f"коллекция недоступна: {exc}"

    try:
        import numpy as np

        vectors = model_manager.encode_texts([topic], batch_size=1, normalize_embeddings=True)
        array = np.asarray(vectors)
        vector = (array[0] if array.ndim == 2 else array).astype("float32")
        norm = float(np.linalg.norm(vector))
        if norm and abs(norm - 1.0) > 0.01:
            vector = vector / norm
        if not norm:
            return [], 0, "модель вернула нулевой вектор запроса"
    except Exception as exc:
        return [], 0, f"не удалось получить вектор запроса: {exc}"

    # Часть коллекций построена без HNSW-индекса (m=0) — тогда нужен точный поиск, иначе пусто
    search_params = None
    try:
        from qdrant_client.http import models as qmodels

        search_params = qmodels.SearchParams(exact=True)
    except Exception:
        search_params = None

    hits = []
    try:
        hits = client.search(
            collection_name=collection,
            query_vector=vector.tolist(),
            limit=int(limit),
            with_payload=True,
            with_vectors=False,
            search_params=search_params,
        )
    except Exception as exc:
        try:
            res = client.query_points(
                collection_name=collection, query=vector.tolist(), limit=int(limit), with_payload=True
            )
            hits = list(getattr(res, "points", []) or [])
        except Exception:
            return [], 0, f"поиск не выполнен: {exc}"

    ids: List[str] = []
    best_score = 0.0
    for hit in hits or []:
        try:
            score = float(getattr(hit, "score", 0) or 0)
        except Exception:
            score = 0.0
        best_score = max(best_score, score)
        payload = getattr(hit, "payload", None) or {}
        meta = payload.get("metadata") or {}
        value = meta.get("id")
        if value is None:
            continue
        text_id = str(value)
        if text_id not in ids:
            ids.append(text_id)
    return ids, len(ids), f"ok (лучший score {best_score:.3f})"


def fetch_docs_by_ids(index_name: str, ids: List[str], limit: int = 200) -> Tuple[List[dict], str]:
    """Достаёт полные документы из Elasticsearch по id, найденным векторным поиском.

    Возвращает (документы, причина отказа) — причина нужна, чтобы в отчёте было видно,
    работал ли векторный слой.
    """
    from .tools_data import _es, _iso, _sample

    if not ids:
        return [], "нет векторных попаданий"
    body = {
        "size": min(len(ids), int(limit)),
        "_source": ["text", "timeCreate", "hub", "url", "likesCount", "commentsCount", "toneMark", "city", "authorObject"],
        "query": {"ids": {"values": [str(i) for i in ids]}},
    }
    try:
        res = _es().search(index=index_name, body=body)
    except Exception as exc:
        return [], f"Elasticsearch не отдал документы: {exc}"

    docs = []
    for hit in (res.get("hits") or {}).get("hits") or []:
        sample = _sample(hit)
        sample["time"] = _iso((hit.get("_source") or {}).get("timeCreate"))
        sample["city"] = (hit.get("_source") or {}).get("city") or ""
        sample["origin"] = "semantic"
        docs.append(sample)
    if not docs:
        return [], "документы по векторным id не найдены в Elasticsearch"
    return docs, ""


def scope_note(terms: Dict[str, int], vector_used: bool, vector_count: int, vector_note: str = "") -> str:
    """Строка для отчёта: по каким формулировкам шёл поиск и работал ли векторный слой."""
    top = sorted(terms.items(), key=lambda kv: -kv[1])[:12]
    listed = ", ".join(f"«{phrase}» ({count})" for phrase, count in top if count) or "нет совпадений"
    note = f"Поиск вёлся по формулировкам: {listed}."
    if vector_used:
        note += f" Дополнительно применён смысловой (векторный) поиск: {vector_count} совпадений."
    elif vector_count:
        note += f" Векторный поиск дал совпадения ({vector_count}), но они уже попали в выборку по формулировкам."
    else:
        reason = f" Причина: {vector_note}." if vector_note else ""
        note += f" Смысловой (векторный) поиск для этого датасета недоступен.{reason}"
    return note
