# -*- coding: utf-8 -*-
"""Период данных датасета: минимальное и максимальное время сообщений, с кэшем.

Нужен для подписи «Признаки ОРВИ 01.08.2026-01.10.2026»: в имени файла периода может не
быть (выгрузки вида ``BA_Признаки_ОРВИ_20261003_175906``), а в данных он есть. Замеры
кэшируются на десять минут: списки тем запрашиваются на каждой странице, а агрегация по
индексу — не бесплатная.
"""
from __future__ import annotations

import time
from typing import Dict, Tuple

ES_HOST = "http://localhost:9200"
ES_USER = "elastic"
ES_PASS = "biz8z5i1w0nLPmEweKgP"
_TTL = 600.0

_client = None
_cache: Dict[str, Tuple[float, Tuple[float, float]]] = {}


def _es():
    global _client
    if _client is None:
        from elasticsearch import Elasticsearch

        _client = Elasticsearch(hosts=[ES_HOST], basic_auth=(ES_USER, ES_PASS), request_timeout=20)
    return _client


def data_period(index_name: str) -> Tuple[float, float]:
    """(min, max) времени сообщений индекса; (0, 0), если данных нет или индекс недоступен."""
    name = str(index_name or "").strip()
    if not name:
        return 0.0, 0.0
    now = time.time()
    cached = _cache.get(name)
    if cached and now - cached[0] < _TTL:
        return cached[1]
    bounds = (0.0, 0.0)
    try:
        res = _es().search(index=name, size=0, body={"aggs": {
            "min_t": {"min": {"field": "timeCreate"}},
            "max_t": {"max": {"field": "timeCreate"}},
        }})
        agg = res.get("aggregations") or {}
        bounds = (float(agg.get("min_t", {}).get("value") or 0),
                  float(agg.get("max_t", {}).get("value") or 0))
    except Exception:
        bounds = (0.0, 0.0)
    _cache[name] = (now, bounds)
    return bounds
