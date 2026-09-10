# -*- coding: utf-8 -*-
"""
Лёгкий учёт токенов LLM (задел под биллинг).

Gateway остаётся единственной точкой входа к моделям; здесь пишем usage после ответа.
Если позже понадобится LiteLLM - его можно вставить за тем же gateway-интерфейсом.
"""
import contextvars
import json
import threading

_CTX = contextvars.ContextVar('llm_usage_ctx', default=None)

_PRICES_LOCK = threading.Lock()
_PRICES: dict = {}


def _db():
    import psycopg2
    from config import DB_HOST, DB_NAME, DB_PASS, DB_PORT, DB_USER
    return psycopg2.connect(host=DB_HOST, port=DB_PORT or 5432,
                            dbname=DB_NAME, user=DB_USER, password=DB_PASS,
                            connect_timeout=5)


def _ensure_tables(cur):
    cur.execute("""CREATE TABLE IF NOT EXISTS llm_pricing (
        provider TEXT NOT NULL,
        model TEXT NOT NULL,
        input_price_usd_per_1m DOUBLE PRECISION NOT NULL DEFAULT 0,
        output_price_usd_per_1m DOUBLE PRECISION NOT NULL DEFAULT 0,
        currency TEXT NOT NULL DEFAULT 'USD',
        PRIMARY KEY (provider, model)
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS llm_usage (
        id BIGSERIAL PRIMARY KEY,
        ts TIMESTAMPTZ NOT NULL DEFAULT now(),
        user_id INTEGER,
        case_id TEXT,
        provider TEXT,
        model TEXT,
        status TEXT DEFAULT 'ok',
        prompt_tokens BIGINT DEFAULT 0,
        completion_tokens BIGINT DEFAULT 0,
        total_tokens BIGINT DEFAULT 0,
        cost_usd DOUBLE PRECISION DEFAULT 0,
        latency_ms INTEGER DEFAULT 0,
        extra TEXT
    )""")
    cur.execute("CREATE INDEX IF NOT EXISTS ix_llm_usage_user_ts ON llm_usage(user_id, ts)")
    cur.execute("CREATE INDEX IF NOT EXISTS ix_llm_usage_case_ts ON llm_usage(case_id, ts)")


def _reload_prices(cur):
    global _PRICES
    with _PRICES_LOCK:
        _PRICES = {}
        try:
            cur.execute("SELECT provider, model, input_price_usd_per_1m, output_price_usd_per_1m FROM llm_pricing")
            for provider, model, ip, op in cur.fetchall():
                _PRICES[(str(provider), str(model))] = (float(ip or 0), float(op or 0))
        except Exception as exc:
            print("llm usage prices reload err:", exc)


def _pricing_seeds() -> list[tuple]:
    """Цены берём из mlops/lock.yaml (секция pricing), иначе — разумные значения по умолчанию."""
    default = [
        ("vllm", "*", 0.0, 0.0),
        ("aitunnel", "*", 3.0, 15.0),
    ]
    try:
        from .lock import load_lock

        pricing = (load_lock() or {}).get("pricing") or {}
        rows = []
        for provider, models in pricing.items():
            if not isinstance(models, dict):
                continue
            for model, price in models.items():
                if isinstance(price, dict):
                    rows.append(
                        (str(provider), str(model), float(price.get("in") or 0), float(price.get("out") or 0))
                    )
        return rows or default
    except Exception:
        return default


def ensure():
    try:
        conn = _db()
        cur = conn.cursor()
        _ensure_tables(cur)
        seeds = _pricing_seeds()
        for provider, model, ip, op in seeds:
            cur.execute(
                "INSERT INTO llm_pricing (provider, model, input_price_usd_per_1m, output_price_usd_per_1m) "
                "VALUES (%s, %s, %s, %s) ON CONFLICT (provider, model) DO UPDATE SET "
                "input_price_usd_per_1m = EXCLUDED.input_price_usd_per_1m, "
                "output_price_usd_per_1m = EXCLUDED.output_price_usd_per_1m",
                (provider, model, ip, op))
        conn.commit()
        _reload_prices(cur)
        cur.close()
        conn.close()
    except Exception as exc:
        print("llm usage ensure err:", exc)


def write(user_id=None, case=None, provider="", model="", status="ok",
          prompt_tokens=0, completion_tokens=0, total_tokens=0,
          latency_ms=0, extra=None):
    try:
        conn = _db()
        cur = conn.cursor()
        _ensure_tables(cur)
        key = (str(provider or ""), str(model or ""))
        pr = _PRICES.get(key) or _PRICES.get((str(provider or ""), "*")) or (0.0, 0.0)
        iprice, oprice = pr
        cost = (float(prompt_tokens or 0) * iprice + float(completion_tokens or 0) * oprice) / 1_000_000.0
        cur.execute(
            "INSERT INTO llm_usage (user_id, case_id, provider, model, status, "
            "prompt_tokens, completion_tokens, total_tokens, cost_usd, latency_ms, extra) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (int(user_id) if user_id is not None else None,
             case, provider, model, status,
             int(prompt_tokens or 0), int(completion_tokens or 0), int(total_tokens or 0),
             round(cost, 8), int(latency_ms or 0),
             json.dumps(extra, ensure_ascii=False) if extra else None))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as exc:
        print("llm usage write err:", exc)


def aggregate(date_from=None, date_to=None):
    out = []
    try:
        conn = _db()
        cur = conn.cursor()
        _ensure_tables(cur)
        sql = (
            "SELECT COALESCE(user_id, 0), COALESCE(case_id, ''), COALESCE(provider, ''), COALESCE(model, ''), "
            "COUNT(*), COALESCE(SUM(prompt_tokens),0), COALESCE(SUM(completion_tokens),0), "
            "COALESCE(SUM(total_tokens),0), COALESCE(SUM(cost_usd),0) "
            "FROM llm_usage "
        )
        conds = []
        params = []
        if date_from:
            conds.append("(ts AT TIME ZONE 'UTC')::date >= %s::date")
            params.append(date_from)
        if date_to:
            conds.append("(ts AT TIME ZONE 'UTC')::date <= %s::date")
            params.append(date_to)
        if conds:
            sql += "WHERE " + " AND ".join(conds) + " "
        sql += "GROUP BY 1,2,3,4 ORDER BY 5 DESC"
        cur.execute(sql, params)
        for r in cur.fetchall():
            out.append({
                "user_id": r[0], "case_id": r[1], "provider": r[2], "model": r[3],
                "requests": int(r[4]),
                "prompt_tokens": int(r[5]), "completion_tokens": int(r[6]),
                "total_tokens": int(r[7]), "cost_usd": round(float(r[8]), 6),
            })
        cur.close()
        conn.close()
    except Exception as exc:
        print("llm usage aggregate err:", exc)
    return out


ensure()
def set_ctx(user_id=None, case=None):
    return _CTX.set({"user_id": user_id, "case": case})


def reset_ctx(token):
    try:
        _CTX.reset(token)
    except Exception:
        pass


def current():
    return _CTX.get() or {}
def aggregate_days(user_id=None, case=None, provider=None, model=None, days=30, date_from=None, date_to=None):
    out = []
    try:
        conn = _db()
        cur = conn.cursor()
        _ensure_tables(cur)
        sql = ("SELECT to_char(ts AT TIME ZONE 'UTC', 'YYYY-MM-DD') AS day, "
               "COALESCE(user_id, 0), COALESCE(case_id, ''), COALESCE(provider, ''), COALESCE(model, ''), "
               "COUNT(*), COALESCE(SUM(prompt_tokens),0), COALESCE(SUM(completion_tokens),0), "
               "COALESCE(SUM(total_tokens),0), COALESCE(SUM(cost_usd),0) "
               "FROM llm_usage WHERE ")
        conds = []
        params = []
        if date_from or date_to:
            if date_from:
                conds.append("(ts AT TIME ZONE 'UTC')::date >= %s::date")
                params.append(date_from)
            if date_to:
                conds.append("(ts AT TIME ZONE 'UTC')::date <= %s::date")
                params.append(date_to)
        else:
            conds.append("ts >= now() - (%s || ' days')::interval ")
            params.append(int(days or 30))
        if user_id is not None:
            conds.append("user_id = %s")
            params.append(int(user_id))
        if case:
            conds.append("case_id = %s")
            params.append(case)
        if provider:
            conds.append("provider = %s")
            params.append(provider)
        if model:
            conds.append("model = %s")
            params.append(model)
        sql += " AND ".join(conds)
        sql += " GROUP BY 1,2,3,4,5 ORDER BY 1"
        cur.execute(sql, params)
        for r in cur.fetchall():
            out.append({"day": r[0], "user_id": r[1], "case_id": r[2], "provider": r[3], "model": r[4],
                        "requests": int(r[5]), "prompt_tokens": int(r[6]), "completion_tokens": int(r[7]),
                        "total_tokens": int(r[8]), "cost_usd": round(float(r[9]), 6)})
        cur.close()
        conn.close()
    except Exception as exc:
        print("llm usage days err:", exc)
    return out
