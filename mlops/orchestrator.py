# -*- coding: utf-8 -*-
"""Модель-оркестратор: кто планирует шаги и вызывает инструменты Tellscope.

Настройка живёт в ``mlops/lock.yaml`` (секция ``agent``) и переопределяется окружением:

    TELLSCOPE_ORCHESTRATOR           — основной ключ модели (gpt | deepseek | qwen | claude)
    TELLSCOPE_ORCHESTRATOR_FALLBACKS — порядок отказов через запятую
    TELLSCOPE_ORCHESTRATOR_PROBE     — tools | reachability

Читается на каждый запуск (см. ``mlops.lock.agent_cfg``), поэтому переключение действует
без правки кода и без рестарта. Дорогая часть — проба доступности модели: её результат
кэшируется на ``PROBE_TTL_SEC`` секунд, чтобы один запуск не платил за проверку дважды.

Цепочка отказов: если профиль недоступен (нет ключа, 401/403/429/402, «insufficient balance»,
таймаут, ошибка пробы tool-calling), берём следующий ключ из ``orchestrator_fallbacks`` и
работаем на первом доступном. Причина отказа всегда попадает в метаданные запуска и в логи.
"""
from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Статусы, при которых модель считается недоступной и цепочка идёт дальше.
UNAVAILABLE_STATUSES = {0, 400, 401, 402, 403, 404, 405, 408, 409, 413, 422, 429, 500, 501, 502, 503, 504}
# Подсказки в тексте ошибки (aitunnel отдаёт причину словами).
UNAVAILABLE_MARKERS = (
    "insufficient", "not enough", "balance", "quota", "credit", "payment", "billing",
    "not set", "no api key", "unauthor", "forbidden", "invalid api key", "rate limit",
    "too many requests", "timeout", "timed out", "connection", "no healthy",
)

PROBE_TTL_SEC = 25.0
PROBE_TIMEOUT_SEC = 40.0

PROBE_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "tellscope_probe",
            "description": "Служебная проверка поддержки вызова инструментов",
            "parameters": {
                "type": "object",
                "properties": {"ok": {"type": "boolean", "description": "всегда true"}},
                "required": ["ok"],
            },
        },
    }
]

_PROBE_CACHE: Dict[str, Dict[str, Any]] = {}
_CACHE_LOCK = threading.Lock()
_LAST: Optional[Dict[str, Any]] = None


@dataclass
class OrchestratorState:
    """Результат выбора оркестратора: кто работает и почему не взяли предыдущих."""

    choice: str
    label: str = ""
    requested: str = ""
    chain: List[str] = field(default_factory=list)
    reasons: List[Dict[str, Any]] = field(default_factory=list)
    source: str = ""
    probe_mode: str = "tools"
    checked: bool = False
    fallback_used: bool = False
    available: bool = True

    def reason_text(self) -> str:
        if not self.reasons:
            return ""
        parts = []
        for item in self.reasons:
            parts.append(f"{item.get('model')}: {item.get('reason')}")
        return "; ".join(parts)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "orchestrator": self.choice,
            "orchestrator_label": self.label,
            "orchestrator_requested": self.requested,
            "orchestrator_chain": list(self.chain),
            "orchestrator_source": self.source,
            "orchestrator_probe": self.probe_mode,
            "orchestrator_checked": self.checked,
            "orchestrator_fallback_used": self.fallback_used,
            "orchestrator_reason": self.reason_text(),
            "orchestrator_reasons": list(self.reasons),
        }


def _model_choices() -> Dict[str, Dict[str, Any]]:
    """MODEL_CHOICES из цикла агента. Импорт ленивый: mlops не должен зависеть от agent_engine."""
    from agent_engine.loop import MODEL_CHOICES

    return MODEL_CHOICES


def label_for(key: str) -> str:
    return str((_model_choices().get(key) or {}).get("label") or key)


def known_keys() -> List[str]:
    return list(_model_choices().keys())


def chain_keys() -> List[str]:
    """Настроенная цепочка оркестратора: только известные ключи, с сохранением порядка."""
    from mlops.lock import agent_cfg

    cfg = agent_cfg()
    keys = [key for key in (cfg.get("chain") or []) if key in _model_choices()]
    return keys or ["gpt"]


def failure_reason(exc: BaseException) -> Optional[str]:
    """Считать ли ошибку вызова модели поводом уйти на следующего кандидата."""
    status = getattr(exc, "status_code", None)
    try:
        status = int(status or 0)
    except Exception:
        status = 0
    message = str(exc or "")
    low = message.lower()
    marker = next((item for item in UNAVAILABLE_MARKERS if item in low), "")
    if status in UNAVAILABLE_STATUSES or marker:
        if status:
            return f"HTTP {status}" + (f" ({message})" if message else "")
        return message or "модель недоступна"
    return None


def _precheck(key: str, choice: Dict[str, Any]) -> Optional[str]:
    """Дешёвая проверка до сетевого обращения: ключ и адрес."""
    provider = str(choice.get("provider") or "")
    if provider == "aitunnel":
        try:
            from mlops.gateway import _aitunnel_key

            if not _aitunnel_key():
                return "нет ключа AITUNNEL_API_KEY"
        except Exception:
            pass
    if provider == "vllm":
        try:
            from mlops.lock import generate_cfg

            if not (generate_cfg().get("base_url") or "").strip():
                return "не задан адрес локального vLLM"
        except Exception:
            pass
    return None


def _cache_get(key: str, mode: str, force: bool = False) -> Optional[Dict[str, Any]]:
    if force:
        return None
    with _CACHE_LOCK:
        item = _PROBE_CACHE.get(f"{mode}|{key}")
    if not item:
        return None
    if time.time() - float(item.get("ts") or 0) > PROBE_TTL_SEC:
        return None
    return item


def _cache_put(key: str, mode: str, payload: Dict[str, Any]) -> None:
    with _CACHE_LOCK:
        _PROBE_CACHE[f"{mode}|{key}"] = {**payload, "ts": time.time()}


def probe_cache() -> Dict[str, Any]:
    """Снимок кэша проб — для /harness/info и админского эндпоинта."""
    with _CACHE_LOCK:
        items = dict(_PROBE_CACHE)
    out: Dict[str, Any] = {}
    for cache_key, item in items.items():
        out[cache_key] = {
            "ok": bool(item.get("ok")),
            "reason": item.get("reason") or "",
            "latency_ms": item.get("latency_ms"),
            "age_sec": round(time.time() - float(item.get("ts") or 0), 1),
            "fresh": (time.time() - float(item.get("ts") or 0)) <= PROBE_TTL_SEC,
        }
    return out


async def _probe_once(key: str, choice: Dict[str, Any], mode: str) -> Dict[str, Any]:
    """Одна сетевая проверка кандидата: отвечает ли модель и умеет ли вызывать инструменты."""
    from mlops import gateway

    provider = str(choice.get("provider") or "")
    extra: Dict[str, Any] = {}
    messages = [{"role": "user", "content": "Ответь одним словом: готов."}]
    if mode == "tools":
        extra = {
            "tools": PROBE_TOOL,
            "tool_choice": {"type": "function", "function": {"name": "tellscope_probe"}},
            "parallel_tool_calls": False,
        }
        messages = [{"role": "user", "content": "Вызови инструмент tellscope_probe с ok=true."}]
    if provider == "vllm":
        extra["chat_template_kwargs"] = {"enable_thinking": False}
    started = time.perf_counter()
    try:
        result = await gateway.achat(
            provider=provider,
            messages=messages,
            temperature=0.0,
            max_tokens=96,
            timeout=PROBE_TIMEOUT_SEC,
            extra=extra or None,
            profile=str(choice.get("profile") or "dashboard_qa"),
            usage_ctx={"user_id": None, "case": "orchestrator-probe"},
        )
    except Exception as exc:  # noqa: BLE001
        reason = failure_reason(exc) or f"{type(exc).__name__}: {exc}"
        return {"ok": False, "reason": f"проба не прошла: {reason}", "stage": "probe",
                "latency_ms": int((time.perf_counter() - started) * 1000)}
    latency = int((time.perf_counter() - started) * 1000)
    if mode == "reachability":
        text = (getattr(result, "content", "") or "").strip()
        if text:
            return {"ok": True, "reason": "", "stage": "probe", "latency_ms": latency}
        return {"ok": False, "reason": "проба не прошла: модель вернула пустой ответ",
                "stage": "probe", "latency_ms": latency}
    raw = getattr(result, "raw", None) or {}
    message = ((raw.get("choices") or [{}])[0] or {}).get("message") or {}
    calls = message.get("tool_calls") or []
    if calls:
        return {"ok": True, "reason": "", "stage": "probe", "latency_ms": latency}
    return {"ok": False, "reason": "проба tool-calling: модель не вызвала инструмент",
            "stage": "probe", "latency_ms": latency}


async def check_candidate(key: str, mode: Optional[str] = None, force: bool = False) -> Dict[str, Any]:
    """Доступность кандидата с кэшем на PROBE_TTL_SEC секунд."""
    choices = _model_choices()
    choice = choices.get(key)
    if choice is None:
        return {"ok": False, "reason": f"неизвестный ключ модели: {key}", "stage": "config"}
    if mode is None:
        from mlops.lock import agent_cfg

        mode = str(agent_cfg().get("orchestrator_probe") or "tools")
    cached = _cache_get(key, mode, force=force)
    if cached is not None:
        return cached
    pre = _precheck(key, choice)
    if pre:
        payload = {"ok": False, "reason": pre, "stage": "precheck", "latency_ms": 0}
        _cache_put(key, mode, payload)
        return payload
    payload = await _probe_once(key, choice, mode)
    _cache_put(key, mode, payload)
    return payload


async def resolve_orchestrator(probe: bool = True, force: bool = False) -> OrchestratorState:
    """Первый доступный оркестратор из цепочки. Результат пишется в _LAST для /harness/info."""
    global _LAST
    from mlops.lock import agent_cfg

    cfg = agent_cfg()
    mode = str(cfg.get("orchestrator_probe") or "tools")
    candidates = [key for key in (cfg.get("chain") or []) if key in _model_choices()]
    if not candidates:
        candidates = ["gpt"]
    reasons: List[Dict[str, Any]] = []
    for key in candidates:
        if probe:
            check = await check_candidate(key, mode, force=force)
            if not check.get("ok"):
                reasons.append({"model": key, "label": label_for(key), "reason": check.get("reason") or "недоступна",
                                "stage": check.get("stage") or "probe"})
                continue
        else:
            pre = _precheck(key, _model_choices()[key])
            if pre:
                reasons.append({"model": key, "label": label_for(key), "reason": pre, "stage": "precheck"})
                continue
        state = OrchestratorState(
            choice=key,
            label=label_for(key),
            requested=str(cfg.get("orchestrator") or ""),
            chain=candidates,
            reasons=reasons,
            source=str(cfg.get("source") or ""),
            probe_mode=mode,
            checked=bool(probe),
            fallback_used=bool(reasons),
            available=True,
        )
        _LAST = {**state.as_dict(), "orchestrator_at": datetime.now().isoformat(timespec="seconds")}
        return state
    # Никто не прошёл проверку: работаем на основном и честно фиксируем причину,
    # чтобы запуск не падал целиком, а в метаданных было видно, что проверка не прошла.
    state = OrchestratorState(
        choice=candidates[0],
        label=label_for(candidates[0]),
        requested=str(cfg.get("orchestrator") or ""),
        chain=candidates,
        reasons=reasons,
        source=str(cfg.get("source") or ""),
        probe_mode=mode,
        checked=bool(probe),
        fallback_used=bool(reasons),
        available=False,
    )
    if state.reasons:
        state.reasons.append({
            "model": candidates[0],
            "label": state.label,
            "reason": "ни один профиль не прошёл проверку — работаем на основном как есть",
            "stage": "fallback-exhausted",
        })
    _LAST = {**state.as_dict(), "orchestrator_at": datetime.now().isoformat(timespec="seconds")}
    return state


def last_resolution() -> Optional[Dict[str, Any]]:
    return dict(_LAST) if _LAST else None


def describe() -> Dict[str, Any]:
    """Состояние без сетевых проб: настройка, цепочка, кэш проб и последний выбор."""
    from mlops.lock import agent_cfg

    cfg = agent_cfg()
    choices = _model_choices()
    chain = [key for key in (cfg.get("chain") or []) if key in choices] or ["gpt"]
    last = last_resolution() or {}
    return {
        "config": {
            "orchestrator": cfg.get("orchestrator"),
            "orchestrator_fallbacks": cfg.get("orchestrator_fallbacks"),
            "orchestrator_probe": cfg.get("orchestrator_probe"),
            "chain": chain,
            "source": cfg.get("source"),
            "fallbacks_source": cfg.get("fallbacks_source"),
            "lock_path": cfg.get("lock_path"),
            "env_file": cfg.get("env_file"),
            "probe_ttl_sec": PROBE_TTL_SEC,
            "env_vars": {
                "orchestrator": "TELLSCOPE_ORCHESTRATOR",
                "fallbacks": "TELLSCOPE_ORCHESTRATOR_FALLBACKS",
                "probe": "TELLSCOPE_ORCHESTRATOR_PROBE",
            },
        },
        "chain": [{"key": key, "label": label_for(key), "provider": (choices.get(key) or {}).get("provider"),
                   "profile": (choices.get(key) or {}).get("profile")} for key in chain],
        "models": [{"key": key, "label": value.get("label"), "tier": value.get("tier")}
                   for key, value in choices.items()],
        "last_resolution": last or None,
        "probe_cache": probe_cache(),
    }


def reset_cache() -> None:
    """Сбросить кэш проб: админский эндпоинт после смены настройки."""
    with _CACHE_LOCK:
        _PROBE_CACHE.clear()


async def first_available(probe: bool = True) -> Tuple[str, Dict[str, Any]]:
    state = await resolve_orchestrator(probe=probe)
    return state.choice, state.as_dict()


def run_async(coro: Any) -> Any:
    """Утилита для вызова асинхронной проверки из синхронного кода (админка, скрипты)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError("внутри активного цикла событий используйте await")
