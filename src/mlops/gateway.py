"""Single chat client for local vLLM and AITunnel. Keys only from env."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from .lock import external_cfg, generate_cfg

_async_client: httpx.AsyncClient | None = None
_async_loop_id: int | None = None
_ENV_LOADED = False
_METRICS_CALLS: dict[str, int] = {}
_METRICS_ERRORS: dict[str, int] = {}
_METRICS_LATENCY_MS: dict[str, float] = {}


@dataclass
class ChatResult:
    content: str
    provider: str
    model: str
    status_code: int = 200
    raw: dict = field(default_factory=dict)
    finish_reason: str = ""


class GatewayError(RuntimeError):
    def __init__(self, message: str, status_code: int = 0):
        super().__init__(message)
        self.status_code = status_code


def _load_env_files() -> None:
    """Supervisor does not inject .env; fill os.environ without overriding existing keys."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True
    candidates = [
        Path("/home/dev/tellscope_app/tellscope_backend/.env"),
        Path(__file__).resolve().parents[1] / ".env",
        Path.cwd() / ".env",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            for raw in path.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = val
        except Exception:
            continue


def _metric_key(provider: str, profile: str) -> str:
    return f"{provider}|{profile or '-'}"


def _record_call(provider: str, profile: str, ok: bool, elapsed_ms: float) -> None:
    key = _metric_key(provider, profile)
    _METRICS_CALLS[key] = _METRICS_CALLS.get(key, 0) + 1
    _METRICS_LATENCY_MS[key] = _METRICS_LATENCY_MS.get(key, 0.0) + elapsed_ms
    if not ok:
        _METRICS_ERRORS[key] = _METRICS_ERRORS.get(key, 0) + 1


def gateway_metrics() -> dict:
    return {
        "calls": dict(_METRICS_CALLS),
        "errors": dict(_METRICS_ERRORS),
        "latency_ms": dict(_METRICS_LATENCY_MS),
    }


def _aitunnel_key() -> str:
    _load_env_files()
    return os.environ.get("AITUNNEL_API_KEY") or ""


def _build_request(
    *,
    provider: str,
    messages: list[dict],
    model: str | None,
    temperature: float,
    max_tokens: int | None,
    extra: dict | None,
    profile: str,
) -> tuple[str, dict, dict, str]:
    extra = dict(extra or {})
    if provider == "vllm":
        cfg = generate_cfg()
        url = f"{cfg['base_url']}/v1/chat/completions"
        model_id = model or cfg["model"]
        headers: dict = {}
    elif provider == "aitunnel":
        cfg = external_cfg(profile)
        url = f"{cfg['base_url']}/chat/completions"
        model_id = model or cfg["model"]
        key = _aitunnel_key()
        if not key:
            raise GatewayError("AITUNNEL_API_KEY is not set")
        headers = {"Authorization": f"Bearer {key}"}
    else:
        raise GatewayError(f"unknown provider: {provider}")

    body: dict = {
        "model": model_id,
        "temperature": temperature,
        "messages": messages,
        **extra,
    }
    if max_tokens is not None and "max_tokens" not in extra:
        body["max_tokens"] = max_tokens
    return url, headers, body, model_id


def _parse(resp: httpx.Response, provider: str, model_id: str) -> ChatResult:
    if resp.status_code >= 400:
        raise GatewayError(f"{provider} HTTP {resp.status_code}", status_code=resp.status_code)
    try:
        payload = resp.json()
    except Exception as exc:
        raise GatewayError(f"{provider} invalid JSON") from exc
    choice = (payload.get("choices") or [{}])[0] or {}
    content = ((choice.get("message") or {}).get("content")) or ""
    return ChatResult(
        content=content,
        provider=provider,
        model=model_id,
        status_code=resp.status_code,
        raw=payload if isinstance(payload, dict) else {},
        finish_reason=str(choice.get("finish_reason") or ""),
    )


def _timeout(timeout: float) -> httpx.Timeout:
    return httpx.Timeout(timeout, connect=min(15.0, timeout))


async def _get_async_client() -> httpx.AsyncClient:
    global _async_client, _async_loop_id
    import asyncio

    loop = asyncio.get_running_loop()
    loop_id = id(loop)
    if _async_client is not None and (not _async_client.is_closed) and _async_loop_id == loop_id:
        return _async_client
    if _async_client is not None:
        try:
            await _async_client.aclose()
        except Exception:
            pass
        _async_client = None
    _async_client = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=128, max_keepalive_connections=64),
        timeout=_timeout(180),
    )
    _async_loop_id = loop_id
    return _async_client


def chat(
    *,
    provider: str,
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.1,
    max_tokens: int | None = 400,
    timeout: float = 180,
    extra: dict | None = None,
    profile: str = "dashboard_qa",
) -> ChatResult:
    """OpenAI-compatible chat.completions. `provider` is vllm | aitunnel."""
    started = time.perf_counter()
    ok = False
    try:
        url, headers, body, model_id = _build_request(
            provider=provider,
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            extra=extra,
            profile=profile,
        )
        try:
            with httpx.Client(timeout=_timeout(timeout)) as client:
                resp = client.post(url, json=body, headers=headers)
        except httpx.TimeoutException as exc:
            raise GatewayError(f"{provider} timeout", status_code=0) from exc
        except httpx.HTTPError as exc:
            raise GatewayError(f"{provider} HTTP error") from exc
        result = _parse(resp, provider, model_id)
        ok = True
        return result
    finally:
        _record_call(provider, profile, ok, (time.perf_counter() - started) * 1000)


async def achat(
    *,
    provider: str,
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.1,
    max_tokens: int | None = 400,
    timeout: float = 180,
    extra: dict | None = None,
    profile: str = "dashboard_qa",
) -> ChatResult:
    """Async variant of chat() for FastAPI llm-run / smart-agent."""
    started = time.perf_counter()
    ok = False
    try:
        url, headers, body, model_id = _build_request(
            provider=provider,
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            extra=extra,
            profile=profile,
        )
        client = await _get_async_client()
        try:
            resp = await client.post(url, json=body, headers=headers, timeout=_timeout(timeout))
        except httpx.TimeoutException as exc:
            raise GatewayError(f"{provider} timeout", status_code=0) from exc
        except httpx.HTTPError as exc:
            raise GatewayError(f"{provider} HTTP error") from exc
        result = _parse(resp, provider, model_id)
        ok = True
        return result
    finally:
        _record_call(provider, profile, ok, (time.perf_counter() - started) * 1000)


async def ping_vllm(timeout: float = 5) -> bool:
    cfg = generate_cfg()
    url = f"{cfg['base_url']}/v1/models"
    try:
        client = await _get_async_client()
        resp = await client.get(url, timeout=_timeout(timeout))
        return resp.status_code == 200
    except Exception:
        return False


@dataclass
class _CompatMessage:
    content: str
    role: str = "assistant"


@dataclass
class _CompatChoice:
    message: _CompatMessage


@dataclass
class _CompatResponse:
    choices: list
    model: str = ""


class GatewayChatClient:
    """Drop-in for openai.OpenAI().chat.completions.create — no keys in callers."""

    def __init__(self, provider: str = "aitunnel", profile: str = "dashboard_qa"):
        self.provider = provider
        self.profile = profile
        self.chat = _ChatNamespace(self)


class _ChatNamespace:
    def __init__(self, owner: GatewayChatClient):
        self.completions = _Completions(owner)


class _Completions:
    def __init__(self, owner: GatewayChatClient):
        self._owner = owner

    def create(
        self,
        messages,
        model=None,
        temperature=None,
        max_tokens=None,
        timeout=None,
        **kwargs,
    ):
        extra = {}
        for key in ("response_format", "top_p", "chat_template_kwargs", "stream"):
            if key in kwargs:
                extra[key] = kwargs[key]
        result = chat(
            provider=self._owner.provider,
            messages=messages,
            model=model,
            temperature=1.0 if temperature is None else temperature,
            max_tokens=max_tokens,
            timeout=float(timeout) if timeout is not None else 180,
            extra=extra or None,
            profile=self._owner.profile,
        )
        return _CompatResponse(
            choices=[_CompatChoice(message=_CompatMessage(content=result.content))],
            model=result.model,
        )
