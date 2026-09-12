"""Serving lock: pinned models, images, prompt ids. No secrets."""
from __future__ import annotations

import os
import re
import shutil
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

_PKG = Path(__file__).resolve().parent
_LOCK_CANDIDATES = [
    Path(os.environ["MLOPS_LOCK"]) if os.environ.get("MLOPS_LOCK") else None,
    _PKG / "lock.yaml",
    Path("/home/dev/tellscope_app/tellscope_backend/mlops/lock.yaml"),
]
_LOCK_CANDIDATES = [p for p in _LOCK_CANDIDATES if p is not None]


@lru_cache(maxsize=1)
def load_lock() -> dict:
    for path in _LOCK_CANDIDATES:
        if path and path.exists():
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            data["_lock_path"] = str(path)
            return data
    return {"version": 0, "generate": {}, "embed": {}, "external": {}, "prompts": {}}


def generate_cfg() -> dict:
    lock = load_lock().get("generate") or {}
    return {
        **lock,
        "model": os.environ.get("VLLM_MODEL") or lock.get("model") or "Qwen/Qwen3-32B-FP8",
        "base_url": (
            os.environ.get("VLLM_BASE_URL") or lock.get("base_url") or "http://127.0.0.1:8000"
        ).rstrip("/"),
        "revision": os.environ.get("VLLM_REVISION") or lock.get("revision") or "",
    }


def embed_cfg() -> dict:
    lock = load_lock().get("embed") or {}
    return {
        **lock,
        "model": os.environ.get("EMBED_MODEL") or lock.get("model") or "deepvk/USER2-base",
    }


def external_cfg(profile: str = "dashboard_qa") -> dict:
    lock = load_lock().get("external") or {}
    profiles = lock.get("profiles") or {}
    env_profile = os.environ.get(f"AITUNNEL_MODEL_{profile.upper()}")
    env_model = os.environ.get("AITUNNEL_MODEL")
    model = (
        env_profile
        or profiles.get(profile)
        or env_model
        or profiles.get("dashboard_qa")
        or "gpt-4.1-mini"
    )
    return {
        **lock,
        "model": model,
        "profile": profile,
        "base_url": (
            os.environ.get("AITUNNEL_BASE_URL") or lock.get("base_url") or "https://api.aitunnel.ru/v1"
        ).rstrip("/"),
    }


# --- Вовлечённость: веса и шкала ранжирования «ключевых сообщений» (agent_engine/tools_text.py).
# Дефолты живут в коде инструмента, здесь только переопределения: без настройки поведение
# остаётся предсказуемым. Приоритет — окружение, затем lock.yaml:
#   1) TELLSCOPE_ENGAGEMENT_WEIGHTS="commentsCount=3,repostsCount=4,likesCount=2,er=0"
#      TELLSCOPE_ENGAGEMENT_SCALE=log|sqrt|linear
#      TELLSCOPE_ENGAGEMENT_ENABLED=true|false
#   2) lock.yaml:  engagement: {weights: {...}, scale: log, enabled: true}
def _engagement_env_weights(raw: str) -> dict:
    """Разбирает TELLSCOPE_ENGAGEMENT_WEIGHTS: «commentsCount=3,repostsCount=4»."""
    out: dict = {}
    for chunk in str(raw or "").replace(";", ",").split(","):
        if "=" not in chunk:
            continue
        field, _, value = chunk.partition("=")
        field = field.strip()
        value = value.strip().replace(",", ".")
        if not field or not value:
            continue
        try:
            out[field] = float(value)
        except ValueError:
            continue
    return out


@lru_cache(maxsize=64)
def _lock_engagement_cached(path_str: str, mtime: float) -> dict:
    """Секция engagement из lock-файла. Кэш привязан к mtime файла, поэтому правка YAML
    подхватывается на следующем запуске инструмента — без рестарта приложения."""
    try:
        data = yaml.safe_load(Path(path_str).read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    section = data.get("engagement") or {}
    return section if isinstance(section, dict) else {}


def _lock_engagement() -> dict:
    for path in _LOCK_CANDIDATES:
        try:
            if path and path.exists():
                return _lock_engagement_cached(str(path), path.stat().st_mtime)
        except Exception:
            continue
    return {}


def engagement_cfg() -> dict:
    """Активные веса и шкала вовлечённости: lock.yaml + окружение (окружение приоритетнее)."""
    lock = _lock_engagement()
    lock_weights = lock.get("weights") if isinstance(lock.get("weights"), dict) else {}
    env_weights = _engagement_env_weights(os.environ.get("TELLSCOPE_ENGAGEMENT_WEIGHTS") or "")
    env_scale = os.environ.get("TELLSCOPE_ENGAGEMENT_SCALE") or ""
    env_enabled = os.environ.get("TELLSCOPE_ENGAGEMENT_ENABLED")
    enabled = lock.get("enabled")
    if env_enabled not in (None, ""):
        enabled = str(env_enabled).strip().lower() in ("1", "true", "yes", "on", "да")
    return {
        "weights": {**(lock_weights or {}), **env_weights},
        "scale": env_scale or lock.get("scale") or "",
        "enabled": True if enabled is None else bool(enabled),
        "source": {
            "lock": bool(lock),
            "env_weights": bool(env_weights),
            "env_scale": bool(env_scale),
            "env_enabled": env_enabled not in (None, ""),
        },
    }


def prompt_id(name: str, default: str) -> str:
    prompts = load_lock().get("prompts") or {}
    return str(prompts.get(name) or default)


# ------------------------------------------------------- модель-оркестратор (agent)
# Единая настройка того, кто планирует шаги и вызывает инструменты Tellscope.
# Значения — ключи MODEL_CHOICES из agent_engine/loop.py: gpt | deepseek | qwen | claude.
# Настройка читается заново на каждый запуск (файл + переменные окружения), поэтому
# переключение действует без правки кода и без рестарта приложения. Секретов тут нет.

DEFAULT_ORCHESTRATOR = "gpt"
DEFAULT_ORCHESTRATOR_FALLBACKS: Tuple[str, ...] = ("deepseek", "qwen")
DEFAULT_ORCHESTRATOR_PROBE = "tools"
PROBE_MODES = ("tools", "reachability")

ENV_ORCHESTRATOR = "TELLSCOPE_ORCHESTRATOR"
ENV_ORCHESTRATOR_FALLBACKS = "TELLSCOPE_ORCHESTRATOR_FALLBACKS"
ENV_ORCHESTRATOR_PROBE = "TELLSCOPE_ORCHESTRATOR_PROBE"

_ENV_FILE_CANDIDATES = [
    Path("/home/dev/tellscope_app/tellscope_backend/.env"),
    _PKG.parent / ".env",
    Path.cwd() / ".env",
]

_ENV_FILE_AT_START: Optional[Dict[str, str]] = None
_AGENT_WRITE_LOCK = threading.Lock()

_AGENT_BLOCK_RE = re.compile(r"(?ms)(?:^#[^\n]*\n)*^agent:[ \t]*\n(?:[ \t]+[^\n]*\n|\n)*")


def _read_env_file() -> Dict[str, str]:
    """Свежее чтение .env: supervisor не отдаёт переменные, оператор правит именно этот файл."""
    for path in _ENV_FILE_CANDIDATES:
        try:
            if not path or not path.exists():
                continue
            values: Dict[str, str] = {}
            for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                if key:
                    values[key] = val.strip().strip('"').strip("'")
            if values:
                return values
        except Exception:
            continue
    return {}


def _env_file_at_start() -> Dict[str, str]:
    global _ENV_FILE_AT_START
    if _ENV_FILE_AT_START is None:
        _ENV_FILE_AT_START = _read_env_file()
    return _ENV_FILE_AT_START


def _env_override(key: str) -> Tuple[Optional[str], str]:
    """Значение переменной окружения и его источник.

    Если .env отредактировали после старта процесса, свежее значение важнее копии,
    которую шлюз при импорте перенёс в os.environ: иначе переключение через env
    требовало бы рестарта, а это как раз то, чего мы избегаем. Так же ловится и удаление
    строки из .env: копия в os.environ считается устаревшей и настройка возвращается к lock.yaml.
    """
    fresh = _read_env_file().get(key)
    if fresh != _env_file_at_start().get(key):
        return (fresh.strip(), "env-file") if fresh else (None, "")
    proc = os.environ.get(key)
    if proc:
        source = "env-file" if fresh and proc.strip() == fresh.strip() else "env"
        return proc.strip(), source
    if fresh:
        return fresh.strip(), "env-file"
    return None, ""


def _as_key_list(value: Any) -> List[str]:
    if value in (None, "", [], {}):
        return []
    if isinstance(value, (list, tuple, set)):
        raw_items = [str(item) for item in value]
    else:
        raw_items = re.split(r"[,\s;]+", str(value))
    out: List[str] = []
    for item in raw_items:
        key = item.strip().lower()
        if key and key not in out:
            out.append(key)
    return out


def lock_file_path() -> Optional[Path]:
    for path in _LOCK_CANDIDATES:
        try:
            if path and path.exists():
                return path
        except Exception:
            continue
    return None


def read_lock_fresh() -> dict:
    """lock.yaml без кэша: нужен там, где настройку меняют на ходу."""
    path = lock_file_path()
    if path is None:
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def agent_cfg() -> dict:
    """Настройка модели-оркестратора: что планирует шаги и вызывает инструменты.

    Приоритет: переменные окружения → секция agent в lock.yaml → значения по умолчанию.
    Файл читается на каждый вызов (это ~2 КБ), поэтому смена настройки применяется сразу.
    """
    section = read_lock_fresh().get("agent") or {}
    path = lock_file_path()

    primary = str(section.get("orchestrator") or "").strip().lower() or DEFAULT_ORCHESTRATOR
    source = "lock.yaml" if section.get("orchestrator") else "default"
    fallbacks = _as_key_list(section.get("orchestrator_fallbacks")) or list(DEFAULT_ORCHESTRATOR_FALLBACKS)
    fallbacks_source = "lock.yaml" if section.get("orchestrator_fallbacks") else "default"
    probe = str(section.get("orchestrator_probe") or "").strip().lower() or DEFAULT_ORCHESTRATOR_PROBE
    probe_source = "lock.yaml" if section.get("orchestrator_probe") else "default"

    env_primary, env_primary_source = _env_override(ENV_ORCHESTRATOR)
    if env_primary:
        primary, source = env_primary.lower(), env_primary_source
    env_fallbacks, env_fallbacks_source = _env_override(ENV_ORCHESTRATOR_FALLBACKS)
    if env_fallbacks is not None:
        fallbacks, fallbacks_source = _as_key_list(env_fallbacks), env_fallbacks_source
    env_probe, env_probe_source = _env_override(ENV_ORCHESTRATOR_PROBE)
    if env_probe:
        probe, probe_source = env_probe.lower(), env_probe_source
    if probe not in PROBE_MODES:
        probe = DEFAULT_ORCHESTRATOR_PROBE

    chain = [primary] + [key for key in fallbacks if key != primary]
    return {
        "orchestrator": primary,
        "orchestrator_fallbacks": fallbacks,
        "chain": chain,
        "orchestrator_probe": probe,
        "source": source,
        "fallbacks_source": fallbacks_source,
        "probe_source": probe_source,
        "lock_path": str(path) if path else None,
        "env_file": str(next((p for p in _ENV_FILE_CANDIDATES if p and p.exists()), "") or ""),
    }


def _render_agent_block(orchestrator: str, fallbacks: List[str], probe: str) -> str:
    return (
        "# Модель-оркестратор: планирует шаги и вызывает инструменты Tellscope.\n"
        "# Значения — ключи MODEL_CHOICES из agent_engine/loop.py (gpt | deepseek | qwen | claude).\n"
        "# Настройка читается на каждый запуск, поэтому переключается без правки кода и рестарта.\n"
        "# Переопределения окружением: TELLSCOPE_ORCHESTRATOR, TELLSCOPE_ORCHESTRATOR_FALLBACKS,\n"
        "# TELLSCOPE_ORCHESTRATOR_PROBE (tools | reachability). Приоритет у окружения.\n"
        "agent:\n"
        f"  orchestrator: {orchestrator}\n"
        "  orchestrator_fallbacks: [" + ", ".join(fallbacks) + "]\n"
        f"  orchestrator_probe: {probe}\n"
        "\n"
    )


def save_agent_cfg(orchestrator: str, fallbacks: Optional[List[str]] = None,
                   probe: Optional[str] = None, path: Optional[Path] = None) -> dict:
    """Пишет настройку оркестратора в lock.yaml, сохраняя остальной файл как есть.

    Рядом один раз создаётся копия lock.yaml.bak_orch — состояние до первой правки.
    Запись атомарная (временный файл + replace), поэтому читатели не увидят половину файла.
    """
    target = Path(path) if path else lock_file_path()
    if target is None or not target.exists():
        raise FileNotFoundError("lock.yaml не найден: некуда записывать настройку оркестратора")
    primary = str(orchestrator or "").strip().lower()
    if not primary:
        raise ValueError("нужен ключ модели-оркестратора")
    keys = _as_key_list(fallbacks if fallbacks is not None else DEFAULT_ORCHESTRATOR_FALLBACKS)
    probe_mode = str(probe or DEFAULT_ORCHESTRATOR_PROBE).strip().lower()
    if probe_mode not in PROBE_MODES:
        raise ValueError(f"orchestrator_probe должен быть одним из {PROBE_MODES}")

    with _AGENT_WRITE_LOCK:
        text = target.read_text(encoding="utf-8")
        block = _render_agent_block(primary, keys, probe_mode)
        if _AGENT_BLOCK_RE.search(text):
            text = _AGENT_BLOCK_RE.sub(block, text, count=1)
        else:
            text = text.rstrip("\n") + "\n\n" + block
        backup = target.with_name(target.name + ".bak_orch")
        if not backup.exists():
            shutil.copy2(target, backup)
        tmp = target.with_name(target.name + ".tmp_orch")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, target)
        load_lock.cache_clear()
    return agent_cfg()


def public_lock() -> dict:
    """Safe to return from an API — no env secrets."""
    lock = dict(load_lock())
    lock.pop("_lock_path", None)
    gen = generate_cfg()
    emb = embed_cfg()
    ext = load_lock().get("external") or {}
    agent = agent_cfg()
    return {
        "version": lock.get("version"),
        "generate": {
            "provider": gen.get("provider"),
            "model": gen.get("model"),
            "revision": gen.get("revision") or None,
            "image": gen.get("image"),
            "image_digest": gen.get("image_digest"),
            "pipeline_parallel_size": gen.get("pipeline_parallel_size"),
            "max_model_len": gen.get("max_model_len"),
            "gpu": gen.get("gpu"),
        },
        "embed": {
            "provider": emb.get("provider"),
            "model": emb.get("model"),
            "gpu": emb.get("gpu"),
        },
        "external": {
            "provider": ext.get("provider"),
            "profiles": ext.get("profiles") or {},
        },
        "agent": {
            "orchestrator": agent.get("orchestrator"),
            "orchestrator_fallbacks": agent.get("orchestrator_fallbacks"),
            "orchestrator_probe": agent.get("orchestrator_probe"),
            "source": agent.get("source"),
        },
        "prompts": lock.get("prompts") or {},
        # Прозрачность ранжирования «ключевых сообщений»: что реально применено.
        "engagement": {
            "weights": (engagement_cfg().get("weights") or {}),
            "scale": engagement_cfg().get("scale") or None,
            "enabled": bool(engagement_cfg().get("enabled")),
        },
    }
