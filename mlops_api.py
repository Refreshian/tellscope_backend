"""Read-only MLOps surface: serving lock, jobs, prompts, drain, Prometheus metrics.

Журнал задач (``GET /mlops/jobs``) персональный:

* по умолчанию отдаются только задачи текущего пользователя;
* суперпользователь может запросить весь журнал — ``?all=1`` (синоним ``?scope=all``);
  обычному пользователю такой запрос отклоняется с 403;
* автор задачи берётся из поля ``user_id`` записи журнала (Redis-хеш
  ``tellscope:job:{id}``). Его пишут производители: agent-mode (``agent_engine/runs.py``),
  ИИ-анализ (``mlops/runtime.py``), smart-agent, загрузка файлов, Мосинформ
  (``mosinform_api.py``);
* у старых записей ``user_id`` пуст. Такие задачи показываются только суперпользователю
  и помечаются ``author_unknown: true``: доказать, что запись принадлежит обычному
  пользователю, нечем, а текст постановки в ней чужой. Поэтому обычному пользователю они
  не отдаются.

``GET /mlops/busy`` отвечает на вопрос «кто держит GPU»: чужие задачи видны, потому что
занятость GPU общая, но обычному пользователю у чужих записей скрыты ``job_id`` и текст
постановки (остаются продукт и статус).

Права проверяются собственной зависимостью (тот же приём, что в ``reports_api.py``):
импортировать ``main`` отсюда нельзя — ``main.py`` сам импортирует этот роутер.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response

from mlops.gateway import gateway_metrics
from mlops.jobs import list_jobs
from mlops.lock import public_lock
from mlops.prompts import list_prompts
from mlops.runtime import reap_stale_jobs, running_batches, stale_jobs

router = APIRouter(prefix="/mlops", tags=["mlops"])


# --------------------------------------------------------------------------- #
# авторизация
# --------------------------------------------------------------------------- #
def _extract_token(request: Request) -> str:
    """Токен из заголовка Bearer либо из cookie (скачивание идёт обычной навигацией)."""
    auth = request.headers.get("authorization") or ""
    if auth[:7].lower() == "bearer ":
        return auth[7:].strip()
    for name in ("token", "access_token", "tellscope_refresh_token"):
        value = request.cookies.get(name)
        if value:
            return value.strip()
    return ""


async def current_user(request: Request):
    """Пользователь по токену. Без токена или с недействительным — 401."""
    token = _extract_token(request)
    if not token or token.lower() in ("null", "undefined"):
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        import jwt as _jwt
        from auth.auth import SECRET as _SECRET

        payload = _jwt.decode(
            token, _SECRET, algorithms=["HS256"], options={"verify_aud": False}
        )
        user_id = int(payload.get("sub"))
    except Exception:
        raise HTTPException(status_code=401, detail="Unauthorized")

    from sqlalchemy import select

    from auth.database import User as AuthUser, async_session_maker

    async with async_session_maker() as session:
        user = (
            await session.execute(select(AuthUser).where(AuthUser.id == user_id))
        ).scalars().first()
    if user is None or not getattr(user, "is_active", False):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return user


def _is_superuser(user: Any) -> bool:
    return bool(getattr(user, "is_superuser", False))


def _owner_of(job: Dict[str, Any]) -> str:
    """Автор задачи: поле ``user_id`` записи журнала (пусто у старых задач)."""
    return str(job.get("user_id") or "").strip()


def _scope_requested(all_flag: int, scope: str) -> bool:
    return bool(all_flag) or str(scope or "").strip().lower() == "all"


def _guard_scope(user: Any, all_flag: int, scope: str) -> bool:
    """True — запрошен весь журнал. Обычному пользователю это запрещено (403)."""
    if not _scope_requested(all_flag, scope):
        return False
    if not _is_superuser(user):
        raise HTTPException(
            status_code=403, detail="Показ всех задач доступен только администратору"
        )
    return True


def _with_author(job: Dict[str, Any]) -> Dict[str, Any]:
    item = dict(job)
    owner = _owner_of(job)
    item["user_id"] = owner
    item["author_unknown"] = not owner
    return item


def _scoped_jobs(user: Any, all_flag: int, scope: str) -> Dict[str, Any]:
    every = _guard_scope(user, all_flag, scope)
    mine = str(getattr(user, "id", "") or "")
    jobs = list_jobs()
    if every:
        visible = [_with_author(job) for job in jobs]
    else:
        visible = [
            _with_author(job)
            for job in jobs
            if _owner_of(job) and _owner_of(job) == mine
        ]
    return {
        "jobs": visible,
        "scope": "all" if every else "mine",
        "can_see_all": _is_superuser(user),
        "total": len(visible),
        "hidden": max(0, len(jobs) - len(visible)),
    }


@router.get("/lock")
def serving_lock():
    return public_lock()


@router.get("/jobs")
def jobs(
    all_: int = Query(
        0,
        alias="all",
        description="1 — показать задачи всего тенанта (только суперпользователь)",
    ),
    scope: str = Query("", description="mine | all — синоним ?all=1"),
    user: Any = Depends(current_user),
):
    """Журнал задач: свои — всем, все — только суперпользователю."""
    return _scoped_jobs(user, all_, scope)


@router.get("/busy")
def busy(user: Any = Depends(current_user)):
    """Кто держит GPU. Чужие задачи обычному пользователю — без id и текста постановки."""
    holders = running_batches()
    if _is_superuser(user):
        return {"jobs": [_with_author(job) for job in holders]}
    mine = str(getattr(user, "id", "") or "")
    masked: List[Dict[str, Any]] = []
    for job in holders:
        if _owner_of(job) and _owner_of(job) == mine:
            masked.append(_with_author(job))
        else:
            masked.append(
                {
                    "product": job.get("product") or "",
                    "status": job.get("status") or "",
                    "job_id": "",
                    "message": "",
                    "user_id": "",
                    "author_unknown": True,
                    "foreign": True,
                }
            )
    return {"jobs": masked}


@router.get("/ready")
def ready(response: Response):
    """503 if a live GPU batch holds Qwen — check before FastAPI restart."""
    holders = running_batches()
    if holders:
        response.status_code = 503
        return {"ready": False, "jobs": holders}
    return {"ready": True, "jobs": []}


@router.get("/health")
def health():
    """Живость сервиса: 200 и {"status": "ok"} без токена.

    Отдельно от /mlops/ready: ready отвечает про занятость GPU (503, если держит батч),
    а health — просто «сервис жив». Оба пути открыты в белом списке гейта (main.PUBLIC_EXACT),
    поэтому внешние проверки не нуждаются в токене.
    """
    return {"status": "ok"}


@router.get("/prompts")
def prompts():
    return {"prompts": list_prompts()}


@router.post("/drain")
def drain():
    """Mark stale journal leftovers as stale. Live GPU holders are left alone."""
    return {"reaped": reap_stale_jobs(), "stale": stale_jobs()}


def _prom_line(name: str, labels: dict, value) -> str:
    parts = ",".join(f'{k}="{v}"' for k, v in labels.items())
    inner = f"{{{parts}}}" if parts else ""
    return f"{name}{inner} {value}"


@router.get("/metrics")
def metrics():
    jobs = list_jobs()
    busy_jobs = running_batches()
    by_ps = Counter((job.get("product") or "unknown", job.get("status") or "unknown") for job in jobs)
    stale = len(stale_jobs())
    gw = gateway_metrics()
    lines = [
        "# HELP tellscope_jobs Jobs in the unified journal",
        "# TYPE tellscope_jobs gauge",
    ]
    for (product, status), count in sorted(by_ps.items()):
        lines.append(_prom_line("tellscope_jobs", {"product": product, "status": status}, count))
    lines += [
        "# HELP tellscope_gpu_busy 1 if a live heavy GPU job holds Qwen",
        "# TYPE tellscope_gpu_busy gauge",
        f"tellscope_gpu_busy {1 if busy_jobs else 0}",
        "# HELP tellscope_gpu_holders Live heavy GPU jobs holding Qwen",
        "# TYPE tellscope_gpu_holders gauge",
        f"tellscope_gpu_holders {len(busy_jobs)}",
        "# HELP tellscope_jobs_stale Active-looking jobs without a fresh heartbeat",
        "# TYPE tellscope_jobs_stale gauge",
        f"tellscope_jobs_stale {stale}",
        "# HELP tellscope_gateway_calls_total LLM gateway calls since process start",
        "# TYPE tellscope_gateway_calls_total counter",
    ]
    for key, count in sorted(gw["calls"].items()):
        provider, profile = (key.split("|", 1) + ["-"])[:2]
        lines.append(_prom_line("tellscope_gateway_calls_total", {"provider": provider, "profile": profile}, count))
    lines += [
        "# HELP tellscope_gateway_errors_total LLM gateway errors since process start",
        "# TYPE tellscope_gateway_errors_total counter",
    ]
    for key, count in sorted(gw["errors"].items()):
        provider, profile = (key.split("|", 1) + ["-"])[:2]
        lines.append(_prom_line("tellscope_gateway_errors_total", {"provider": provider, "profile": profile}, count))
    lines += [
        "# HELP tellscope_gateway_latency_ms_sum Sum of gateway call latency",
        "# TYPE tellscope_gateway_latency_ms_sum counter",
    ]
    for key, total in sorted(gw["latency_ms"].items()):
        provider, profile = (key.split("|", 1) + ["-"])[:2]
        lines.append(_prom_line("tellscope_gateway_latency_ms_sum", {"provider": provider, "profile": profile}, round(total, 1)))
    body = "\n".join(lines) + "\n"
    return Response(content=body, media_type="text/plain; version=0.0.4")
