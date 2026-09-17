"""FastAPI-роутер: загрузка выгрузок Медиалогии → PPTX рейтинга ОИВ.

Права доступа (расчёты Мосинформ.Рейтинг персональные):

* ``GET /mosinform/jobs`` — по умолчанию только расчёты текущего пользователя;
  суперпользователь может запросить все ``?all=1`` (синоним ``?scope=all``), обычному
  пользователю такой запрос отклоняется с 403;
* ``GET /mosinform/jobs/{job_id}`` — статус расчёта, ``GET .../pptx`` и ``.../xlsx`` —
  скачивание файлов: чужой ``job_id`` даёт 403;
* владелец расчёта берётся из поля ``user_id`` записи (Redis-хеш ``mosinform:{job_id}``
  и ``job.json``), его пишет ``POST /mosinform/jobs`` при создании. У расчётов, созданных
  до появления учёта автора, поле пусто: такие записи доступны только суперпользователю
  и помечаются ``author_unknown: true`` — иначе чужой расчёт остался бы виден обычному
  пользователю, а доказать его принадлежность нечем.

Права проверяются собственной зависимостью (тот же приём, что в ``reports_api.py``):
импортировать ``main`` отсюда нельзя — ``main.py`` сам импортирует этот роутер.
"""
from __future__ import annotations

import json
import os
import re
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import redis
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse

from mosinform.pipeline import run_pipeline

try:
    from mlops.jobs import register as mlops_register
except ImportError:
    mlops_register = None

load_dotenv()

router = APIRouter(prefix="/mosinform", tags=["mosinform rating"])

DATA_ROOT = Path("/home/dev/tellscope_app/tellscope_backend/data/mosinform")
REDIS = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
JOBS_INDEX = "mosinform:jobs"
ARCHIVE_ID = "__archive__"
SUMMARY_KEYS = ("messages", "objects", "untagged", "top", "missing", "notes")

# Идентификатор расчёта: uuid4().hex[:12]. Ограничение защищает от «..» и «/» в пути.
JOB_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


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


def _user_key(user: Any) -> str:
    return str(getattr(user, "id", "") or "")


def _scope_requested(all_flag: int, scope: str) -> bool:
    return bool(all_flag) or str(scope or "").strip().lower() == "all"


def _guard_scope(user: Any, all_flag: int, scope: str) -> bool:
    """True — запрошены все расчёты. Обычному пользователю это запрещено (403)."""
    if not _scope_requested(all_flag, scope):
        return False
    if not _is_superuser(user):
        raise HTTPException(
            status_code=403, detail="Показ всех задач доступен только администратору"
        )
    return True


def _job_key(job_id: str) -> str:
    return f"mosinform:{job_id}"


def _out_paths(job_id: str) -> tuple[Path, Path]:
    out = DATA_ROOT / job_id / "out"
    return out / "mosinform_rating.pptx", out / "mosinform_rating.xlsx"


def _persist(job_id: str) -> None:
    if job_id == ARCHIVE_ID:
        return
    try:
        REDIS.sadd(JOBS_INDEX, job_id)
        data = REDIS.hgetall(_job_key(job_id)) or {}
        data["job_id"] = job_id
        path = DATA_ROOT / job_id / "job.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def _set(job_id: str, **fields) -> None:
    REDIS.hset(_job_key(job_id), mapping={k: str(v) for k, v in fields.items()})
    _persist(job_id)
    if mlops_register:
        data = _get(job_id)
        mlops_register(
            job_id,
            product="mosinform",
            route="/mosinform-rating",
            status=data.get("status") or "",
            message=data.get("message") or "",
            period=data.get("period") or "",
            files=data.get("files") or "",
            user_id=data.get("user_id") or "",
            created_at=data.get("created_at") or "",
            updated_at=data.get("updated_at") or datetime.now().isoformat(),
        )


def _get(job_id: str) -> dict:
    return REDIS.hgetall(_job_key(job_id)) or {}


def _load_disk(job_id: str) -> dict:
    path = DATA_ROOT / job_id / "job.json"
    if not path.exists():
        return {}
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
        return saved if isinstance(saved, dict) else {}
    except Exception:
        return {}


def _merged(job_id: str) -> dict:
    """Запись расчёта: Redis, дополненный полями из job.json."""
    data = dict(_get(job_id))
    if not data:
        return dict(_load_disk(job_id))
    for key, value in _load_disk(job_id).items():
        if not data.get(key):
            data[key] = value
    return data


def _owner_of(job_id: str) -> str:
    """Автор расчёта: поле ``user_id`` записи (пусто у расчётов без автора)."""
    return str(_merged(job_id).get("user_id") or "").strip()


def _valid_job_id(job_id: str) -> str:
    value = str(job_id or "")
    if value == ARCHIVE_ID or not JOB_ID_RE.match(value):
        raise HTTPException(status_code=404, detail="Задача не найдена")
    return value


def _guard_job(job_id: str, user: Any) -> str:
    """Расчёт существует (иначе 404) и доступен пользователю (иначе 403)."""
    value = _valid_job_id(job_id)
    data = _merged(value)
    if not data and not (DATA_ROOT / value).is_dir():
        raise HTTPException(status_code=404, detail="Задача не найдена")
    if _is_superuser(user):
        return value
    owner = str(data.get("user_id") or "").strip()
    if not owner or owner != _user_key(user):
        raise HTTPException(
            status_code=403, detail="Нет доступа к расчёту другого пользователя"
        )
    return value


def _parse_summary(raw) -> dict:
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def _resolve_file(job_id: str, kind: str) -> Path | None:
    data = _merged(job_id)
    stored = data.get(kind)
    if stored and Path(stored).exists():
        return Path(stored)
    pptx, xlsx = _out_paths(job_id)
    candidate = pptx if kind == "pptx" else xlsx
    return candidate if candidate.exists() else None


def _known_job_ids() -> set[str]:
    ids: set[str] = set()
    try:
        ids.update(REDIS.smembers(JOBS_INDEX) or [])
        for key in REDIS.keys("mosinform:*"):
            if key in {JOBS_INDEX, _job_key(ARCHIVE_ID)}:
                continue
            if key.startswith("mosinform:") and key.count(":") == 1:
                ids.add(key.split(":", 1)[1])
    except Exception:
        pass
    if DATA_ROOT.exists():
        for path in DATA_ROOT.iterdir():
            if path.is_dir() and path.name not in {ARCHIVE_ID}:
                ids.add(path.name)
    ids.discard(ARCHIVE_ID)
    return {value for value in ids if JOB_ID_RE.match(str(value))}


def _public_job(job_id: str) -> dict | None:
    if job_id == ARCHIVE_ID:
        return None
    data = _merged(job_id)
    has_pptx = bool(_resolve_file(job_id, "pptx"))
    has_xlsx = bool(_resolve_file(job_id, "xlsx"))
    if not data and not (DATA_ROOT / job_id).is_dir():
        return None
    owner = str(data.get("user_id") or "").strip()
    status = data.get("status") or ("done" if has_pptx else "unknown")
    summary = _parse_summary(data.get("summary"))
    lineage = _parse_summary(data.get("lineage"))
    stale = False
    updated = data.get("updated_at") or ""
    if status in {"running", "queued"} and updated:
        try:
            age = (datetime.now() - datetime.fromisoformat(updated)).total_seconds()
            stale = age > 2 * 3600
        except ValueError:
            stale = False
    return {
        "job_id": job_id,
        "status": status,
        "stale": stale,
        "message": data.get("message") or "",
        "progress": data.get("progress") or "",
        "period": data.get("period") or "",
        "files": data.get("files") or "",
        "created_at": data.get("created_at") or "",
        "updated_at": data.get("updated_at") or "",
        "user_id": owner,
        "author_unknown": not owner,
        "summary": {key: summary[key] for key in SUMMARY_KEYS if key in summary},
        "lineage": {
            k: lineage[k]
            for k in (
                "git_sha",
                "model_id",
                "model_rev",
                "image_digest",
                "prompt_id",
                "catalog_version",
                "catalog_hash",
                "cache_key",
                "aitunnel_model",
            )
            if k in lineage
        },
        "has_pptx": has_pptx,
        "has_xlsx": has_xlsx,
    }


def _run(job_id: str, input_dir: Path, output_dir: Path, period: str) -> None:
    def progress(msg: str) -> None:
        _set(job_id, status="running", message=msg, updated_at=datetime.now().isoformat())

    try:
        _set(job_id, status="running", message="старт", progress="5")
        os.environ.setdefault("MOSINFORM_VLLM", "1")
        result = run_pipeline(
            input_dir, output_dir, period=period, tellscope=True, progress=progress, job_id=job_id
        )
        payload = dict(
            status="done",
            message="готово",
            progress="100",
            summary=json.dumps(result, ensure_ascii=False),
            pptx=result.get("pptx", ""),
            xlsx=result.get("xlsx", ""),
            updated_at=datetime.now().isoformat(),
        )
        if result.get("lineage"):
            payload["lineage"] = json.dumps(result["lineage"], ensure_ascii=False)
        _set(job_id, **payload)
    except Exception as exc:
        _set(
            job_id,
            status="error",
            message=str(exc)[:500],
            progress="0",
            updated_at=datetime.now().isoformat(),
        )


@router.get("/jobs")
def list_jobs(
    all_: int = Query(
        0,
        alias="all",
        description="1 — показать расчёты всего тенанта (только суперпользователь)",
    ),
    scope: str = Query("", description="mine | all — синоним ?all=1"),
    user: Any = Depends(current_user),
):
    """Архив расчётов: свои — всем, все — только суперпользователю."""
    every = _guard_scope(user, all_, scope)
    mine = _user_key(user)
    items = []
    for job_id in sorted(_known_job_ids()):
        item = _public_job(job_id)
        if item:
            items.append(item)
    jobs = items if every else [item for item in items if item["user_id"] and item["user_id"] == mine]
    jobs.sort(key=lambda item: item.get("created_at") or item.get("updated_at") or "", reverse=True)
    return {
        "jobs": jobs,
        "scope": "all" if every else "mine",
        "can_see_all": _is_superuser(user),
        "total": len(jobs),
        "hidden": max(0, len(items) - len(jobs)),
    }


@router.post("/jobs")
async def create_job(
    period: str = Form(default=""),
    files: list[UploadFile] = File(...),
    user: Any = Depends(current_user),
):
    if not files:
        raise HTTPException(400, "Нужно загрузить хотя бы один файл")
    try:
        from mlops.runtime import GpuBusy, assert_can_start

        assert_can_start("mosinform")
    except GpuBusy as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    job_id = uuid.uuid4().hex[:12]
    input_dir = DATA_ROOT / job_id / "in"
    output_dir = DATA_ROOT / job_id / "out"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for item in files:
        name = Path(item.filename or "file.bin").name
        target = input_dir / name
        data = await item.read()
        target.write_bytes(data)
        saved.append(name)
    _set(
        job_id,
        status="queued",
        message="файлы приняты",
        progress="0",
        period=period,
        files=", ".join(saved),
        user_id=_user_key(user),
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat(),
    )
    thread = threading.Thread(
        target=_run, args=(job_id, input_dir, output_dir, period), daemon=True
    )
    thread.start()
    return {"job_id": job_id, "files": saved}


@router.get("/jobs/{job_id}")
def job_status(job_id: str, user: Any = Depends(current_user)):
    job_id = _guard_job(job_id, user)
    data = _public_job(job_id)
    if not data:
        raise HTTPException(404, "Задача не найдена")
    return data


@router.get("/jobs/{job_id}/pptx")
def download_pptx(job_id: str, user: Any = Depends(current_user)):
    job_id = _guard_job(job_id, user)
    path = _resolve_file(job_id, "pptx")
    if not path:
        raise HTTPException(404, "Презентация ещё не готова")
    return FileResponse(
        path,
        filename=f"mosinform_rating_{job_id}.pptx",
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
    )


@router.get("/jobs/{job_id}/xlsx")
def download_xlsx(job_id: str, user: Any = Depends(current_user)):
    job_id = _guard_job(job_id, user)
    path = _resolve_file(job_id, "xlsx")
    if not path:
        raise HTTPException(404, "Excel ещё не готов")
    return FileResponse(
        path,
        filename=f"mosinform_rating_{job_id}.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
