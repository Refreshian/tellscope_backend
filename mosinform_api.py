"""FastAPI-роутер: загрузка выгрузок Медиалогии → PPTX рейтинга ОИВ."""
from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime
from pathlib import Path

import redis
from dotenv import load_dotenv
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from mosinform.pipeline import run_pipeline

load_dotenv()

router = APIRouter(prefix="/mosinform", tags=["mosinform rating"])

DATA_ROOT = Path("/home/dev/tellscope_app/tellscope_backend/data/mosinform")
REDIS = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)


def _job_key(job_id: str) -> str:
    return f"mosinform:{job_id}"


def _set(job_id: str, **fields) -> None:
    REDIS.hset(_job_key(job_id), mapping={k: str(v) for k, v in fields.items()})


def _get(job_id: str) -> dict:
    data = REDIS.hgetall(_job_key(job_id))
    return data or {}


def _run(job_id: str, input_dir: Path, output_dir: Path, period: str) -> None:
    def progress(msg: str) -> None:
        _set(job_id, status="running", message=msg, updated_at=datetime.now().isoformat())

    try:
        _set(job_id, status="running", message="старт", progress="5")
        os.environ.setdefault("MOSINFORM_VLLM", "1")
        os.environ.setdefault("VLLM_BASE_URL", "http://127.0.0.1:8000")
        os.environ.setdefault("VLLM_MODEL", "Qwen/Qwen3-32B-FP8")
        result = run_pipeline(input_dir, output_dir, period=period, tellscope=True, progress=progress)
        _set(
            job_id,
            status="done",
            message="готово",
            progress="100",
            summary=json.dumps(result, ensure_ascii=False),
            pptx=result.get("pptx", ""),
            xlsx=result.get("xlsx", ""),
        )
    except Exception as exc:
        _set(job_id, status="error", message=str(exc)[:500], progress="0")


@router.post("/jobs")
async def create_job(
    period: str = Form(default=""),
    files: list[UploadFile] = File(...),
):
    if not files:
        raise HTTPException(400, "Нужно загрузить хотя бы один файл")
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
        created_at=datetime.now().isoformat(),
    )
    thread = threading.Thread(
        target=_run, args=(job_id, input_dir, output_dir, period), daemon=True
    )
    thread.start()
    return {"job_id": job_id, "files": saved}


@router.get("/jobs/{job_id}")
def job_status(job_id: str):
    data = _get(job_id)
    if not data:
        raise HTTPException(404, "Задача не найдена")
    summary = {}
    if data.get("summary"):
        try:
            summary = json.loads(data["summary"])
        except json.JSONDecodeError:
            summary = {}
    return {
        "job_id": job_id,
        "status": data.get("status"),
        "message": data.get("message"),
        "progress": data.get("progress"),
        "period": data.get("period"),
        "files": data.get("files"),
        "summary": {
            k: summary.get(k)
            for k in ("messages", "objects", "untagged", "top", "missing", "notes")
            if k in summary
        },
        "has_pptx": bool(data.get("pptx") and Path(data["pptx"]).exists()),
        "has_xlsx": bool(data.get("xlsx") and Path(data["xlsx"]).exists()),
    }


@router.get("/jobs/{job_id}/pptx")
def download_pptx(job_id: str):
    data = _get(job_id)
    path = data.get("pptx")
    if not path or not Path(path).exists():
        raise HTTPException(404, "Презентация ещё не готова")
    return FileResponse(
        path,
        filename=f"mosinform_rating_{job_id}.pptx",
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
    )


@router.get("/jobs/{job_id}/xlsx")
def download_xlsx(job_id: str):
    data = _get(job_id)
    path = data.get("xlsx")
    if not path or not Path(path).exists():
        raise HTTPException(404, "Excel ещё не готов")
    return FileResponse(
        path,
        filename=f"mosinform_rating_{job_id}.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
