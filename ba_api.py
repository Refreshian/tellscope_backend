
# -*- coding: utf-8 -*-
"""Brand Analytics import API (P1): /ba/*"""
from __future__ import annotations
import json, os, re, shutil, subprocess, threading, uuid
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import redis
from ba_import import (BE, DATA, THEMES, creds, slug, run_ba_export,
                       register_dataset, load_indexes)

router = APIRouter(prefix="/ba", tags=["brand analytics"])
REDIS = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
ARCHIVE_DIR = DATA / "ba_archive"
REGISTRY = ARCHIVE_DIR / "imports.jsonl"
VENV_PY = BE / "venv_py312_clean" / "bin" / "python3"

def _jid(key, **fields):
    REDIS.hset(key, mapping={k: str(v) for k, v in fields.items()})

def _jget(jid):
    return REDIS.hgetall(f"ba:job:{jid}") or {}

def load_registry():
    out = []
    if REGISTRY.exists():
        for line in REGISTRY.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try: out.append(json.loads(line))
                except Exception: pass
    return out

def append_registry(rec):
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    with REGISTRY.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

class ImportBody(BaseModel):
    theme_id: str
    user_id: str = Field(default=os.environ.get("BA_USER_ID", "1"))
    folder: str = ""
    date_from: str = ""
    date_to: str = ""
    force: bool = False

def _index_subprocess(user_id, folder, filename, task_id, jid):
    """Запускает индексацию в отдельном процессе (как add-file/Celery, но надёжно)."""
    env = dict(os.environ); env["PYTHONPATH"] = str(BE)
    cmd = [str(VENV_PY), str(BE / "ba_import.py"), "index",
           "--user", str(user_id), "--folder", folder, "--file", filename]
    return subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3600)

def _run_import(job_id: str, body: ImportBody):
    jid = f"ba:job:{job_id}"
    theme_id = body.theme_id
    folder = body.folder or THEMES.get(theme_id, "BA_theme_" + theme_id)
    try:
        _jid(jid, status="running", message="экспорт из Brand Analytics", progress="10", started=datetime.now().isoformat())
        run_dir = Path("/tmp") / ("ba_run_" + job_id)
        raw = run_ba_export(theme_id, run_dir, body.date_from, body.date_to)
        # архив
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        arch = ARCHIVE_DIR / theme_id
        arch.mkdir(parents=True, exist_ok=True)
        arch_file = arch / ("%s_%s_%s.json" % (slug(THEMES.get(theme_id, theme_id)), stamp, raw.name.split("_")[-1]))
        shutil.copyfile(raw, arch_file)

        _jid(jid, status="running", message="регистрация датасета", progress="40")
        json_filename = "BA_%s_%s.json" % (slug(THEMES.get(theme_id, theme_id)), stamp)
        indexes = load_indexes()
        nk = max(indexes.keys()) + 1 if indexes else 1
        folder_dir, nk = register_dataset(body.user_id, folder, json_filename, raw, nk)

        _jid(jid, status="running", message="индексация (Elasticsearch/Qdrant)", progress="55")
        r = _index_subprocess(body.user_id, folder, json_filename, job_id, jid)
        tail = (r.stdout or "") + (r.stderr or "")
        if r.returncode != 0:
            raise RuntimeError("индексация не удалась: " + tail[-1500:])
        last = tail.strip().splitlines()[-1] if tail.strip() else ""
        rec = {
            "job_id": job_id, "theme_id": theme_id, "theme": THEMES.get(theme_id, theme_id),
            "user_id": str(body.user_id), "folder": folder, "file": json_filename,
            "archive": str(arch_file), "date_from": body.date_from, "date_to": body.date_to,
            "bytes": raw.stat().st_size, "index_key": nk,
            "index_name": json_filename.replace(".json", "").lower(),
            "created": datetime.now().isoformat(),
        }
        append_registry(rec)
        shutil.rmtree(run_dir, ignore_errors=True)
        _jid(jid, status="done", message="готово", progress="100", summary=json.dumps(rec, ensure_ascii=False))
    except Exception as e:
        _jid(jid, status="error", message=str(e)[:500], progress="0")
    finally:
        shutil.rmtree(Path("/tmp") / ("ba_run_" + job_id), ignore_errors=True)

@router.post("/import")
async def import_data(body: ImportBody):
    if body.theme_id not in THEMES:
        raise HTTPException(400, "Неизвестная тема: %s (допустимые: %s)" % (body.theme_id, ", ".join(THEMES)))
    if not body.force:
        for rec in load_registry():
            if rec.get("theme_id") == body.theme_id and rec.get("date_from", "") == body.date_from and rec.get("date_to", "") == body.date_to:
                raise HTTPException(409, "Такая выгрузка уже есть (файл %s). Повторите с force=true" % rec.get("file"))
    job_id = uuid.uuid4().hex[:12]
    _jid(f"ba:job:{job_id}", status="queued", message="поставлено в очередь", progress="0")
    threading.Thread(target=_run_import, args=(job_id, body), daemon=True).start()
    return {"job_id": job_id}

@router.get("/jobs/{job_id}")
def job_status(job_id: str):
    data = _jget(job_id)
    if not data:
        raise HTTPException(404, "Задача не найдена")
    out = {"job_id": job_id, "status": data.get("status"), "message": data.get("message"),
           "progress": data.get("progress"), "summary": None}
    if data.get("summary"):
        try: out["summary"] = json.loads(data["summary"])
        except Exception: pass
    return out

@router.get("/themes")
def themes():
    regs = load_registry()
    last = {}
    for r in regs:
        cur = last.get(r["theme_id"])
        if not cur or r.get("created", "") > cur.get("created", ""):
            last[r["theme_id"]] = r
    items = [{"theme_id": k, "title": v, "last_import": last.get(k)} for k, v in THEMES.items()]
    return {"themes": items}

@router.get("/registry")
def registry():
    regs = load_registry()
    regs.reverse()
    return {"imports": regs}
