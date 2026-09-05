
# -*- coding: utf-8 -*-
"""Brand Analytics API (P1): /ba/* — импорт, аккаунты пользователей, темы, реестр."""
from __future__ import annotations
import json, os, re, shutil, subprocess, threading, uuid
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import redis
from ba_import import BE, DATA, THEMES, creds, slug, run_ba_export, register_dataset, load_indexes

router = APIRouter(prefix="/ba", tags=["brand analytics"])
REDIS = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
ARCHIVE_DIR = DATA / "ba_archive"
REGISTRY = ARCHIVE_DIR / "imports.jsonl"
ACCOUNTS_FILE = DATA / "ba_accounts.json"
VENV_PY = BE / "venv_py312_clean" / "bin" / "python3"

def _jid(jid, **fields):
    REDIS.hset(f"ba:job:{jid}", mapping={k: str(v) for k, v in fields.items()})

def _jget(jid):
    return REDIS.hgetall(f"ba:job:{jid}") or {}

def safe_folder(title: str) -> str:
    return re.sub(r"[\\/:*?\"<>|]+", "_", title).strip(" _") or "BA theme"

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

_ACCOUNTS_LOCK = None

def _accounts_lock():
    global _ACCOUNTS_LOCK
    import threading
    if _ACCOUNTS_LOCK is None:
        _ACCOUNTS_LOCK = threading.Lock()
    return _ACCOUNTS_LOCK

def _accounts_db():
    import psycopg2
    from config import DB_HOST, DB_NAME, DB_PASS, DB_PORT, DB_USER
    return psycopg2.connect(host=DB_HOST, port=DB_PORT or 5432, dbname=DB_NAME, user=DB_USER, password=DB_PASS, connect_timeout=5)

def _accounts_init(cur):
    cur.execute("CREATE TABLE IF NOT EXISTS tellscope_ba_accounts (user_id INTEGER PRIMARY KEY, login TEXT NOT NULL DEFAULT '', password TEXT NOT NULL DEFAULT '', updated TEXT)")

def load_accounts() -> dict:
    with _accounts_lock():
        try:
            conn = _accounts_db()
            cur = conn.cursor()
            _accounts_init(cur)
            conn.commit()
            cur.execute("SELECT user_id, login, password, COALESCE(updated, '') FROM tellscope_ba_accounts ORDER BY user_id")
            rows = {str(r[0]): {'login': r[1], 'password': r[2], 'updated': r[3]} for r in cur.fetchall()}
            cur.close()
            conn.close()
            if not rows and ACCOUNTS_FILE.exists():
                try:
                    frows = json.loads(ACCOUNTS_FILE.read_text(encoding='utf-8'))
                    if frows:
                        save_accounts(frows)
                        return frows
                except Exception:
                    pass
            return rows
        except Exception as e:
            print('ba accounts: PG load unavailable, fallback file:', e)
            if ACCOUNTS_FILE.exists():
                try:
                    return json.loads(ACCOUNTS_FILE.read_text(encoding='utf-8'))
                except Exception:
                    return {}
            return {}

def save_accounts(accounts: dict):
    with _accounts_lock():
        ACCOUNTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        ACCOUNTS_FILE.write_text(json.dumps(accounts, ensure_ascii=False), encoding='utf-8')
        try:
            os.chmod(ACCOUNTS_FILE, 0o600)
        except Exception:
            pass
        try:
            conn = _accounts_db()
            cur = conn.cursor()
            _accounts_init(cur)
            conn.commit()
            cur.execute('DELETE FROM tellscope_ba_accounts')
            for uid, acc in accounts.items():
                cur.execute('INSERT INTO tellscope_ba_accounts (user_id, login, password, updated) VALUES (%s, %s, %s, %s)', (int(uid), acc.get('login') or '', acc.get('password') or '', acc.get('updated') or ''))
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            print('ba accounts: PG save unavailable, file only:', e)
def _fernet():
    from cryptography.fernet import Fernet
    key_file = ACCOUNTS_FILE.parent / ".ba_creds_key"
    if key_file.exists():
        key = key_file.read_bytes()
    else:
        key = Fernet.generate_key()
        key_file.write_bytes(key)
        try: os.chmod(key_file, 0o600)
        except Exception: pass
    return Fernet(key)


def _enc_secret(v) -> str:
    if not v:
        return v if v is not None else ""
    return "enc:" + _fernet().encrypt(str(v).encode("utf-8")).decode("ascii")


def _dec_secret(v) -> str:
    if not v:
        return ""
    if isinstance(v, str) and v.startswith("enc:"):
        try:
            return _fernet().decrypt(v[4:].encode("ascii")).decode("utf-8")
        except Exception:
            return ""
    return str(v)


def account_creds(user_id: str) -> dict:
    acc = load_accounts().get(str(user_id), {})
    c = creds()
    return {"BA_LOGIN": _dec_secret(acc.get("login")) or c["BA_LOGIN"],
            "BA_PASS": _dec_secret(acc.get("password")) or c["BA_PASS"]}

def ensure_theme_folders(user_id: str):
    """Создаёт в папках пользователя папки по темам BA, если их ещё нет."""
    folders = {}
    raw = REDIS.hget(str(user_id), "json_files_directory")
    if raw:
        try: folders = json.loads(raw)
        except Exception: folders = {}
    changed = False
    for theme_id, title in THEMES.items():
        fname = safe_folder(title)
        if fname not in folders:
            folders[fname] = []
            changed = True
            (DATA / str(user_id) / "json_files_directory" / fname).mkdir(parents=True, exist_ok=True)
    if changed:
        REDIS.hset(str(user_id), "json_files_directory", json.dumps(folders, ensure_ascii=False))

def _range_includes_today(date_to: str) -> bool:
    """True, если период выгрузки покрывает текущий день (или дата не указана)."""
    if not date_to or not str(date_to).strip():
        return True
    try:
        to = int(float(str(date_to)))
    except Exception:
        return True
    import time as _t
    now = _t.localtime()
    today_start = int(_t.mktime((now.tm_year, now.tm_mon, now.tm_mday, 0, 0, 0, 0, 0, -1)))
    return to >= today_start


class ImportBody(BaseModel):
    theme_id: str
    user_id: str = Field(default=os.environ.get("BA_USER_ID", "1"))
    folder: str = ""
    date_from: str = ""
    date_to: str = ""
    force: bool = False

class AccountBody(BaseModel):
    user_id: str = Field(default=os.environ.get("BA_USER_ID", "1"))
    login: str
    password: str
    create_folders: bool = True

def _index_subprocess(user_id, folder, filename):
    env = dict(os.environ); env["PYTHONPATH"] = str(BE)
    cmd = [str(VENV_PY), str(BE / "ba_import.py"), "index",
           "--user", str(user_id), "--folder", folder, "--file", filename]
    return subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3600)

def _run_import(job_id: str, body: ImportBody):
    folder = body.folder or safe_folder(THEMES.get(body.theme_id, "BA theme"))
    cc = account_creds(body.user_id)
    try:
        _jid(job_id, status="running", message="экспорт из Brand Analytics", progress="10", started=datetime.now().isoformat())
        run_dir = Path("/tmp") / ("ba_run_" + job_id)
        raw = run_ba_export(body.theme_id, run_dir, body.date_from, body.date_to, login=cc["BA_LOGIN"], passw=cc["BA_PASS"])
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        arch = ARCHIVE_DIR / body.theme_id
        arch.mkdir(parents=True, exist_ok=True)
        base_arc = "%s_%s_%s" % (slug(THEMES.get(body.theme_id, body.theme_id)), stamp, raw.name)
        arch_file = arch / base_arc
        if arch_file.suffix.lower() != ".json":
            arch_file = arch / (base_arc + ".json")
        shutil.copyfile(raw, arch_file)

        _jid(job_id, status="running", message="регистрация датасета", progress="40")
        json_filename = "BA_%s_%s.json" % (slug(THEMES.get(body.theme_id, body.theme_id)), stamp)
        indexes = load_indexes()
        nk = max(indexes.keys()) + 1 if indexes else 1
        register_dataset(body.user_id, folder, json_filename, raw, nk)

        _jid(job_id, status="running", message="индексация (Elasticsearch/Qdrant)", progress="55")
        r = _index_subprocess(body.user_id, folder, json_filename)
        tail = (r.stdout or "") + (r.stderr or "")
        if r.returncode != 0:
            raise RuntimeError("индексация не удалась: " + tail[-1500:])
        rec = {
            "job_id": job_id, "theme_id": body.theme_id, "theme": THEMES.get(body.theme_id, body.theme_id),
            "user_id": str(body.user_id), "folder": folder, "file": json_filename,
            "archive": str(arch_file), "date_from": body.date_from, "date_to": body.date_to,
            "bytes": raw.stat().st_size, "index_key": nk,
            "index_name": json_filename.replace(".json", "").lower(),
            "created": datetime.now().isoformat(),
        }
        append_registry(rec)
        shutil.rmtree(run_dir, ignore_errors=True)
        _jid(job_id, status="done", message="готово", progress="100", summary=json.dumps(rec, ensure_ascii=False))
    except Exception as e:
        _jid(job_id, status="error", message=str(e)[:500], progress="0")
    finally:
        shutil.rmtree(Path("/tmp") / ("ba_run_" + job_id), ignore_errors=True)

@router.post("/import")
async def import_data(body: ImportBody):
    if body.theme_id not in THEMES:
        raise HTTPException(400, "Неизвестная тема: %s (допустимые: %s)" % (body.theme_id, ", ".join(THEMES)))
    if not body.force and not _range_includes_today(body.date_to):
        for rec in load_registry():
            if rec.get("theme_id") == body.theme_id and rec.get("user_id") == str(body.user_id) and rec.get("date_from", "") == body.date_from and rec.get("date_to", "") == body.date_to:
                raise HTTPException(409, "Данные за этот период уже выгружены (файл %s). Для уже завершившихся дней повторная загрузка не выполняется; если период включает текущий день, запустите ещё раз — свежие сообщения добавятся." % rec.get("file"))
    job_id = uuid.uuid4().hex[:12]
    _jid(job_id, status="queued", message="поставлено в очередь", progress="0")
    threading.Thread(target=_run_import, args=(job_id, body), daemon=True).start()
    return {"job_id": job_id}

@router.post("/account")
async def save_account(body: AccountBody):
    accounts = load_accounts()
    if not body.login.strip():
        raise HTTPException(400, "Введите логин Brand Analytics")
    accounts[str(body.user_id)] = {"login": _enc_secret(body.login.strip()), "password": _enc_secret(body.password)}
    save_accounts(accounts)
    if body.create_folders:
        ensure_theme_folders(str(body.user_id))
    acc = account_creds(str(body.user_id))
    return {"status": "ok", "configured": bool(acc.get("BA_LOGIN")), "themes": len(THEMES)}

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
def themes(user_id: str = "1"):
    regs = load_registry()
    last = {}
    for r in regs:
        if r.get("user_id") != str(user_id):
            continue
        cur = last.get(r["theme_id"])
        if not cur or r.get("created", "") > cur.get("created", ""):
            last[r["theme_id"]] = r
    acc = load_accounts().get(str(user_id))
    items = [{"theme_id": k, "title": v, "last_import": last.get(k)} for k, v in THEMES.items()]
    return {"themes": items, "account_configured": bool(acc and _dec_secret(acc.get("login")))}

@router.get("/registry")
def registry():
    regs = load_registry()
    regs.reverse()
    return {"imports": regs}
