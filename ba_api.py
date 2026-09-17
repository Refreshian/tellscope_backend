
# -*- coding: utf-8 -*-
"""Brand Analytics API (P1/P2): /ba/* — импорт, персональные аккаунты, темы, реестр.

Персональность: у каждого пользователя Tellscope своё подключение Brand Analytics и свой
снапшот тем. Общий аккаунт тенанта (``.env_ba``) рабочим фолбэком больше НЕ является:
если у пользователя нет своего подключения, эндпоинты возвращают понятную ошибку
(«Подключите свой аккаунт Brand Analytics»), а не работают под чужой учёткой.
"""
from __future__ import annotations
import json, os, re, shutil, subprocess, threading, time, uuid
from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional
from fastapi import APIRouter, Depends, HTTPException
from fastapi_users import FastAPIUsers
from pydantic import BaseModel, Field

from auth.auth import auth_backend
from auth.database import User as AuthUser
from auth.manager import get_user_manager

import redis
from ba_import import (
    BE, DATA, STATUS_ERROR, STATUS_UNVERIFIED, STATUS_VERIFIED,
    account_creds, account_delete, account_status, account_touch, account_put,
    load_themes, save_themes, themes_cache_reset, fetch_ba_themes,
    slug, run_ba_export, register_dataset, load_indexes,
)

router = APIRouter(prefix="/ba", tags=["brand analytics"])

# Текст-подсказка для всех операций BA без своего подключения (единый для API и интерфейса).
NO_ACCOUNT_HINT = "Аккаунт Brand Analytics не подключён: подключите свой аккаунт (логин и пароль BA) в разделе «Наборы данных»"

# Свой экземпляр зависимостей fastapi-users: у модуля собственный APIRouter,
# поэтому «кто зовёт» нужно получать здесь, а не в main.py.
_ba_users = FastAPIUsers[AuthUser, int](get_user_manager, [auth_backend])
require_user = _ba_users.current_user()

# Как читать «кто зовёт» в эндпоинтах BA.
#
# Зависимость объявлена через Annotated, а НЕ как значение по умолчанию ``= Depends(...)``.
# Это принципиально: модуль вызывают не только через HTTP, но и напрямую из кода
# (agent_engine/tools_ba.py зовёт ``ba_api.themes(uid, refresh=0)`` и ``ba_api.job_status(jid)``).
# При ``= Depends(...)`` прямой вызов получал бы в аргументе объект Depends, владелец
# определялся бы как «пустой id» и внутренний вызов падал с 403 — агентский инструмент
# выгрузки темы из BA при этом молча не работал. С Annotated HTTP-запросы так же требуют
# авторизацию, а прямой вызов получает user=None («внутренний вызов»).
CurrentUser = Annotated[Optional[AuthUser], Depends(require_user)]


def _owner_for(requested, user) -> str:
    """Владелец данных Brand Analytics: всегда свой id.

    Чужой ``user_id`` принимается только от суперпользователя (админские операции).
    Раньше ``user_id`` приходил от клиента без проверки: любой новый пользователь мог
    прочитать темы и реестр импортов другого пользователя и перезаписать его креды BA.

    ``user is None`` — внутренний вызов (агент Tellscope, tools_ba.py): владелец уже
    определён вызывающим кодом, поэтому берём переданный ``user_id`` как есть.
    """
    want = str(requested or "").strip()
    if user is None:
        if not want:
            raise HTTPException(400, "Не указан user_id")
        return want
    me = str(getattr(user, "id", "") or "")
    if getattr(user, "is_superuser", False):
        return want or me
    if not want or want == me:
        return me
    raise HTTPException(status_code=403, detail="Нет доступа к данным Brand Analytics другого пользователя")

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


def require_account(user_id: str) -> dict:
    """Свои креды BA пользователя или понятная ошибка (без похода под общий .env_ba)."""
    cc = account_creds(user_id)
    if not cc.get("configured"):
        raise HTTPException(status_code=400, detail=NO_ACCOUNT_HINT)
    return cc


def user_themes(user_id: str) -> dict:
    """Темы конкретного пользователя (пер-пользовательский снапшот)."""
    return load_themes(str(user_id))


# ---------------------------------------------------------------------------
# Автозагрузка тем: подключение есть, а снапшота тем ещё нет
# ---------------------------------------------------------------------------
# Раньше GET /ba/themes в таком случае отдавал пустой список без объяснения: пользователь
# с настроенным подключением видел «тем нет» и не понимал, что делать (снапшот может
# отсутствовать у того, кто подключился до появления пер-пользовательских снапшотов, или
# после отключения/повторного подключения аккаунта). Теперь темы подтягиваются сами:
# запрос в BA идёт в фоне (~40 с), эндпоинт сразу отдаёт состояние themes_loading=true,
# а интерфейс опрашивает его и показывает «загружаю темы».
#
# Ограничение частоты (THEMES_AUTOFETCH_COOLDOWN) не даёт долбить BA: после неудачной
# попытки автозагрузка для этого пользователя молчит кулдаун, дальше — только «Обновить».
# Параллельные запросы не плодят входы в BA: на пользователя одновременно одна попытка.
_THEMES_FETCH_LOCK = threading.Lock()
_THEMES_FETCH = {}  # uid -> {"status": "loading"|"done"|"error", "started": float, "finished": float, "error": str}
THEMES_AUTOFETCH_COOLDOWN = 120.0


def themes_fetch_state(user_id: str) -> dict:
    """Состояние автозагрузки тем пользователя (без запуска новых попыток)."""
    with _THEMES_FETCH_LOCK:
        return dict(_THEMES_FETCH.get(str(user_id)) or {})


def _themes_autofetch_worker(uid: str):
    err = ""
    try:
        cc = account_creds(uid)
        if not cc.get("configured"):
            err = NO_ACCOUNT_HINT
        else:
            themes = fetch_ba_themes(login=cc["BA_LOGIN"], passw=cc["BA_PASS"], user_id=uid)
            account_touch(uid, status=STATUS_VERIFIED, error="", themes_count=len(themes))
    except Exception as exc:
        err = str(exc)[:400]
        try:
            account_touch(uid, status=STATUS_ERROR, error=err)
        except Exception:
            pass
    with _THEMES_FETCH_LOCK:
        prev = dict(_THEMES_FETCH.get(uid) or {})
        prev.update({"status": "error" if err else "done", "finished": time.time(), "error": err})
        _THEMES_FETCH[uid] = prev


def start_themes_autofetch(user_id: str) -> dict:
    """Запускает фоновую загрузку тем, если её ещё нет и не истёк кулдаун.

    Возвращает текущее состояние: ``loading`` — уже загружаем, ``done``/``error`` — результат
    последней попытки, ``{}`` — попытка не запускалась (снапшот уже есть).
    """
    uid = str(user_id)
    now = time.time()
    with _THEMES_FETCH_LOCK:
        st = dict(_THEMES_FETCH.get(uid) or {})
        if st.get("status") == "loading":
            return st
        if st.get("started") and (now - float(st["started"])) < THEMES_AUTOFETCH_COOLDOWN:
            return st
        st = {"status": "loading", "started": now, "finished": 0.0, "error": ""}
        _THEMES_FETCH[uid] = st
    threading.Thread(target=_themes_autofetch_worker, args=(uid,), daemon=True).start()
    return dict(st)


def ensure_theme_folders(user_id: str):
    """Создаёт в папках пользователя папки по ЕГО темам BA, если их ещё нет."""
    folders = {}
    raw = REDIS.hget(str(user_id), "json_files_directory")
    if raw:
        try: folders = json.loads(raw)
        except Exception: folders = {}
    changed = False
    for theme_id, title in user_themes(user_id).items():
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
    user_id: str = Field(default="", description="id владельца; пусто = текущий пользователь")
    folder: str = ""
    date_from: str = ""
    date_to: str = ""
    force: bool = False

class AccountBody(BaseModel):
    user_id: str = Field(default="", description="id владельца; пусто = текущий пользователь")
    login: str
    password: str
    create_folders: bool = True

def _index_subprocess(user_id, folder, filename):
    env = dict(os.environ); env["PYTHONPATH"] = str(BE)
    cmd = [str(VENV_PY), str(BE / "ba_import.py"), "index",
           "--user", str(user_id), "--folder", folder, "--file", filename]
    return subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3600)

def _run_import(job_id: str, body: ImportBody):
    uid = str(body.user_id)
    themes = user_themes(uid)
    theme_title = themes.get(body.theme_id, body.theme_id)
    folder = body.folder or safe_folder(theme_title)
    cc = account_creds(uid)
    if not cc.get("configured"):
        _jid(job_id, status="error", message=NO_ACCOUNT_HINT, progress="0", owner=uid)
        return
    try:
        _jid(job_id, status="running", message="Экспорт данных", progress="10", started=datetime.now().isoformat(), owner=uid)
        run_dir = Path("/tmp") / ("ba_run_" + job_id)
        raw = run_ba_export(body.theme_id, run_dir, body.date_from, body.date_to,
                            login=cc["BA_LOGIN"], passw=cc["BA_PASS"], user_id=uid)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        arch = ARCHIVE_DIR / body.theme_id
        arch.mkdir(parents=True, exist_ok=True)
        base_arc = "%s_%s_%s" % (slug(theme_title), stamp, raw.name)
        arch_file = arch / base_arc
        if arch_file.suffix.lower() != ".json":
            arch_file = arch / (base_arc + ".json")
        shutil.copyfile(raw, arch_file)

        _jid(job_id, status="running", message="Сохранение данных", progress="40")
        json_filename = "BA_%s_%s.json" % (slug(theme_title), stamp)
        indexes = load_indexes()
        nk = max(indexes.keys()) + 1 if indexes else 1
        register_dataset(uid, folder, json_filename, raw, nk)

        _jid(job_id, status="running", message="Обработка данных", progress="55")
        r = _index_subprocess(uid, folder, json_filename)
        tail = (r.stdout or "") + (r.stderr or "")
        if r.returncode != 0:
            raise RuntimeError("индексация не удалась: " + tail[-1500:])
        rec = {
            "job_id": job_id, "theme_id": body.theme_id, "theme": theme_title,
            "user_id": uid, "folder": folder, "file": json_filename,
            "archive": str(arch_file), "date_from": body.date_from, "date_to": body.date_to,
            "bytes": raw.stat().st_size, "index_key": nk,
            "index_name": json_filename.replace(".json", "").lower(),
            "created": datetime.now().isoformat(),
        }
        append_registry(rec)
        shutil.rmtree(run_dir, ignore_errors=True)
        _jid(job_id, status="done", message="Готово", progress="100", summary=json.dumps(rec, ensure_ascii=False))
    except Exception as e:
        err = str(e)[:500]
        # Проблема с входом в BA — помечаем подключение как ошибочное, чтобы это было видно в UI.
        if "не принял логин" in err or "BA_AUTH" in err or "cookies" in err.lower():
            try:
                account_touch(uid, status=STATUS_ERROR, error=err)
            except Exception:
                pass
        _jid(job_id, status="error", message=err, progress="0")
    finally:
        shutil.rmtree(Path("/tmp") / ("ba_run_" + job_id), ignore_errors=True)

@router.post("/import")
def import_data(body: ImportBody, user: CurrentUser = None):
    body.user_id = _owner_for(body.user_id, user)
    require_account(body.user_id)
    themes = user_themes(body.user_id)
    if not themes:
        raise HTTPException(400, "Список тем Brand Analytics пуст: нажмите «Обновить» в блоке Brand Analytics — темы загрузятся из вашего аккаунта")
    if body.theme_id not in themes:
        raise HTTPException(400, "Неизвестная тема: %s (допустимые: %s)" % (body.theme_id, ", ".join(themes)))
    if not body.force and not _range_includes_today(body.date_to):
        for rec in load_registry():
            if rec.get("theme_id") == body.theme_id and rec.get("user_id") == str(body.user_id) and rec.get("date_from", "") == body.date_from and rec.get("date_to", "") == body.date_to:
                raise HTTPException(409, "Данные за этот период уже выгружены (файл %s). Для уже завершившихся дней повторная загрузка не выполняется; если период включает текущий день, запустите ещё раз — свежие сообщения добавятся." % rec.get("file"))
    job_id = uuid.uuid4().hex[:12]
    _jid(job_id, status="queued", message="Экспорт данных", progress="0", owner=str(body.user_id))
    threading.Thread(target=_run_import, args=(job_id, body), daemon=True).start()
    return {"job_id": job_id}

@router.get("/account")
def get_account(user_id: str = "", user: CurrentUser = None):
    """Статус своего подключения BA (логин маскируется, пароль не отдаётся никогда)."""
    uid = _owner_for(user_id, user)
    st = account_status(uid)
    st["themes_count_snapshot"] = len(user_themes(uid))
    st["hint"] = "" if st["configured"] else NO_ACCOUNT_HINT
    return st

@router.post("/account")
def save_account(body: AccountBody, user: CurrentUser = None):
    """Проверяет логин/пароль реальным входом в BA и только потом сохраняет подключение.

    Неверные креды не сохраняются: пользователь получает текст ошибки от Brand Analytics.
    (Эндпоинт синхронный: проверка занимает до ~1 минуты — FastAPI выполняет его в threadpool
    и не блокирует event loop.)
    """
    uid = _owner_for(body.user_id, user)
    login = (body.login or "").strip()
    if not login:
        raise HTTPException(400, "Введите логин Brand Analytics")
    if not (body.password or "").strip():
        raise HTTPException(400, "Введите пароль Brand Analytics")
    try:
        themes = fetch_ba_themes(login=login, passw=body.password, user_id=uid)
    except Exception as exc:
        err = str(exc)[:400]
        raise HTTPException(400, "Подключение не сохранено. " + err)
    account_put(uid, login, body.password, status=STATUS_VERIFIED, error="", themes_count=len(themes))
    if body.create_folders:
        ensure_theme_folders(uid)
    st = account_status(uid)
    return {"status": "ok", "verified": True, "themes_count": len(themes), "account": st,
            "message": "Аккаунт проверен и сохранён: доступно тем %d" % len(themes)}

@router.post("/account/disconnect")
def disconnect_account(user_id: str = "", user: CurrentUser = None):
    """Отключает аккаунт BA пользователя: удаляет его запись и его снапшот тем."""
    uid = _owner_for(user_id, user)
    existed = account_delete(uid)
    themes_cache_reset(uid)
    try:
        path = DATA / "ba_themes" / ("%s.json" % uid)
        if path.exists():
            path.unlink()
    except Exception:
        pass
    try:
        ck = DATA / "ba_cookies" / ("u%s.json" % uid)
        if ck.exists():
            ck.unlink()
    except Exception:
        pass
    return {"status": "ok", "disconnected": bool(existed), "account_configured": False,
            "hint": NO_ACCOUNT_HINT}

@router.get("/jobs/{job_id}")
def job_status(job_id: str, user: CurrentUser = None):
    data = _jget(job_id)
    if not data:
        raise HTTPException(404, "Задача не найдена")
    owner = str(data.get("owner") or "")
    if user is not None and owner and owner != str(getattr(user, "id", "")) and not getattr(user, "is_superuser", False):
        raise HTTPException(status_code=403, detail="Нет доступа к задаче другого пользователя")
    out = {"job_id": job_id, "status": data.get("status"), "message": data.get("message"),
           "progress": data.get("progress"), "summary": None}
    if data.get("summary"):
        try: out["summary"] = json.loads(data["summary"])
        except Exception: pass
    return out

@router.get("/themes")
def themes(user_id: str = "", refresh: int = 0, user: CurrentUser = None):
    """Темы СВОЕГО аккаунта BA (пер-пользовательский снапшот) + статус подключения.

    Если подключение настроено, а снапшота тем нет, темы подтягиваются автоматически в
    фоне: ответ приходит сразу с ``themes_loading=true`` и понятным hint, интерфейс
    показывает «загружаю темы» и опрашивает эндпоинт, пока снапшот не появится.
    Явный ``refresh=1`` («Обновить» в интерфейсе) работает как раньше и ждёт результат.
    """
    uid = _owner_for(user_id, user)
    st = account_status(uid)
    snapshot = user_themes(uid)
    if not st["configured"]:
        return {"themes": [], "account_configured": False, "account_status": "none",
                "login_masked": "", "account_error": "", "verified_at": "",
                "themes_loading": False, "hint": NO_ACCOUNT_HINT}
    refresh_error = ""
    if refresh:
        cc = account_creds(uid)
        try:
            snapshot = fetch_ba_themes(login=cc["BA_LOGIN"], passw=cc["BA_PASS"], user_id=uid)
            account_touch(uid, status=STATUS_VERIFIED, error="", themes_count=len(snapshot))
        except Exception as exc:
            refresh_error = str(exc)[:400]
            account_touch(uid, status=STATUS_ERROR, error=refresh_error)
        st = account_status(uid)
    auto = {}
    if not refresh and not snapshot:
        auto = start_themes_autofetch(uid)
        if auto.get("status") == "done":
            snapshot = user_themes(uid)
            st = account_status(uid)
    regs = load_registry()
    last = {}
    for r in regs:
        if r.get("user_id") != str(uid):
            continue
        cur = last.get(r["theme_id"])
        if not cur or r.get("created", "") > cur.get("created", ""):
            last[r["theme_id"]] = r
    items = [{"theme_id": k, "title": v, "last_import": last.get(k)} for k, v in snapshot.items()]
    loading = bool(auto.get("status") == "loading")
    out = {"themes": items, "account_configured": True, "account_status": st["status"],
           "login_masked": st["login_masked"], "account_error": st["error"],
           "verified_at": st["verified_at"], "themes_count": len(items),
           "themes_loading": loading,
           "themes_fetch_status": (auto.get("status") or ("cached" if items else ""))}
    if not items:
        if loading:
            out["hint"] = ("Загружаю список тем из вашего аккаунта Brand Analytics — "
                           "это занимает до минуты, список появится здесь автоматически")
        elif auto.get("error"):
            out["hint"] = ("Не удалось загрузить темы автоматически: %s — нажмите «Обновить»"
                           % auto["error"])
        else:
            out["hint"] = ("В вашем аккаунте Brand Analytics темы не найдены — проверьте "
                           "логин/пароль и нажмите «Обновить»")
    if refresh_error:
        out["refresh_error"] = refresh_error
    return out

@router.get("/registry")
def registry(user: CurrentUser = None):
    regs = load_registry()
    if user is not None and not getattr(user, "is_superuser", False):
        me = str(getattr(user, "id", ""))
        regs = [r for r in regs if str(r.get("user_id") or "") == me]
    regs.reverse()
    return {"imports": regs}
