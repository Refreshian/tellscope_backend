
# -*- coding: utf-8 -*-
"""
BA -> Tellscope manual import (P1).
Экспорт JSON из Brand Analytics, размещение в датасете и (опционально) индексация.

Персональность (P2): у каждого пользователя Tellscope своё подключение Brand Analytics
и свой список тем:

  * креды — таблица PG ``tellscope_ba_accounts`` (user_id, login, password, updated, status,
    error, verified_at, themes_count) с файловым фолбэком ``data/ba_accounts.json``;
    пароль шифруется Fernet-ключом ``data/.ba_creds_key``;
  * темы — пер-пользовательский снапшот ``data/ba_themes/<user_id>.json`` и кэш в памяти по uid;
  * ``data/ba_themes.json`` и ``.env_ba`` оставлены только как источник одноразовой
    миграции/сида (``migrate_legacy_snapshot``) — рабочим фолбэком они больше не являются.

Запуск (на сервере, от dev):
  NODE_PATH=/tmp/tshot/node_modules python3 ba_import.py export --theme 14075092 \
      --user 1 --folder "BA ORVI" [--no-index]
"""
from __future__ import annotations
import argparse, hashlib, json, logging, os, pickle, re, shutil, subprocess, sys, threading, time, uuid
from datetime import datetime
from pathlib import Path
import redis

log = logging.getLogger("ba_import")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

BE = Path(__file__).resolve().parent
DATA = BE / "data"
INDEXES_PKL = DATA / "indexes.pkl"
REDIS = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

BA_DEFAULT_PERIOD = ("1788037200", "1788641999")  # последние дни, как в UI по умолчанию


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-zА-Яа-я0-9]+", "_", text).strip("_")[:60]


# Дефолтный набор тем из кода: используется только как сид, если у пользователя ещё нет
# собственного снапшота и нет файла миграции. Рабочим источником тем не является.
DEFAULT_THEMES = {
    "12394607": "Мониторинг тем",
    "12466084": "Морская вода, Аллергия",
    "14075092": "Признаки ОРВИ",
    "12505577": "Риномарис",
    "13947576": "Строительство / Реконструкция",
    "14166164": "Энергострой",
}

# ---------------------------------------------------------------------------
# Аккаунты Brand Analytics (пер-пользовательские)
# ---------------------------------------------------------------------------
ACCOUNTS_FILE = DATA / "ba_accounts.json"
COOKIES_DIR = DATA / "ba_cookies"
ACCOUNT_COLUMNS = ("user_id", "login", "password", "updated", "status", "error", "verified_at", "themes_count")

STATUS_VERIFIED = "verified"
STATUS_UNVERIFIED = "unverified"
STATUS_ERROR = "error"

_ACCOUNTS_LOCK = threading.Lock()


def _accounts_lock():
    return _ACCOUNTS_LOCK


def _accounts_db():
    import psycopg2
    from config import DB_HOST, DB_NAME, DB_PASS, DB_PORT, DB_USER
    return psycopg2.connect(host=DB_HOST, port=DB_PORT or 5432, dbname=DB_NAME, user=DB_USER,
                            password=DB_PASS, connect_timeout=5)


def _accounts_init(cur):
    cur.execute("CREATE TABLE IF NOT EXISTS tellscope_ba_accounts (user_id INTEGER PRIMARY KEY, "
                "login TEXT NOT NULL DEFAULT '', password TEXT NOT NULL DEFAULT '', updated TEXT)")
    # Колонки статуса проверки подключения добавлены позже (P2) — досоздаём на месте.
    cur.execute("ALTER TABLE tellscope_ba_accounts ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT ''")
    cur.execute("ALTER TABLE tellscope_ba_accounts ADD COLUMN IF NOT EXISTS error TEXT NOT NULL DEFAULT ''")
    cur.execute("ALTER TABLE tellscope_ba_accounts ADD COLUMN IF NOT EXISTS verified_at TEXT NOT NULL DEFAULT ''")
    cur.execute("ALTER TABLE tellscope_ba_accounts ADD COLUMN IF NOT EXISTS themes_count INTEGER NOT NULL DEFAULT 0")


def _norm_record(rec) -> dict:
    rec = rec or {}
    out = {
        "login": rec.get("login") or "",
        "password": rec.get("password") or "",
        "updated": rec.get("updated") or "",
        "status": rec.get("status") or "",
        "error": rec.get("error") or "",
        "verified_at": rec.get("verified_at") or "",
    }
    try:
        out["themes_count"] = int(rec.get("themes_count") or 0)
    except Exception:
        out["themes_count"] = 0
    return out


def accounts_load() -> dict:
    """Все подключения BA: {user_id: record}. PG, с файловым фолбэком."""
    with _accounts_lock():
        try:
            conn = _accounts_db()
            cur = conn.cursor()
            _accounts_init(cur)
            conn.commit()
            cur.execute("SELECT user_id, login, password, COALESCE(updated, ''), COALESCE(status, ''), "
                        "COALESCE(error, ''), COALESCE(verified_at, ''), COALESCE(themes_count, 0) "
                        "FROM tellscope_ba_accounts ORDER BY user_id")
            rows = {str(r[0]): _norm_record({"login": r[1], "password": r[2], "updated": r[3],
                                             "status": r[4], "error": r[5], "verified_at": r[6],
                                             "themes_count": r[7]}) for r in cur.fetchall()}
            cur.close()
            conn.close()
            if not rows and ACCOUNTS_FILE.exists():
                try:
                    frows = json.loads(ACCOUNTS_FILE.read_text(encoding='utf-8'))
                    if frows:
                        accounts_save(frows)
                        return {str(k): _norm_record(v) for k, v in frows.items()}
                except Exception:
                    pass
            return rows
        except Exception as e:
            print('ba accounts: PG load unavailable, fallback file:', e)
            if ACCOUNTS_FILE.exists():
                try:
                    return {str(k): _norm_record(v) for k, v in json.loads(ACCOUNTS_FILE.read_text(encoding='utf-8')).items()}
                except Exception:
                    return {}
            return {}


def accounts_save(accounts: dict):
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
                acc = _norm_record(acc)
                cur.execute('INSERT INTO tellscope_ba_accounts (user_id, login, password, updated, status, error, verified_at, themes_count) '
                            'VALUES (%s, %s, %s, %s, %s, %s, %s, %s)',
                            (int(uid), acc['login'], acc['password'], acc['updated'], acc['status'],
                             acc['error'], acc['verified_at'], acc['themes_count']))
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            print('ba accounts: PG save unavailable, file only:', e)


def _fernet():
    from cryptography.fernet import Fernet
    key_file = DATA / ".ba_creds_key"
    if key_file.exists():
        key = key_file.read_bytes()
    else:
        key = Fernet.generate_key()
        key_file.write_bytes(key)
        try:
            os.chmod(key_file, 0o600)
        except Exception:
            pass
    return Fernet(key)


def enc_secret(v) -> str:
    if not v:
        return v if v is not None else ""
    return "enc:" + _fernet().encrypt(str(v).encode("utf-8")).decode("ascii")


def dec_secret(v) -> str:
    if not v:
        return ""
    if isinstance(v, str) and v.startswith("enc:"):
        try:
            return _fernet().decrypt(v[4:].encode("ascii")).decode("utf-8")
        except Exception:
            return ""
    return str(v)


def mask_login(login: str) -> str:
    """Маскировка логина для интерфейса: ale***@list.ru."""
    login = str(login or "").strip()
    if not login:
        return ""
    if "@" in login:
        name, _, domain = login.partition("@")
        head = name[:3] if len(name) > 3 else name[:1]
        return "%s***@%s" % (head, domain)
    return login[:3] + "***" if len(login) > 3 else login[:1] + "***"


def account_get(user_id) -> dict:
    """Запись подключения BA конкретного пользователя (без фолбэка на общий аккаунт)."""
    return accounts_load().get(str(user_id)) or {}


def account_put(user_id, login: str, password: str, status: str = STATUS_VERIFIED,
                error: str = "", themes_count: int = 0, verified_at: str = "") -> dict:
    """Сохраняет подключение пользователя (логин/пароль шифруются)."""
    accounts = accounts_load()
    rec = {
        "login": enc_secret(str(login).strip()),
        "password": enc_secret(password),
        "updated": datetime.now().isoformat(),
        "status": status,
        "error": error or "",
        "verified_at": verified_at or (datetime.now().isoformat() if status == STATUS_VERIFIED else ""),
        "themes_count": int(themes_count or 0),
    }
    accounts[str(user_id)] = rec
    accounts_save(accounts)
    return rec


def account_mark_error(user_id, error: str) -> dict:
    """Помечает подключение как непроверенное/ошибочное, не трогая сохранённые креды."""
    accounts = accounts_load()
    rec = accounts.get(str(user_id))
    if not rec:
        return {}
    rec["status"] = STATUS_ERROR
    rec["error"] = str(error or "")[:500]
    rec["updated"] = datetime.now().isoformat()
    accounts[str(user_id)] = rec
    accounts_save(accounts)
    return rec


def account_touch(user_id, status: str = None, error: str = None, themes_count: int = None) -> dict:
    """Обновляет статус проверки подключения, НЕ меняя сохранённые логин/пароль."""
    accounts = accounts_load()
    rec = accounts.get(str(user_id))
    if not rec:
        return {}
    if status is not None:
        rec["status"] = status
        if status == STATUS_VERIFIED:
            rec["verified_at"] = datetime.now().isoformat()
    if error is not None:
        rec["error"] = str(error or "")[:500]
    if themes_count is not None:
        rec["themes_count"] = int(themes_count)
    accounts[str(user_id)] = rec
    accounts_save(accounts)
    return rec


def account_status(user_id) -> dict:
    """Публичный статус подключения пользователя (без логина/пароля) для API и интерфейса."""
    rec = account_get(user_id)
    login = dec_secret(rec.get("login")) if rec else ""
    passw = dec_secret(rec.get("password")) if rec else ""
    configured = bool(login and passw)
    status = "none"
    if configured:
        status = rec.get("status") or STATUS_UNVERIFIED
        if status not in (STATUS_VERIFIED, STATUS_UNVERIFIED, STATUS_ERROR):
            status = STATUS_UNVERIFIED
    return {
        "configured": configured,
        "login_masked": mask_login(login),
        "status": status,
        "error": (rec.get("error") or "") if rec else "",
        "verified_at": (rec.get("verified_at") or "") if rec else "",
        "updated": (rec.get("updated") or "") if rec else "",
        "themes_count": int(rec.get("themes_count") or 0) if rec else 0,
    }


def account_delete(user_id) -> bool:
    accounts = accounts_load()
    existed = str(user_id) in accounts
    if existed:
        accounts.pop(str(user_id), None)
        accounts_save(accounts)
    return existed


def account_creds(user_id) -> dict:
    """Свои креды BA пользователя. Фолбэка на общий .env_ba здесь нет.

    ``configured=False`` означает «подключите свой аккаунт Brand Analytics» — вызывающий код
    обязан вернуть пользователю понятную ошибку, а не работать под чужой учёткой.
    """
    rec = account_get(user_id)
    login = dec_secret(rec.get("login"))
    passw = dec_secret(rec.get("password"))
    return {
        "BA_LOGIN": login, "BA_PASS": passw,
        "login": login, "login_masked": mask_login(login),
        "configured": bool(login and passw),
        "status": rec.get("status") or (STATUS_UNVERIFIED if login else ""),
        "error": rec.get("error") or "",
        "verified_at": rec.get("verified_at") or "",
        "themes_count": int(rec.get("themes_count") or 0),
        "updated": rec.get("updated") or "",
    }


def legacy_creds() -> dict:
    """Общие креды из .env_ba: ТОЛЬКО для одноразовой миграции/сида, не рабочий фолбэк."""
    env_file = BE / ".env_ba"
    out = {"BA_LOGIN": os.environ.get("BA_LOGIN", ""), "BA_PASS": os.environ.get("BA_PASS", "")}
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    return out


# Обратная совместимость: старое имя функции. Использовать не в рабочем пути.
def creds() -> dict:
    return legacy_creds()


def cookies_path(user_id=None, login: str = "") -> Path:
    """Пер-пользовательский файл cookies BA: сессии разных аккаунтов не смешиваются."""
    COOKIES_DIR.mkdir(parents=True, exist_ok=True)
    if user_id:
        return COOKIES_DIR / ("u%s.json" % re.sub(r"[^0-9A-Za-z_-]", "", str(user_id)))
    key = hashlib.sha1(str(login or "anon").encode("utf-8")).hexdigest()[:16]
    return COOKIES_DIR / ("login_%s.json" % key)


# ---------------------------------------------------------------------------
# Темы Brand Analytics: пер-пользовательские снапшоты
# ---------------------------------------------------------------------------
THEMES_DIR = DATA / "ba_themes"
LEGACY_SNAPSHOT = DATA / "ba_themes.json"   # старый общий снапшот: только источник миграции

_THEMES_LOCK = threading.RLock()
_THEMES_CACHE = {}  # uid -> {"mtime": float, "themes": {...}}


def _uid(user_id) -> str:
    return str(user_id or "").strip()


def themes_path(user_id) -> Path:
    return THEMES_DIR / ("%s.json" % (_uid(user_id) or "unknown"))


def _read_json_dict(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items() if str(k).strip() and str(v).strip()}


def load_themes(user_id, use_cache: bool = True) -> dict:
    """Темы конкретного пользователя. Пустой словарь — «снапшота нет».

    Общий ``data/ba_themes.json`` здесь НЕ читается: иначе новый пользователь увидел бы
    темы владельца общего аккаунта. Единственный путь получить общий снапшот — миграция
    (``migrate_legacy_snapshot``).
    """
    uid = _uid(user_id)
    if not uid:
        return {}
    path = themes_path(uid)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        with _THEMES_LOCK:
            _THEMES_CACHE.pop(uid, None)
        return {}
    with _THEMES_LOCK:
        cached = _THEMES_CACHE.get(uid)
        if use_cache and cached and cached.get("mtime") == mtime:
            return dict(cached["themes"])
    themes = _read_json_dict(path)
    with _THEMES_LOCK:
        _THEMES_CACHE[uid] = {"mtime": mtime, "themes": dict(themes)}
    return themes


def save_themes(user_id, themes: dict) -> dict:
    """Сохраняет снапшот тем ОДНОГО пользователя (других не трогает)."""
    uid = _uid(user_id)
    if not uid:
        raise ValueError("save_themes: не указан user_id")
    clean = {str(k): str(v) for k, v in (themes or {}).items() if str(k).strip() and str(v).strip()}
    THEMES_DIR.mkdir(parents=True, exist_ok=True)
    path = themes_path(uid)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    with _THEMES_LOCK:
        _THEMES_CACHE[uid] = {"mtime": path.stat().st_mtime, "themes": dict(clean)}
    return clean


def themes_cache_reset(user_id=None):
    """Сбрасывает кэш тем (одного пользователя или весь)."""
    with _THEMES_LOCK:
        if user_id is None:
            _THEMES_CACHE.clear()
        else:
            _THEMES_CACHE.pop(_uid(user_id), None)


def legacy_snapshot_themes() -> dict:
    """Содержимое старого общего ``data/ba_themes.json`` (для миграции) или дефолты кода."""
    snap = _read_json_dict(LEGACY_SNAPSHOT) if LEGACY_SNAPSHOT.exists() else {}
    if snap:
        return snap
    return dict(DEFAULT_THEMES)


def migrate_legacy_snapshot(user_ids=None) -> dict:
    """Одноразовая миграция общего снапшота в пер-пользовательские файлы.

    ``user_ids`` — список пользователей, которым полагается старый общий набор (владельцы
    существующих подключений BA). Файл, который уже есть у пользователя, не перезаписывается.
    """
    source = legacy_snapshot_themes()
    ids = [_uid(u) for u in (user_ids or []) if _uid(u)]
    created = {}
    for uid in ids:
        path = themes_path(uid)
        if path.exists():
            continue
        save_themes(uid, source)
        created[uid] = len(source)
    return created


def merge_theme_snapshot() -> dict:
    """Совместимость: обновляет справочный ``THEMES`` из общего файла. Рабочий путь не использует."""
    global THEMES
    THEMES = legacy_snapshot_themes()
    return THEMES


# Справочный набор «как было» — только для сида/миграции (рабочие темы берутся из снапшота пользователя).
THEMES = legacy_snapshot_themes()


def theme_title(user_id, theme_id: str, fallback: str = "") -> str:
    return load_themes(user_id).get(str(theme_id)) or fallback or str(theme_id)


# ---------------------------------------------------------------------------
# Работа с Brand Analytics (node-CLI)
# ---------------------------------------------------------------------------
def _ba_env(login: str, passw: str, cookies: Path = None) -> dict:
    if not login or not passw:
        raise RuntimeError(
            "Brand Analytics: не переданы логин/пароль. Подключите свой аккаунт Brand Analytics в Tellscope")
    env = dict(os.environ)
    env["BA_LOGIN"] = login
    env["BA_PASS"] = passw
    env["NODE_PATH"] = env.get("NODE_PATH", "/tmp/tshot/node_modules")
    if cookies:
        cookies.parent.mkdir(parents=True, exist_ok=True)
        env["BA_COOKIES"] = str(cookies)
    return env


def run_ba_export(theme_id: str, out_dir: Path, tsf: str, tst: str, login=None, passw=None,
                  user_id=None) -> Path:
    """Выгрузка темы под учёткой конкретного пользователя Tellscope."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cookies = cookies_path(user_id, login or "")
    env = _ba_env(login or "", passw or "", cookies)
    script = BE / "ba_worker" / "export_cli.js"
    cmd = ["node", str(script), theme_id, str(out_dir), tsf, tst]
    log.info("BA export: %s", " ".join(cmd[:3]) + " ...")
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
    tail = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0:
        raise RuntimeError("BA export failed: " + tail[-2000:])
    out = r.stdout.strip().splitlines()
    if not out:
        raise RuntimeError("BA export: empty output: " + tail[-1000:])
    p = Path(out[-1])
    if not p.exists():
        raise RuntimeError("BA export file not found: " + str(p))
    return p


def fetch_ba_themes(login=None, passw=None, user_id=None) -> dict:
    """Получает список тем BA под указанной учёткой и (если задан user_id) сохраняет её снапшот.

    Ошибка входа в BA (неверный логин/пароль) — это RuntimeError с понятным текстом:
    сохранение такого подключения запрещено, вызывающий код показывает ошибку пользователю.
    """
    cookies = cookies_path(user_id, login or "")
    env = _ba_env(login or "", passw or "", cookies)
    script = BE / "ba_worker" / "themes_cli.js"
    r = subprocess.run(["node", str(script)], env=env, capture_output=True, text=True, timeout=300)
    tail = (r.stdout or "") + (r.stderr or "")
    auth_ok = None
    out = {}
    for line in (r.stdout or "").splitlines():
        s = line.strip()
        if s.startswith("AUTH_FAILED"):
            auth_ok = False
            reason = s[len("AUTH_FAILED"):].strip(" :")
            raise RuntimeError("Brand Analytics не принял логин или пароль: " + (reason or "проверьте данные"))
        if s.startswith("AUTH_OK"):
            auth_ok = True
        if s == "RESULT_JSON":
            continue
        if auth_ok and s.startswith("{"):
            try:
                data = json.loads(s)
            except Exception:
                continue
            out = {str(k): str(v) for k, v in data.items() if k and v}
    if r.returncode != 0:
        raise RuntimeError("BA themes scrape failed: " + tail[-1500:])
    if auth_ok is None:
        raise RuntimeError("BA themes scrape: нет подтверждения входа в Brand Analytics: " + tail[-800:])
    if not out:
        raise RuntimeError("В аккаунте Brand Analytics не найдено ни одной темы")
    if user_id:
        save_themes(user_id, out)
    return out


# ---------------------------------------------------------------------------
# Датасеты и индексация
# ---------------------------------------------------------------------------
def load_indexes() -> dict:
    if INDEXES_PKL.exists():
        try:
            with open(INDEXES_PKL, "rb") as f:
                d = pickle.load(f)
            return d if isinstance(d, dict) else {}
        except Exception:
            return {}
    return {}


def save_indexes(indexes: dict) -> None:
    INDEXES_PKL.parent.mkdir(parents=True, exist_ok=True)
    with open(INDEXES_PKL, "wb") as f:
        pickle.dump(indexes, f)


def register_dataset(user_id: str, folder_name: str, json_filename: str, json_path: Path, next_key: int) -> None:
    """Регистрирует json как датасет (как add-file): indexes.pkl + Redis json_files_directory."""
    folder_dir = DATA / user_id / "json_files_directory" / folder_name
    folder_dir.mkdir(parents=True, exist_ok=True)
    target = folder_dir / json_filename
    shutil.copyfile(json_path, target)

    indexes = load_indexes()
    nk = int(next_key) if next_key else (max(indexes.keys()) + 1 if indexes else 1)
    indexes[nk] = json_filename.replace(".json", "").lower()
    save_indexes(indexes)

    folders = {}
    raw = REDIS.hget(user_id, "json_files_directory")
    if raw:
        try:
            folders = json.loads(raw)
        except Exception:
            folders = {}
    files = folders.get(folder_name, [])
    if json_filename in files:
        files.remove(json_filename)
    files.append(json_filename)
    folders[folder_name] = files
    REDIS.hset(user_id, "json_files_directory", json.dumps(folders, ensure_ascii=False))
    return folder_dir, nk


def index_file(user_id: str, folder_name: str, json_filename: str, folder_dir: Path, nk: int) -> dict:
    from load_data_elastic import load_file_to_elstic
    class FileObject:
        def __init__(self, filename): self.filename = filename
    # loader внутри делает os.chdir(path) — сохраняем и восстанавливаем cwd
    cwd = os.getcwd()
    try:
        result = load_file_to_elstic(FileObject(json_filename), path=str(folder_dir))
    finally:
        os.chdir(cwd)
    return {"status": "ok", "result": result, "index_key": nk, "index_name": json_filename.replace(".json", "").lower()}


def cmd_export(args) -> int:
    theme_id = args.theme
    user_id = str(args.user)
    themes = load_themes(user_id)
    folder = args.folder or themes.get(theme_id, "BA_theme_" + theme_id)
    tsf, tst = (args.tsf, args.tst) if args.tsf and args.tst else BA_DEFAULT_PERIOD
    cc = account_creds(user_id)
    if not cc["configured"]:
        print(json.dumps({"status": "error", "error": "Аккаунт Brand Analytics не подключён: подключите свой аккаунт BA в Tellscope"}, ensure_ascii=False))
        return 1
    run_dir = Path("/tmp") / ("ba_run_" + uuid.uuid4().hex[:8])
    try:
        log.info("Запрашиваю экспорт темы %s ...", theme_id)
        raw = run_ba_export(theme_id, run_dir, tsf, tst, login=cc["BA_LOGIN"], passw=cc["BA_PASS"], user_id=user_id)
        size = raw.stat().st_size
        log.info("Скачан файл %s (%d байт)", raw.name, size)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = "BA_%s_%s.json" % (slug(themes.get(theme_id, theme_id)), stamp)
        indexes = load_indexes()
        nk = max(indexes.keys()) + 1 if indexes else 1
        folder_dir, nk = register_dataset(user_id, folder, json_filename, raw, nk)
        log.info("Файл размещён: %s (индекс key=%s)", folder_dir / json_filename, nk)
        if not args.no_index:
            log.info("Индексация в Elasticsearch/Qdrant ...")
            res = index_file(user_id, folder, json_filename, folder_dir, nk)
            log.info("Индексация завершена: %s", res["result"])
        else:
            res = {"status": "skipped"}
        print(json.dumps({"json_file": str(folder_dir / json_filename), "index_key": nk,
                          "bytes": size, "index_status": res.get("status")}, ensure_ascii=False))
        return 0
    except Exception as e:
        log.error("Импорт не удался: %s", e)
        print(json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False))
        return 1
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def cmd_index(args) -> int:
    user_id = str(args.user)
    folder = args.folder
    filename = args.file
    folder_dir = DATA / user_id / "json_files_directory" / folder
    if not (folder_dir / filename).exists():
        print(json.dumps({"status": "error", "error": "file not found: %s" % (folder_dir / filename)}, ensure_ascii=False))
        return 1
    indexes = load_indexes()
    nk = None
    base = filename.replace(".json", "").lower()
    for k, v in indexes.items():
        if v == base:
            nk = k
            break
    try:
        res = index_file(user_id, folder, filename, folder_dir, nk)
        print(json.dumps({"status": "ok", "index_name": base, "index_key": nk, "result": res.get("result")}, ensure_ascii=False))
        return 0
    except Exception as e:
        print(json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False))
        return 1


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("export")
    ex.add_argument("--theme", required=True, help="id темы в BA (report/<id>/summary)")
    ex.add_argument("--user", default=os.environ.get("BA_USER_ID", "1"))
    ex.add_argument("--folder", default="", help="имя папки-датасета в Tellscope")
    ex.add_argument("--tsf", default="")
    ex.add_argument("--tst", default="")
    ex.add_argument("--no-index", action="store_true")
    ix = sub.add_parser("index")
    ix.add_argument("--user", default=os.environ.get("BA_USER_ID", "1"))
    ix.add_argument("--folder", required=True)
    ix.add_argument("--file", required=True)
    args = p.parse_args()
    if args.cmd == "export":
        return cmd_export(args)
    if args.cmd == "index":
        return cmd_index(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
