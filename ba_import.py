
# -*- coding: utf-8 -*-
"""
BA -> Tellscope manual import (P1).
Экспорт JSON из Brand Analytics, размещение в датасет и (опционально) индексация.
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

THEMES = {
    "12394607": "Мониторинг тем",
    "12466084": "Морская вода, Аллергия",
    "14075092": "Признаки ОРВИ",
    "12505577": "Риномарис",
    "13947576": "Строительство / Реконструкция",
    "14166164": "Энергострой",
}
BA_DEFAULT_PERIOD = ("1788037200", "1788641999")  # последние дни, как в UI по умолчанию

def creds() -> dict:
    env_file = BE / ".env_ba"
    out = {"BA_LOGIN": os.environ.get("BA_LOGIN", "alexmisis@list.ru"),
           "BA_PASS": os.environ.get("BA_PASS", "")}
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    return out

def run_ba_export(theme_id: str, out_dir: Path, tsf: str, tst: str, login=None, passw=None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    c = creds()
    env["BA_LOGIN"] = login or c["BA_LOGIN"]
    env["BA_PASS"] = passw or c["BA_PASS"]
    env["NODE_PATH"] = env.get("NODE_PATH", "/tmp/tshot/node_modules")
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

def slug(text: str) -> str:
    return re.sub(r"[^A-Za-zА-Яа-я0-9]+", "_", text).strip("_")[:60]


THEMES_SNAPSHOT = DATA / "ba_themes.json"


def merge_theme_snapshot():
    try:
        if THEMES_SNAPSHOT.exists():
            snap = json.loads(THEMES_SNAPSHOT.read_text(encoding="utf-8"))
            if isinstance(snap, dict) and snap:
                THEMES.clear()
                for k, v in snap.items():
                    if str(k) and v:
                        THEMES[str(k)] = str(v)
    except Exception as e:
        log.warning("merge theme snapshot failed: %s", e)


def fetch_ba_themes(login=None, passw=None) -> dict:
    c = creds()
    env = dict(os.environ)
    env["BA_LOGIN"] = login or c["BA_LOGIN"]
    env["BA_PASS"] = passw or c["BA_PASS"]
    env["NODE_PATH"] = env.get("NODE_PATH", "/tmp/tshot/node_modules")
    script = BE / "ba_worker" / "themes_cli.js"
    r = subprocess.run(["node", str(script)], env=env, capture_output=True, text=True, timeout=300)
    tail = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0:
        raise RuntimeError("BA themes scrape failed: " + tail[-1500:])
    out = {}
    started = False
    for line in (r.stdout or "").splitlines():
        if line.strip() == "RESULT_JSON":
            started = True
            continue
        if started and line.strip():
            try:
                data = json.loads(line)
                out = {str(k): str(v) for k, v in data.items() if k and v}
            except Exception:
                pass
            break
    if not out:
        raise RuntimeError("BA themes scrape: no data: " + tail[-1000:])
    THEMES_SNAPSHOT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    THEMES.clear()
    for k, v in out.items():
        THEMES[k] = v
    return out

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
    folder = args.folder or THEMES.get(theme_id, "BA_theme_" + theme_id)
    tsf, tst = (args.tsf, args.tst) if args.tsf and args.tst else BA_DEFAULT_PERIOD
    run_dir = Path("/tmp") / ("ba_run_" + uuid.uuid4().hex[:8])
    try:
        log.info("Запрашиваю экспорт темы %s ...", theme_id)
        raw = run_ba_export(theme_id, run_dir, tsf, tst)
        size = raw.stat().st_size
        log.info("Скачан файл %s (%d байт)", raw.name, size)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = "BA_%s_%s.json" % (slug(THEMES.get(theme_id, theme_id)), stamp)
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

merge_theme_snapshot()

if __name__ == "__main__":
    sys.exit(main())
