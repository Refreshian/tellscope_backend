# -*- coding: utf-8 -*-
"""Последовательный прогон задач на годовые и межгодовой отчёты KFC (живёт на сервере).

Для каждой задачи пишет в лог: статус, время, вызвал ли агент read_reports, какие файлы собрал.
Итог — data/1/kfc_annual_result.json.
"""
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

BASE = "http://127.0.0.1:5000"
B = "/home/dev/tellscope_app/tellscope_backend"
LOG = B + "/data/1/kfc_annual_runner.log"
LIST = B + "/data/1/kfc_annual_list.json"
RESULT = B + "/data/1/kfc_annual_result.json"
DONE = {"completed", "failed", "cancelled", "interrupted"}
TOKEN = {"value": ""}


def log(msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S") + "  " + msg
    print(line, flush=True)
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def login(force=False):
    if TOKEN["value"] and not force:
        return TOKEN["value"]
    body = urllib.parse.urlencode({"username": "test@test.ru", "password": "1245"}).encode()
    req = urllib.request.Request(BASE + "/auth/login", data=body,
                                 headers={"Content-Type": "application/x-www-form-urlencoded"})
    with urllib.request.urlopen(req, timeout=60) as r:
        TOKEN["value"] = json.load(r)["access_token"]
    return TOKEN["value"]


def call(method, path, data=None, timeout=180, retries=4):
    last = None
    for attempt in range(retries):
        req = urllib.request.Request(BASE + path, data=data, method=method,
                                     headers={"Authorization": "Bearer " + login()})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, json.load(r)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:200]
            last = "HTTP %s %s" % (e.code, body)
            if e.code == 401:
                login(force=True)
                continue
            if e.code in (409, 429):
                log("  слот занят (%s) — жду 60 с" % last)
                time.sleep(60)
                continue
            return e.code, last
        except Exception as e:
            last = "%s: %s" % (type(e).__name__, e)
            time.sleep(20)
    return "ERR", last


def run_task(label, task_id, min_seconds=90):
    started = time.time()
    for attempt in (1, 2):
        status, resp = call("POST", "/harness/task/%s/run" % task_id, b"{}")
        run_id = (resp or {}).get("run_id") if isinstance(resp, dict) else None
        if not run_id:
            log("%s: запуск не удался (%s %s)" % (label, status, str(resp)[:120]))
            time.sleep(60)
            continue
        log("%s: запущена, run=%s (попытка %d)" % (label, run_id, attempt))
        while time.time() - started < 3 * 3600:
            time.sleep(30)
            code, doc = call("GET", "/agent/run/%s" % run_id)
            run = (doc or {}).get("run") if isinstance(doc, dict) else None
            run = run or (doc if isinstance(doc, dict) else {})
            st = str(run.get("status") or "")
            if st in DONE:
                duration = time.time() - started
                tools = sorted({str(c.get("name")) for c in (run.get("tool_calls") or [])})
                arts = [a.get("name") for a in (run.get("artifacts") or []) if str(a.get("kind")) == "report"]
                used = "read_reports" in tools
                log("%s: статус %s, %.0f с, read_reports=%s, инструментов %d, файлов %d"
                    % (label, st, duration, "да" if used else "НЕТ", len(tools), len(arts)))
                log("%s: инструменты: %s" % (label, ", ".join(tools) or "-"))
                log("%s: файлы: %s" % (label, "; ".join(arts) or "-"))
                ok = st == "completed" and duration >= min_seconds
                return {"label": label, "task_id": task_id, "run_id": run_id, "status": st,
                        "duration": round(duration, 1), "read_reports": used, "tools": tools,
                        "files": arts, "ok": ok, "answer": str(run.get("answer") or "")[:2000]}
        log("%s: больше 3 часов — пропускаю" % label)
        return {"label": label, "task_id": task_id, "status": "timeout", "ok": False}
    return {"label": label, "task_id": task_id, "status": "no-run", "ok": False}


def main():
    items = json.load(open(LIST, encoding="utf-8"))
    log("раннер годовых отчётов стартовал: задач %d (%s)"
        % (len(items), ", ".join(str(i.get("label")) for i in items)))
    results = []
    for item in items:
        try:
            results.append(run_task(str(item.get("label")), str(item.get("id"))))
        except Exception as exc:
            log("%s: ошибка запуска: %s: %s" % (item.get("label"), type(exc).__name__, exc))
            results.append({"label": str(item.get("label")), "task_id": item.get("id"),
                            "status": "exception", "ok": False})
        time.sleep(20)
    with open(RESULT, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=1)
    log("раннер годовых отчётов завершён: готово %d из %d"
        % (len([r for r in results if r.get("ok")]), len(results)))
    for row in results:
        log("  %s: %s, read_reports=%s, файлов %d" % (row.get("label"), row.get("status"),
                                                     row.get("read_reports"), len(row.get("files") or [])))


if __name__ == "__main__":
    main()
