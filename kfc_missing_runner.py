# -*- coding: utf-8 -*-
"""Догон недостающих месячных задач KFC (2024-11, 2024-12, 2025-04, 2025-11, 2026-03..08)."""
import json, time, urllib.request, urllib.error

BASE = "http://127.0.0.1:5000"
B = "/home/dev/tellscope_app/tellscope_backend"
LOG = B + "/data/1/kfc_missing_runner.log"
LIST = B + "/data/1/kfc_missing_list.json"
DONE = {"completed", "failed", "cancelled", "interrupted"}


def log(msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S") + "  " + msg
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def call(method, path, data=None, headers=None, timeout=180):
    req = urllib.request.Request(BASE + path, data=data, method=method)
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
        return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except Exception as e:
        return "ERR:" + type(e).__name__, ""


def login():
    c, b = call("POST", "/auth/login", b"username=test@test.ru&password=1245",
                {"Content-Type": "application/x-www-form-urlencoded"})
    return json.loads(b)["access_token"]


def run_month(label, tid):
    run_id = None
    for _ in range(20):
        tok = login()
        c, b = call("POST", "/harness/task/%s/run" % tid, None, {"Authorization": "Bearer " + tok})
        if c == 200:
            try:
                run_id = json.loads(b).get("run_id")
            except Exception:
                run_id = None
            break
        log("%s: слот занят (%s), жду 60 с" % (label, c))
        time.sleep(60)
    else:
        log("%s: не удалось запустить" % label)
        return False
    log("%s: запущена, run=%s" % (label, run_id))
    started = time.time()
    while True:
        time.sleep(30)
        st = None
        if run_id:
            c, b = call("GET", "/agent/run/%s" % run_id, None, {"Authorization": "Bearer " + tok})
            try:
                st = json.loads(b).get("status")
            except Exception:
                st = None
        if st in DONE:
            dur = time.time() - started
            quick = dur < 180 and st == "completed"
            log("%s: статус %s, время %.0f с%s" % (label, st, dur,
                "  <-- СЛИШКОМ БЫСТРО, проверить!" if quick else ""))
            return st == "completed" and not quick
        if time.time() - started > 3 * 3600:
            log("%s: больше 3 часов, пропускаю" % label)
            return False


def main():
    items = json.load(open(LIST, encoding="utf-8"))
    log("догон недостающих месяцев: задач %d — %s" % (len(items), ", ".join(i[0] for i in items)))
    for label, tid in items:
        ok = run_month(label, tid)
        if not ok:
            log("%s: повтор через 120 с" % label)
            time.sleep(120)
            ok = run_month(label, tid)
        log("%s: ИТОГ — %s" % (label, "готово" if ok else "не удалось/подозрительно быстро"))
        time.sleep(20)
    log("догон недостающих месяцев завершён")


if __name__ == "__main__":
    main()
