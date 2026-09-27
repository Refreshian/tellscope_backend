# -*- coding: utf-8 -*-
"""Последовательный запуск месячных задач KFC 2026. Работает на сервере, не зависит от браузера и ноутбука."""
import json, time, urllib.request, urllib.error

BASE = "http://127.0.0.1:5000"
LOG = "/home/dev/tellscope_app/tellscope_backend/data/1/kfc_2026_runner.log"
TASKS = "/home/dev/tellscope_app/tellscope_backend/data/1/kfc_2026_months.json"
TERMINAL = {"done", "failed", "cancelled", "interrupted"}


def log(msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S") + "  " + msg
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def call(method, path, data=None, headers=None, timeout=120):
    req = urllib.request.Request(BASE + path, data=data, method=method)
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except Exception as e:
        return "ERR:" + type(e).__name__, ""


def login():
    c, b = call("POST", "/auth/login", b"username=test@test.ru&password=1245",
                {"Content-Type": "application/x-www-form-urlencoded"})
    return json.loads(b)["access_token"]


def states(tok):
    c, b = call("GET", "/harness/tasks", None, {"Authorization": "Bearer " + tok}, timeout=60)
    try:
        items = json.loads(b)
        items = items.get("tasks", items) if isinstance(items, dict) else items
        return {t.get("id"): (t.get("status"), t.get("run_status")) for t in items}
    except Exception:
        return {}


def run_month(label, tid):
    """Запускает месяц, ждёт завершения, при 409/сбое повторяет. Возвращает True, если done."""
    for waiting in range(40):          # до ~40 минут ожидания свободного слота
        tok = login()
        c, b = call("POST", "/harness/task/%s/run" % tid, None,
                    {"Authorization": "Bearer " + tok}, timeout=300)
        if c == 200:
            break
        log("%s: запуск отложен (%s %s), жду 60 с" % (label, c, b[:90].replace("\n", " ")))
        time.sleep(60)
    else:
        log("%s: не удалось запустить — слот занят" % label)
        return False

    log("%s: запущена" % label)
    started = time.time()
    while True:
        time.sleep(30)
        st = states(tok).get(tid, ("?", "?"))
        if st[0] in TERMINAL and (st[1] in TERMINAL or st[1] is None):
            break
        if time.time() - started > 4 * 3600:
            log("%s: жду больше 4 часов, перехожу к следующей" % label)
            return False
    log("%s: статус %s/%s, время %.0f с" % (label, st[0], st[1], time.time() - started))
    return st[0] == "done"


def main():
    tasks = json.load(open(TASKS, encoding="utf-8"))
    log("раннер стартовал, задач: %d" % len(tasks))
    for label, tid, _ in tasks:
        ok = run_month(label, tid)
        if not ok:
            log("%s: повтор через 90 с" % label)
            time.sleep(90)
            ok = run_month(label, tid)
        log("%s: итог — %s" % (label, "готово" if ok else "не удалось"))
        time.sleep(20)
    log("раннер завершил все задачи")


if __name__ == "__main__":
    main()