# -*- coding: utf-8 -*-
"""Последовательный прогон месячных задач KFC 2026. Живёт на сервере, следит за статусом ЗАПУСКА."""
import json, time, urllib.request, urllib.error

BASE = "http://127.0.0.1:5000"
LOG = "/home/dev/tellscope_app/tellscope_backend/data/1/kfc_2026_runner2.log"
TASKS = [("январь", "ht_7542b0592b"), ("февраль", "ht_77118458ce"), ("март", "ht_b9677bc83a"),
         ("апрель", "ht_ce42ae6f7d"), ("май", "ht_5c06703327"), ("июнь", "ht_97d339244c"),
         ("июль", "ht_9d82a8b604"), ("август", "ht_000b4dcf47")]
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
    for _ in range(20):                     # ждём свободный слот до ~20 минут
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
            log("%s: статус %s, время %.0f с" % (label, st, time.time() - started))
            return st == "completed"
        if time.time() - started > 3 * 3600:
            log("%s: больше 3 часов, перехожу к следующей" % label)
            return False


def main():
    log("раннер-2 стартовал, задач: %d" % len(TASKS))
    for label, tid in TASKS:
        ok = run_month(label, tid)
        if not ok:
            log("%s: повтор через 120 с" % label)
            time.sleep(120)
            ok = run_month(label, tid)
        log("%s: ИТОГ — %s" % (label, "готово" if ok else "не удалось"))
        time.sleep(20)
    log("раннер-2 завершил все задачи")


if __name__ == "__main__":
    main()