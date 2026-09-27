# -*- coding: utf-8 -*-
"""Стадия после догона: дождаться раннера, перезапустить сервис, почистить папки,
создать задачи на годовые отчёты и запустить их последовательный прогон.

Запускается на сервере в фоне (setsid nohup), чтобы не зависеть от SSH-сессии.
"""
import json
import os
import subprocess
import sys
import time

B = "/home/dev/tellscope_app/tellscope_backend"
PY = B + "/venv_py312_clean/bin/python"
SUDO = "echo WCsEGHqXkbpts5rYqbBjCDVop | sudo -S "
LOG = B + "/data/1/kfc_annual_stage.log"
STAGE = B + "/kfc_annual_stage.json"


def log(msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S") + "  " + msg
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def sh(cmd, timeout=3600):
    proc = subprocess.run(cmd, shell=True, cwd=B, capture_output=True, text=True, timeout=timeout)
    return proc.returncode, (proc.stdout or "")[-4000:], (proc.stderr or "")[-2000:]


def main():
    state = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "steps": []}
    log("стадия годовых отчётов: старт")
    # 1) ждём завершения раннера догона
    waited = 0
    while waited < 4 * 3600:
        # Квадратные скобки в шаблоне: pgrep -f иначе находит сам себя (команда содержит ту же строку)
        rc, out, _ = sh("pgrep -f \"[k]fc_missing_runner.py\" | head -3", timeout=60)
        if not out.strip():
            break
        time.sleep(60)
        waited += 60
        if waited % 600 == 0:
            log("ждём догон: %d мин" % (waited // 60))
    log("догон завершён (ждали %d мин)" % (waited // 60))
    state["steps"].append({"wait_done": waited})

    # 2) перезапуск сервиса: подхватываем правки tools_reports (ключ межгодового итога)
    rc, out, err = sh(SUDO + "supervisorctl restart fastapi_app", timeout=600)
    log("рестарт сервиса: rc=%s %s" % (rc, out.strip().replace("\n", " ")[:200]))
    time.sleep(30)
    rc, out, _ = sh("curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:5000/mlops/ready", timeout=120)
    log("сервис /mlops/ready = %s" % out.strip())
    state["steps"].append({"restart": out.strip()})
    if out.strip() != "200":
        log("СЕРВИС НЕ ГОТОВ — стадия остановлена")
        json.dump(state, open(STAGE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        return

    # 3) чистка папок отчётов KFC
    rc, out, err = sh("%s -u kfc_cleanup.py --apply" % PY, timeout=600)
    log("чистка отчётов: rc=%s" % rc)
    log(out[-2500:])
    if err.strip():
        log("чистка stderr: %s" % err[-800:])
    state["steps"].append({"cleanup_rc": rc, "cleanup_tail": out[-1500:]})

    # 4) задачи на годовые и межгодовой отчёты
    rc, out, err = sh("%s -u kfc_make_annual.py --apply" % PY, timeout=900)
    log("создание задач: rc=%s" % rc)
    log(out[-2500:])
    if err.strip():
        log("создание задач stderr: %s" % err[-800:])
    state["steps"].append({"annual_tasks_rc": rc, "tail": out[-1500:]})

    # 5) прогон годовых отчётов
    log("старт раннера годовых отчётов")
    rc, out, err = sh("%s -u kfc_annual_runner.py" % PY, timeout=6 * 3600)
    log("раннер годовых отчётов: rc=%s" % rc)
    log(out[-3000:])
    state["steps"].append({"annual_runner_rc": rc, "tail": out[-2000:]})
    state["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(state, open(STAGE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log("стадия годовых отчётов завершена")


if __name__ == "__main__":
    main()
