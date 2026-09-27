# -*- coding: utf-8 -*-
"""Чистка папок отчётов KFC: по месяцу — один самый свежий комплект (DOCX+PDF+summary).

Скрипт живёт на сервере и не импортирует main (только файловые операции).
По умолчанию показывает план; с --apply выполняет его. Манифест всех действий — в JSON,
чтобы всё можно было вернуть.
"""
import io
import json
import os
import re
import shutil
import sys
import time

B = "/home/dev/tellscope_app/tellscope_backend"
R = B + "/data/1/reports_directory"
DATASET = "kfc_13.05.2024-22.09.2026 Отчёты"
DATASET_DIR = R + "/" + DATASET
MANIFEST = B + "/data/1/kfc_cleanup_manifest.json"

MONTH_STEM = ["январ", "феврал", "март", "апрел", "ма", "июн", "июл",
              "август", "сентябр", "октябр", "ноябр", "декабр"]

# Папки, где лежат именно месячные отчёты KFC. «KFC», «KFC_агент», чужие темы (Озон, Платон)
# и файлы других тем внутри «Центр задач» не трогаем.
SCOPE_DIRS = ["kfc_13.05.2024-22.09.2026 Отчёты", "1102 Отчёты", "kfc_feb_2026_reports",
              "KFC_2025_01_Отчёты", "kfc_январь_2025", "Центр задач"]
# Месяцы, которые удаляем целиком: тестовый январь 2024 и ошибочный сентябрь 2026.
DROP_MONTHS = {"2024-01", "2026-09"}

SUMMARY_RE = re.compile(r"^(\d{4})-(\d{2})_summary\.json$")


def is_kfc(name):
    low = name.lower()
    return "kfc" in low or "кфс" in low


def month_of(name):
    match = SUMMARY_RE.match(name)
    if match:
        return "%s-%s" % match.groups()
    if not is_kfc(name):
        return None
    low = name.lower().replace("ё", "е")
    stamp = re.search(r"(20\d{2})-(\d{2})-\d{2}", low)
    if stamp:
        return "%s-%s" % (stamp.group(1), stamp.group(2))
    for index, stem in enumerate(MONTH_STEM):
        match = re.search(stem + r"[а-я]*\s+(20\d{2})", low)
        if match:
            return "%s-%02d" % (match.group(1), index + 1)
    return None


def kind_of(name):
    low = name.lower()
    if low.endswith("_summary.json"):
        return "summary"
    if low.endswith(".docx"):
        return "docx"
    if low.endswith(".pdf"):
        return "pdf"
    return None


def collect():
    """Все месячные файлы KFC в области видимости: month -> folder -> kind -> [(name, mtime, size)]."""
    data = {}
    for folder in SCOPE_DIRS:
        fdir = os.path.join(R, folder)
        if not os.path.isdir(fdir):
            continue
        for name in sorted(os.listdir(fdir)):
            path = os.path.join(fdir, name)
            if not os.path.isfile(path):
                continue
            kind = kind_of(name)
            if not kind:
                continue
            month = month_of(name)
            if not month:
                continue
            st = os.stat(path)
            data.setdefault(month, {}).setdefault(folder, {}).setdefault(kind, []).append(
                (name, st.st_mtime, st.st_size))
    return data


def plan(data):
    """Для каждого месяца: какие файлы оставить (и куда перенести), какие удалить."""
    keep, delete, move = [], [], []
    for month in sorted(data):
        if month in DROP_MONTHS:
            for folder, kinds in data[month].items():
                for kind, files in kinds.items():
                    for name, _, _ in files:
                        delete.append((folder, name))
            continue
        # Папка-победитель: полный комплект (docx+pdf+summary) с самым свежим файлом,
        # иначе — самое свежее из того, что есть.
        def rank(folder):
            kinds = data[month][folder]
            complete = {"docx", "pdf", "summary"} <= set(kinds)
            newest = max(mtime for files in kinds.values() for _, mtime, _ in files)
            return (1 if complete else 0, newest)

        folder = max(data[month], key=rank)
        chosen = {}
        for kind, files in data[month][folder].items():
            chosen[kind] = sorted(files, key=lambda item: item[1])[-1]
        for other, kinds in data[month].items():
            for kind, files in kinds.items():
                for name, mtime, _ in files:
                    if other == folder and kind in chosen and name == chosen[kind][0]:
                        continue
                    delete.append((other, name))
        if folder != DATASET:
            for kind, (name, _, _) in sorted(chosen.items()):
                move.append((folder, name))
        keep.append((month, folder, sorted(chosen)))
    return keep, delete, move


def main():
    apply = "--apply" in sys.argv
    data = collect()
    keep, delete, move = plan(data)

    print("=== ПЛАН ЧИСТКИ (%s) ===" % ("ПРИМЕНЕНИЕ" if apply else "сухой прогон"))
    print("\n-- остаётся по месяцам (один комплект) --")
    for month, folder, kinds in keep:
        target = DATASET if (folder, "") and any(m[0] == folder for m in move) else folder
        print("  %s  %-34s %s" % (month, target, kinds))
    print("\n-- перенос в «%s» --" % DATASET)
    for folder, name in move:
        print("  %-24s %s" % (folder, name))
    print("\n-- удаление (%d файлов) --" % len(delete))
    for folder, name in sorted(delete):
        print("  %-34s %s" % (folder, name))

    months_keep = sorted(month for month, _, _ in keep)
    print("\nмесяцев с комплектом: %d — %s" % (len(months_keep), ", ".join(months_keep)))
    missing = [k for k in ("%s-%02d" % (y, m) for y in (2024, 2025, 2026) for m in range(1, 13))
               if k not in months_keep and k not in DROP_MONTHS and k >= "2024-05"
               and not (k >= "2026-09")]
    print("без комплекта (ожидаемо/проверить): %s" % ", ".join(missing) if missing else "нет")

    if not apply:
        print("\n(сухой прогон: ничего не изменено; запустите с --apply)")
        return

    manifest = {"at": time.strftime("%Y-%m-%d %H:%M:%S"), "moved": [], "deleted": []}
    for folder, name in move:
        src = os.path.join(R, folder, name)
        dst = os.path.join(DATASET_DIR, name)
        if os.path.isfile(dst):
            print("  пропуск переноса (уже есть): %s" % name)
            continue
        shutil.move(src, dst)
        manifest["moved"].append({"from": src, "to": dst})
        print("  перенесён: %s -> %s" % (name, DATASET))
    for folder, name in sorted(delete):
        path = os.path.join(R, folder, name)
        if os.path.isfile(path):
            os.remove(path)
            manifest["deleted"].append(path)
    with io.open(MANIFEST, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=1)
    print("\nперенесено: %d, удалено: %d, манифест: %s"
          % (len(manifest["moved"]), len(manifest["deleted"]), MANIFEST))


if __name__ == "__main__":
    main()
