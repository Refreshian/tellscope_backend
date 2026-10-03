#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Переименование датасета: файл, индекс Elasticsearch, коллекция Qdrant, реестр.

Зачем: имена вида ``BA_Озон_отзывы_20261003_120318.json`` ничего не говорят о содержимом —
в списке тем нужен «Озон отзывы 01.09.2026-30.09.2026». Переименование затрагивает:

* файл данных в ``data/<user_id>/json_files_directory/<папка>/``;
* имя в Redis (``json_files_directory``), по нему строится список файлов в интерфейсе;
* индекс Elasticsearch (создаётся новый, данные переливаются ``_reindex``, старый удаляется
  после сверки количества документов — файл данных остаётся источником для повторной индексации);
* коллекцию Qdrant ``<индекс>__bge`` (векторный поиск), если она есть;
* реестр ``data/indexes.pkl`` (номер датасета → имя индекса).

Примеры:
    python rename_dataset.py --list
    python rename_dataset.py --key 1105 --name "Озон отзывы 01.09.2026-30.09.2026"
    python rename_dataset.py --key 1105 --name "..." --dry-run
    python rename_dataset.py --auto --user 1        # все BA_* → «Тема ДД.ММ.ГГГГ-ДД.ММ.ГГГГ»
"""
from __future__ import annotations

import argparse
import datetime
import io
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path

import redis
from elasticsearch import Elasticsearch
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

BE = Path(__file__).resolve().parent
DATA = BE / "data"
INDEXES_PKL = DATA / "indexes.pkl"
REDIS = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
MSK = datetime.timezone(datetime.timedelta(hours=3))


def log(message: str) -> None:
    print("[%s] %s" % (datetime.datetime.now().strftime("%H:%M:%S"), message), flush=True)


def es_client() -> Elasticsearch:
    """Клиент Elasticsearch с доступами проекта (как в приложении)."""
    sys.path.insert(0, str(BE))
    from agent_engine.tools_data import _es

    return _es()


def index_map() -> dict:
    if not INDEXES_PKL.exists():
        return {}
    with INDEXES_PKL.open("rb") as fh:
        return pickle.load(fh)


def save_index_map(mapping: dict) -> None:
    backup = INDEXES_PKL.with_suffix(".pkl.bak_%s" % datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    if INDEXES_PKL.exists():
        backup.write_bytes(INDEXES_PKL.read_bytes())
    with INDEXES_PKL.open("wb") as fh:
        pickle.dump(mapping, fh)
    log("реестр сохранён (бэкап: %s)" % backup.name)


def dataset_index_name(file_name: str) -> str:
    base = str(file_name or "").replace(".json", "").lower()
    base = re.sub(r"[^0-9a-zа-яё._-]+", "_", base).strip("_.-")
    return base or "dataset"


def find_file(user_id: str, index_name: str):
    """Ищет файл по имени индекса: в Redis имя файла может отличаться регистром и хвостом .json."""
    raw = REDIS.hget(str(user_id), "json_files_directory")
    if not raw:
        return None, None, None
    wanted = str(index_name).lower()
    for folder, files in (json.loads(raw) or {}).items():
        for file_name in files or []:
            stem = str(file_name)
            if stem.lower() == wanted or stem.lower() == wanted + ".json" \
                    or dataset_index_name(stem) == wanted:
                return folder, DATA / str(user_id) / "json_files_directory" / folder / stem, stem
    return None, None, None


def unique_name(folder_dir: Path, base_name: str) -> str:
    """«Озон отзывы 10.09.2026-11.09.2026 (2).json», если такое имя уже занято."""
    path = Path(folder_dir) / base_name
    if not path.exists():
        return base_name
    stem = base_name[:-5] if base_name.endswith(".json") else base_name
    counter = 2
    while (Path(folder_dir) / ("%s (%d).json" % (stem, counter))).exists():
        counter += 1
    return "%s (%d).json" % (stem, counter)


def es_period(es: Elasticsearch, index_name: str) -> str:
    """Период данных датасета «ДД.ММ.ГГГГ-ДД.ММ.ГГГГ» по timeCreate."""
    try:
        res = es.search(index=index_name, body={"size": 0, "aggs": {
            "lo": {"min": {"field": "timeCreate"}}, "hi": {"max": {"field": "timeCreate"}}}})
        agg = res.get("aggregations") or {}
        lo = int((agg.get("lo") or {}).get("value") or 0)
        hi = int((agg.get("hi") or {}).get("value") or 0)
    except Exception:
        return ""
    if not (lo and hi):
        return ""
    first = datetime.datetime.fromtimestamp(lo, MSK).strftime("%d.%m.%Y")
    last = datetime.datetime.fromtimestamp(hi, MSK).strftime("%d.%m.%Y")
    return first if first == last else "%s-%s" % (first, last)


def human_name(theme: str, period: str) -> str:
    base = " ".join(part for part in (str(theme or "").strip(), period) if part)
    base = re.sub(r'[\\/:*?"<>|]+', "-", base).strip()
    return (base or "Тема") + ".json"


def rename_es(es: Elasticsearch, old: str, new: str, dry_run: bool) -> bool:
    """Переименовывает индекс Elasticsearch через _clone: индекс копируется целиком.

    Переиндексация документ-за-документом спотыкалась о поля с ignore_malformed
    («failed to parse field [geoObject.properties.external_id] of type [long]»), _clone
    копирует настройки, маппинг и данные без пересборки документов.
    """
    old_exists = es.indices.exists(index=old)
    new_exists = es.indices.exists(index=new)

    if not old_exists and new_exists:
        log("индекс уже переименован ранее: %s" % new)
        return True
    if not old_exists:
        log("индекса %s нет — пропускаю Elasticsearch" % old)
        return False

    old_count = es.count(index=old)["count"]
    if dry_run:
        log("[проба] Elasticsearch: %s → %s (документов %d)" % (old, new, old_count))
        return True

    if new_exists:
        # след неудачной попытки: целевой индекс есть, но данные в источнике — источник истины
        log("удаляю неполный индекс %s от прошлой попытки" % new)
        es.indices.delete(index=new, ignore=[400, 404])

    try:
        es.indices.put_settings(index=old, body={"index": {"blocks": {"write": True}}})
        es.indices.clone(index=old, target=new)
    except Exception as exc:  # noqa: BLE001
        log("_clone не сработал (%s: %s), перехожу к переиндексации" % (type(exc).__name__, str(exc)[:120]))
        try:
            es.indices.put_settings(index=old, body={"index": {"blocks": {"write": False}}})
        except Exception:  # noqa: BLE001
            pass
        es.indices.delete(index=new, ignore=[400, 404])
        src = es.indices.get(index=old)[old]
        es.indices.create(index=new, body={"settings": src.get("settings") or {},
                                           "mappings": src.get("mappings") or {}})
        es.reindex(body={"source": {"index": old}, "dest": {"index": new}},
                   wait_for_completion=True, request_timeout=7200)

    es.indices.refresh(index=new)
    new_count = es.count(index=new)["count"]
    if new_count != old_count:
        raise SystemExit("перенесено %d из %d документов — старый индекс не удаляю" % (new_count, old_count))
    try:
        es.indices.put_settings(index=old, body={"index": {"blocks": {"write": False}}})
    except Exception:  # noqa: BLE001
        pass
    es.indices.delete(index=old)
    log("Elasticsearch: %s → %s (документов %d), старый индекс удалён" % (old, new, new_count))
    return True


def rename_qdrant(old: str, new: str, dry_run: bool) -> bool:
    """Копирует коллекцию векторов <индекс>__bge под новым именем."""
    client = QdrantClient(host="localhost", port=6333)
    existing = {c.name for c in client.get_collections().collections}
    old_col, new_col = old + "__bge", new + "__bge"
    if old_col not in existing:
        log("коллекции Qdrant %s нет — пропускаю" % old_col)
        return False
    if new_col in existing:
        raise SystemExit("коллекция Qdrant %s уже существует" % new_col)
    info = client.get_collection(old_col)
    total = getattr(info, "points_count", 0) or 0
    if dry_run:
        log("[проба] Qdrant: %s → %s (точек %s)" % (old_col, new_col, total))
        return True
    vectors = info.config.params.vectors
    client.create_collection(collection_name=new_col, vectors_config=vectors)
    offset = None
    moved = 0
    while True:
        points, offset = client.scroll(collection_name=old_col, limit=256, offset=offset,
                                       with_payload=True, with_vectors=True)
        if points:
            # Собираем точки заново: объекты из scroll содержат служебные поля (order_value,
            # shard_key), и клиент отвергает их при повторной вставке.
            batch = [qmodels.PointStruct(id=point.id, vector=point.vector, payload=point.payload or {})
                     for point in points]
            client.upsert(collection_name=new_col, points=batch, wait=True)
            moved += len(batch)
        if offset is None:
            break
    client.delete_collection(old_col)
    log("Qdrant: %s → %s (точек %d), старая коллекция удалена" % (old_col, new_col, moved))
    return True


def rename_one(es: Elasticsearch, key: int, new_name: str, dry_run: bool = False) -> None:
    mapping = index_map()
    old_name = mapping.get(key) or mapping.get(str(key))
    if not old_name:
        raise SystemExit("датасет с номером %s не найден в реестре" % key)
    old_index = str(old_name)

    user_id = None
    folder = None
    file_old = None
    for candidate in sorted(os.listdir(DATA)):
        if not candidate.isdigit():
            continue
        found_folder, path, found_file = find_file(candidate, old_index)
        if found_folder:
            user_id, folder, file_old = candidate, found_folder, found_file
            break
    if not user_id:
        raise SystemExit("файл с индексом %s не найден ни у одного пользователя" % old_index)

    if not new_name.endswith(".json"):
        new_name += ".json"
    folder_dir = DATA / user_id / "json_files_directory" / folder
    if not dry_run:
        new_name = unique_name(folder_dir, new_name)
    new_index = dataset_index_name(new_name)
    log("датасет %s: %s (папка «%s», пользователь %s)" % (key, old_index, folder, user_id))
    log("новое имя: %s → индекс %s" % (new_name, new_index))
    target = folder_dir / new_name
    if target.exists() and target.name != file_old:
        raise SystemExit("файл %s уже существует" % target)

    if dry_run:
        log("[проба] файл: %s → %s" % (file_old, new_name))
        log("[проба] Redis, реестр — без изменений")
        rename_es(es, old_index, new_index, dry_run=True)
        rename_qdrant(old_index, new_index, dry_run=True)
        return

    # 1) индекс и векторы
    rename_es(es, old_index, new_index, dry_run=False)
    rename_qdrant(old_index, new_index, dry_run=False)

    # 2) файл данных
    source = folder_dir / file_old
    source.rename(target)
    log("файл переименован: %s" % target.name)

    # 3) Redis: имя файла в папке пользователя
    raw = REDIS.hget(str(user_id), "json_files_directory")
    data = json.loads(raw) if raw else {}
    files = data.get(folder) or []
    data[folder] = [new_name if str(item) == file_old else item for item in files]
    REDIS.hset(str(user_id), "json_files_directory", json.dumps(data, ensure_ascii=False))
    log("Redis обновлён: в папке «%s» теперь %s" % (folder, new_name))

    # 4) реестр датасетов
    mapping[key] = new_index
    save_index_map(mapping)
    log("готово: %s → %s" % (old_index, new_index))


def auto_names(es: Elasticsearch, user_id: str, dry_run: bool) -> None:
    """Переименовывает все «технические» имена (BA_*, с хешем) в «Тема период»."""
    mapping = index_map()
    raw = REDIS.hget(str(user_id), "json_files_directory")
    if not raw:
        raise SystemExit("у пользователя %s нет файлов данных" % user_id)
    folders = json.loads(raw) or {}
    for folder, files in folders.items():
        for file_name in list(files or []):
            stem = str(file_name)
            if not re.match(r"^(BA_|ba_)", stem) and "_6aa15d0f" not in stem:
                continue
            key = next((k for k, v in mapping.items() if str(v) == dataset_index_name(stem)), None)
            if key is None:
                log("датчик %s не найден в реестре — пропускаю" % stem)
                continue
            period = es_period(es, dataset_index_name(stem))
            target = human_name(folder, period)
            log("=== %s → %s" % (stem, target))
            try:
                rename_one(es, key, target, dry_run=dry_run)
            except SystemExit as exc:
                log("пропущено: %s" % exc)
            except Exception as exc:  # noqa: BLE001
                log("ошибка при переименовании %s: %s" % (stem, exc))


def main() -> None:
    parser = argparse.ArgumentParser(description="Переименование датасета")
    parser.add_argument("--key", type=int, help="номер датасета из реестра")
    parser.add_argument("--name", help="новое имя, например «Озон отзывы 01.09.2026-30.09.2026»")
    parser.add_argument("--auto", action="store_true", help="переименовать технические имена (BA_*, с хешем)")
    parser.add_argument("--user", default="1", help="пользователь для --auto (по умолчанию 1)")
    parser.add_argument("--list", action="store_true", help="показать датасеты, у которых имя техническое")
    parser.add_argument("--dry-run", action="store_true", help="только показать, что будет сделано")
    args = parser.parse_args()

    es = es_client()
    if args.list:
        mapping = index_map()
        for user in sorted(os.listdir(DATA)):
            if not user.isdigit():
                continue
            raw = REDIS.hget(user, "json_files_directory")
            for folder, files in (json.loads(raw) if raw else {}).items():
                for file_name in files or []:
                    stem = str(file_name)
                    if re.match(r"^(BA_|ba_)", stem) or "_6aa15d0f" in stem:
                        key = next((k for k, v in mapping.items() if str(v) == dataset_index_name(stem)), "—")
                        print("user %-3s | ключ %-5s | %-22s | %s" % (user, key, folder[:22], stem))
        return

    if args.auto:
        auto_names(es, str(args.user), args.dry_run)
        return
    if args.key and args.name:
        rename_one(es, args.key, args.name, dry_run=args.dry_run)
        return
    parser.print_help()


if __name__ == "__main__":
    main()
