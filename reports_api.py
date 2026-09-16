# -*- coding: utf-8 -*-
"""Удаление файлов и папок во вкладке «Отчёты».

Эндпоинты (владелец или суперпользователь):

* ``DELETE /reports/file/{user_id}/{folder_name}/{file_name}`` — удалить файл отчёта;
* ``DELETE /reports/folder/{user_id}/{folder_name}`` — удалить папку отчётов
  (непустую — только с ``?force=true``, иначе 409).

Правила доступа и безопасности:

* без токена — 401; чужой ``user_id`` — 403 (суперпользователю можно);
* файл и папка обязаны быть прямыми потомками папки отчётов пользователя
  ``data/<user_id>/reports_directory`` — имена с ``/``, ``\\``, ``..`` и абсолютные
  пути отклоняются (400), симлинки за пределы папки тоже;
* удаление возможно только внутри папки отчётов: файлы датасетов, итогов и
  прочих каталогов пользователя этим API тронуть нельзя;
* нет файла/папки — 404; непустая папка без ``force`` — 409 с числом файлов.

Модуль самостоятельный (собственный ``APIRouter``), в ``main.py`` подключается
одной строкой ``app.include_router(reports_api_router)``.
"""
from __future__ import annotations

import os
import shutil
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Query, Request

router = APIRouter(tags=["reports"])

BACKEND_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BACKEND_ROOT, "data")
REPORTS_DIR_NAME = "reports_directory"

# Имя папки отчётов не может быть длиннее этого: так же ограничивает сборщик отчётов
# (agent_engine/tools_reports._safe_name), поэтому чужие имена сюда и так не попадают.
MAX_COMPONENT_LEN = 255


# --------------------------------------------------------------------------- #
# авторизация
# --------------------------------------------------------------------------- #
def _extract_token(request: Request) -> str:
    """Токен из заголовка Bearer либо из cookie (скачивание идёт обычной навигацией)."""
    auth = request.headers.get("authorization") or ""
    if auth[:7].lower() == "bearer ":
        return auth[7:].strip()
    for name in ("token", "access_token", "tellscope_refresh_token"):
        value = request.cookies.get(name)
        if value:
            return value.strip()
    return ""


async def _current_user(request: Request):
    """Пользователь по токену. Без токена или с недействительным — 401."""
    token = _extract_token(request)
    if not token or token.lower() in ("null", "undefined"):
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        import jwt as _jwt
        from auth.auth import SECRET as _SECRET

        payload = _jwt.decode(
            token, _SECRET, algorithms=["HS256"], options={"verify_aud": False}
        )
        user_id = int(payload.get("sub"))
    except Exception:
        raise HTTPException(status_code=401, detail="Unauthorized")

    from sqlalchemy import select

    from auth.database import User as AuthUser, async_session_maker

    async with async_session_maker() as session:
        user = (
            await session.execute(select(AuthUser).where(AuthUser.id == user_id))
        ).scalars().first()
    if user is None or not getattr(user, "is_active", False):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return user


def _guard_owner(user: Any, user_id: str) -> str:
    """Свой ``user_id`` или суперпользователь — иначе 403."""
    target = str(user_id or "").strip()
    if not target or not target.isdigit():
        raise HTTPException(status_code=400, detail="Некорректный user_id")
    if str(getattr(user, "id", "")) == target:
        return target
    if getattr(user, "is_superuser", False):
        return target
    raise HTTPException(status_code=403, detail="Нет доступа к отчётам другого пользователя")


# --------------------------------------------------------------------------- #
# пути и защита от выхода за пределы папки отчётов
# --------------------------------------------------------------------------- #
def _safe_component(value: Any, field: str) -> str:
    """Имя файла или папки без разделителей пути.

    Отклоняем ``..``, ``.``, абсолютные пути, разделители ``/`` и ``\\`` и ``\\0``:
    имя обязано быть ровно одним сегментом пути.
    """
    raw = str(value or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail=f"{field}: пустое имя")
    if "\x00" in raw:
        raise HTTPException(status_code=400, detail=f"{field}: недопустимый символ в имени")
    if len(raw) > MAX_COMPONENT_LEN:
        raise HTTPException(status_code=400, detail=f"{field}: имя слишком длинное")
    if raw in (".", ".."):
        raise HTTPException(status_code=400, detail=f"{field}: недопустимое имя")
    if os.path.isabs(raw) or os.path.splitdrive(raw)[0]:
        raise HTTPException(status_code=400, detail=f"{field}: абсолютные пути запрещены")
    unified = raw.replace("\\", "/")
    if "/" in unified:
        raise HTTPException(
            status_code=400, detail=f"{field}: имя не может содержать разделители пути"
        )
    if unified in (".", ".."):
        raise HTTPException(status_code=400, detail=f"{field}: недопустимое имя")
    return unified


def _user_reports_root(user_id: str) -> str:
    return os.path.join(DATA_ROOT, str(user_id), REPORTS_DIR_NAME)


def _resolve_folder_path(user_id: str, folder_name: Any) -> str:
    """Путь к папке отчётов пользователя. Папка обязана быть прямым потомком корня."""
    root = _user_reports_root(user_id)
    folder = _safe_component(folder_name, "folder")
    path = os.path.join(root, folder)
    real_root = os.path.realpath(root)
    real_path = os.path.realpath(path)
    if os.path.dirname(real_path) != real_root:
        raise HTTPException(status_code=400, detail="Папка вне каталога отчётов")
    return path


def _resolve_file_path(user_id: str, folder_name: Any, file_name: Any) -> str:
    """Путь к файлу внутри папки отчётов пользователя."""
    folder_path = _resolve_folder_path(user_id, folder_name)
    name = _safe_component(file_name, "file")
    path = os.path.join(folder_path, name)
    real_folder = os.path.realpath(folder_path)
    real_path = os.path.realpath(path)
    if os.path.dirname(real_path) != real_folder:
        raise HTTPException(status_code=400, detail="Файл вне папки отчётов")
    return path


# --------------------------------------------------------------------------- #
# эндпоинты
# --------------------------------------------------------------------------- #
@router.delete("/reports/file/{user_id}/{folder_name}/{file_name}")
async def delete_report_file(
    user_id: str,
    folder_name: str,
    file_name: str,
    request: Request,
) -> Dict[str, Any]:
    """Удаляет файл отчёта. 404 — файла нет, 400 — путь вне папки отчётов."""
    user = await _current_user(request)
    owner_id = _guard_owner(user, user_id)

    path = _resolve_file_path(owner_id, folder_name, file_name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Файл не найден")
    if os.path.isdir(path):
        raise HTTPException(
            status_code=400, detail="Это папка, а не файл — используйте удаление папки"
        )
    if not os.path.isfile(path):
        raise HTTPException(status_code=400, detail="Это не обычный файл")

    try:
        size = os.path.getsize(path)
    except OSError:
        size = 0
    try:
        os.remove(path)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Не удалось удалить файл: {exc}")

    return {
        "ok": True,
        "kind": "file",
        "user_id": owner_id,
        "folder": os.path.basename(os.path.dirname(path)),
        "name": os.path.basename(path),
        "size": size,
    }


@router.delete("/reports/folder/{user_id}/{folder_name}")
async def delete_report_folder(
    user_id: str,
    folder_name: str,
    request: Request,
    force: bool = Query(False, description="удалить папку вместе с содержимым"),
) -> Dict[str, Any]:
    """Удаляет папку отчётов.

    404 — папки нет; 409 — папка не пуста и ``force`` не передан (в ответе число
    файлов и первые имена); с ``force=true`` папка удаляется со всем содержимым.
    """
    user = await _current_user(request)
    owner_id = _guard_owner(user, user_id)

    path = _resolve_folder_path(owner_id, folder_name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Папка не найдена")
    if not os.path.isdir(path) or os.path.islink(path):
        raise HTTPException(status_code=400, detail="Это не папка отчётов")

    try:
        entries: List[str] = sorted(os.listdir(path))
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Не удалось прочитать папку: {exc}")

    if entries and not force:
        raise HTTPException(
            status_code=409,
            detail={
                "message": "Папка не пуста",
                "folder": os.path.basename(path),
                "count": len(entries),
                "files": entries[:20],
            },
        )

    removed = len(entries)
    try:
        if force:
            shutil.rmtree(path)
        else:
            os.rmdir(path)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Не удалось удалить папку: {exc}")

    return {
        "ok": True,
        "kind": "folder",
        "user_id": owner_id,
        "folder": os.path.basename(path),
        "removed_files": removed,
    }
