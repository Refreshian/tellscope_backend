# -*- coding: utf-8 -*-
"""Уровни доступа для маршрутов, которым нужен контроль владельца/роли.

Глобальный ``main.AuthGate`` проверяет только сам факт валидного токена (JWT или
сервисного), поэтому для приватных и деструктивных маршрутов нужны более точные
зависимости:

* ``current_user_any`` — любой активный пользователь; токен берётся из заголовка
  ``Authorization: Bearer`` либо из cookie (нужно для навигации браузера:
  ``iframe`` / ``window.open`` / ``fetch`` без заголовка);
* ``current_superuser_any`` — администратор (тот же приём с cookie): остальным 403;
* ``current_user_or_service`` — пользователь **или** владелец сервисного токена
  (``X-Service-Token``) — для машинных вызовов из Dify / n8n;
* ``current_service_or_superuser`` — служебные маршруты (метрики, промпты, сброс
  журнала задач): сервисный токен мониторинга либо администратор.

Модуль самостоятельный и не импортирует ``main`` (тот подключает роутеры сам),
поэтому его можно использовать и из ``mlops_api.py``.
"""
from __future__ import annotations

from typing import Any, Optional

from fastapi import Depends, HTTPException, Request
from sqlalchemy import select

from auth.auth import SECRET as AUTH_SECRET
from auth.database import User as AuthUser, async_session_maker

COOKIE_NAMES = ("token", "access_token", "tellscope_refresh_token")


def extract_token(request: Request) -> str:
    """JWT из заголовка Bearer, иначе из cookie (скачивание идёт обычной навигацией)."""
    auth = request.headers.get("authorization") or ""
    if auth[:7].lower() == "bearer ":
        token = auth[7:].strip()
        if token:
            return token
    for name in COOKIE_NAMES:
        value = (request.cookies.get(name) or "").strip()
        if value and value.lower() not in ("null", "undefined"):
            return value
    return ""


async def user_by_id(user_id: Any) -> Optional[Any]:
    """Активный пользователь по идентификатору (в т.ч. владелец сервисного токена)."""
    try:
        uid = int(user_id)
    except (TypeError, ValueError):
        return None
    try:
        async with async_session_maker() as session:
            user = (
                await session.execute(select(AuthUser).where(AuthUser.id == uid))
            ).scalars().first()
    except Exception:
        return None
    if user is None or not getattr(user, "is_active", False):
        return None
    return user


async def user_from_token(token: str) -> Optional[Any]:
    """Активный пользователь по JWT — иначе ``None``."""
    token = (token or "").strip()
    if not token or token.lower() in ("null", "undefined"):
        return None
    try:
        import jwt as _jwt

        payload = _jwt.decode(token, AUTH_SECRET, algorithms=["HS256"], options={"verify_aud": False})
        user_id = payload.get("sub")
    except Exception:
        return None
    return await user_by_id(user_id)


async def current_user_any(request: Request):
    """Активный пользователь по Bearer-заголовку ИЛИ cookie с токеном; иначе 401."""
    user = await user_from_token(extract_token(request))
    if user is None:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return user


async def current_superuser_any(user: Any = Depends(current_user_any)):
    """Администратор (токен из заголовка или cookie); обычному пользователю — 403."""
    if not getattr(user, "is_superuser", False):
        raise HTTPException(status_code=403, detail="Доступно только администратору")
    return user


def service_token_info(request: Request) -> Optional[dict]:
    """Данные владельца сервисного токена из ``X-Service-Token`` — иначе ``None``."""
    token = (request.headers.get("x-service-token") or "").strip()
    if not token:
        return None
    try:
        from agent_engine.service_api import match_service_token

        return match_service_token(token)
    except Exception:
        return None


async def current_user_or_service(request: Request):
    """JWT (заголовок/cookie) либо сервисный токен с существующим владельцем; иначе 401."""
    user = await user_from_token(extract_token(request))
    if user is not None:
        return user
    info = service_token_info(request)
    if info:
        owner = await user_by_id(info.get("user_id"))
        if owner is not None:
            return owner
    raise HTTPException(status_code=401, detail="Unauthorized")


async def current_service_or_superuser(request: Request):
    """Сервисный токен (мониторинг/инструменты) либо администратор; остальным 401/403."""
    info = service_token_info(request)
    if info:
        owner = await user_by_id(info.get("user_id"))
        return owner if owner is not None else {"id": info.get("user_id"), "service": info.get("name")}
    user = await user_from_token(extract_token(request))
    if user is None:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not getattr(user, "is_superuser", False):
        raise HTTPException(status_code=403, detail="Доступно только администратору или сервисному токену")
    return user
