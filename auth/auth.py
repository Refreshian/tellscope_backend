from fastapi_users.authentication import CookieTransport, AuthenticationBackend, BearerTransport
from fastapi_users.authentication import JWTStrategy

# cookie_transport = CookieTransport(cookie_name="analytics", cookie_max_age=3600)
bearer_transport = BearerTransport(tokenUrl="auth/jwt/login")

import os as _os
from pathlib import Path as _Path

SECRET_FILE = _Path(_os.getenv("AUTH_SECRET_FILE") or "/home/dev/tellscope_app/tellscope_backend/data/auth_secret.key")


def _load_secret() -> str:
    """JWT-секрет: из AUTH_SECRET, иначе из файла (создаётся один раз, chmod 600)."""
    env = (_os.getenv("AUTH_SECRET") or "").strip()
    if env:
        return env
    try:
        if SECRET_FILE.exists():
            value = SECRET_FILE.read_text(encoding="utf-8").strip()
            if value:
                return value
    except Exception:
        pass
    import secrets
    value = secrets.token_urlsafe(48)
    try:
        SECRET_FILE.parent.mkdir(parents=True, exist_ok=True)
        SECRET_FILE.write_text(value, encoding="utf-8")
        _os.chmod(SECRET_FILE, 0o600)
    except Exception:
        pass
    return value


SECRET = _load_secret()

# Для access-токена (короткоживущий)
def get_jwt_strategy() -> JWTStrategy:
    return JWTStrategy(secret=SECRET, lifetime_seconds=36000)  # 1 час

# Для refresh-токена (долгоживущий)
def get_refresh_strategy() -> JWTStrategy:
    return JWTStrategy(secret=SECRET, lifetime_seconds=2592000)  # 30 дней

auth_backend = AuthenticationBackend(
    name="jwt",
    transport=bearer_transport,
    get_strategy=get_jwt_strategy,
)
