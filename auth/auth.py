from fastapi_users.authentication import CookieTransport, AuthenticationBackend, BearerTransport
from fastapi_users.authentication import JWTStrategy

# cookie_transport = CookieTransport(cookie_name="analytics", cookie_max_age=3600)
bearer_transport = BearerTransport(tokenUrl="auth/jwt/login")

SECRET = "SECRET"

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
