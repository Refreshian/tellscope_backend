from datetime import datetime

from sqlalchemy import MetaData, Table, Column, Integer, String, TIMESTAMP, ForeignKey, JSON, Boolean
from sqlalchemy import *

metadata = MetaData()

role = Table(
    "role",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String, nullable=False),
    Column("permissions", JSON),
) 

user = Table(
    "user",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("email", String, nullable=False),
    Column("username", String, nullable=False),
    Column("registered_at", TIMESTAMP, default=datetime.utcnow),
    Column("role_id", Integer, ForeignKey(role.c.id)),
    Column("hashed_password", String, nullable=False),
    Column("is_active", Boolean, default=True, nullable=False),
    Column("is_superuser", Boolean, default=False, nullable=False),
    Column("is_verified", Boolean, default=False, nullable=False),
    Column("theme_rules", JSON),
)

embeddings = Table(
    "embeddings_pg",
    metadata,
    Column("id", Integer, primary_key=True, index=True),
    Column("user_id", Integer, nullable=False),  # Указан идентификатор пользователя
    Column("filename", String(255), nullable=False),  # Имя файла
    Column("folder_name", String(255), nullable=False),  # Имя папки
    Column("vectors", JSON, nullable=False),  # Поле для хранения эмбеддингов в формате JSON
)


tasks = Table(
    "tasks",
    metadata,
    Column("id", Integer, primary_key=True, index=True),
    Column("title", String, nullable=False),  # Добавил имя "title"
    Column("status", String, nullable=False),  # Добавил имя "status"
    Column("error", String, nullable=True),   # Добавил имя "error"
    Column("created_at", DateTime(timezone=True), server_default=func.now()),
    Column("updated_at", DateTime(timezone=True), onupdate=func.now()),
)

processing_results = Table(
    "processing_results",
    metadata,
    Column("id", Integer, primary_key=True, index=True),
    Column("task_id", Integer, ForeignKey("tasks.id"), nullable=False),
    Column("result_data", JSON, nullable=True),
    Column("created_at", DateTime(timezone=True), server_default=func.now()),
)