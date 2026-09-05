import ast
import asyncio
import subprocess
from datetime import datetime, timedelta
from enum import Enum
import gc
import glob
import itertools
import re
import shutil
import tempfile
from typing import List, Optional, Union, Dict, Tuple
from collections import ChainMap, defaultdict
import time
from os import listdir 
from os.path import isfile, join 

import psutil
from fastapi.security import OAuth2PasswordBearer
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse

import aiofiles
from sklearn import manifold
from fastapi_users import fastapi_users, FastAPIUsers
import pandas as pd
from pydantic import BaseModel, Field, validator, ValidationError
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from fastapi import BackgroundTasks, FastAPI, File, Request, UploadFile, WebSocket, logger, status, Depends, WebSocketDisconnect
from fastapi.encoders import jsonable_encoder
# from fastapi.exceptions import ValidationError
from fastapi.responses import JSONResponse, HTMLResponse
import uvicorn
import numpy as np

import functools as ft
import io

import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from operator import itemgetter
import codecs, json

import websocket
from asyncio import Event

from auth.auth import auth_backend
from auth.auth import get_jwt_strategy, get_refresh_strategy, SECRET
from auth.database import User
from auth.manager import get_user_manager
from auth.schemas import UserRead, UserCreate
from fastapi.middleware.cors import CORSMiddleware 
from elasticsearch import Elasticsearch, helpers
import sys, json, os
from load_data_elastic import load_file_to_elstic
# from search_data_elastic import elastic_query
from operator import itemgetter
from transformers import AutoTokenizer, pipeline
import torch

import tensorflow_hub as hub
import tensorflow_text
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware

import jwt
from sqlalchemy.orm import Session 
from fastapi import HTTPException, status
from fastapi import FastAPI, Depends, Form
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.future import select
from fastapi_users.db import SQLAlchemyBaseUserTable
from sqlalchemy import Column, String, Boolean, Integer, TIMESTAMP, ForeignKey

from datetime import datetime
from typing import AsyncGenerator
from sqlalchemy.ext.declarative import DeclarativeMeta, declarative_base
from config import DB_HOST, DB_NAME, DB_PASS, DB_PORT, DB_USER
from model.models import role
from tensorflow.keras.preprocessing.sequence import pad_sequences

import tarfile
import time
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from torch import cuda

from torch import bfloat16
import transformers
from contextlib import asynccontextmanager

from umap import UMAP
from hdbscan import HDBSCAN
import gc
import torch, os, json
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, TextGeneration
from celery_app import celery_app

from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
from pathlib import Path
from PIL import Image
import joblib  # import pickle
import tensorflow as tf
from prometheus_fastapi_instrumentator import Instrumentator
from embedding_model_manager import model_manager

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Скрыть INFO и WARNING сообщения TensorFlow

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:50"
os.environ["SUNO_USE_SMALL_MODELS"] = "True"

# Локальный vLLM (OpenAI-compatible). Если VLLM_MODEL не задан — берётся первый id из GET {base}/v1/models
_VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000").rstrip("/")
VLLM_CHAT_COMPLETIONS_URL = f"{_VLLM_BASE_URL}/v1/chat/completions"
VLLM_MODELS_URL = f"{_VLLM_BASE_URL}/v1/models"
VLLM_MODEL_ENV = os.environ.get("VLLM_MODEL")
_VLLM_FALLBACK_MODEL_ID = "Qwen/Qwen3-32B-FP8"
 
DATABASE_URL = f"postgresql+asyncpg://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_async_engine(DATABASE_URL)
async_session_maker = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# Секретный ключ
SECRET_KEY = "SECRET"
ALGORITHM = "HS256"  # Указание алгоритма

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

import logging
# Настройка логирования для записи в файл
logging.basicConfig(filename='app.log', level=logging.INFO)

import redis.asyncio as redis
# redis_db = redis.StrictRedis(host="localhost", port=6379, db=0, decode_responses=True) # БД  для прогресс-бара с LLM расчетами
# Инициализация клиента Redis
redis_db = redis.Redis(host='localhost', port=6379, db=0)


es = Elasticsearch(
    hosts=["http://localhost:9200"],
    basic_auth=("elastic", "biz8z5i1w0nLPmEweKgP"),
    verify_certs=False,
    headers={"Accept": "application/vnd.elasticsearch+json; compatible-with=9"}
)

path_json_files = '/home/dev/tellscope_app/tellscope_backend/data/json_files'

torch.cuda.empty_cache() 
gc.collect()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

# Инициализация модели при запуске приложения
@asynccontextmanager
async def model_lifespan():
    try:
        print("Инициализация модели при запуске приложения...")
        model_manager.initialize_model()
        print("Модель успешно инициализирована при запуске")
        yield
    finally:
        print("Очистка модели при завершении приложения...")
        model_manager.cleanup()
        print("Модель успешно очищена")

@asynccontextmanager
async def redis_lifespan():
    try:
        await redis_db.ping()
        logging.info("Redis подключен!")
        existing_status = await redis_db.get("gpu:status")
        if not existing_status:
            logging.info("Инициализация статуса GPU как 'idle'.")
            await redis_db.set("gpu:status", "idle")
        yield
    finally:
        await redis_db.close()

@asynccontextmanager
async def combined_lifespan(app: FastAPI):
    async with model_lifespan():
        async with redis_lifespan():
            yield

app = FastAPI(
    title="Analytics App",
    lifespan=combined_lifespan
)

# Настройка CORS
origins = [
    "http://localhost",
    "http://localhost:5000",
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:4000",
    "http://localhost:5175",
    "http://194.146.113.124",
    "http://194.146.113.124:3000",
    "http://194.146.113.124:4000",
    "http://194.146.113.124:5000",
    "http://194.146.113.124:5173",
    "http://194.146.113.124:5175",
    "http://194.146.113.124:8000",
    "http://194.146.113.124:8080",
    "https://194.146.113.124",
    "https://194.146.113.124:4000",
    "https://localhost:4000",
    "https://tellscope.headsmade.com",
    "https://tellscope40.headsmade.com",  # ← ДОБАВЬТЕ ЭТО
    "https://tsdoc.headsmade.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"]
)

Instrumentator().instrument(app).expose(app)

fastapi_users = FastAPIUsers[User, int]( 
    get_user_manager,
    [auth_backend], 
)
 
### TonalityLandscape Models
class TonalityValues(BaseModel):
    negative_count: int
    positive_count: int

class NegativeHub(BaseModel):
    name: str
    values: int
    comments_sum: int
    likes_sum: int
    views_sum: int
    audience_sum: int

class PositiveHub(BaseModel):
    name: str
    values: int
    comments_sum: int
    likes_sum: int
    views_sum: int
    audience_sum: int

class ModelAuthorsTonalityLandscape(BaseModel):
    negative_hubs: List[NegativeHub]
    positive_hubs: List[PositiveHub]
    neutral_hubs: List[NegativeHub] = Field(default_factory=list)

class TextData(BaseModel):
    hub: str
    url: str
    er: Optional[int]
    viewsCount: Optional[Union[int, str]]
    commentsCount: Optional[Union[int, str]]
    audienceCount: Optional[Union[int, str]]
    likesCount: Optional[Union[int, str]]
    region: Optional[str] = None
    elastic_id: Optional[Union[int, str]]

class AuthorDatum(BaseModel):
    fullname: Optional[str]
    url: Optional[str]
    author_type: Optional[str]
    sex: Optional[str]
    age: Optional[int]
    count_texts: Optional[int]
    texts: List[TextData]
 
class ModeAuthorValues(BaseModel):
    author_data: List[AuthorDatum]

class Model_TonalityLandscape(BaseModel):
    tonality_values: TonalityValues
    tonality_hubs_values: ModelAuthorsTonalityLandscape
    negative_authors_values: List[ModeAuthorValues]
    positive_authors_values: List[ModeAuthorValues]
###=====###

### Information Graph Models
class AuthorInfGraph(BaseModel):
    fullname: str
    url: str
    author_type: str
    hub: Optional[str] = ''
    sex: str
    age: str
    audienceCount: int
    er: int
    viewsCount: Union[int, str]
    timeCreate: str
    es_id: Union[int, str]
    text: Optional[str] = ''
    commentsCount: Optional[int] = 0
    likesCount: Optional[int] = 0

    @validator("commentsCount", "likesCount", pre=True)
    def convert_engagement_counts(cls, value):
        if value in ('', '-', None):
            return 0
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return 0

    @validator("timeCreate", pre=True)
    def convert_time_create(cls, value):
        # если приходит int, приводим к строке
        if isinstance(value, int):
            return str(value)
        return value
    
    @validator("viewsCount", pre=True)
    def convert_views_count(cls, value):
        if isinstance(value, int):
            return str(value)
        return value
    
    @validator("audienceCount", pre=True)
    def convert_audience_count(cls, value):
        if value == '':
            return 0
        try:
            return int(value)
        except (ValueError, TypeError):
            return 0
    
    @validator("er", pre=True)
    def convert_er(cls, value):
        if value == '':
            return 0
        try:
            return int(value)
        except (ValueError, TypeError):
            return 0


class RepostInfGraph(BaseModel):
    fullname: str
    url: str
    author_type: str
    hub: Optional[str] = ''
    sex: str
    age: str
    audienceCount: int
    er: int
    viewsCount: str
    timeCreate: str
    es_id: Union[int, str]
    commentsCount: Optional[int] = 0
    likesCount: Optional[int] = 0

    @validator("audienceCount", pre=True)
    def convert_audience_count(cls, value):
        if value == '':
            return 0
        try:
            return int(value)
        except (ValueError, TypeError):
            return 0

    @validator("commentsCount", "likesCount", pre=True)
    def convert_repost_engagement(cls, value):
        if value in ('', '-', None):
            return 0
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return 0
    
    @validator("er", pre=True)
    def convert_er(cls, value):
        if value == '':
            return 0
        try:
            return int(value)
        except (ValueError, TypeError):
            return 0
    
    @validator("viewsCount", pre=True)
    def convert_views_count(cls, value):
        if isinstance(value, int):
            return str(value)
        return value
    
    @validator("timeCreate", pre=True)
    def convert_time_create(cls, value):
        if isinstance(value, int):
            return str(value)
        return value


class AuthorsStream(BaseModel):
    author: AuthorInfGraph
    reposts: Optional[List[RepostInfGraph]]

 
class ModelInfGraph(BaseModel):
    values: List[AuthorsStream]
    dynamicdata_audience: dict
    post: bool
    repost: bool
    SMI: bool
    num_messages: int 
    num_unique_authors: int


# Themes Model
class ThemesValues(BaseModel):
    description: str
    count: int
    audience: str
    er: str
    viewsCount: str
    texts: str


class ThemesModel(BaseModel):
    values: List[ThemesValues]

from typing import List, Optional, Any

# Customer Voice Model
class TonalityVoice(BaseModel):
    source: str
    Нейтрал: int
    Позитив: int
    Негатив: int
    elastic_id: List[Union[str, int]]

class SunkeyDatum(BaseModel):
    hub: str
    type: str
    tonality: str
    count: int
    search: str
    commentsCount: int
    audienceCount: int
    repostsCount: int
    viewsCount: int
    likesCount: int = 0
    author_type: str = ""
    elastic_id: Any # str или List[str]

class VoiceModel(BaseModel):
    name: str
    tonality: List[TonalityVoice]
    sunkey_data: List[SunkeyDatum]

class ModelVoice(BaseModel):
    values: List[VoiceModel]


# Mediarating Model
class NegativeSmiMediaRating(BaseModel):
    name: str
    index: int
    message_count: int
    elastic_id: Optional[Union[int, str]] = None  # Заменяем _id на id

class PositiveSmiMediaRating(BaseModel):
    name: str
    index: int
    message_count: int
    elastic_id: Optional[Union[int, str]] = None  # Заменяем _id на id


class FirstGraphMediaRating(BaseModel):
    negative_smi: List[NegativeSmiMediaRating]
    positive_smi: List[PositiveSmiMediaRating]


class SecondGraphItemMediaRating(BaseModel):
    name: str
    time: int
    index: int
    url: str
    color: str
    elastic_id: Union[int, str]  # Заменяем _id на id
    categoryName: Optional[str] = ""
    duplicateCount: Optional[int] = 1


class MediaRatingModel(BaseModel):
    first_graph: FirstGraphMediaRating
    second_graph: List[SecondGraphItemMediaRating]


class ModelItemAIAnalyticsNone(BaseModel):
    id: int
    timeCreate: int
    text: str
    hub: str
    audienceCount: int
    commentsCount: int
    er: int
    url: str

class ModelAiAnalyticsItem(BaseModel):
    id: int
    timeCreate: int
    text: str
    hub: str
    audienceCount: Optional[Union[int, str]] = None
    commentsCount: Optional[Union[int, str]] = None
    er: Optional[float] = None
    url: str

class ModelAiAnalytics(BaseModel):
    data: List[ModelAiAnalyticsItem]
    total_rows: int  # Добавляем новое поле


# class ModelAIPostAnalytics(BaseModel):
#     id: int
#     text: str
#     llm_text: str


# class ModelAIAnalyticsPost(BaseModel):
#     promt: str
#     texts: List[ModelAIPostAnalytics]


class QueryAiLLM(BaseModel):
    index: int=None
    min_date: int=None
    max_date: int=None
    promt: str = None
    texts_ids: list[int] = None


### Model Competitors
class QueryCompetitors(BaseModel):
    themes_ind: List[int] = Field(default_factory=list)
    min_date: Optional[int] = None
    max_date: Optional[int] = None


class FirstGraphItem(BaseModel):
    index_name: str
    values: List


class NegItem(BaseModel):
    hub: str
    audienceCount: int


class Po(BaseModel):
    hub: str
    audienceCount: int


class SMI(BaseModel):
    name: str
    neg: List[NegItem]
    pos: List[Po]


class SecondGraphItem(BaseModel):
    index_name: str
    SMI: SMI


class SMIItem(BaseModel):
    name: str
    count: int
    rating: Optional[int]


class SocmediaItem(BaseModel):
    name: str
    count: int
    rating: Optional[int]


class ThirdGraphItem(BaseModel):
    index_name: str
    SMI: List[SMIItem]
    Socmedia: List[SocmediaItem]


class CompetitorsModel(BaseModel):
    first_graph: List[FirstGraphItem]
    second_graph: List[SecondGraphItem]
    third_graph: List[ThirdGraphItem]


class DataFolder(BaseModel):
    name: str
    values: List[str]


class ModelDataFolder(BaseModel):
    values: List[DataFolder]

###=====###

app.include_router(
    fastapi_users.get_auth_router(auth_backend),
    prefix="/auth/jwt",
    tags=["auth"],
)

app.include_router(
    fastapi_users.get_register_router(UserRead, UserCreate),
    prefix="/auth",
    tags=["auth"], 
)


current_user = fastapi_users.current_user()

# indexes = {1: "rosbank_01.02.2024-07.02.2024", 2: "skillfactory_zaprosy_na_obuchenie_15.01.2024-21.01.2024", 3:'rosbank_19.02.2024-29.02.2024', 
#            4: "rosbank_14.03.2024-14.03.2024_fullday", 5: "r_13.03.2024-14.03.2024_full", 6: "rosbank_22.03.2024-24.03.2024", 
#            7: "monitoring_tem_19.03.2024-25.03.2024", 8: 'rosbank_26.03.2024-01.04.2024', 9: 'tehfob', 10: 'transport_01.01.2024-09.04.2024', 
#            11: 'moskovskiy_transport_01.01.2024_09.04.2024_2b', 12: 'rosbank_01.04.2024-15.04.2024', 13: 'rosbank_14.05.2024-16.05_чистая прибыль',
#            14: 'contented_smi_01.04.2024-26.05.2024', 15: 'skillbox_smi_01.04.2024-26.05.2024', 16: 'rb_smi', 17: 'geekbrains', 18: 'eduson', 
#            19: 'maley_nlmk_boevaya_tema_17.06.2024-21.06.2024_66757eb24cb15033866ecdd8', 20: 'maley_nlmk_boevaya_tema_17_06_2024_21_06_2024',
#            21: 'platon_test_31.07.2024-06.08.2024', 22: 'platon_test', 23: 'avtomobili_01.09.2023-02.09.2024', 24: 'cennosti_01.08.2024-31.08.2024', 
#            25: 'cennosti_01.07.2024-31.07.2024', 26: 'cennosti_data_year', 27: 'cennosti_data_year_without_doubles', 28: 'irkutsk', 
#            29: 'platon_22.11.2024-21.12.2024'}

# сохранение начального словаря всех файлов/тем
def load_dict_from_pickle(file_path):
    """Загружает словарь из pickle файла"""
    try:
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                return data if isinstance(data, dict) else {}
        else:
            logger.warning(f"Файл {file_path} не существует, возвращаем пустой словарь")
            return {}
    except Exception as e:
        logger.error(f"Ошибка при загрузке {file_path}: {str(e)}")
        return {}

def save_dict_to_pickle(file_path, data):
    """Сохраняет словарь в pickle файл"""
    try:
        # Создаем директорию если её нет
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Данные успешно сохранены в {file_path}")
        return True
    except Exception as e:
        logger.error(f"Ошибка при сохранении в {file_path}: {str(e)}")
        return False


def update_max_result_window(index_name: str, max_window: int = 1000000):
    try:
        es.indices.put_settings(
            index=index_name,
            body={"index": {"max_result_window": max_window}}
        )
    except Exception as e:
        print(f"Ошибка при обновлении настроек индекса '{index_name}': {e}")

def build_query(query_str: str, default_fields: List[str] = ["text", "Текст сообщения"]) -> dict:
    """
    Формирует сложный запрос для Эластика:
    - Если строка 'all' или пустая — match_all (все документы).
    - Если строка содержит ~N (пример: "инженер данных~3") — ищем фразу с расстоянием (slop).
    - Иначе — ищем все слова из запроса, независимо от порядка, с морфологией.
    Поддерживает поиск по нескольким полям (text и Текст сообщения).
    """
    if query_str is None or query_str.strip().lower() == "all":
        return {"match_all": {}}

    query_str = query_str.strip()
    # Фразовый поиск с расстоянием (пример "инженер данных~3")
    phrase_match = re.match(r'^(.*?)~(\d+)$', query_str)
    if phrase_match:
        phrase = phrase_match.group(1).strip()
        slop = int(phrase_match.group(2))
        return {
            "multi_match": {
                "query": phrase,
                "type": "phrase",
                "slop": slop,
                "fields": default_fields
            }
        }
    
    # Булевский AND для всех слов (морфология — предполагается статсномный анализатор на индексе)
    words = query_str.split()
    must_clauses = []
    for w in words:
        must_clauses.append({
            "multi_match": {
                "query": w,
                "fields": default_fields,
                "operator": "and"  # <= для поддержки русского можно опустить, если индекс морфологический
            }
        })
    return {"bool": {"must": must_clauses}}

def search_single_subquery(
    theme_index: str,
    query_str: str,
    min_date: Optional[int],
    max_date: Optional[int],
    scroll_time: str,
    batch_size: int,
    default_fields: List[str] = ["text", "Текст сообщения"]
) -> List[dict]:
    user_query = build_query(query_str, default_fields)
    es_query = {"query": user_query}

    # Фильтр по дате (если задан)
    if min_date is not None or max_date is not None:
        date_filter = {"range": {"timeCreate": {}}}
        if min_date is not None:
            date_filter['range']['timeCreate']['gte'] = min_date
        if max_date is not None:
            date_filter['range']['timeCreate']['lte'] = max_date

        es_query = {
            "query": {
                "bool": {
                    "must": user_query,
                    "filter": date_filter
                }
            }
        }
    try:
        response = es.search(
            index=theme_index,
            body=es_query,
            scroll=scroll_time,
            size=batch_size
        )
    except Exception as e:
        print(f"Ошибка при выполнении запроса: {e}")
        return []

    scroll_id = response.get('_scroll_id')
    results = response['hits']['hits']
    total_hits = response['hits']['total']['value'] if isinstance(response['hits']['total'], dict) else response['hits']['total']

    # Получаем все страницы scroll-батчей
    while True:
        try:
            response = es.scroll(scroll_id=scroll_id, scroll=scroll_time)
        except Exception as e:
            print(f"Ошибка при выполнении scroll-запроса: {e}")
            break

        hits = response['hits']['hits']
        if not hits:
            break
        results.extend(hits)
        scroll_id = response.get('_scroll_id')

    try:
        es.clear_scroll(scroll_id=scroll_id)
    except Exception:
        pass

    # Преобразуем к формату с _id внутри и нормализуем текстовое поле
    normalized_results = []
    for hit in results:
        doc = dict(**hit['_source'], _id=hit['_id'])
        # Нормализуем текстовое поле (объединяем оба варианта)
        if 'Текст сообщения' in doc and 'text' not in doc:
            doc['text'] = doc['Текст сообщения']
        elif 'text' in doc and 'Текст сообщения' not in doc:
            doc['Текст сообщения'] = doc['text']
        normalized_results.append(doc)
    
    return normalized_results

def elastic_query(
    theme_index: str,
    query_str: Optional[str] = None,  # делаем параметр опциональным с None по умолчанию
    min_date: Optional[int] = None,
    max_date: Optional[int] = None,
    scroll_time: str = '5m',
    batch_size: int = 10000,
    default_fields: List[str] = ["text", "Текст сообщения"]
) -> List[Dict]:
    """
    Выполняет поиск в индексе theme_index:
      - query_str: поисковая строка, поддерживает запятые как ИЛИ поиска ("one, two, three").
        Если None или пустая строка - возвращает все документы.
      - min_date, max_date — фильтрация по unix-таймштампу в поле timeCreate (опционально)
      - scroll_time, batch_size — параметры скроллинга
      - default_fields — поля для поиска (обычно ['text', 'Текст сообщения'], поля должны быть с русским анализатором)
    Возвращает: list[dict] — все найденные документы, каждый содержит _id и нормализованные текстовые поля.
    """
    update_max_result_window(theme_index)

    # Обработка случая, когда query_str is None или пустая строка
    if query_str is None or query_str.strip() == "":
        # Используем "all" как значение запроса, чтобы получить все документы
        subqueries = ["all"]
    # Разделяем на подзапросы по запятым, если есть
    elif "," in query_str:
        subqueries = [q.strip() for q in query_str.split(",")]
    else:
        subqueries = [query_str.strip()]
    
    all_results = {}
    total_found = 0

    for idx, subquery in enumerate(subqueries):
        if not subquery:  # пропускаем пустые подстроки после split
            continue
        data = search_single_subquery(
            theme_index,
            subquery,
            min_date=min_date,
            max_date=max_date,
            scroll_time=scroll_time,
            batch_size=batch_size,
            default_fields=default_fields
        )
        print(f"[{idx+1}/{len(subqueries)}] По выражению '{subquery}' найдено: {len(data)} документов")

        for item in data:
            all_results[item['_id']] = item  # переопределение ничего страшного, если дубль

        total_found += len(data)

    print(f"Без дубликатов найдено документов: {len(all_results)} (всего найдено {total_found})")
    return list(all_results.values())


@app.get("/tonality_landscape", tags=['data analytics'])
async def tonality_landscape(
    index: int = None,
    min_date: Optional[int] = None,
    max_date: Optional[int] = None
) -> Model_TonalityLandscape:
    file_path = '/home/dev/tellscope_app/tellscope_backend/data/indexes.pkl'
    indexes = load_dict_from_pickle(file_path)
    data = elastic_query(theme_index=indexes[index], min_date=min_date, max_date=max_date, query_str='all')

    # Преобразуем hub
    for entry in data:
        if 'hub' in entry:
            hub = entry['hub']
            if hub == 'telegram.org':
                entry['hub'] = 'telegram.me'
            elif hub == 'maps.yandex.ru':
                entry['hub'] = 'yandex.ru'
            elif hub == 'tinkoff.ru':
                entry['hub'] = 'tbank.ru'

    pos = [entry for entry in data if entry.get('toneMark') == 1]
    neg = [entry for entry in data if entry.get('toneMark') == -1]
    neutral = [entry for entry in data if int(entry.get('toneMark', 0) or 0) == 0]

    print(len(pos))
    print(len(neg))

    def aggregate_metrics(entries):
        metrics_by_hub = {}
        for entry in entries:
            hub = entry.get('hub', 'unknown')
            if hub not in metrics_by_hub:
                metrics_by_hub[hub] = {
                    "posts_count": 0,
                    "comments_sum": 0,
                    "likes_sum": 0,
                    "views_sum": 0,
                    "audience_sum": 0,
                }
            metrics_by_hub[hub]["posts_count"] += 1
            
            try:
                comments = int(entry.get('commentsCount', 0) or 0)
            except (TypeError, ValueError):
                comments = 0
                
            try:
                likes = int(entry.get('likesCount', 0) or 0)
            except (TypeError, ValueError):
                likes = 0
                
            try:
                views = int(entry.get('viewsCount', 0) or 0)
            except (TypeError, ValueError):
                views = 0
                
            try:
                audience = int(entry.get('audienceCount', 0) or 0)
            except (TypeError, ValueError):
                audience = 0
                
            metrics_by_hub[hub]["comments_sum"] += comments
            metrics_by_hub[hub]["likes_sum"] += likes
            metrics_by_hub[hub]["views_sum"] += views
            metrics_by_hub[hub]["audience_sum"] += audience
            
        return metrics_by_hub

    def prepare_hub_response(metrics_by_hub):
        return [
            {
                "name": hub,
                "values": metrics["posts_count"],
                "comments_sum": metrics["comments_sum"],
                "likes_sum": metrics["likes_sum"],
                "views_sum": metrics["views_sum"],
                "audience_sum": metrics["audience_sum"],
            }
            for hub, metrics in sorted(metrics_by_hub.items(), key=lambda x: x[1]["posts_count"], reverse=True)
        ]

    # ИСПРАВЛЕНИЕ: убираем дублирование
    # Используем только neg и pos, без добавления neg_authors и pos_authors
    neg_hub_metrics = aggregate_metrics(neg)
    pos_hub_metrics = aggregate_metrics(pos)

    neg_hub_response = prepare_hub_response(neg_hub_metrics)
    pos_hub_response = prepare_hub_response(pos_hub_metrics)
    neutral_hub_metrics = aggregate_metrics(neutral)
    neutral_hub_response = prepare_hub_response(neutral_hub_metrics)

    def process_author_object(entry):
        if 'authorObject' in entry and entry['authorObject']:
            author_obj = entry['authorObject']
            age_value = author_obj.get('age')
            try:
                if isinstance(age_value, str) and age_value.strip():
                    age_value = int(age_value)
                elif isinstance(age_value, (int, float)):
                    age_value = int(age_value)
                else:
                    age_value = None
            except Exception:
                age_value = None

            return {
                'fullname': author_obj.get('fullname', ''),
                'url': author_obj.get('url', '') or entry.get('author_url', '') or entry.get('url', ''),
                'author_type': author_obj.get('author_type', ''),
                'sex': author_obj.get('sex'),
                'age': age_value,
            }
        else:
            age_value = entry.get('age')
            if age_value is not None:
                try:
                    if isinstance(age_value, str) and age_value.strip():
                        age_value = int(age_value)
                    elif isinstance(age_value, (int, float)):
                        age_value = int(age_value)
                    else:
                        age_value = None
                except (ValueError, TypeError):
                    age_value = None
            author_type = entry.get('author_type')
            if author_type is None:
                hubtype = entry.get('hubtype')
                if hubtype:
                    author_type = hubtype
                else:
                    author_type = 'unknown'

            return {
                'fullname': entry.get('fullname', ''),
                'url': entry.get('author_url', '') or entry.get('url', ''),
                'author_type': author_type,
                'sex': entry.get('sex'),
                'age': age_value,
            }

    def build_text_item(entry):
        def safeint(x): 
            try: 
                return int(x) 
            except: 
                return 0

        comments_count = entry.get('commentsCount', 0)
        if comments_count is not None and comments_count != '':
            comments_count = safeint(comments_count)
        else:
            comments_count = 0
                
        audience_count = entry.get('audienceCount', 0)
        if audience_count is not None and audience_count != '':
            audience_count = safeint(audience_count)
        else:
            audience_count = 0

        likes_count = entry.get('likesCount', 0)
        if likes_count is not None and likes_count != '':
            likes_count = safeint(likes_count)
        else:
            likes_count = 0

        views_count = entry.get('viewsCount', 0)
        if views_count is not None and views_count != '':
            views_count = safeint(views_count)
        else:
            views_count = 0

        elastic_id = entry.get('_id')
        try:
            if isinstance(elastic_id, str) and elastic_id.strip():
                elastic_id = int(elastic_id)
        except Exception:
            pass

        return TextData(
            hub=entry.get('hub', ''),
            url=entry.get('url', ''),
            er=entry.get('er', 0),
            commentsCount=comments_count,
            audienceCount=audience_count,
            likesCount=likes_count,
            viewsCount=views_count,
            region=entry.get('region', ''),
            elastic_id=elastic_id,
        )

    def build_authors_groups(entries):
        """Группировать по (fullname + url) и сделать итоговый список объектов ModeAuthorValues"""
        groups = defaultdict(list)
        
        # ИСПРАВЛЕНИЕ: добавляем отслеживание уникальных elastic_id
        seen_ids = defaultdict(set)
        
        for entry in entries:
            author_obj = process_author_object(entry)
            author_id = (author_obj['fullname'], author_obj['url'])
            
            # Проверяем, не добавляли ли мы уже этот текст для данного автора
            elastic_id = entry.get('_id')
            if elastic_id not in seen_ids[author_id]:
                groups[author_id].append(entry)
                seen_ids[author_id].add(elastic_id)

        res = []
        author_data_list = []
        for author_id, texts in groups.items():
            author_obj = process_author_object(texts[0])
            texts_data = [build_text_item(entry) for entry in texts]
            author_data_list.append(
                AuthorDatum(
                    **author_obj,
                    count_texts=len(texts_data),
                    texts=texts_data
                )
            )
        
        for author_data in author_data_list:
            res.append(ModeAuthorValues(author_data=[author_data]))
        return res

    # ИСПРАВЛЕНИЕ: передаём только neg и pos, без дубликатов
    negative_authors_values = build_authors_groups(neg)
    positive_authors_values = build_authors_groups(pos)

    values = Model_TonalityLandscape(
        tonality_values=TonalityValues(
            negative_count=len(neg),
            positive_count=len(pos)
        ),
        tonality_hubs_values=ModelAuthorsTonalityLandscape(
            negative_hubs=neg_hub_response,
            positive_hubs=pos_hub_response,
            neutral_hubs=neutral_hub_response
        ),
        negative_authors_values=negative_authors_values,
        positive_authors_values=positive_authors_values,
    )
    return values


@app.get('/information_graph', tags=['data analytics'])
async def information_graph(# user: User = Depends(current_user),
                          index: int=None,
                          min_date: int=None, max_date: int=None, query_str: Optional[str] = 'карта',
                          post: Optional[bool] = None, repost: Optional[bool] = None,
                          SMI: Optional[bool] = None) -> ModelInfGraph:
    # Путь к файлу с темами
    file_path = '/home/dev/tellscope_app/tellscope_backend/data/indexes.pkl'
    # Загрузка словаря с темами
    indexes = load_dict_from_pickle(file_path)

    repost = bool(repost) if repost is not None else False
    post = bool(post) if post is not None else False
    SMI = bool(SMI) if SMI is not None else False
    repost_value = bool(repost) if repost is not None else False

    # делаем запрос на текстовый поиск
    data = elastic_query(theme_index=indexes[index], query_str=query_str)

    # отфильтровываем по необходимой дате из календаря
    data = [x for x in data if x['timeCreate'] is not None and min_date <= x['timeCreate'] <= max_date]
    num_messages = len(data)

    # предобработка данных
    df_meta = pd.DataFrame(data)

    count_vectorizer = CountVectorizer()
    vector_matrix = count_vectorizer.fit_transform(
        df_meta['text'].values)

    cosine_similarity_matrix = cosine_similarity(vector_matrix)

    dff = pd.DataFrame(cosine_similarity_matrix)

    val_dff = dff.values
    # заменяем значения по главной диагонали на 0
    for i in range(len(val_dff)):
        val_dff[i][i] = 0

    dff = pd.DataFrame(val_dff)

    # Обработка случая, когда в df_meta нет ключа 'authorObject'
    if 'authorObject' in df_meta.columns:
        author_data = pd.DataFrame(list(df_meta['authorObject'].values),
                                  columns=['fullname', 'text_url', 'author_type', 'sex', 'age'])
        df_meta = df_meta.join(author_data)
        # заменяем пустые fullname в СМИ на значения из hub
        df_meta['fullname'].fillna(df_meta['hub'], inplace=True)
    else:
        # Создаем необходимые столбцы, если их нет
        if 'fullname' not in df_meta.columns:
            df_meta['fullname'] = df_meta['Кто пишет'] if 'Кто пишет' in df_meta.columns else df_meta['hub']
        if 'author_type' not in df_meta.columns:
            df_meta['author_type'] = df_meta['Тип автора'] if 'Тип автора' in df_meta.columns else ''
        if 'sex' not in df_meta.columns:
            df_meta['sex'] = df_meta['Пол'] if 'Пол' in df_meta.columns else ''
        if 'age' not in df_meta.columns:
            df_meta['age'] = df_meta['Возраст'] if 'Возраст' in df_meta.columns else ''
        if 'er' not in df_meta.columns:
            df_meta['er'] = 0
        if 'viewsCount' not in df_meta.columns:
            df_meta['viewsCount'] = 0

    df = df_meta.copy()

    # создаем словарь похожих текстов с устранением дублирования
    fin_dict = {}
    threashhold = 0.8
    processed_indices = set()  # множество для отслеживания уже обработанных индексов

    # выявляем список строк с похожими текстами
    for i in range(dff.shape[0]):
        if i in processed_indices:
            continue
            
        similar_indices = list(np.where(dff.loc[i].values >= threashhold)[0])
        
        if similar_indices:
            # Добавляем текущий индекс и все похожие в обработанные
            processed_indices.add(i)
            processed_indices.update(similar_indices)
            
            # Основной автор - это тот, у кого наиболее ранняя дата (самое раннее время)
            candidates = [i] + similar_indices
            main_author_idx = min(candidates, key=lambda x: df_meta.loc[x, 'timeCreate'])
            
            # Репосты - это все остальные, отсортированные по времени
            reposts = [idx for idx in candidates if idx != main_author_idx]
            reposts.sort(key=lambda x: df_meta.loc[x, 'timeCreate'])
            fin_dict[main_author_idx] = reposts
        else:
            # Если нет похожих текстов, добавляем как отдельный элемент
            fin_dict[i] = []

    df_meta.fillna('', inplace=True)

    # Проверяем и создаем недостающие столбцы для модели Pydantic
    required_columns = ['id', 'fullname', 'url', 'author_type', 'hub', 'sex', 'age',
                       'audienceCount', 'er', 'viewsCount', 'commentsCount', 'likesCount', 'timeCreate', '_id']

    # Добавляем столбец 'id', если его нет в df_meta
    if 'id' not in df_meta.columns:
        df_meta['id'] = ''

    for col in required_columns:
        if col not in df_meta.columns:
            if col in ['audienceCount', 'er', 'viewsCount', 'commentsCount', 'likesCount']:
                df_meta[col] = 0
            else:
                df_meta[col] = ''

    # оставляем необходимую мету
    df_meta = df_meta[required_columns]

    # Сортируем fin_dict по времени основного автора для хронологического порядка
    sorted_fin_dict = dict(sorted(fin_dict.items(), key=lambda x: df_meta.loc[x[0], 'timeCreate']))

    # получение итогового массива данных с последовательностями авторов распространения информации и репостами (похожими текстами)
    data = []

    for key, val in sorted_fin_dict.items():
        author_dct = {}
        author_data = df_meta.loc[key].to_dict()

        # Преобразование числовых значений в строки для модели Pydantic
        if isinstance(author_data['age'], (int, float)) and not pd.isna(author_data['age']):
            author_data['age'] = str(author_data['age'])
        if isinstance(author_data['viewsCount'], (int, float)):
            author_data['viewsCount'] = str(author_data['viewsCount'])
        if isinstance(author_data['timeCreate'], (int, float)):
            author_data['timeCreate'] = str(author_data['timeCreate'])

        # Безопасное преобразование audienceCount и er в целые числа
        try:
            author_data['audienceCount'] = int(author_data['audienceCount']) if author_data['audienceCount'] not in ['', '-'] else 0
        except (ValueError, TypeError):
            author_data['audienceCount'] = 0
            
        try:
            author_data['er'] = int(author_data['er']) if author_data['er'] not in ['', '-'] else 0
        except (ValueError, TypeError):
            author_data['er'] = 0

        for _metric in ('commentsCount', 'likesCount'):
            try:
                raw = author_data.get(_metric, 0)
                author_data[_metric] = int(raw) if raw not in ['', '-', None] else 0
            except (ValueError, TypeError):
                author_data[_metric] = 0

        # Создаем структуру author для выходного формата
        author_struct = {
            "fullname": author_data['fullname'],
            "url": author_data['url'],
            "author_type": author_data['author_type'],
            "hub": author_data['hub'],
            "sex": author_data['sex'],
            "age": author_data['age'],
            "audienceCount": author_data['audienceCount'],
            "er": author_data['er'],
            "viewsCount": author_data['viewsCount'],
            "timeCreate": author_data['timeCreate'],
            "es_id": author_data['_id'],
            "commentsCount": author_data.get('commentsCount', 0),
            "likesCount": author_data.get('likesCount', 0),
            "text": str(df.loc[key, 'text'] if 'text' in df.columns else '')[:2500]
        }

        author_dct['author'] = author_struct
        author_dct['reposts'] = []

        if len(val) > 0:
            for i in range(len(val)):
                repost_data = df_meta.loc[val[i]].to_dict()

                # Преобразование числовых значений в строки для модели Pydantic
                if isinstance(repost_data['age'], (int, float)) and not pd.isna(repost_data['age']):
                    repost_data['age'] = str(repost_data['age'])
                if isinstance(repost_data['viewsCount'], (int, float)):
                    repost_data['viewsCount'] = str(repost_data['viewsCount'])
                if isinstance(repost_data['timeCreate'], (int, float)):
                    repost_data['timeCreate'] = str(repost_data['timeCreate'])

                # Безопасное преобразование audienceCount и er в целые числа
                try:
                    repost_data['audienceCount'] = int(repost_data['audienceCount']) if repost_data['audienceCount'] not in ['', '-'] else 0
                except (ValueError, TypeError):
                    repost_data['audienceCount'] = 0
                    
                try:
                    repost_data['er'] = int(repost_data['er']) if repost_data['er'] not in ['', '-'] else 0
                except (ValueError, TypeError):
                    repost_data['er'] = 0

                for _metric in ('commentsCount', 'likesCount'):
                    try:
                        raw = repost_data.get(_metric, 0)
                        repost_data[_metric] = int(raw) if raw not in ['', '-', None] else 0
                    except (ValueError, TypeError):
                        repost_data[_metric] = 0

                # Создаем структуру repost для выходного формата
                repost_struct = {
                    "fullname": repost_data['fullname'],
                    "url": repost_data['url'],
                    "author_type": repost_data['author_type'],
                    "hub": repost_data['hub'],
                    "sex": repost_data['sex'],
                    "age": repost_data['age'],
                    "audienceCount": repost_data['audienceCount'],
                    "er": repost_data['er'],
                    "viewsCount": repost_data['viewsCount'],
                    "timeCreate": repost_data['timeCreate'],
                    "es_id": repost_data['_id'],
                    "commentsCount": repost_data.get('commentsCount', 0),
                    "likesCount": repost_data.get('likesCount', 0)
                }

                author_dct['reposts'].append(repost_struct)

        data.append(author_dct)

    # Остальная часть кода для динамического графика остается без изменений
    ### данные для динамического графика
    def to_datetime(unixtime):
        return datetime.fromtimestamp(unixtime)

    df['timeCreate'] = df['timeCreate'].apply(to_datetime)
    df.sort_values(by='timeCreate', inplace=True)
    df.reset_index(inplace=True)
    if 'index' in df.columns:
        df.drop('index', axis=1, inplace=True)

    bins = pd.date_range(np.min(df['timeCreate'].values), np.max(df['timeCreate'].values), freq='600T') # по 10 минут

    df['cut'] = pd.cut(df['timeCreate'], bins, right=False)
    df = df.astype(str)
    df['cut'] = [x.replace('nan', str(bins[-1])) if x == 'nan' else x for x in df['cut'].values]
    df['cut'] = [x.split(',')[0].replace("[", '') for x in df['cut'].values]

    # мержинг данных на 10 минутки
    df_bins = pd.DataFrame(bins, columns=['cut']).astype(str).set_index('cut')
    df_bins['cut'] = list(df_bins.index)

    df = df_bins.set_index('cut').join(df.set_index('cut'))
    df.fillna('', inplace=True)

    df['timeCreate'] = list(df.index)
    df.reset_index(inplace=True)
    df.reset_index(inplace=True)
    df.drop(['index', 'cut'], axis=1, inplace=True)

    # Убедимся, что нужные столбцы существуют
    if 'hub' not in df.columns:
        df['hub'] = df['fullname'] if 'fullname' in df.columns else ''
    if 'audienceCount' not in df.columns:
        df['audienceCount'] = 0

    df = df[['hub', 'timeCreate', 'audienceCount']]

    # Заменяем строку с ошибкой на безопасное преобразование
    def safe_convert(x):
        try:
            if x == '' or x == '-':
                return 0
            return int(float(x))
        except (ValueError, TypeError):
            return 0

    df['audienceCount'] = [safe_convert(x) for x in df['audienceCount'].values]

    listhubs = [x for x in list(set(df['hub'].values)) if x != '']
    set_timeCreate = set(df['timeCreate'].values)

    # добавляем не заполненные N-минутки по источнику данными по времени и 0 по аудитории (т.е. в этот период 10 мин не было сообщ)
    for i in range(len(listhubs)):
        df_ban = df[df['hub'] == listhubs[i]]
        # недостающие временные отрезки
        delta_set = set_timeCreate - set(df_ban['timeCreate'].values)

        if delta_set != set():
            df_need = pd.DataFrame(zip([listhubs[i]]*len(delta_set), delta_set, [0]*len(delta_set)))
            df_need.columns = ['hub', 'timeCreate', 'audienceCount']
            df = pd.concat([df, df_need], ignore_index=True)

        else:
            df_need = pd.DataFrame(zip([listhubs[i]]*len(set_timeCreate), set_timeCreate, [0]*len(set_timeCreate)))
            df_need.columns = ['hub', 'timeCreate', 'audienceCount']
            df = pd.concat([df, df_need], ignore_index=True)

    df.sort_values(by='timeCreate', inplace=True)

    # подготовка итогового словаря с hub и аудиторией
    hub_dcts = []
    for hub in listhubs:
        hub_df = df[df['hub'] == hub][['timeCreate', 'audienceCount']]
        if not hub_df.empty:
            hub_dict = hub_df.set_index('timeCreate')['audienceCount'].to_dict()
            hub_dcts.append({hub: hub_dict})

    dynamicdata_audience = {}
    
    for hub_dict in hub_dcts:
        for hub, time_data in hub_dict.items():
            hub_data = {}
            cumulative_audience = 0
            for key, val in time_data.items():
                try:
                    unix_time = int(time.mktime(datetime.strptime(key, "%Y-%m-%d %H:%M:%S").timetuple()))
                    cumulative_audience += val
                    hub_data[str(unix_time)] = str(cumulative_audience)
                except Exception as e:
                    print(f"Error converting time for {key}: {e}")
            dynamicdata_audience[hub] = hub_data

    # Если dynamicdata_audience пуст, добавляем заглушку
    if not dynamicdata_audience:
        dynamicdata_audience = {"default": {"0": "0"}}

    # Подсчет количества сообщений
    print(f"Количество сообщений: {num_messages}")

    def count_unique_authors(data):
        authors = set()
        try:
            for item in data:
                # Добавляем автора из основного сообщения
                if 'author' in item and isinstance(item['author'], dict) and 'fullname' in item['author']:
                    authors.add(item['author']['fullname'])
                # Добавляем авторов из репостов
                if 'reposts' in item and isinstance(item['reposts'], list):
                    for repost in item['reposts']:
                        if isinstance(repost, dict) and 'fullname' in repost:
                            authors.add(repost['fullname'])
        except Exception as e:
            print(f"Error counting unique authors: {e}")
        
        return len(authors)

    # Обновленная функция подсчета уникальных авторов
    num_unique_authors = count_unique_authors(data)

    # Проверка на корректность boolean значение
    repost_value = bool(repost) if repost is not None else False

    if repost == False:
        repost = None

    # Формирование результата
    values = ModelInfGraph(
        values=data,
        post=post,
        repost=repost_value,
        SMI=SMI,
        dynamicdata_audience=dynamicdata_audience,
        num_messages=num_messages,
        num_unique_authors=num_unique_authors
    )

    return values


@app.get("/voice", tags=['data analytics'])
async def voice_analize(
    index: int = None,
    min_date: int = None,
    max_date: int = None,
    query_str: str = None
) -> ModelVoice:
    file_path = '/home/dev/tellscope_app/tellscope_backend/data/indexes.pkl'
    indexes = load_dict_from_pickle(file_path)
    raw = (query_str or "").strip()
    if raw.lower() in ("", "all", "yes", "none", "null"):
        terms = ["all"]
        labels = ["Все сообщения"]
    else:
        terms = [part.strip() for part in raw.split(",") if part.strip()] or ["all"]
        labels = ["Все сообщения" if term.lower() == "all" else term for term in terms]
    topn = 20
    values = []

    def _n(value):
        try:
            if value is None or value == "":
                return 0
            return int(float(value))
        except Exception:
            return 0

    def _tone(mark):
        text_mark = str(mark)
        if text_mark in ("1", "1.0"):
            return "Позитив"
        if text_mark in ("-1", "-1.0"):
            return "Негатив"
        return "Нейтрал"

    for term, label in zip(terms, labels):
        data = elastic_query(theme_index=indexes[index], query_str=term)
        data = [
            item for item in data
            if item.get("timeCreate") is not None and min_date <= item["timeCreate"] <= max_date
        ]
        for item in data:
            if "toneMark" not in item:
                item["toneMark"] = 0
            if not item.get("type"):
                item["type"] = "other"
            if item.get("hub") == "telegram.org":
                item["hub"] = "telegram.me"

        source_ids_by_tonality = defaultdict(lambda: defaultdict(list))
        grouped = defaultdict(lambda: {
            "count": 0,
            "comments": 0,
            "audience": 0,
            "reposts": 0,
            "views": 0,
            "likes": 0,
            "author_types": Counter(),
            "elastic_ids": [],
        })
        hub_counts = Counter()
        for item in data:
            hub = item.get("hub") or "unknown"
            typ = item.get("type") or "other"
            tone = _tone(item.get("toneMark", 0))
            _id = item.get("_id")
            author_type = (item.get("author_type") or "").strip() or "не указан"
            source_ids_by_tonality[hub][tone].append(_id)
            bucket = grouped[(hub, typ, tone)]
            bucket["count"] += 1
            bucket["comments"] += _n(item.get("commentsCount"))
            bucket["audience"] += _n(item.get("audienceCount"))
            bucket["reposts"] += _n(item.get("repostsCount"))
            bucket["views"] += _n(item.get("viewsCount"))
            bucket["likes"] += _n(item.get("likesCount"))
            bucket["author_types"][author_type] += 1
            if _id is not None:
                bucket["elastic_ids"].append(_id)
            hub_counts[hub] += 1

        dcts = []
        for source, tones in source_ids_by_tonality.items():
            ids = tones.get("Нейтрал", []) + tones.get("Позитив", []) + tones.get("Негатив", [])
            dcts.append({
                "source": source,
                "Нейтрал": len(tones.get("Нейтрал", [])),
                "Позитив": len(tones.get("Позитив", [])),
                "Негатив": len(tones.get("Негатив", [])),
                "elastic_id": ids,
            })

        list_topn_hubs = [hub for hub, _ in hub_counts.most_common(topn)]
        hub_tonality_type_list = []
        for (hub_val, type_val, tonality_val), bucket in grouped.items():
            if hub_val not in list_topn_hubs:
                continue
            author_type = bucket["author_types"].most_common(1)[0][0] if bucket["author_types"] else ""
            ids = bucket["elastic_ids"]
            hub_tonality_type_list.append({
                "hub": hub_val,
                "type": type_val,
                "tonality": tonality_val,
                "count": bucket["count"],
                "search": label,
                "commentsCount": bucket["comments"],
                "audienceCount": bucket["audience"],
                "repostsCount": bucket["reposts"],
                "viewsCount": bucket["views"],
                "likesCount": bucket["likes"],
                "author_type": author_type,
                "elastic_id": ids if len(ids) != 1 else (ids[0] if ids else []),
            })
        hub_tonality_type_list = sorted(hub_tonality_type_list, key=lambda row: row["count"], reverse=True)
        values.append({
            "name": label,
            "tonality": dcts,
            "sunkey_data": hub_tonality_type_list,
        })

    return ModelVoice(values=values)


@app.get("/media-rating", tags=["data analytics"])
def media_rating(index: int = None, min_date: int = None, max_date: int = None) -> MediaRatingModel:
    # 1. Загружаем словарь с темами
    file_path = "/home/dev/tellscope_app/tellscope_backend/data/indexes.pkl"
    indexes = load_dict_from_pickle(file_path)

    # 2. Запрашиваем из Elasticsearch
    data = elastic_query(theme_index=indexes[index], query_str="all") 

    print(777999)
    print(f'data: {data[:2]})')
    # Оставляем только записи с timeCreate в нужном диапазоне
    data = [
        x for x in data
        if x.get("timeCreate") is not None and min_date <= x["timeCreate"] <= max_date
    ]

    # 3. Собираем DataFrame и нормализуем citeIndex
    df = pd.DataFrame(data)
    if "_id" not in df.columns:
        df["_id"] = df.index.astype(str)  # на случай, если _id отсутствует
    if "citeIndex" in df.columns:
        df["citeIndex"] = df["citeIndex"].apply(lambda x: 0 if x == "" else x)

    # 4. Собираем общий df_meta
    #    Мы делаем два варианта: только СМИ (нет hubtype) и когда hubtype есть
    df_meta = pd.DataFrame()

    if "hubtype" not in df.columns:
        # только СМИ
        dff = df.copy()
        dff["timeCreate"] = dff["timeCreate"].apply(
            lambda ts: datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
        )
        # типовой набор колонок
        available = ["timeCreate", "hub", "url", "text"]
        dff.setdefault("toneMark", None)
        available.append("toneMark")
        dff["audience"] = dff.get("audienceCount", 0)
        available.append("audience")
        if "citeIndex" not in dff.columns:
            dff["citeIndex"] = dff["audience"]
        available.append("citeIndex")
        # обязательно захватываем _id
        available.append("_id")
        for extra in ("categoryName", "duplicateCount"):
            if extra in dff.columns:
                available.append(extra)

        df_meta_smi_only = dff[available].copy()
        df_meta_smi_only["fullname"] = df_meta_smi_only["hub"]
        df_meta_smi_only["author_type"] = "Онлайн-СМИ"
        df_meta_smi_only["hubtype"] = "Онлайн-СМИ"
        df_meta_smi_only["type"] = "Онлайн-СМИ"
        df_meta_smi_only["er"] = 0
        df_meta_smi_only.dropna(subset=["timeCreate"], inplace=True)
        df_meta_smi_only = df_meta_smi_only.set_index("timeCreate")
        df_meta_smi_only["date"] = df_meta_smi_only.index
        df_meta = df_meta_smi_only

    else:
        # есть hubtype — соцмедиа и СМИ
        parts = []
        # 4.1. Соцмедиа (hubtype != Онлайн-СМИ)
        socm = df[df["hubtype"] != "Онлайн-СМИ"]
        if not socm.empty:
            socm = socm.copy()
            socm["timeCreate"] = socm["timeCreate"].apply(
                lambda ts: datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
            )
            socm_df = socm[[
                "timeCreate", "hub", "toneMark", "audienceCount",
                "url", "er", "hubtype", "text", "_id"
            ]].copy()
            socm_df["fullname"] = pd.json_normalize(socm["authorObject"])["fullname"]
            socm_df["author_type"] = pd.json_normalize(socm["authorObject"])["author_type"]
            socm_df.dropna(subset=["timeCreate"], inplace=True)
            socm_df = socm_df.set_index("timeCreate")
            socm_df["date"] = socm_df.index.str[:10]
            parts.append(socm_df)

        # 4.2. Онлайн-СМИ (hubtype == Онлайн-СМИ)
        smi = df[df["hubtype"] == "Онлайн-СМИ"]
        if not smi.empty:
            smi = smi.copy()
            smi["timeCreate"] = smi["timeCreate"].apply(
                lambda ts: datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
            )
            smi_cols = [
                "timeCreate", "hub", "toneMark", "audienceCount",
                "url", "er", "hubtype", "text", "citeIndex", "_id"
            ]
            for extra in ("categoryName", "duplicateCount"):
                if extra in smi.columns:
                    smi_cols.append(extra)
            smi_df = smi[smi_cols].copy()
            smi_df["fullname"] = smi_df["hub"]
            smi_df["author_type"] = "Онлайн-СМИ"
            smi_df["type"] = "Онлайн-СМИ"
            smi_df.dropna(subset=["timeCreate"], inplace=True)
            smi_df = smi_df.set_index("timeCreate")
            smi_df["date"] = smi_df.index.str[:10]
            parts.append(smi_df)

        # объединяем
        df_meta = pd.concat(parts, axis=0)

    # дополнительная фильтрация для telegram.org, если надо
    if set(df_meta["hub"].unique()) == {"telegram.org"}:
        df_meta = df_meta[
            (df_meta["hubtype"] == "Мессенджеры каналы") &
            (df_meta["hub"] == "telegram.org")
        ]

    # 5. Для мессенджерных каналов строим bobble — но он нам потом перезапишется для Онлайн‑СМИ
    # (просто для примера оставил код, но он будет перезаписан ниже)
    # ...

    # 6. Обрабатываем только Онлайн‑СМИ для построения first_graph и второго bobble
    df_online = df_meta[df_meta["hubtype"] == "Онлайн-СМИ"].copy()

    # 6.1. Отрицательные онлайн-СМИ
    neg_df = df_online[df_online["toneMark"] == -1]
    # группируем по hub и собираем list citeIndex
    dict_neg = defaultdict(list)
    for _, row in neg_df.iterrows():
        dict_neg[row["hub"]].append(int(row.get("citeIndex", 0)))

    # приводим к уникальному int
    neg_smi = []
    for hub, cites in dict_neg.items():
        uniq_cites = set(cites)
        # из примера вы берёте именно одно число из этого списка
        idx_value = max(uniq_cites) if uniq_cites else 0
        count_msgs = len(cites)
        neg_smi.append((hub, idx_value, count_msgs))

    # 6.2. Положительные онлайн-СМИ
    pos_df = df_online[df_online["toneMark"] == 1]
    dict_pos = defaultdict(list)
    for _, row in pos_df.iterrows():
        dict_pos[row["hub"]].append(int(row.get("citeIndex", 0)))

    pos_smi = []
    for hub, cites in dict_pos.items():
        uniq_cites = set(cites)
        idx_value = max(uniq_cites) if uniq_cites else 0
        count_msgs = len(cites)
        pos_smi.append((hub, idx_value, count_msgs))

    # 7. Сортируем и берём топ‑1000
    max_size = 1000
    neg_smi_sorted = sorted(neg_smi, key=lambda x: x[1], reverse=True)[:max_size]
    pos_smi_sorted = sorted(pos_smi, key=lambda x: x[1], reverse=True)[:max_size]

    # 8. Добавляем в каждый элемент _id: находим в df_online первую запись с таким hub и citeIndex
    first_negative = []
    for hub, idx_val, msg_cnt in neg_smi_sorted:
        # Ищем все записи с таким hub и citeIndex
        matches = df_online[
            (df_online["hub"] == hub) & 
            (df_online["citeIndex"] == idx_val)
        ]
        elastic_id = None
        if not matches.empty:
            elastic_id = matches.iloc[0]["_id"]
        
        first_negative.append(NegativeSmiMediaRating(
            name=hub,
            index=idx_val,
            message_count=msg_cnt,
            elastic_id=elastic_id  # Добавляем elastic_id
        ))

    first_positive = []
    for hub, idx_val, msg_cnt in pos_smi_sorted:
        # Ищем все записи с таким hub и citeIndex
        matches = df_online[
            (df_online["hub"] == hub) & 
            (df_online["citeIndex"] == idx_val)
        ]
        elastic_id = None
        if not matches.empty:
            elastic_id = matches.iloc[0]["_id"]
        
        first_positive.append(PositiveSmiMediaRating(
            name=hub,
            index=idx_val,
            message_count=msg_cnt,
            elastic_id=elastic_id  # Добавляем elastic_id
        ))

    # 9. Строим second_graph (bobble) для Онлайн‑СМИ
    # вытягиваем все ненулевые toneMark
    ton_df = df_online[df_online["toneMark"] != 0].copy()
    # переводим дату в миллисекунды
    times = ton_df.index.to_series().apply(
        lambda ts: int((datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") -
                        datetime(1970, 1, 1)).total_seconds() * 1000)
    ).tolist()
    bobble = []
    for (i, (_, row)) in enumerate(ton_df.iterrows()):
        color = "#32ff32" if row["toneMark"] == 1 else "#FF3232"
        try:
            id = int(row["_id"])
        except:
            id = row["_id"]
        cat = row["categoryName"] if "categoryName" in row.index else ""
        if pd.isna(cat):
            cat = ""
        dup = row["duplicateCount"] if "duplicateCount" in row.index else 1
        try:
            dup = int(dup) if pd.notna(dup) else 1
        except Exception:
            dup = 1
        if dup < 1:
            dup = 1
        bobble.append({
            "name": row["hub"],
            "time": times[i],
            "index": int(row.get("citeIndex", 0) or 0),
            "url": row["url"],
            "color": color,
            "elastic_id": id,
            "categoryName": str(cat) if cat else "",
            "duplicateCount": dup,
        })

    # 10. Собираем итог и возвращаем
    values = {
        "first_graph": {
            "negative_smi": first_negative,
            "positive_smi": first_positive
        },
        "second_graph": bobble
    }
    
    # print(values)

    return MediaRatingModel(
        first_graph=values["first_graph"],
        second_graph=values["second_graph"]
    )


@app.get('/ai-analytics', tags=['ai analytics'])
async def ai_analytics_get(
    index: int = None,
    min_date: int = None,
    max_date: int = None,
    query_str: str = None
) -> ModelAiAnalytics:
    file_path = '/home/dev/tellscope_app/tellscope_backend/data/indexes.pkl'
    indexes = load_dict_from_pickle(file_path)
    
    # Получаем все данные (без фильтрации по дате сначала)
    all_data = elastic_query(theme_index=indexes[index], query_str=query_str)
    
    # Фильтруем по дате и считаем общее количество
    filtered_data = [x for x in all_data if x['timeCreate'] is not None and min_date <= x['timeCreate'] <= max_date]
    total_rows = len(filtered_data)  # Общее количество строк после фильтрации
    
    # Ограничение на 10 000 записей
    if len(filtered_data) > 10000:
        filtered_data = filtered_data[:10000]
    
    # Подготовка данных
    keys = ['id', 'timeCreate', 'text', 'hub', 'audienceCount', 'commentsCount', 'er', 'url']
    data = [{k: y.get(k, None) for k in keys} for y in filtered_data]
    ranges = list(np.arange(0, len(data)))
    [x.update({'id': y.item()}) for x, y in zip(data, ranges)]

    return ModelAiAnalytics(
        data=data,
        total_rows=total_rows  # Возвращаем общее количество строк
    )


# Определение модели запроса
class QueryCompetitors(BaseModel):
    themes_ind: list
    min_date: int
    max_date: int


class ValueCompetitor(BaseModel):
    timestamp: int
    count: int


class FirstGraphCompetitor(BaseModel):
    index_name: str
    values: List[ValueCompetitor]


class NegItem(BaseModel):
    hub: str
    count: int
    rating: int
    url: str


class Po(BaseModel):
    hub: str
    count: int
    rating: Union[int, str]
    url: str


class SMICompetitor(BaseModel):
    neg: List[NegItem]
    pos: List[Po]


class Po1(BaseModel):
    hub: str
    count: int
    rating: int
    url: str


class SocmediaCompetitor(BaseModel):
    neg: List[NegItem]
    pos: List[Po1]


class SecondGraphCompetitor(BaseModel):
    index_name: str
    SMI: SMICompetitor
    Socmedia: SocmediaCompetitor


class SMIItem(BaseModel):
    name: str
    count: int
    rating: Union[int, str]
    url: str


class SocmediaItem(BaseModel):
    name: str
    count: int
    rating: int
    url: str


class ThirdGraphCompetitor(BaseModel):
    index_name: str
    SMI: List[SMIItem]
    Socmedia: List[SocmediaItem]


class CompetitorsModel(BaseModel):
    first_graph: List[FirstGraphCompetitor]
    second_graph: List[SecondGraphCompetitor]
    third_graph: List[ThirdGraphCompetitor]


@app.post('/competitors', response_model=CompetitorsModel, tags=['data analytics'])
async def competitors(query: QueryCompetitors): # , user: User = Depends(current_user)
    # Путь к файлу с темами
    file_path = '/home/dev/tellscope_app/tellscope_backend/data/indexes.pkl'
    indexes = load_dict_from_pickle(file_path)

    another_graph = []
    min_date = []
    max_date = []
    themes_ind = query.themes_ind

    # Обработка данных для каждого theme_ind
    for i in range(len(themes_ind)):
        data = elastic_query(theme_index=indexes[themes_ind[i]], query_str='all')

        # Проверяем состав данных и дополняем отсутствующие поля
        ind_df = []
        for x in data:
            # Создаем новый элемент с нужными полями
            item = {}
            # Копируем имеющиеся ключи
            for k, v in x.items():
                if k == "audience":
                    item["audienceCount"] = v
                else:
                    item[k] = v
            
            # Добавляем отсутствующие поля
            if "audienceCount" not in item:
                item["audienceCount"] = 0
            
            if "hubtype" not in item:
                # Предполагаем, что все ресурсы без hubtype - это "Онлайн-СМИ"
                item["hubtype"] = "Онлайн-СМИ"
            
            if "citeIndex" not in item:
                # Используем audienceCount вместо citeIndex, если его нет
                item["citeIndex"] = item["audienceCount"]
            
            if "toneMark" not in item:
                # Если нет тональности, предполагаем нейтральную (0)
                item["toneMark"] = 0
            
            ind_df.append(item)

        # Формирование цензури для SMI
        for item in ind_df:
            if item['hubtype'] == 'Онлайн-СМИ':
                item['rating'] = item.get('citeIndex', 0)
            else:
                item['rating'] = item.get('audienceCount', 0)

        min_date.append(np.min([x['timeCreate'] for x in ind_df]))
        max_date.append(np.max([x['timeCreate'] for x in ind_df]))
        another_graph.append(ind_df)

    # Получение общей мин и макс даты
    dates = [min_date, max_date]
    min_date = np.min(dates[0])
    max_date = np.max(dates[1])
    filenames = [indexes[x] for x in themes_ind]

    # Формирование первого графика
    first_graph = []
    for theme_data, filename in zip(another_graph, filenames):
        df = pd.DataFrame(theme_data)
        df['timeCreate'] = pd.to_datetime(df['timeCreate'], unit='s')
        min_date_dt = pd.to_datetime(min_date, unit='s')
        max_date_dt = pd.to_datetime(max_date, unit='s')
        df['bins'] = pd.cut(df['timeCreate'], pd.date_range(min_date_dt, max_date_dt, freq='30T'))
        aggregated_data = df.groupby('bins').size().reset_index(name='count')
        aggregated_data['time'] = aggregated_data['bins'].apply(lambda x: x.left.timestamp())

        first_graph.append({
            'index_name': filename,
            'values': [{'timestamp': int(row.time * 1000), 'count': row.count} for row in aggregated_data.itertuples()]
        })

    # Функция для безопасного преобразования в int
    def safe_to_int(value):
        try:
            # Преобразуем в целое число
            return int(value)
        except (ValueError, TypeError):  
            # Если преобразование невозможно, возвращаем 0
            return 0

    # Формирование второго графика (second_graph)
    second_graph = []
    for theme_data, filename in zip(another_graph, filenames):
        df = pd.DataFrame(theme_data)

        # Добавляем проверку наличия колонки toneMark
        if 'toneMark' not in df.columns:
            df['toneMark'] = 0  # Используем нейтральную тональность по умолчанию

        # Данные только по SMI (hubtype == 'Онлайн-СМИ')
        smi_data = df[df['hubtype'] == 'Онлайн-СМИ']

        neg_smi = smi_data[smi_data['toneMark'] == -1].groupby('hub').agg(
            count=('hub', 'size'),
            citeIndex=('citeIndex', 'first'),
            url=('url', 'first')
        ).reset_index()

        pos_smi = smi_data[smi_data['toneMark'] == 1].groupby('hub').agg(
            count=('hub', 'size'),
            citeIndex=('citeIndex', 'first'),
            url=('url', 'first')
        ).reset_index()
        
        # Обработка данных SMI
        second_graph.append({
            'index_name': filename,
            'SMI': {
                'neg': [
                    {
                        'hub': row['hub'],
                        'count': row['count'],
                        'rating': safe_to_int(row['citeIndex']),
                        'url': row['url']
                    }
                    for _, row in neg_smi.iterrows()
                ],
                'pos': [
                    {
                        'hub': row['hub'],
                        'count': row['count'],
                        'rating': safe_to_int(row['citeIndex']),
                        'url': row['url']
                    }
                    for _, row in pos_smi.iterrows()
                ],
            }
        })

        # Данные только по Соцмедиа (hubtype != 'Онлайн-СМИ')
        socmedia_data = df[df['hubtype'] != 'Онлайн-СМИ']

        # Если нет соцмедиа данных (в новом формате), добавляем пустые списки
        if len(socmedia_data) == 0:
            second_graph[-1]['Socmedia'] = {
                'neg': [],
                'pos': []
            }
        else:
            neg_socmedia = socmedia_data[socmedia_data['toneMark'] == -1].groupby('hub').agg(
                count=('hub', 'size'),
                audienceCount=('audienceCount', 'first'),
                url=('url', 'first')
            ).reset_index()

            pos_socmedia = socmedia_data[socmedia_data['toneMark'] == 1].groupby('hub').agg(
                count=('hub', 'size'),
                audienceCount=('audienceCount', 'first'),
                url=('url', 'first')
            ).reset_index()

            # Обработка данных Socmedia
            second_graph[-1]['Socmedia'] = {
                'neg': [
                    {
                        'hub': row['hub'],
                        'count': row['count'],
                        'rating': safe_to_int(row['audienceCount']),
                        'url': row['url']
                    }
                    for _, row in neg_socmedia.iterrows()
                ],
                'pos': [
                    {
                        'hub': row['hub'],
                        'count': row['count'],
                        'rating': safe_to_int(row['audienceCount']),
                        'url': row['url']
                    }
                    for _, row in pos_socmedia.iterrows()
                ],
            }

    # Формирование третьего графика (third_graph)
    third_graph = []
    for theme_data, filename in zip(another_graph, filenames):
        df = pd.DataFrame(theme_data)

        # SMI данные
        df_smi = df[df['hubtype'] == 'Онлайн-СМИ']
        smi_data = df_smi.groupby('hub').agg(
            hub_count=('hub', 'size'),
            citeIndex=('citeIndex', 'first'),
            url=('url', 'first')
        ).reset_index()

        smi_results = [{
            'name': row['hub'],
            'count': row['hub_count'],
            'rating': safe_to_int(row['citeIndex']),
            'url': row['url']
        } for _, row in smi_data.iterrows()]

        # Данные Socmedia
        third_graph_item = {
            'index_name': filename,
            'SMI': smi_results,
        }
        
        # Для нового формата данных, если нет соцмедиа, добавляем пустой список
        df_socmedia = df[df['hubtype'] != 'Онлайн-СМИ']
        if len(df_socmedia) == 0:
            third_graph_item['Socmedia'] = []
        else:
            socmedia_data = df_socmedia.groupby('hub').agg(
                hub_count=('hub', 'size'),
                audienceCount=('audienceCount', 'first'),
                url=('url', 'first')
            ).reset_index()

            socmedia_results = [{
                'name': row['hub'],
                'count': row['hub_count'],
                'rating': safe_to_int(row['audienceCount']),
                'url': row['url']
            } for _, row in socmedia_data.iterrows()]
            
            third_graph_item['Socmedia'] = socmedia_results
            
        third_graph.append(third_graph_item)

    return {
        'first_graph': first_graph,
        'second_graph': second_graph,
        'third_graph': third_graph,
    }


class LCAExamplesRequest(BaseModel):
    """Запрос для получения синтетических примеров по типу/подтипу метафоры."""
    frame_type: Optional[str] = None
    frame_subtype: Optional[str] = None
    limit_per_cluster: int = 4
    custom_topic: Optional[str] = None
    # Явный выбор модели LLM для генерации примеров; по умолчанию используется глобальный ai_model
    model_name: Optional[str] = None
    # Дополнительные настройки генерации
    author_gender: Optional[str] = None      # 'Мужской' / 'Женский' / другое
    author_age_group: Optional[str] = None   # '18-25', '26-35', '36-50', '51+' и т.п.
    person: Optional[str] = None             # 'я', 'мы', 'они', 'он/она', 'ты/вы'
    max_chars: Optional[int] = Field(
        default=None,
        ge=1,
        le=8000,
        description="Ограничение на длину текста в символах",
    )


FIXED_LCA_PROFILES: Dict[Tuple[str, str], Dict[str, str]] = {
    # Тип → Подтип → (пол, возраст, платформа) по таблице 8.1 Word-отчёта
    ("4. СЕМЕЙНЫЕ", "Семья/род"): {
        "author_gender": "Женский",
        "author_age_group": "51+",
        "platform": "VK",
    },
    ("1. ПРОСТРАНСТВЕННЫЕ", "Дом/жилище"): {
        "author_gender": "Женский",
        "author_age_group": "51+",
        "platform": "VK",
    },
    ("6. ВОЕННЫЕ", "Крепость/оборона"): {
        "author_gender": "Женский",
        "author_age_group": "51+",
        "platform": "VK",
    },
    ("2. ОРГАНИЧЕСКИЕ", "Рождение/рост"): {
        "author_gender": "Женский",
        "author_age_group": "36-50",
        "platform": "VK",
    },
    ("10. ИСТОРИЧЕСКИЕ", "Наследие/памятник"): {
        "author_gender": "Женский",
        "author_age_group": "51+",
        "platform": "VK",
    },
    ("3. МЕХАНИЧЕСКИЕ", "Строительство"): {
        "author_gender": "Мужской",
        "author_age_group": "36-50",
        "platform": "VK",
    },
    ("9. ИГРОВЫЕ", "Игра/партия"): {
        "author_gender": "Женский",
        "author_age_group": "51+",
        "platform": "VK",
    },
    ("5. САКРАЛЬНЫЕ", "Миссия/предназначение"): {
        "author_gender": "Мужской",
        "author_age_group": "51+",
        "platform": "VK",
    },
    ("7. ПРИРОДНЫЕ", "Почва/земля"): {
        "author_gender": "Мужской",
        "author_age_group": "51+",
        "platform": "VK",
    },
    ("8. МЕДИЦИНСКИЕ", "Болезнь/здоровье"): {
        "author_gender": "Женский",
        "author_age_group": "51+",
        "platform": "VK",
    },
}

# Кэш статистики по полу/возрасту для типов метафор
METAPHOR_DEM_STATS: Dict[str, Dict[str, Dict]] = {}


@app.get("/metaphor-taxonomy", tags=['ai analytics'])
async def get_metaphor_taxonomy():
    """
    Возвращает полную типологию метафор (типы и подтипы),
    чтобы фронтенд мог показывать все варианты, независимо от результатов LCA.
    """
    taxonomy = [
        {
            "frame_type": "1. ПРОСТРАНСТВЕННЫЕ",
            "subtypes": ["Путь/дорога", "Центр-периферия", "Дом/жилище", "Граница/рубеж"],
        },
        {
            "frame_type": "2. ОРГАНИЧЕСКИЕ",
            "subtypes": ["Организм/тело", "Дерево/корни", "Рождение/рост"],
        },
        {
            "frame_type": "3. МЕХАНИЧЕСКИЕ",
            "subtypes": ["Машина/механизм", "Строительство", "Инструмент"],
        },
        {
            "frame_type": "4. СЕМЕЙНЫЕ",
            "subtypes": ["Мать/отец", "Братство", "Семья/род"],
        },
        {
            "frame_type": "5. САКРАЛЬНЫЕ",
            "subtypes": ["Храм/святыня", "Жертва", "Миссия/предназначение"],
        },
        {
            "frame_type": "6. ВОЕННЫЕ",
            "subtypes": ["Крепость/оборона", "Битва/война", "Армия/строй"],
        },
        {
            "frame_type": "7. ПРИРОДНЫЕ",
            "subtypes": ["Стихия/поток", "Почва/земля", "Климат/погода"],
        },
        {
            "frame_type": "8. МЕДИЦИНСКИЕ",
            "subtypes": ["Болезнь/здоровье", "Вирус/зараза", "Хирургия/ампутация"],
        },
        {
            "frame_type": "9. ИГРОВЫЕ",
            "subtypes": ["Игра/партия", "Спектакль/роль"],
        },
        {
            "frame_type": "10. ИСТОРИЧЕСКИЕ",
            "subtypes": ["Эпоха/век", "Наследие/памятник"],
        },
    ]
    return {"taxonomy": taxonomy}


def _age_to_group(age_val) -> Optional[str]:
    """Группировка возраста в интервалы, совпадающие с фронтендом."""
    if age_val is None:
        return None
    try:
        age_int = int(str(age_val).strip())
    except (ValueError, TypeError):
        return None
    if age_int <= 25:
        return "18-25"
    if age_int <= 35:
        return "26-35"
    if age_int <= 50:
        return "36-50"
    return "51+"


def _load_metaphor_dem_stats() -> Dict[str, Dict]:
    """
    Загружает файл patriotism_text_clusters.xlsx и считает,
    сколько авторов каждого пола/возрастной группы есть в каждом типе метафор.
    """
    global METAPHOR_DEM_STATS
    if METAPHOR_DEM_STATS:
        return METAPHOR_DEM_STATS

    path = "/home/dev/tellscope_app/tellscope_backend/data/patriotism_text_clusters.xlsx"
    try:
        df = pd.read_excel(path)
    except FileNotFoundError:
        return {}

    stats: Dict[str, Dict] = {}

    for _, row in df.iterrows():
        ft = row.get("frame_type")
        if not isinstance(ft, str) or not ft.strip():
            continue

        gender_raw = row.get("author_gender", "")
        gender_str = str(gender_raw).strip().lower()
        # Обрабатываем NaN / пустые значения как "не указан"
        if not gender_str or gender_str in ("nan", "none"):
            gender_norm = "не указан"
        else:
            gender_norm = gender_str

        age_raw = row.get("author_age", None)
        age_group = _age_to_group(age_raw)
        if not age_group:
            # Пустой или некорректный возраст считаем как отдельную группу
            age_group = "Не указана"

        if ft not in stats:
            stats[ft] = {"total": 0, "by_gender_age": {}, "by_gender": {}, "by_age": {}}

        stats[ft]["total"] += 1
        key = (gender_norm, age_group)
        stats[ft]["by_gender_age"][key] = stats[ft]["by_gender_age"].get(key, 0) + 1
        # агрегаты отдельно по полу и по возрасту
        stats[ft]["by_gender"][gender_norm] = stats[ft]["by_gender"].get(gender_norm, 0) + 1
        stats[ft]["by_age"][age_group] = stats[ft]["by_age"].get(age_group, 0) + 1

    METAPHOR_DEM_STATS = stats
    return stats


@app.get("/metaphor-dominant-demographics", tags=['ai analytics'])
async def get_metaphor_dominant_demographics(frame_type: str):
    """
    Возвращает наиболее частое сочетание пола и возрастной группы
    для заданного типа метафор (по данным patriotism_text_clusters.xlsx),
    игнорируя случаи, когда пол или возраст не указаны.
    """
    if not frame_type:
        raise HTTPException(status_code=400, detail="Параметр frame_type обязателен.")

    stats = _load_metaphor_dem_stats()
    ft_stats = stats.get(frame_type)
    if not ft_stats or ft_stats.get("total", 0) == 0:
        return {
            "frame_type": frame_type,
            "author_gender": None,
            "author_age_group": None,
            "count": 0,
            "total": 0,
            "percent": 0.0,
        }

    by_ga = ft_stats["by_gender_age"]
    best_key = None
    best_count = 0

    # Ищем просто самую частую комбинацию пола и возраста
    for (g, ag), cnt in by_ga.items():
        if cnt > best_count:
            best_count = cnt
            best_key = (g, ag)

    if not best_key:
        return {
            "frame_type": frame_type,
            "author_gender": None,
            "author_age_group": None,
            "count": 0,
            "total": int(ft_stats["total"]),
            "percent": 0.0,
        }

    gender_norm, age_group = best_key
    total = ft_stats["total"]
    percent = round(100.0 * best_count / total, 1) if total > 0 else 0.0

    # Нормализуем отображение пола
    if gender_norm == "не указан":
        display_gender = "Не указан"
    elif gender_norm in ("женский", "мужской"):
        display_gender = gender_norm.capitalize()
    else:
        display_gender = str(gender_norm)

    return {
        "frame_type": frame_type,
        "author_gender": display_gender,
        "author_age_group": age_group,
        "count": int(best_count),
        "total": int(total),
        "percent": percent,
    }


@app.get("/metaphor-demographics", tags=['ai analytics'])
async def get_metaphor_demographics(
    frame_type: str,
    author_gender: Optional[str] = None,
    author_age_group: Optional[str] = None,
):
    """
    Возвращает процент сообщений в заданном типе метафор,
    написанных авторами с указанным полом и возрастной группой.
    """
    if not frame_type:
        raise HTTPException(status_code=400, detail="Параметр frame_type обязателен.")

    # Должен быть указан хотя бы один из параметров (пол или возраст)
    if not author_gender and not author_age_group:
        raise HTTPException(
            status_code=400,
            detail="Для расчёта процента нужно указать хотя бы пол или возрастную группу.",
        )

    stats = _load_metaphor_dem_stats()
    ft_stats = stats.get(frame_type)
    if not ft_stats or ft_stats.get("total", 0) == 0:
        return {
            "frame_type": frame_type,
            "author_gender": author_gender,
            "author_age_group": author_age_group,
            "count": 0,
            "total": 0,
            "percent": 0.0,
        }

    gender_norm = str(author_gender).strip().lower() if author_gender else None
    age_group = author_age_group
    by_ga = ft_stats["by_gender_age"]

    # 3 режима: и пол, и возраст; только пол; только возраст
    if gender_norm and age_group:
        # Особый случай: «не указан» / «Не указана» — считаем долю таких записей
        # среди всех сообщений данного типа (включая известные и неизвестные).
        if gender_norm == "не указан" or age_group == "Не указана":
            count = by_ga.get((gender_norm, age_group), 0)
            denom = ft_stats["total"]
        else:
            # считаем только по записям с непустым полом в этой возрастной группе
            count = by_ga.get((gender_norm, age_group), 0)
            denom = sum(v for (g, ag), v in by_ga.items() if ag == age_group and g)
    elif gender_norm:
        # только пол
        if gender_norm == "не указан":
            # доля неизвестного пола среди всех сообщений данного типа
            count = sum(v for (g, ag), v in by_ga.items() if g == gender_norm)
            denom = ft_stats["total"]
        else:
            # делим на все записи с известным полом
            count = sum(v for (g, ag), v in by_ga.items() if g == gender_norm)
            denom = sum(v for (g, ag), v in by_ga.items() if g)
    elif age_group:
        # только возраст: считаем долю этой возрастной группы среди всех сообщений
        # данного типа (включая и известный, и неизвестный возраст)
        count = sum(v for (g, ag), v in by_ga.items() if ag == age_group)
        denom = ft_stats["total"]
    else:
        count = 0
        denom = 0

    percent = round(100.0 * count / denom, 1) if denom > 0 else 0.0

    return {
        "frame_type": frame_type,
        "author_gender": author_gender,
        "author_age_group": author_age_group,
        "count": int(count),
        "total": int(denom),
        "percent": percent,
    }


@app.post("/lca-examples", tags=['ai analytics'])
async def get_lca_examples(query: LCAExamplesRequest):
    """
    Возвращает сгенерированные примеры сообщений по результатам LCA-кластеризации.
    Источником служит файл patriotism_lca_synthetic_examples.json, который создаётся
    скриптом metaphor_typology_analyzer_27.02.py.
    """
    lca_path = "/home/dev/tellscope_app/tellscope_backend/data/patriotism_lca_synthetic_examples.json"

    try:
        with open(lca_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Файл с синтетическими примерами не найден. Сначала запустите анализатор метафор."
        )

    records = data if isinstance(data, list) else []

    # Фильтрация по типу / подтипу, если заданы
    if query.frame_type:
        records = [r for r in records if r.get("frame_type") == query.frame_type]
    if query.frame_subtype:
        records = [r for r in records if r.get("frame_subtype") == query.frame_subtype]

    # Нормализуем желаемое количество примеров
    desired_n = max(1, query.limit_per_cluster or 1)
    # Определяем модель: либо явно запрошенная, либо глобальная по умолчанию
    used_model = query.model_name or ai_model

    # Функция для расчёта max_tokens по ограничению в символах
    def compute_max_tokens(max_chars: Optional[int]) -> int:
        if max_chars is None or max_chars <= 0:
            return 300
        # Для русского BPE чаще 2–3 символа на токен; деление на 3 занижало бюджет
        # (300 символов → 100 токенов → фактически ~200 знаков у обрыва по max_tokens).
        ceil_half = (max_chars + 1) // 2
        slack = max(8, max_chars // 12)
        approx_tokens = ceil_half + slack
        return max(50, min(approx_tokens, 8192))

    # Более жёсткое описание грамматического лица / позиции
    def person_instruction(person: Optional[str]) -> str:
        if not person:
            return ""
        mapping = {
            "я": "Говори от первого лица единственного числа («я»), используй соответствующие формы глаголов и местоимений.",
            "мы": "Говори от первого лица множественного числа («мы»), подчёркивай коллективную позицию.",
            "они": "Говори о ситуации в третьем лице множественного числа («они»), как наблюдатель.",
            "он/она": "Говори о персонаже в третьем лице единственного числа («он» / «она»), как сторонний наблюдатель.",
            "ты/вы": "Обращайся к читателю на «ты» или «вы», используй форму обращения во втором лице.",
        }
        base = mapping.get(person, "")
        if not base:
            return ""
        return base + " Всегда соблюдай это лицо при построении фраз."

    # Если пользователь задал свой текст-контекст, генерируем примеры только по нему,
    # не опираясь на оффлайн-файл с примерами.
    if query.custom_topic and query.frame_type:
        ft = query.frame_type
        sub = query.frame_subtype

        # Если подтип не указан, ищем его в FIXED_LCA_PROFILES по типу
        if ft and not sub:
            for (t, s) in FIXED_LCA_PROFILES.keys():
                if t == ft:
                    sub = s
                    break

        key = (ft, sub) if ft and sub else None
        if not key or key not in FIXED_LCA_PROFILES:
            raise HTTPException(
                status_code=400,
                detail="Для пользовательского текста нужен корректный тип/подтип из таксономии."
            )

        profile = FIXED_LCA_PROFILES[key]
        portrait = {
            "author_gender": profile.get("author_gender", "не указан"),
            "author_age_group": profile.get("author_age_group", "не указана"),
            "platform": profile.get("platform", "VK"),
        }
        # Переопределяем портрет, если пользователь задал пол / возраст явно
        if query.author_gender:
            portrait["author_gender"] = query.author_gender
        if query.author_age_group:
            portrait["author_age_group"] = query.author_age_group

        topic = query.custom_topic[:500]  # ограничим длину контекста
        person_hint = ""
        if query.person:
            person_hint = f"\nГрамматическое лицо / позиция говорящего: {query.person}."
        person_rule = person_instruction(query.person)

        max_tokens = compute_max_tokens(query.max_chars)

        system_msg = (
            "Ты — симулятор пользователя российских социальных сетей. "
            "Пиши естественные, правдоподобные посты/комментарии на русском языке, "
            "без объяснений и метакомментариев. "
            "Строго соблюдай заданный портрет аудитории и грамматическое лицо."
        )

        examples = []
        for i in range(desired_n):
            user_prompt = f"""
Профиль автора:
- Пол: {portrait['author_gender']}
- Возрастная группа: {portrait['author_age_group']}
- Платформа: {portrait['platform']}
{person_hint}

Метафорический фрейм:
- Тип: {ft}
- Подтип: {sub}

Текст-пример (контекст):
\"\"\"{topic}\"\"\"

ЗАДАНИЕ:
На основе этого текста-примера сгенерируй новый текст (1–4 предложения), который:
- выглядел бы как реальный пост или комментарий в {portrait['platform']};
- сохраняет основную тему и интонацию примера, но не копирует его дословно;
- использует образность фрейма «{ft} → {sub}» (на уровне смысла).

Если указано лицо/позиция говорящего, строго следуй ему:
{person_rule or 'если лицо не задано, выбери его сам, но делай текст естественным для данного пола и возраста.'}

Ограничение по длине: не более {query.max_chars or 'примерно 300'} символов.

Сделай этот пример стилистически отличным от возможных других сообщений:
можно варьировать тон (более ироничный, более официально-деловой, более эмоциональный и т.п.).

Ответь только текстом сообщения без каких-либо пояснений.
"""
            try:
                resp = client.chat.completions.create(
                    model=used_model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=0.9,
                )
                text = resp.choices[0].message.content.strip()
            except Exception as e:
                text = f"Ошибка генерации: {e}"

            examples.append(
                {
                    "example_idx": i + 1,
                    "topic_context": topic,
                    "source_url": "",
                    "source_id": "",
                    "generated_post": text,
                    "cluster_id": -1,
                }
            )

        return {
            "frames": [
                {
                    "frame_type": ft,
                    "frame_subtype": sub,
                    "portrait": portrait,
                    "examples": examples,
                }
            ]
        }

    # Если в файле нет примеров для запрошенного типа/подтипа —
    # пробуем сгенерировать их «на лету» по FIXED_LCA_PROFILES.
    if not records:
        ft = query.frame_type
        sub = query.frame_subtype

        # Если подтип не указан, ищем его в FIXED_LCA_PROFILES по типу
        if ft and not sub:
            for (t, s) in FIXED_LCA_PROFILES.keys():
                if t == ft:
                    sub = s
                    break

        key = (ft, sub) if ft and sub else None
        if not key or key not in FIXED_LCA_PROFILES:
            raise HTTPException(
                status_code=404,
                detail="Нет примеров для указанного типа/подтипа метафор."
            )

        profile = FIXED_LCA_PROFILES[key]
        portrait = {
            "author_gender": profile.get("author_gender", "не указан"),
            "author_age_group": profile.get("author_age_group", "не указана"),
            "platform": profile.get("platform", "VK"),
        }
        # Переопределяем портрет при явном выборе
        if query.author_gender:
            portrait["author_gender"] = query.author_gender
        if query.author_age_group:
            portrait["author_age_group"] = query.author_age_group

        # Генерируем примеры на лету через LLM
        n = desired_n
        examples = []
        topic = f"Обсуждение патриотизма в рамках фрейма {ft} → {sub}"

        system_msg = (
            "Ты — симулятор пользователя российских социальных сетей. "
            "Пиши естественные, правдоподобные посты/комментарии на русском языке, "
            "без объяснений и метакомментариев. "
            "Строго соблюдай заданный портрет аудитории и грамматическое лицо."
        )

        person_hint = ""
        if query.person:
            person_hint = f"\nГрамматическое лицо / позиция говорящего: {query.person}."
        person_rule = person_instruction(query.person)

        max_tokens = compute_max_tokens(query.max_chars)

        for i in range(n):
            user_prompt = f"""
Профиль автора:
- Пол: {portrait['author_gender']}
- Возрастная группа: {portrait['author_age_group']}
- Платформа: {portrait['platform']}
{person_hint}

Метафорический фрейм:
- Тип: {ft}
- Подтип: {sub}

Тема сообщения:
- {topic}

ЗАДАНИЕ:
Напиши короткий, но содержательный текст (1–4 предложения), который:
- выглядел бы как реальный пост или комментарий в {portrait['platform']};
- использует образность фрейма «{ft} → {sub}» (на уровне смысла, а не обязательного повторения слов);
- отражает патриотический дискурс (история, единство, общая семья, долг, память и т.п.).

Если указано лицо/позиция говорящего, строго следуй ему:
{person_rule or 'если лицо не задано, выбери его сам, но делай текст естественным для данного пола и возраста.'}

Ограничение по длине: не более {query.max_chars or 'примерно 300'} символов.

Сделай этот пример стилистически отличным от возможных других сообщений:
можно варьировать тон (более ироничный, более официально-деловой, более эмоциональный и т.п.).

Ответь только текстом сообщения без каких-либо пояснений.
"""
            try:
                resp = client.chat.completions.create(
                    model=used_model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=0.85,
                )
                text = resp.choices[0].message.content.strip()
            except Exception as e:
                text = f"Ошибка генерации: {e}"

            examples.append(
                {
                    "example_idx": i + 1,
                    "topic_context": topic,
                    "source_url": "",
                    "source_id": "",
                    "generated_post": text,
                    "cluster_id": -1,
                }
            )

        return {
            "frames": [
                {
                    "frame_type": ft,
                    "frame_subtype": sub,
                    "portrait": portrait,
                    "examples": examples,
                }
            ]
        }

    from collections import Counter  # локальный импорт

    # Группируем по (frame_type, frame_subtype)
    grouped: Dict[Tuple[str, str], list] = {}
    for r in records:
        key = (r.get("frame_type"), r.get("frame_subtype"))
        grouped.setdefault(key, []).append(r)

    def most_common(counter: Counter, default: str) -> str:
        return counter.most_common(1)[0][0] if counter else default

    response_frames = []
    for (frame_type, frame_subtype), items in grouped.items():
        key = (frame_type, frame_subtype)

        # 1) Пытаемся взять портрет прямо из LCA-кластера (как в Word-отчёте)
        lca_portrait = None
        for it in items:
            p = it.get("portrait")
            if isinstance(p, dict) and p:
                lca_portrait = p
                break

        if lca_portrait:
            def extract_field(key_p: str, default: str) -> str:
                raw = lca_portrait.get(key_p, "")
                if isinstance(raw, str):
                    return raw.split(" (")[0].strip() or default
                return str(raw) or default

            portrait = {
                "author_gender": extract_field("author_gender", "не указан"),
                "author_age_group": extract_field("author_age_group", "не указана"),
                "platform": extract_field("platform", "VK"),
            }
        elif key in FIXED_LCA_PROFILES:
            # 2) ЖЁСТКОЕ соответствие Word-таблице 8.1, если нет портрета в данных
            fixed = FIXED_LCA_PROFILES[key]
            portrait = {
                "author_gender": fixed.get("author_gender", "не указан"),
                "author_age_group": fixed.get("author_age_group", "не указана"),
                "platform": fixed.get("platform", "VK"),
            }
        else:
            # 3) Fallback: считаем по самим примерам
            gender_counter = Counter(i.get("author_gender") for i in items if i.get("author_gender"))
            age_counter = Counter(i.get("author_age_group") for i in items if i.get("author_age_group"))
            platform_counter = Counter(i.get("platform") for i in items if i.get("platform"))

            portrait = {
                "author_gender": most_common(gender_counter, "не указан"),
                "author_age_group": most_common(age_counter, "не указана"),
                "platform": most_common(platform_counter, "VK"),
            }

        # Берём не более desired_n примеров из файла
        limited_items = items[: desired_n]
        examples = [
            {
                "example_idx": it.get("example_idx"),
                "topic_context": it.get("topic_context", ""),
                "source_url": it.get("source_url", ""),
                "source_id": it.get("source_id", ""),
                "generated_post": it.get("generated_post", ""),
                "cluster_id": it.get("cluster_human_id", it.get("cluster_id")),
            }
            for it in limited_items
        ]

        # Если примеров меньше, чем нужно — досинтезируем недостающие на лету
        if len(examples) < desired_n:
            ft = frame_type
            sub = frame_subtype
            topic = (
                limited_items[0].get("topic_context", "")
                if limited_items and limited_items[0].get("topic_context")
                else f"Обсуждение патриотизма в рамках фрейма {ft} → {sub}"
            )

            system_msg = (
                "Ты — симулятор пользователя российских социальных сетей. "
                "Пиши естественные, правдоподобные посты/комментарии на русском языке, "
                "без объяснений и метакомментариев. "
                "Строго соблюдай заданный портрет аудитории и грамматическое лицо."
            )

            person_hint = ""
            if query.person:
                person_hint = f"\nГрамматическое лицо / позиция говорящего: {query.person}."
            person_rule = person_instruction(query.person)

            max_tokens = compute_max_tokens(query.max_chars)

            for extra_idx in range(len(examples), desired_n):
                user_prompt = f"""
Профиль автора:
- Пол: {portrait['author_gender']}
- Возрастная группа: {portrait['author_age_group']}
- Платформа: {portrait['platform']}
{person_hint}

Метафорический фрейм:
- Тип: {ft}
- Подтип: {sub}

Тема сообщения:
- {topic}

ЗАДАНИЕ:
Напиши короткий, но содержательный текст (1–4 предложения), который:
- выглядел бы как реальный пост или комментарий в {portrait['platform']};
- использует образность фрейма «{ft} → {sub}» (на уровне смысла, а не обязательного повторения слов);
- отражает патриотический дискурс (история, единство, общая семья, долг, память и т.п.).

Если указано лицо/позиция говорящего, строго следуй ему:
{person_rule or 'если лицо не задано, выбери его сам, но делай текст естественным для данного пола и возраста.'}

Ограничение по длине: не более {query.max_chars or 'примерно 300'} символов.

Сделай этот пример стилистически отличным от возможных других сообщений:
можно варьировать тон (более ироничный, более официально-деловой, более эмоциональный и т.п.).

Ответь только текстом сообщения без каких-либо пояснений.
"""
                try:
                    resp = client.chat.completions.create(
                        model=used_model,
                        messages=[
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": user_prompt},
                        ],
                        max_tokens=max_tokens,
                        temperature=0.9,
                    )
                    text = resp.choices[0].message.content.strip()
                except Exception as e:
                    text = f"Ошибка генерации: {e}"

                examples.append(
                    {
                        "example_idx": extra_idx + 1,
                        "topic_context": topic,
                        "source_url": "",
                        "source_id": "",
                        "generated_post": text,
                        "cluster_id": -1,
                    }
                )

        response_frames.append(
            {
                "frame_type": frame_type,
                "frame_subtype": frame_subtype,
                "portrait": portrait,
                "examples": examples,
            }
        )

    return {"frames": response_frames}


@app.get("/configs", tags=['utils'], response_class=HTMLResponse)
async def get_configs_page():
    """
    Отдаёт HTML-страницу с конфигами сервисов.
    Файл лежит в фронтенд-проекте: src/components/configs.html.
    """
    cfg_path = "/home/dev/tellscope_app/tellscope_frontend/src/components/configs.html"
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            html = f.read()
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Файл configs.html не найден на сервере.",
        )

    return HTMLResponse(content=html, status_code=200)


@app.get("/create-data-projector/{user_id}/{folder_name}/{file_name}")
async def create_data_projector(user_id: str, folder_name: str, file_name: str, user: User = Depends(current_user)):
    # Путь к файлу с темами 
    file_path = '/home/dev/tellscope_app/tellscope_backend/data/indexes.pkl'
    indexes = load_dict_from_pickle(file_path)

    # Отключаем использование GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    embed = hub.load("/home/dev/tellscope_app/tellscope_backend/data/embed_files/universal-sentence-encoder-multilingual_3")

    # Полный путь к файлу
    file_path = f'/home/dev/tellscope_app/tellscope_backend/data/{user_id}/json_files_directory/{folder_name}/{file_name}' + '.json'

    try:
        with io.open(file_path, encoding='utf-8', mode='r') as train_file:
            dict_train = json.load(train_file, strict=False)

    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Ошибка при чтении JSON: {e}")
        
        a = []
        try:
            with open(file_path, encoding='utf-8', mode='r') as file:
                for line in file:
                    a.append(line)

            dict_train = []
            for i in range(len(a)):
                try:
                    dict_train.append(ast.literal_eval(a[i]))
                except (SyntaxError, ValueError):
                    continue
            dict_train = [x[0] for x in dict_train]

        except FileNotFoundError: 
            raise HTTPException(status_code=404, detail="File not found")

    df = pd.DataFrame(dict_train)
    df_meta = pd.DataFrame()

    if 'hubtype' not in df.columns:
        dff = df
        dff['timeCreate'] = [datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S') for x in dff['timeCreate'].values]
        df_meta_smi_only = dff[['timeCreate', 'hub', 'toneMark', 'audience', 'url', 'text', 'citeIndex']]
        df_meta_smi_only['fullname'] = dff['hub']
        df_meta_smi_only['author_type'] = 'Онлайн-СМИ'
        df_meta_smi_only['hubtype'] = 'Онлайн-СМИ'
        df_meta_smi_only['type'] = 'Онлайн-СМИ'
        df_meta_smi_only['er'] = 0
        df_meta = df_meta_smi_only

    if 'hubtype' in df.columns:
        for i in range(2):
            if i == 0:
                dff = df[df['hubtype'] != 'Онлайн-СМИ']
                if dff.shape[0] != 0:
                    dff['timeCreate'] = [datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S') for x in dff['timeCreate'].values]
                    df_meta_socm = dff[['timeCreate', 'hub', 'toneMark', 'audienceCount', 'url', 'er', 'hubtype', 'text', 'type']]
                    df_meta_socm['fullname'] = pd.DataFrame.from_records(dff['authorObject'].values)['fullname'].values
                    df_meta_socm['author_type'] = pd.DataFrame.from_records(dff['authorObject'].values)['author_type'].values

            if i == 1:
                dff = df[df['hubtype'] == 'Онлайн-СМИ']
                if dff.shape[0] != 0:
                    dff['timeCreate'] = [datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S') for x in dff['timeCreate'].values]
                    df_meta_smi = dff[['timeCreate', 'hub', 'toneMark', 'audienceCount', 'url', 'er', 'hubtype', 'text', 'citeIndex']]
                    df_meta_smi['fullname'] = dff['hub']
                    df_meta_smi['author_type'] = 'Онлайн-СМИ'
                    df_meta_smi['hubtype'] = 'Онлайн-СМИ'
                    df_meta_smi['type'] = 'Онлайн-СМИ'

        if 'df_meta_smi' in locals() and 'df_meta_socm' in locals():
            df_meta = pd.concat([df_meta_socm, df_meta_smi])
        elif 'df_meta_smi' and 'df_meta_socm' not in locals():
            df_meta = df_meta_smi
        else:
            df_meta = df_meta_socm

    df_text = df_meta[['text']]
    
    regex = re.compile(r"[А-Яа-я:=!\)\()A-z_\%/|]+")

    def words_only(text, regex=regex):
        try:
            return " ".join(regex.findall(text))
        except:
            return ""

    mystopwords = ['это', 'наш', 'тыс', 'млн', 'млрд', 'также', 'т', 'д', 'URL',
                   'i', 's', 'v', 'info', 'a', 'подробнее', 'который', 'год',
                   ' - ', '-', 'В', '—', '–', '-', 'в', 'который']

    def preprocess_text(text):
        text = text.lower().replace("ё", "е")
        text = re.sub(r'((www[^\s]+)|(https?://[^\s]+))', 'URL', text)
        text = re.sub(r'@[^\s]+', 'USER', text)
        text = re.sub('[^a-zA-Zа-яА-Я1-9]+', ' ', text)
        text = re.sub(' +', ' ', text)
        return text.strip()

    def remove_stopwords(text, mystopwords=mystopwords):
        try:
            return " ".join([token for token in text.split() if not token in mystopwords])
        except:
            return ""

    df_text['text'] = df_text['text'].apply(words_only)
    df_text['text'] = df_text['text'].apply(preprocess_text)
    # df_text['text'] = df_text['text'].apply(remove_stopwords)
    df_text = df_text[df_text['text'].notna()]
    df_text = df_text[df_text['text'] != '']

    sent_ru = df_text['text'].values
    sent_ru = sent_ru[:50]

    # Обработка по партиям
    batch_size = 8
    embeddings = []
    
    for i in range(0, len(sent_ru), batch_size):
        await asyncio.sleep(0.01)
        batch = sent_ru[i:i + batch_size]
        # Для надежности оборачиваем выполнение на CPU
        with tf.device('/CPU:0'):
            embeddings.append(embed(batch))
    
    # Объединение эмбеддингов в один массив
    embeddings = tf.concat(embeddings, axis=0)

    embed_list = embeddings

    dff = pd.DataFrame(embeddings)

    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    x_tsne = tsne.fit_transform(pd.DataFrame(embed_list).values)

    coord_list = [', '.join(map(str, x)) for x in x_tsne.tolist()]
    names_list = [re.sub('\n', ' ', name) for name in df_meta['fullname'].fillna('None').values.tolist()]

    # Создание директории для сохранения файлов, если она не существует
    project_files_dir = f'/home/dev/tellscope_app/tellscope_backend/data/{user_id}/projector_files_directory/{folder_name}/'
    os.makedirs(project_files_dir, exist_ok=True)

    # сохранение данных для tsne
    dict_tsne = {
        'author_name_str': '\n'.join(names_list),
        'coord_list_str': '\n'.join(coord_list)
    }

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    tsv_filename = f"{file_name}_authors_point_{timestamp}.tsv"
    txt_filename = f"{file_name}_authors_name_{timestamp}.txt"

    # Сохранение файлов
    try:
        # Сохранение tsv файла
        with open(os.path.join(project_files_dir, tsv_filename), 'w') as f:
            for line in embed_list:
                f.write('\t'.join(map(str, line)) + '\n')

        # Сохранение txt файла
        with open(os.path.join(project_files_dir, txt_filename), 'w', encoding='utf-8') as f:
            for line in names_list:
                f.write(line + '\n')
    except Exception as e:
        print(f"Ошибка при сохранении файлов: {e}")

    # Сохранение данных о папке и файлах в Redis
    user_data = await redis_db.hgetall(user_id)
    # Если данные возвращаются в формате 'dict' с байтовыми строками, декодируйте их
    user_data = {key.decode('utf-8'): value.decode('utf-8') for key, value in user_data.items()}

    if not user_data:  # Проверяем, есть ли данные
        raise Exception("User data does not exist.")

    # Проверяем, существует ли field для projector_files_directory
    if "projector_files_directory" in user_data:
        user_folders = json.loads(user_data["projector_files_directory"])
    else:
        user_folders = {}  # Инициализируем пустой словарь, если поле отсутствует

    # Добавляем информацию о новых файлах в соответствующую папку
    file_info = {
        "tsv-file": tsv_filename,
        "txt-file": txt_filename,
        "creation_date": timestamp
    }

    # Проверяем данные пользователя
    if user_data:
        # Проверка на наличие ключа bertopic_files_directory
        if "projector_files_directory" in user_data:
            print(111)
            # Если ключ bertopic_files_directory существует — загружаем его содержимое
            user_folders = json.loads(user_data["projector_files_directory"])
        else:
            # Если ключа нет — создаём пустой словарь
            user_folders = {}

        # Проверяем существование папки, переданной в user_data['folder_name']
        if folder_name in user_folders:
            # Если папка существует, добавляем новый file_info в уже имеющийся список
            user_folders[folder_name].append(file_info)
        else:
            # Если папка не существует, создаём её и добавляем file_info в список
            user_folders[folder_name] = [file_info]

        # Сериализуем обновлённый объект папок (user_folders) в JSON
        serialized_folders = json.dumps(user_folders)

        # Сохраняем обновлённые данные в Redis
        await redis_db.hset(user_id, "projector_files_directory", serialized_folders)
    else:
        # Если данных пользователя нет, выбрасываем исключение
        raise Exception("User data does not exist.")

    # # Добавляем новый файл в соответствующую папку
    # if folder_name not in user_folders:
    #     user_folders[folder_name] = []

    # user_folders[folder_name].append(file_info)

    # # Сохраняем обновленные данные обратно в Redis
    # await redis_db.hset(user_id, "projector_files_directory", json.dumps(user_folders))

    return f"Файлы авторов для прожектора темы {file_name} созданы и сохранены в папку {folder_name}!"


@app.get('/file-load/{user_id}/{file_type}/{folder_name}/{file_name}', tags=['files'])
def load_file(user_id: str, file_type: str, folder_name: str, file_name: str, user: User = Depends(current_user)):
    if str(user.id) != str(user_id) and not user.is_superuser:
        _allowed = [it for it in _load_shares()
                    if it["owner_user_id"] == int(user_id)
                    and it["folder"] == folder_name
                    and it["user_id"] == user.id]
        if not _allowed:
            raise HTTPException(status_code=403, detail="Нет доступа к файлам этого пользователя")
    # Логируем параметры запроса для отладки
    print(f"Received request with parameters: user_id={user_id}, file_type={file_type}, folder_name={folder_name}, file_name={file_name}")

    BASE_DIR = '/home/dev/tellscope_app/tellscope_backend/data'
    PROJECTOR_DIR = os.path.join(BASE_DIR, user_id, 'projector_files_directory', folder_name)
    JSON_DIR = os.path.join(BASE_DIR, user_id, 'json_files_directory', folder_name)
    BERTOPIC_DIR = os.path.join(BASE_DIR, user_id, 'bertopic_files_directory', folder_name)

    # Определяем полный путь к файлу на основе типа файла
    if file_type == 'projector_files_directory': 
        file_path = os.path.join(PROJECTOR_DIR, file_name)
    elif file_type == 'bertopic_files_directory': 
        file_path = os.path.join(BERTOPIC_DIR, file_name)
    elif file_type == 'json_files_directory':
        if '.json' not in file_name:
            file_name += '.json'
        file_path = os.path.join(JSON_DIR, file_name)
    else:
        raise HTTPException(status_code=400, detail="Invalid file type. Use 'projector_files_directory', 'bertopic_files_directory' or 'json_files_directory'.")
    

    # Проверка существования файла
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    # Возврат файла
    return FileResponse(file_path, media_type='application/octet-stream', filename=file_name)

######################################## Запросы к LLM моделям #######################################

def update_task_progress(task_key, progress, queries):
    # Здесь queries - это список задач для данного пользователя
    for query in queries:
        # Проверяем, есть ли ключ задачи в текущем словаре
        if task_key in query:
            # Обновляем данные о прогрессе
            query[task_key] = {**query[task_key], **progress}
            return queries  # Возвращаем обновленные данные по мере нахождения задачи
    
    # Если задача не найдена, возвращаем исходные данные без изменений
    return queries

def update_progress(user_id, task_id, progress):
    os.chdir('/home/dev/tellscope_app/tellscope_backend/data')
    
    # Получаем текущую дату
    current_date = datetime.now().date().strftime('%Y-%m-%d')

    with open('llm_history_progress.pickle', 'rb') as file:
        llm_history = pickle.load(file)

    # Обновляем прогресс только для пользователя с соответствующим user_id
    for entry in llm_history:
        if entry['user_id'] == user_id:
            values = entry['values']
            date_queries = values.get('llm_queries', {})
            
            # Проверяем, есть ли у данного пользователя данные для текущей даты
            if isinstance(date_queries, dict):
                # Проверяем наличие задач для текущей даты
                if current_date in date_queries:
                    queries_for_date = date_queries[current_date]
                    updated_queries = update_task_progress(task_id, progress, queries_for_date)
                    date_queries[current_date] = updated_queries  # Обновляем список с задачами
            elif isinstance(date_queries, list):
                updated_queries = update_task_progress(task_id, progress, date_queries)
                values['llm_queries'] = updated_queries  # Обновляем данные

    # Сохраняем обновленные данные в файл
    with open('llm_history_progress.pickle', 'wb') as file:
        pickle.dump(llm_history, file)


from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return HTMLResponse(content=f"Ошибка валидации: {exc.errors()}", status_code=422)


# def sanitize_string(input_string):
#     if input_string is None:
#         return input_string
#     return input_string.replace("'", "\\'").replace('"', '\\"')

# Модель для задачи
class AnalysisRequest(BaseModel):
    user_id: int
    folder_name: str
    index: int
    min_date: int
    max_date: int
    query_str: Optional[str] = None
    system_prompt: Optional[str] = None
    # example_text: str  # Текст примера
    # example_thematics: str  # Тематики в тексте-примере
    # example_question_keywords: str  # Вопрос для ключевых слов текста
    # example_keywords: str  # Ключевые слова
    promt_question: str  # Вопрос

    def __init__(self, **data):
        super().__init__(**data)  # Вызываем родительский конструктор
        # Очищаем строковые поля
        # self.example_text = self.clean_string(self.example_text)
        # self.example_thematics = self.clean_string(self.example_thematics)
        # self.example_question_keywords = self.clean_string(self.example_question_keywords)
        # self.example_keywords = self.clean_string(self.example_keywords)
        self.promt_question = self.clean_string(self.promt_question)

    @staticmethod
    def clean_string(value: str) -> str:
        # Удаляем все нежелательные символы (в данном случае управляющие символы)
        if value is not None:
            # Удаляем неразрешенные управляющие символы
            value = re.sub(r'[\u0001-\u001F\u007F-\u009F]', '', value)
            # Дополнительно можно экранировать одинарные кавычки
            value = value.replace("'", "")
        return value


# Путь к файлу истории
HISTORY_FILE = '/home/dev/tellscope_app/tellscope_backend/data/llm_history_progress.pickle'

def load_history(user_id):
    """Загружает историю выполнения задач пользователя из файла."""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'rb') as file:
            history = pickle.load(file)
            # Ищем запись для указанного user_id
            for entry in history:
                if entry['user_id'] == user_id:
                    return entry['values']
    return {}

def save_history(user_id, history_data):
    """Сохраняет данные о задачах пользователя в файл."""
    all_history = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'rb') as file:
            all_history = pickle.load(file)

    user_found = False
    for entry in all_history:
        if entry['user_id'] == user_id:
            entry['values'].update(history_data)
            user_found = True
            break

    if not user_found:
        all_history.append({'user_id': user_id, 'values': history_data})

    with open(HISTORY_FILE, 'wb') as file:
        pickle.dump(all_history, file)


# from run_llm_query import run_llm_query
# from run_llm_query_new import run_llm_query
# from test_interactive_embed import run_llm_query

import logging
import os
import re
import time
import pickle
import asyncio
import gc
import json
import traceback
import aiohttp
import torch
from tqdm import tqdm

_vllm_resolved_model_id: Optional[str] = None


async def get_vllm_model_id() -> str:
    """Модель из serving lock, иначе VLLM_MODEL / /v1/models."""
    try:
        from mlops.lock import generate_cfg
        mid = (generate_cfg().get("model") or "").strip()
        if mid:
            return mid
    except Exception:
        pass
    global _vllm_resolved_model_id
    if VLLM_MODEL_ENV is not None and str(VLLM_MODEL_ENV).strip():
        return str(VLLM_MODEL_ENV).strip()
    if _vllm_resolved_model_id is not None:
        return _vllm_resolved_model_id
    try:
        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(timeout=timeout) as s:
            async with s.get(VLLM_MODELS_URL) as r:
                if r.status == 200:
                    data = await r.json()
                    for m in data.get("data") or []:
                        mid = m.get("id")
                        if mid:
                            _vllm_resolved_model_id = str(mid)
                            logging.info("vLLM model (auto from /v1/models): %s", _vllm_resolved_model_id)
                            return _vllm_resolved_model_id
    except Exception as e:
        logging.warning("vLLM /v1/models auto-detect failed: %s", e)
    return _VLLM_FALLBACK_MODEL_ID


from datetime import datetime
from transformers import AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics import silhouette_score
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
import pandas as pd
import datamapplot

from torch import bfloat16
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, TextGeneration
# from search_data_elastic import elastic_query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import sessionmaker
from collections import defaultdict
from sqlalchemy import Column, Integer, String, JSON, select
from config import DB_HOST, DB_NAME, DB_PASS, DB_PORT, DB_USER

import numpy as np
import redis.asyncio as redis
from ollama import AsyncClient
from collections import OrderedDict
from typing import List, Dict
# Инициализация клиента Redis
redis_db = redis.Redis(host='localhost', port=6379, db=0)

# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# import nltk

# # Загружаем списки стоп-слов и токенайзер
# nltk.download('stopwords')
# nltk.download('punkt')

# # Получаем список стоп-слов для русского языка
# russian_stopwords = stopwords.words("russian")

# from sqlalchemy.orm import sessionmaker, declarative_base

# Определяем базовый класс для моделей
from sqlalchemy.orm import declarative_base
Base = declarative_base()

# Определяем модель для хранения эмбеддингов
class Embedding(Base):
    __tablename__ = 'embedding'
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)
    filename = Column(String, nullable=False) 
    # Например, поле для хранения эмбеддингов
    vectors = Column(JSON, nullable=False)

DATABASE_URL = f"postgresql+asyncpg://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_async_engine(DATABASE_URL)
async_session_maker = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

from sqlalchemy import Column, Integer, String, JSON, Table, MetaData, Text
from sqlalchemy.future import select
from sqlalchemy import insert

# Определение метаданных
metadata = MetaData()

# Определение модели таблицы embeddings_pg
embeddings = Table(
    "embeddings_pg",
    metadata,
    Column("id", Integer, primary_key=True, index=True),
    Column("user_id", Integer, nullable=False),  # Указан идентификатор пользователя
    Column("filename", String(255), nullable=False),  # Имя файла
    Column("folder_name", String(255), nullable=False),  # Имя папки
    Column("vectors", JSON, nullable=False),  # Поле для хранения эмбеддингов в формате JSON
)

async def save_embedding_to_pgvector(session: AsyncSession, user_id: int, filename: str, folder_name: str, vectors):
    # Преобразуем каждый массив NumPy в стандартный список Python
    vectors_list = []
    for vector in vectors:
        if isinstance(vector, np.ndarray):
            vectors_list.append(vector.tolist())  # Преобразуем в список
        else:
            raise TypeError("Каждый элемент векторов должен быть массивом NumPy (ndarray).")
    
    # Создаем объект для вставки
    new_embedding = {
        "user_id": user_id,
        "filename": filename,
        "folder_name": folder_name,
        "vectors": vectors_list  # Сохраняем список векторов
    }
    
    # Выполняем вставку в базу данных
    try:
        await session.execute(insert(embeddings).values(new_embedding))  # Замените your_table на реальную таблицу
        await session.commit()
    except Exception as e:
        print(f"Ошибка при сохранении векторов: {e}")

from elasticsearch import Elasticsearch
from typing import Optional, List, Dict

def update_max_result_window(index_name: str, max_window: int = 1000000):
    try:
        es.indices.put_settings(
            index=index_name,
            body={"index": {"max_result_window": max_window}}
        )
    except Exception as e:
        print(f"Ошибка при обновлении настроек индекса '{index_name}': {e}")

def build_query(query_str: str, default_fields: List[str] = ["text", "Текст сообщения"]) -> dict:
    """
    Формирует сложный запрос для Эластика:
    - Если строка 'all' или пустая — match_all (все документы).
    - Если строка содержит ~N (пример: "инженер данных~3") — ищем фразу с расстоянием (slop).
    - Иначе — ищем все слова из запроса, независимо от порядка, с морфологией.
    Поддерживает поиск по нескольким полям (text и Текст сообщения).
    """
    if query_str is None or query_str.strip().lower() == "all":
        return {"match_all": {}}

    query_str = query_str.strip()
    # Фразовый поиск с расстоянием (пример "инженер данных~3")
    phrase_match = re.match(r'^(.*?)~(\d+)$', query_str)
    if phrase_match:
        phrase = phrase_match.group(1).strip()
        slop = int(phrase_match.group(2))
        return {
            "multi_match": {
                "query": phrase,
                "type": "phrase",
                "slop": slop,
                "fields": default_fields
            }
        }
    
    # Булевский AND для всех слов (морфология — предполагается статсномный анализатор на индексе)
    words = query_str.split()
    must_clauses = []
    for w in words:
        must_clauses.append({
            "multi_match": {
                "query": w,
                "fields": default_fields,
                "operator": "and"  # <= для поддержки русского можно опустить, если индекс морфологический
            }
        })
    return {"bool": {"must": must_clauses}}

def search_single_subquery(
    theme_index: str,
    query_str: str,
    min_date: Optional[int],
    max_date: Optional[int],
    scroll_time: str,
    batch_size: int,
    default_fields: List[str] = ["text", "Текст сообщения"]
) -> List[dict]:
    user_query = build_query(query_str, default_fields)
    es_query = {"query": user_query}

    # Фильтр по дате (если задан)
    if min_date is not None or max_date is not None:
        date_filter = {"range": {"timeCreate": {}}}
        if min_date is not None:
            date_filter['range']['timeCreate']['gte'] = min_date
        if max_date is not None:
            date_filter['range']['timeCreate']['lte'] = max_date

        es_query = {
            "query": {
                "bool": {
                    "must": user_query,
                    "filter": date_filter
                }
            }
        }
    try:
        response = es.search(
            index=theme_index,
            body=es_query,
            scroll=scroll_time,
            size=batch_size
        )
    except Exception as e:
        print(f"Ошибка при выполнении запроса: {e}")
        return []

    scroll_id = response.get('_scroll_id')
    results = response['hits']['hits']
    total_hits = response['hits']['total']['value'] if isinstance(response['hits']['total'], dict) else response['hits']['total']

    # Получаем все страницы scroll-батчей
    while True:
        try:
            response = es.scroll(scroll_id=scroll_id, scroll=scroll_time)
        except Exception as e:
            print(f"Ошибка при выполнении scroll-запроса: {e}")
            break

        hits = response['hits']['hits']
        if not hits:
            break
        results.extend(hits)
        scroll_id = response.get('_scroll_id')

    try:
        es.clear_scroll(scroll_id=scroll_id)
    except Exception:
        pass

    # Преобразуем к формату с _id внутри и нормализуем текстовое поле
    normalized_results = []
    for hit in results:
        doc = dict(**hit['_source'], _id=hit['_id'])
        # Нормализуем текстовое поле (объединяем оба варианта)
        if 'Текст сообщения' in doc and 'text' not in doc:
            doc['text'] = doc['Текст сообщения']
        elif 'text' in doc and 'Текст сообщения' not in doc:
            doc['Текст сообщения'] = doc['text']
        normalized_results.append(doc)
    
    return normalized_results

def elastic_query(
    theme_index: str,
    query_str: Optional[str] = None,  # делаем параметр опциональным с None по умолчанию
    min_date: Optional[int] = None,
    max_date: Optional[int] = None,
    scroll_time: str = '5m',
    batch_size: int = 10000,
    default_fields: List[str] = ["text", "Текст сообщения"]
) -> List[Dict]:
    """
    Выполняет поиск в индексе theme_index:
      - query_str: поисковая строка, поддерживает запятые как ИЛИ поиска ("one, two, three").
        Если None или пустая строка - возвращает все документы.
      - min_date, max_date — фильтрация по unix-таймштампу в поле timeCreate (опционально)
      - scroll_time, batch_size — параметры скроллинга
      - default_fields — поля для поиска (обычно ['text', 'Текст сообщения'], поля должны быть с русским анализатором)
    Возвращает: list[dict] — все найденные документы, каждый содержит _id и нормализованные текстовые поля.
    """
    update_max_result_window(theme_index)

    # Обработка случая, когда query_str is None или пустая строка
    if query_str is None or query_str.strip() == "":
        # Используем "all" как значение запроса, чтобы получить все документы
        subqueries = ["all"]
    # Разделяем на подзапросы по запятым, если есть
    elif "," in query_str:
        subqueries = [q.strip() for q in query_str.split(",")]
    else:
        subqueries = [query_str.strip()]
    
    all_results = {}
    total_found = 0

    for idx, subquery in enumerate(subqueries):
        if not subquery:  # пропускаем пустые подстроки после split
            continue
        data = search_single_subquery(
            theme_index,
            subquery,
            min_date=min_date,
            max_date=max_date,
            scroll_time=scroll_time,
            batch_size=batch_size,
            default_fields=default_fields
        )
        print(f"[{idx+1}/{len(subqueries)}] По выражению '{subquery}' найдено: {len(data)} документов")

        for item in data:
            all_results[item['_id']] = item  # переопределение ничего страшного, если дубль

        total_found += len(data)

    print(f"Без дубликатов найдено документов: {len(all_results)} (всего найдено {total_found})")
    return list(all_results.values())


# Установка статуса GPU
async def set_gpu_status(status: str):
    logging.info(f"Устанавливается статус GPU: {status}")
    await redis_db.set("gpu:status", status)

# Сброс статуса GPU
async def reset_gpu_status():
    await set_gpu_status("idle")

# Загрузка словаря с темами
def load_dict_from_pickle(file_name):
    try:
        with open(file_name, 'rb') as f:
            your_dict = pickle.load(f)
        return your_dict
    except Exception as e:
        print(f"Произошла ошибка при загрузке файла: {e}")
        return None

# async def generate_answers(client, prompt):
#     url = "http://localhost:11434/api/generate"
#     payload = {
#         "model": "erwan2/DeepSeek-R1-Distill-Qwen-14B", # Vikhr_Q3
#         "prompt": prompt,
#         "stream": False
#     }
#     async with aiohttp.ClientSession() as session:
#         async with session.post(url, json=payload) as response:
#             if response.status == 200:
#                 response_json = await response.json()
#                 return response_json.get("response", "")
#             else:
#                 print(f"Ошибка при запросе к Ollama: {response.status}")
#                 return None
            
def clean_texts(texts):
    cleaned_texts = []
    for text in texts:
        # Убираем точку в начале, если есть
        if text.startswith('.'):
            text = text[1:]
        # Остальная очистка
        text = text.replace('"', '').replace('«', '').replace('»', '')
        text = re.sub(r'\s+', ' ', text).strip()
        text = text.lower()
        cleaned_texts.append(text)
    return cleaned_texts

from typing import List, Union
import plotly.graph_objects as go
from sklearn.preprocessing import normalize

def visualize_topics_over_time(topic_model,
                               topics_over_time: pd.DataFrame,
                               top_n_topics: int = None,
                               topics: List[int] = None,
                               normalize_frequency: bool = False,
                               custom_labels: Union[bool, str] = False,
                               title: str = "<b>Topics over Time</b>",
                               width: int = 1250,
                               height: int = 450) -> go.Figure:
    """ Visualize topics over time

    Arguments:
        topic_model: A fitted BERTopic instance.
        topics_over_time: The topics you would like to be visualized with the
                          corresponding topic representation
        top_n_topics: To visualize the most frequent topics instead of all
        topics: Select which topics you would like to be visualized
        normalize_frequency: Whether to normalize each topic's frequency individually
        custom_labels: If bool, whether to use custom topic labels that were defined using 
                       `topic_model.set_topic_labels`.
                       If `str`, it uses labels from other aspects, e.g., "Aspect1".
        title: Title of the plot.
        width: The width of the figure.
        height: The height of the figure.

    Returns:
        A plotly.graph_objects.Figure including all traces

    Examples:

    To visualize the topics over time, simply run:

    ```python
    topics_over_time = topic_model.topics_over_time(docs, timestamps)
    topic_model.visualize_topics_over_time(topics_over_time)
    ```

    Or if you want to save the resulting figure:

    ```python
    fig = topic_model.visualize_topics_over_time(topics_over_time)
    fig.write_html("path/to/file.html")
    ```
    <iframe src="../../getting_started/visualization/trump.html"
    style="width:1000px; height: 680px; border: 0px;""></iframe>
    """
    colors = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#D55E00", "#0072B2", "#CC79A7"]

    # Select topics based on top_n and topics args
    freq_df = topic_model.get_topic_freq()
    freq_df = freq_df.loc[freq_df.Topic != -1, :]
    if topics is not None:
        selected_topics = list(topics)
    elif top_n_topics is not None:
        selected_topics = sorted(freq_df.Topic.to_list()[:top_n_topics])
    else:
        selected_topics = sorted(freq_df.Topic.to_list())

    # Prepare data
    if isinstance(custom_labels, str):
        topic_names = [[[str(topic), None]] + topic_model.topic_aspects_[custom_labels][topic] for topic in topics]
        topic_names = ["_".join([label[0] for label in labels[:4]]) for labels in topic_names]
        topic_names = [label if len(label) < 30 else label[:27] + "..." for label in topic_names]
        topic_names = {key: topic_names[index] for index, key in enumerate(topic_model.topic_labels_.keys())}
    elif topic_model.custom_labels_ is not None and custom_labels:
        topic_names = {key: topic_model.custom_labels_[key + topic_model._outliers] for key, _ in topic_model.topic_labels_.items()}
    else:
        topic_names = {key: value[:40] + "..." if len(value) > 40 else value
                       for key, value in topic_model.topic_labels_.items()}
    topics_over_time["Name"] = topics_over_time.Topic.map(topic_names)
    data = topics_over_time.loc[topics_over_time.Topic.isin(selected_topics), :].sort_values(["Topic", "Timestamp"])

    # Add traces
    fig = go.Figure()
    for index, topic in enumerate(data.Topic.unique()):
        trace_data = data.loc[data.Topic == topic, :]
        topic_name = trace_data.Name.values[0]
        words = trace_data.Words.values
        if normalize_frequency:
            y = normalize(trace_data.Frequency.values.reshape(1, -1))[0]
        else:
            y = trace_data.Frequency
        fig.add_trace(go.Scatter(x=trace_data.Timestamp, y=y,
                                 mode='lines',
                                 marker_color=colors[index % 7],
                                 hoverinfo="text",
                                 name=topic_name,
                                 hovertext=[f'<b>Topic {topic}</b><br>Words: {word}' for word in words]))

    # Styling of the visualization
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    fig.update_layout(
        yaxis_title="Количество", # if normalize_frequency else "Frequency",
        title={
            'text': f"{title}",
            'y': .95,
            'x': 0.40,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                size=22,
                color="Black")
        },
        # template="simple_white",
        # width=width,
        # height=height,
        # hoverlabel=dict(
        #     bgcolor="white",
        #     font_size=16,
        #     font_family="Rockwell"
        # ),
        # legend=dict(
        #     title="<b>Global Topic Representation",
        # )
    )
    return fig

from warnings import warn

try:
    import datamapplot
    from matplotlib.figure import Figure
except ImportError:
    warn("Data map plotting is unavailable unless datamapplot is installed.")

    # Create a dummy figure type for typing
    class Figure(object):
        pass


def visualize_document_datamap(
    topic_model,
    docs: List[str] = None,
    topics: List[int] = None,
    embeddings: np.ndarray = None,
    reduced_embeddings: np.ndarray = None,
    custom_labels: Union[bool, str] = False,
    title: str = "Documents and Topics",
    sub_title: Union[str, None] = None,
    width: int = 1200,
    height: int = 750,
    interactive: bool = False,
    enable_search: bool = False,
    topic_prefix: bool = False,
    datamap_kwds: dict = {},
    int_datamap_kwds: dict = {},
) -> Figure:
    """Visualize documents and their topics in 2D as a static plot for publication using
    DataMapPlot.

    Arguments:
        topic_model:  A fitted BERTopic instance.
        docs: The documents you used when calling either `fit` or `fit_transform`.
        topics: A selection of topics to visualize.
                Not to be confused with the topics that you get from `.fit_transform`.
                For example, if you want to visualize only topics 1 through 5:
                `topics = [1, 2, 3, 4, 5]`. Documents not in these topics will be shown
                as noise points.
        embeddings:  The embeddings of all documents in `docs`.
        reduced_embeddings:  The 2D reduced embeddings of all documents in `docs`.
        custom_labels:  If bool, whether to use custom topic labels that were defined using
                       `topic_model.set_topic_labels`.
                       If `str`, it uses labels from other aspects, e.g., "Aspect1".
        title: Title of the plot.
        sub_title: Sub-title of the plot.
        width: The width of the figure.
        height: The height of the figure.
        interactive: Whether to create an interactive plot using DataMapPlot's `create_interactive_plot`.
        enable_search: Whether to enable search in the interactive plot. Only works if `interactive=True`.
        topic_prefix: Prefix to add to the topic number when displaying the topic name.
        datamap_kwds:  Keyword args be passed on to DataMapPlot's `create_plot` function
                       if you are not using the interactive version.
                       See the DataMapPlot documentation for more details.
        int_datamap_kwds:  Keyword args be passed on to DataMapPlot's `create_interactive_plot` function
                           if you are using the interactive version.
                           See the DataMapPlot documentation for more details.

    Returns:
        figure: A Matplotlib Figure object.

    Examples:
    To visualize the topics simply run:

    ```python
    topic_model.visualize_document_datamap(docs)
    ```

    Do note that this re-calculates the embeddings and reduces them to 2D.
    The advised and preferred pipeline for using this function is as follows:

    ```python
    from sklearn.datasets import fetch_20newsgroups
    from sentence_transformers import SentenceTransformer
    from bertopic import BERTopic
    from umap import UMAP

    # Prepare embeddings
    docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = sentence_model.encode(docs, show_progress_bar=False)

    # Train BERTopic
    topic_model = BERTopic().fit(docs, embeddings)

    # Reduce dimensionality of embeddings, this step is optional
    # reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)

    # Run the visualization with the original embeddings
    topic_model.visualize_document_datamap(docs, embeddings=embeddings)

    # Or, if you have reduced the original embeddings already:
    topic_model.visualize_document_datamap(docs, reduced_embeddings=reduced_embeddings)
    ```

    Or if you want to save the resulting figure:

    ```python
    fig = topic_model.visualize_document_datamap(docs, reduced_embeddings=reduced_embeddings)
    fig.savefig("path/to/file.png", bbox_inches="tight")
    ```
    <img src="../../getting_started/visualization/datamapplot.png",
         alt="DataMapPlot of 20-Newsgroups", width=800, height=800></img>
    """
    topic_per_doc = topic_model.topics_

    df = pd.DataFrame({"topic": np.array(topic_per_doc)})
    df["doc"] = docs
    df["topic"] = topic_per_doc

    # Extract embeddings if not already done
    if embeddings is None and reduced_embeddings is None:
        embeddings_to_reduce = topic_model._extract_embeddings(df.doc.to_list(), method="document")
    else:
        embeddings_to_reduce = embeddings

    # Reduce input embeddings
    if reduced_embeddings is None:
        try:
            from umap import UMAP

            umap_model = UMAP(n_neighbors=15, n_components=2, min_dist=0.15, metric="cosine").fit(embeddings_to_reduce)
            embeddings_2d = umap_model.embedding_
        except (ImportError, ModuleNotFoundError):
            raise ModuleNotFoundError(
                "UMAP is required if the embeddings are not yet reduced in dimensionality. Please install it using `pip install umap-learn`."
            )
    else:
        embeddings_2d = reduced_embeddings

    unique_topics = set(topic_per_doc)

    # Prepare text and names
    if isinstance(custom_labels, str):
        names = [[[str(topic), None]] + topic_model.topic_aspects_[custom_labels][topic] for topic in unique_topics]
        names = [" ".join([label[0] for label in labels[:4]]) for labels in names]
        names = [label if len(label) < 30 else label[:27] + "..." for label in names]
    elif topic_model.custom_labels_ is not None and custom_labels:
        names = [topic_model.custom_labels_[topic + topic_model._outliers] for topic in unique_topics]
    else:
        if topic_prefix:
            names = [
                f"Topic-{topic}: " + " ".join([word for word, value in topic_model.get_topic(topic)][:3])
                for topic in unique_topics
            ]
        else:
            names = [" ".join([word for word, value in topic_model.get_topic(topic)][:3]) for topic in unique_topics]

    topic_name_mapping = {topic_num: topic_name for topic_num, topic_name in zip(unique_topics, names)}
    topic_name_mapping[-1] = "Unlabelled"

    # If a set of topics is chosen, set everything else to "Unlabelled"
    if topics is not None:
        selected_topics = set(topics)
        for topic_num in topic_name_mapping:
            if topic_num not in selected_topics:
                topic_name_mapping[topic_num] = "Unlabelled"

    # Map in topic names and plot
    named_topic_per_doc = pd.Series(topic_per_doc).map(topic_name_mapping).values

    if interactive:
        figure = datamapplot.create_interactive_plot(
            embeddings_2d,
            named_topic_per_doc,
            hover_text=docs,
            enable_search=enable_search,
            width=width,
            height=height,
            **int_datamap_kwds,
        )
    else:
        figure, _ = datamapplot.create_plot(
            embeddings_2d,
            named_topic_per_doc,
            figsize=(width / 100, height / 100),
            dpi=100,
            title=title,
            sub_title=sub_title,
            **datamap_kwds,
        )

    return figure


from qdrant_client import QdrantClient
client_qdrant = QdrantClient(
    url="http://localhost:6333",
    timeout=300,  # 5 минут вместо стандартных 60 секунд
    prefer_grpc=False
)



import asyncio
import aiohttp
import numpy as np
import torch
import gc
import time
import os
import json
import pickle
import logging
from datetime import datetime
from collections import defaultdict
from functools import lru_cache
from typing import Dict, List, Tuple, Optional
import concurrent.futures

# Импорты для ML моделей
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
import datamapplot

# Кэш для переиспользования моделей
model_cache = {}
embedding_cache = {}

# Глобальный пул потоков для CPU операций
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

@lru_cache(maxsize=1000)
def cached_truncate_text(text: str, max_tokens: int = 7000) -> str:
    """Кэшированная обрезка текста до указанного количества токенов"""
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    words = text[:max_chars].split()
    return ' '.join(words[:-1])

def get_or_create_model(model_path: str) -> SentenceTransformer:
    """Получает модель из кэша или создает новую"""
    if model_path not in model_cache:
        model_cache[model_path] = SentenceTransformer(model_path, device='cpu')
    return model_cache[model_path]

def post_process_labels(labels: List[str], topic_model: BERTopic) -> List[str]:
    """Финальная постобработка заголовков для обеспечения уникальности и качества"""
    
    processed = []
    seen = {}
    topics_dict = topic_model.get_topics()
    
    for i, label in enumerate(labels):
        # Если заголовок уже использован
        if label in seen:
            # Пытаемся сделать уникальным через ключевые слова
            topic_id = list(topics_dict.keys())[i]
            if topic_id in topics_dict and topics_dict[topic_id]:
                # Берём уникальное ключевое слово из топа
                unique_word = topics_dict[topic_id][0][0].capitalize()
                new_label = f"{label}: {unique_word}"
                processed.append(new_label)
                seen[new_label] = i
            else:
                processed.append(f"{label} (вариант {seen[label] + 1})")
        else:
            processed.append(label)
            seen[label] = i
    
    return processed

async def run_llm_query(task_data: dict):
    """Оптимизированная обработка LLM-запроса"""
    print(f'🚀 НАЧАЛО ЗАДАЧИ: {task_data}')
    logging.info(f"🚀 Запуск задачи {task_data['task_id']}")
    
    # Устанавливаем начальный статус
    await redis_db.hset(f"task:{task_data['task_id']}", mapping={
        "status": "starting",
        "progress": 0,
        "completed_texts": 0,
        "total_texts": 0
    })
    
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    et = time.time()
    
    try:
        # 1. Предварительная загрузка данных
        logging.info("📂 Загрузка индексов...")
        file_path = '/home/dev/tellscope_app/tellscope_backend/data/indexes.pkl'
        
        loop = asyncio.get_event_loop()
        indexes = await loop.run_in_executor(executor, load_dict_from_pickle, file_path)
        logging.info(f"✅ Индексы загружены: {len(indexes)}")
        
        # Обновляем статус
        await redis_db.hset(f"task:{task_data['task_id']}", mapping={
            "status": "loading_data",
            "progress": 5
        })
        
        # Извлекаем даты
        try:
            min_data = task_data['min_data']
            max_data = task_data['max_data']
        except:
            min_data = task_data['min_date']
            max_data = task_data['max_date']

        # 2. Параллельная загрузка данных
        logging.info("🔍 Загрузка данных из Elasticsearch и Qdrant...")
        elasticsearch_task = asyncio.create_task(load_elasticsearch_data(task_data, indexes))
        qdrant_task = asyncio.create_task(load_qdrant_data(indexes[int(task_data['index'])]))
        
        data, (embeddings, qdrant_hashes, texts_from_qdrant) = await asyncio.gather(
            elasticsearch_task, qdrant_task
        )
        logging.info(f"✅ Загружено: {len(data)} документов, {len(embeddings)} эмбеддингов")

        # Обновляем статус
        await redis_db.hset(f"task:{task_data['task_id']}", mapping={
            "status": "filtering_data",
            "progress": 10
        })

        # 3. Быстрая фильтрация
        logging.info("🔧 Фильтрация данных...")
        qdrant_hash_set = set(qdrant_hashes)
        filtered_data = [x for x in data if x.get('hash') in qdrant_hash_set]
        
        if not filtered_data:
            raise ValueError("Нет данных для обработки после фильтрации")
        
        logging.info(f"✅ После фильтрации: {len(filtered_data)} документов")

        # 4. Подготовка данных
        logging.info("📝 Подготовка текстов...")
        maxdata = 5_000_000
        texts = [x['text'] for x in filtered_data][:maxdata]
        urls = [x.get('url', '') for x in filtered_data][:maxdata]
        total_texts = len(texts)
        
        logging.info(f"✅ Подготовлено {total_texts} текстов")

        # Обновляем статус
        await redis_db.hset(f"task:{task_data['task_id']}", mapping={
            "status": "preparing_embeddings",
            "progress": 15,
            "total_texts": total_texts
        })

        # 5. Фильтрация эмбеддингов
        logging.info("🧮 Фильтрация эмбеддингов...")
        hash_to_idx = {hash_val: idx for idx, hash_val in enumerate(qdrant_hashes)}
        filtered_embeddings = []
        
        for x in filtered_data[:maxdata]:
            hash_val = x.get('hash')
            if hash_val in hash_to_idx:
                idx = hash_to_idx[hash_val]
                if idx < len(embeddings):
                    filtered_embeddings.append(embeddings[idx])

        min_len = min(len(texts), len(filtered_embeddings))
        texts, filtered_embeddings, urls = texts[:min_len], filtered_embeddings[:min_len], urls[:min_len]
        embeddings = np.array(filtered_embeddings)
        
        logging.info(f"✅ Эмбеддинги готовы: {embeddings.shape}")

        # Обновляем статус
        await redis_db.hset(f"task:{task_data['task_id']}", mapping={
            "status": "deduplication",
            "progress": 20
        })

        # 6. Дедупликация
        logging.info("🔍 Дедупликация текстов...")
        unique_texts_dict = defaultdict(list)
        for idx, text in enumerate(texts):
            unique_texts_dict[text].append(idx)

        unique_texts = list(unique_texts_dict.keys())
        unique_total = len(unique_texts)
        llm_labels = [None] * len(texts)
        
        logging.info(f"✅ Уникальных текстов: {unique_total} из {total_texts}")

        # 7. Настройка путей
        index_name = indexes[int(task_data['index'])]
        file_location = f'/home/dev/tellscope_app/tellscope_backend/data/{task_data["user_id"]}/bertopic_files_directory/{task_data["folder_name"]}/{index_name}/'
        os.makedirs(file_location, exist_ok=True)
        
        # Сохраняем имя файла в переменную
        pkl_file_name = f'my_list_llm_ans_{index_name}_{current_time}.pkl'
        file_full_path = os.path.join(file_location, pkl_file_name)

        # Обновляем статус перед LLM обработкой
        await redis_db.hset(f"task:{task_data['task_id']}", mapping={
            "status": "llm_processing",
            "progress": 25,
            "unique_texts": unique_total
        })

        # 8. **КРИТИЧНО: LLM обработка**
        logging.info("🤖 Начало LLM обработки...")
        await process_llm_requests_optimized(
            unique_texts, unique_texts_dict, llm_labels, 
            task_data, total_texts, file_full_path, filtered_data
        )
        logging.info("✅ LLM обработка завершена")

        # 9. Параллельное создание моделей UMAP и HDBSCAN
        umap_hdbscan_task = asyncio.create_task(
            create_clustering_models_async(embeddings, texts, llm_labels)
        )
        
        # 10. Подготовка данных для BERTopic
        valid_data = prepare_valid_data(texts, llm_labels, embeddings, urls)
        if not valid_data['texts']:
            raise ValueError("Нет валидных данных для BERTopic")

        # Ждем завершения кластеризации
        umap_model, hdbscan_model, topic_model = await umap_hdbscan_task

        # 11. Обучение BERTopic
        try:
            topics, probs = topic_model.fit_transform(valid_data['texts'], valid_data['embeddings'])
        except ValueError as e:
            if "min_df" not in str(e) and "max_df" not in str(e):
                raise
            logging.warning(f"BERTopic vectorizer fallback after: {e}")
            topic_model.vectorizer_model = CountVectorizer(
                analyzer='word',
                token_pattern=r'(?u)\b\w+\b',
                lowercase=True,
                min_df=1,
                max_df=1.0,
            )
            topics, probs = topic_model.fit_transform(valid_data['texts'], valid_data['embeddings'])

        # Обновляем статус
        await redis_db.hset(f"task:{task_data['task_id']}", mapping={
            "embedding_status": "done",
            "embedding_completed": len(embeddings),
            "embedding_progress": 100,
            "topics_found": len(set(topics))
        })

        # 12. Параллельная генерация заголовков топиков
        topic_labels = await generate_topic_labels_batch(topic_model)
        # НОВОЕ: Постобработка для финальной проверки
        topic_labels = post_process_labels(topic_labels, topic_model)
        topic_model.set_topic_labels(topic_labels)

        # 13. Параллельное сохранение результатов
        save_tasks = [
            save_visualizations_async(topic_model, valid_data, file_location, index_name, current_time, umap_model),
            save_model_and_labels_async(topic_model, topics, file_location, index_name, current_time)
        ]
        
        await asyncio.gather(*save_tasks)

        # 14. Создание CSV-файла и получение его имени
        csv_filename = await create_result_csv(
            file_full_path=file_full_path,
            filtered_data=filtered_data,
            file_location=file_location,
            index_name=index_name,
            current_time=current_time
        )

        # 15. Обновление пользовательских данных - ИСПРАВЛЕНО: 7 параметров
        await update_user_data(
            task_data=task_data,
            index_name=index_name,
            current_time=current_time,
            start_time=et,
            total_texts=total_texts,
            unique_total=unique_total,
            pkl_filename=pkl_file_name  # 7-й параметр
        )

    except Exception as e:
        logging.error(f"Ошибка при обработке задачи {task_data['task_id']}: {e}")
        await redis_db.hset(f"task:{task_data['task_id']}", mapping={"status": "failed", "error": str(e)})
        raise
    
    finally:
        await cleanup_resources(task_data, indexes, current_time)


async def update_user_data(
    task_data: dict, 
    index_name: str, 
    current_time: str, 
    start_time: float, 
    total_texts: int, 
    unique_total: int,
    pkl_filename: str = None  # 7-й параметр
):
    """
    Обновление данных пользователя с информацией о созданных файлах
    """
    try:
        elapsed_time = time.time() - start_time
        total_seconds = int(elapsed_time)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        execution_all_time = f"{hours} ч. {minutes} мин. {seconds} сек."

        user_data = await redis_db.execute_command('HGETALL', task_data['user_id'])
        user_data = {key.decode('utf-8'): value.decode('utf-8') for key, value in user_data.items()}

        creation_date = datetime.strptime(current_time, "%Y%m%d_%H%M%S")
        
        # Формируем имена файлов
        csv_filename = f'result_graph_{index_name}_{current_time}.csv'
        
        file_info = {
            "html-file": f"{index_name}_{current_time}.html",
            "model-file": f'topic_model_{index_name}_{current_time}',
            "pkl-file": pkl_filename,  # Добавляем PKL файл
            "csv-file": csv_filename,   # Добавляем CSV файл
            "creation_date": str(creation_date.strftime("%Y-%m-%d %H:%M:%S")),
            "execution_all_time": execution_all_time,
            "min_data": task_data.get('min_data', ''),
            "max_data": task_data.get('max_data', ''),
            "index_number": int(task_data['index']),
            "task_id": task_data['task_id'],
            "query_str": task_data.get('query_str', ''),
            "count_texts": total_texts,
            "unique_texts": unique_total,
            "promt_question": task_data.get('promt_question', ''),
            "status": "completed"
        }

        if user_data and "bertopic_files_directory" in user_data:
            user_folders = json.loads(user_data["bertopic_files_directory"])
        else:
            user_folders = {}

        folder_name = task_data['folder_name']
        if folder_name in user_folders:
            user_folders[folder_name].append(file_info)
        else:
            user_folders[folder_name] = [file_info]

        serialized_folders = json.dumps(user_folders)
        await redis_db.hset(task_data["user_id"], "bertopic_files_directory", serialized_folders)
        
        logging.info(f"✅ Данные пользователя обновлены: {file_info}")
        try:
            from mlops.runtime import finish_llm_run
            finish_llm_run(task_data, status="done", artifact_dir=file_location)
        except Exception:
            pass
        
    except Exception as e:
        logging.error(f"❌ Ошибка при обновлении данных пользователя: {e}")
        raise


async def create_result_csv(
    file_full_path: str,
    filtered_data: list,
    file_location: str,
    index_name: str,
    current_time: str
) -> str:  # Указываем, что возвращаем строку
    """
    Асинхронное создание CSV-файла с результатами анализа
    """
    loop = asyncio.get_event_loop()
    
    def _create_csv():
        try:
            # Загружаем метки из pkl файла
            with open(file_full_path, 'rb') as file:
                labels_data = pickle.load(file)
            
            # Обрабатываем разные форматы данных
            if isinstance(labels_data, dict) and 'labels' in labels_data:
                labels = list(zip(labels_data['hashes'], labels_data['labels']))
            else:
                labels = labels_data
            
            # Создаем DataFrame с метками
            labels_df = pd.DataFrame(labels, columns=['hash', 'labels'])
            
            # Создаем DataFrame из filtered_data
            data_df = pd.DataFrame(filtered_data)
            
            # Объединяем данные
            result = data_df.merge(labels_df, on='hash', how='inner')
            
            # Проверяем наличие необходимых колонок и выбираем только существующие
            required_columns = [
                'timeCreate', 'title', 'url', 'hubtype', 'type', 'authorObject',
                'commentsCount', 'audienceCount', 'repostsCount', 'likesCount',
                'er', 'viewsCount', 'duplicateCount', 'country', 'region', 'city', 'labels'
            ]
            
            # Фильтруем только существующие колонки
            available_columns = [col for col in required_columns if col in result.columns]
            result = result[available_columns]
            
            # Преобразуем timeCreate если колонка существует
            if 'timeCreate' in result.columns:
                result['timeCreate'] = pd.to_datetime(result['timeCreate'], unit='s', errors='coerce')
            
            # Обрабатываем authorObject если колонка существует
            if 'authorObject' in result.columns:
                try:
                    author_df = pd.json_normalize(result['authorObject'])
                    
                    # Переименовываем 'url' в 'author_url' если существует
                    if 'url' in author_df.columns:
                        author_df = author_df.rename(columns={'url': 'author_url'})
                    
                    # Объединяем с исходным DataFrame
                    result = pd.concat([result.drop('authorObject', axis=1), author_df], axis=1)
                except Exception as e:
                    logging.warning(f"Не удалось обработать authorObject: {e}")
            
            # Формируем имя файла
            csv_filename = f'result_graph_{index_name}_{current_time}.csv'
            csv_full_path = os.path.join(file_location, csv_filename)
            
            # Сохраняем CSV
            result.to_csv(csv_full_path, index=False, encoding='utf-8')
            
            logging.info(f"CSV файл успешно создан: {csv_full_path}")
            return csv_filename  # Возвращаем имя файла
            
        except Exception as e:
            logging.error(f"Ошибка при создании CSV файла: {e}")
            raise
    
    return await loop.run_in_executor(executor, _create_csv)


# async def update_user_data(
#     task_data: dict, 
#     index_name: str, 
#     current_time: str, 
#     et: float, 
#     total_texts: int, 
#     unique_total: int,
#     pkl_filename: str = None  # 7-й параметр уже есть
# ):
#     """
#     Обновление данных пользователя с информацией о созданных файлах
#     """
#     try:
#         execution_time = time.time() - et
        
#         # Формируем имена файлов
#         csv_filename = f'result_graph_{index_name}_{current_time}.csv'
        
#         user_data = {
#             "task_id": task_data['task_id'],
#             "user_id": task_data['user_id'],
#             "folder_name": task_data['folder_name'],
#             "index_name": index_name,
#             "total_texts": total_texts,
#             "unique_texts": unique_total,
#             "execution_time": round(execution_time, 2),
#             "timestamp": current_time,
#             "pkl_file": pkl_filename,  # Добавляем имя pkl файла
#             "csv_file": csv_filename,   # Добавляем имя csv файла
#             "status": "completed"
#         }
        
#         # Сохраняем в Redis или базу данных
#         await redis_db.hset(
#             f"user_results:{task_data['user_id']}:{task_data['task_id']}", 
#             mapping=user_data
#         )
        
#         logging.info(f"Данные пользователя обновлены: {user_data}")
        
#     except Exception as e:
#         logging.error(f"Ошибка при обновлении данных пользователя: {e}")
#         raise

# Вспомогательные асинхронные функции

async def load_elasticsearch_data(task_data: dict, indexes: dict) -> list:
    """Асинхронная загрузка данных из Elasticsearch"""
    loop = asyncio.get_event_loop()
    
    def _load_data():
        data = []
        if task_data['query_str'] and task_data['query_str'] != 'all':
            search = task_data['query_str'].split(',')
            for query in search:
                data.extend(elastic_query(theme_index=indexes[int(task_data['index'])], query_str=query))
        else:
            min_data = task_data.get('min_data') or task_data.get('min_date')
            max_data = task_data.get('max_data') or task_data.get('max_date')
            data = elastic_query(
                theme_index=indexes[int(task_data['index'])],
                query_str='all',
                min_date=min_data,
                max_date=max_data
            )
        return data
    
    return await loop.run_in_executor(executor, _load_data)

async def load_qdrant_data(collection_name: str) -> Tuple[List, List, List]:
    """Асинхронная загрузка данных из Qdrant"""
    loop = asyncio.get_event_loop()
    
    def _load_qdrant():
        embeddings, qdrant_hashes, texts_from_qdrant = [], [], []
        all_points = []
        next_offset = None

        # Загружаем данные батчами
        while True:
            batch_points, next_offset = client_qdrant.scroll(
                collection_name=collection_name,
                with_vectors=True,
                limit=10000,  # Оптимизированный размер батча
                offset=next_offset,
            )
            all_points.extend(batch_points)
            if not next_offset:
                break

        # Обрабатываем точки
        for point in all_points:
            if point.payload and 'metadata' in point.payload and 'hash' in point.payload['metadata']:
                qdrant_hashes.append(point.payload['metadata']['hash'])
                texts_from_qdrant.append(point.payload.get('text', ''))
                if point.vector is not None:
                    embeddings.append(point.vector)

        return embeddings, qdrant_hashes, texts_from_qdrant
    
    return await loop.run_in_executor(executor, _load_qdrant)


async def process_llm_requests_optimized(unique_texts: List[str], unique_texts_dict: Dict, 
                                       llm_labels: List, task_data: dict, total_texts: int, 
                                       file_full_path: str, filtered_data: List):
    """Оптимизированная обработка LLM запросов"""
    
    logging.info(f"🚀 Начало LLM обработки: {len(unique_texts)} уникальных текстов")
    
    # Проверка доступности LLM API через serving lock
    try:
        from mlops.gateway import ping_vllm
        if await ping_vllm():
            logging.info("✅ LLM API доступен")
        else:
            logging.error("⚠️ LLM API недоступен")
            raise RuntimeError("vLLM /v1/models failed")
    except Exception as e:
        logging.error(f"❌ LLM API недоступен: {e}")
        raise

    semaphore = asyncio.Semaphore(20)
    response_cache = {}
    
    async def generate_answer_cached(text: str, question: str, system_prompt: str = None) -> str:
        """Кэшированная версия generate_answer"""
        cache_key = hash(f"{text[:100]}{question}{system_prompt}")
        
        if cache_key in response_cache:
            return response_cache[cache_key]
        
        if not text or len(text) < 8:
            result = "Короткий текст"
        elif len(text) > 25000:
            result = "Длинный текст"
        else:
            result = await generate_answer_with_retries(text, question, system_prompt)
        
        response_cache[cache_key] = result
        return result

    async def generate_answer_with_retries(text: str, question: str, system_prompt: str = None, max_retries: int = 2) -> str:
        """Улучшенная версия generate_answer с повторными попытками"""
        from mlops.gateway import GatewayError, achat

        if system_prompt and system_prompt.strip():
            system_line = system_prompt.strip()
        else:
            try:
                from mlops.lock import prompt_id as lock_prompt_id
                from mlops.prompts import render_prompt
                system_line = render_prompt(lock_prompt_id("llm_run_default", "llm_run_default_v1"))
            except Exception:
                system_line = "Ты отвечаешь очень кратко, только на поставленный вопрос. Только факт из текста, не повторяй формулировки вопроса."
        user_content = f"Текст: {cached_truncate_text(text, 7500)}\n\nВопрос: {question.strip()}\n\nОтвет:"

        for attempt in range(max_retries + 1):
            try:
                result = await achat(
                    provider="vllm",
                    messages=[
                        {"role": "system", "content": system_line},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=0.7,
                    max_tokens=None,
                    timeout=60,
                    extra={"top_p": 0.8, "chat_template_kwargs": {"enable_thinking": False}},
                )
                answer = result.content.strip().rstrip(".").strip()
                return answer if answer else "Модель не ответила"
            except GatewayError as e:
                if e.status_code == 0 and "timeout" in str(e).lower():
                    if attempt < max_retries:
                        await asyncio.sleep(0.1 * (attempt + 1))
                        continue
                    return "Timeout ошибка"
                if attempt < max_retries:
                    await asyncio.sleep(0.1 * (attempt + 1))
                    continue
                return f"Ошибка API: {e.status_code or str(e)}"
            except Exception as e:
                if attempt < max_retries:
                    await asyncio.sleep(0.1 * (attempt + 1))
                    continue
                return f"Ошибка: {str(e)}"

        return "Модель не ответила"

    async def process_text_with_semaphore(text: str, question: str, system_prompt: str):
        async with semaphore:
            return await generate_answer_cached(text, question, system_prompt)

    # Параллельная обработка с увеличенным размером батча
    BATCH_SIZE = 100  # Увеличено для лучшей производительности
    SAVE_THRESHOLD = 200
    
    completed = 0
    new_count_since_save = 0
    
    for i in range(0, len(unique_texts), BATCH_SIZE):
        batch_texts = unique_texts[i:i+BATCH_SIZE]
        
        # Создаем задачи для параллельной обработки
        tasks = []
        for j, text in enumerate(batch_texts):
            task = process_text_with_semaphore(
                text,
                task_data.get("promt_question", ""),
                task_data.get("system_prompt", "")
            )
            tasks.append((i + j, task))

        # Параллельно выполняем все задачи в батче
        results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)

        # Обрабатываем результаты
        for (original_idx, _), result in zip(tasks, results):
            if isinstance(result, Exception):
                result = "Ошибка обработки"

            original_text = unique_texts[original_idx]
            indices = unique_texts_dict[original_text]

            for idx in indices:
                llm_labels[idx] = result

            completed += len(indices)
            new_count_since_save += len(indices)

        # Обновляем прогресс
        progress = round((completed / total_texts) * 100, 1)
        await redis_db.hset(f"task:{task_data['task_id']}", mapping={
            "status": "in progress",
            "completed_texts": completed,
            "progress": progress
        })

        # Асинхронное сохранение
        if new_count_since_save >= SAVE_THRESHOLD:
            asyncio.create_task(save_labels_async(file_full_path, task_data, llm_labels, filtered_data))  # Добавили filtered_data
            new_count_since_save = 0

        # Короткая пауза между батчами
        await asyncio.sleep(0.01)

    # Финальное сохранение
    if new_count_since_save > 0:
        await save_labels_async(file_full_path, task_data, llm_labels, filtered_data)  # Добавили filtered_data

async def save_labels_async(file_full_path: str, task_data: dict, llm_labels: List, filtered_data: List):
    """Асинхронное сохранение меток"""
    loop = asyncio.get_event_loop()
    
    def _save():
        # Создаем список кортежей (hash, label)
        filtered_data_hashes = [x['hash'] for x in filtered_data]
        labels_to_save = list(zip(filtered_data_hashes, llm_labels))
        
        with open(file_full_path, 'wb') as file:
            pickle.dump(labels_to_save, file)
        
        logging.info(f"Сохранено {len(labels_to_save)} меток в {file_full_path}")
    
    await loop.run_in_executor(executor, _save)

async def create_clustering_models_async(embeddings: np.ndarray, texts: List[str], llm_labels: List):
    """Асинхронное создание моделей кластеризации"""
    loop = asyncio.get_event_loop()
    
    def _create_models():
        # UMAP модель
        n_neighbors = min(20, len(embeddings) - 1)
        umap_model = UMAP(
            n_neighbors=n_neighbors,
            n_components=min(len(embeddings), 2),
            min_dist=0.0,
            metric="cosine",
            random_state=42,
            n_jobs=1  # Ограничиваем использование CPU
        )
        
        # HDBSCAN модель
        min_cluster_size = min(5, len(embeddings) // 2)
        hdbscan_model = HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
            core_dist_n_jobs=1
        )
        
        # BERTopic модель
        representation_model = MaximalMarginalRelevance(diversity=0.8)
        # c-TF-IDF fits CountVectorizer on one concatenated document per topic.
        # min_df=2 + max_df=0.9 raises "max_df corresponds to < documents than min_df"
        # when HDBSCAN finds fewer than 3 topics (typical for short query slices).
        n_docs = max(len(texts) if texts is not None else 0, len(embeddings) if embeddings is not None else 0, 1)
        vectorizer_model = CountVectorizer(
            analyzer='word',
            token_pattern=r'(?u)\b\w+\b',
            lowercase=True,
            min_df=1,
            max_df=1.0,
            ngram_range=(1, 2) if n_docs < 80 else (1, 3),
            stop_words=None
        )
        
        topic_model = BERTopic(
            embedding_model=None,
            verbose=True,
            representation_model=representation_model,
            vectorizer_model=vectorizer_model,
            min_topic_size=2 if n_docs < 40 else 3,
            calculate_probabilities=True
        )
        
        return umap_model, hdbscan_model, topic_model
    
    return await loop.run_in_executor(executor, _create_models)

def prepare_valid_data(texts: List[str], llm_labels: List, embeddings: np.ndarray, urls: List[str]) -> Dict:
    """Подготовка валидных данных для обработки"""
    valid_data = {
        'texts': [],
        'embeddings': [],
        'urls': [],
        'indices': []
    }
    
    for i, (text, label) in enumerate(zip(texts, llm_labels)):
        if label and isinstance(label, str) and len(label.strip()) > 0:
            valid_data['texts'].append(label)
            valid_data['embeddings'].append(embeddings[i])
            valid_data['urls'].append(urls[i])
            valid_data['indices'].append(i)
    
    valid_data['embeddings'] = np.array(valid_data['embeddings'])
    return valid_data



def clean_theme_text(label) -> str:
    """Drop internal hash ids and 'Тематика текста:' prefixes from LLM labels."""
    import re
    if isinstance(label, (tuple, list)):
        label = label[-1] if len(label) >= 2 else (label[0] if label else "")
    if label is None:
        return ""
    text = str(label).strip()
    text = re.sub(r"^[0-9a-fA-F]{32}\d{8}[,:;\s-]*", "", text).strip()
    prefixes = (
        "Тематика текста:",
        "Тематика текста",
        "Тематика:",
        "Тема текста:",
        "Тема:",
    )
    lowered = text.lower()
    for prefix in prefixes:
        if lowered.startswith(prefix.lower()):
            text = text[len(prefix):].strip(" :,-")
            break
    return text


def unpack_saved_llm_labels(saved_data):
    """Normalize pickle formats to (hashes|None, list[str] labels)."""
    hashes, labels = None, []
    if isinstance(saved_data, dict) and "labels" in saved_data:
        hashes = saved_data.get("hashes")
        labels = saved_data.get("labels") or []
    elif isinstance(saved_data, list) and saved_data:
        first = saved_data[0]
        if isinstance(first, (tuple, list)) and len(first) >= 2:
            hashes = [item[0] for item in saved_data]
            labels = [item[1] for item in saved_data]
        else:
            labels = saved_data
    else:
        labels = saved_data or []
    return hashes, [clean_theme_text(x) for x in labels]

def clean_label(label: str) -> str:
    """Очистка и форматирование заголовка"""
    # Удаляем лишние символы
    label = label.strip().strip('"').strip("'").strip()
    
    # Удаляем типичные префиксы
    prefixes_to_remove = [
        "Заголовок:", "Тема:", "Название:", 
        "Краткое название:", "Короткий заголовок:"
    ]
    for prefix in prefixes_to_remove:
        if label.lower().startswith(prefix.lower()):
            label = label[len(prefix):].strip()
    
    # Ограничиваем длину (5-8 слов оптимально)
    words = label.split()
    if len(words) > 8:
        label = ' '.join(words[:8]) + '...'
    elif len(words) < 2:
        return label.capitalize()

    # Капитализация первой буквы
    return label[0].upper() + label[1:] if label else label


async def generate_topic_labels_batch(topic_model: BERTopic) -> List[str]:
    """Батчевая генерация заголовков топиков с контролем уникальности"""
    topics_dict = topic_model.get_topics()
    
    # Для контроля уникальности
    used_labels = set()
    semaphore = asyncio.Semaphore(10)
    
    async def generate_single_label(topic_id, topic_words, semaphore):
        async with semaphore:
            # Обработка топика -1 (шум)
            if topic_id == -1:
                return "Разные темы (нет общего)"
            
            # Проверка наличия ключевых слов
            if not topic_words or len(topic_words) == 0:
                return "Общая тема"
            
            # Фильтрация пустых токенов
            valid_tokens = [token[0] for token in topic_words if token[0] and len(token[0].strip()) > 0]
            
            if not valid_tokens:
                return "Разные темы (нет общего)"
            
            # Берём топ-15 ключевых слов с весами
            top_keywords = valid_tokens[:15]
            weights = [token[1] for token in topic_words[:15] if token[0] in top_keywords]
            
            # Формируем контекст с весами
            keywords_with_weights = [
                f"{word} ({weight:.2f})" 
                for word, weight in zip(top_keywords, weights)
            ]
            
            return await generate_topic_label_optimized(
                key_words=" | ".join(top_keywords),
                keywords_with_weights=keywords_with_weights,
                topic_id=topic_id,
                used_labels=used_labels
            )

    # Параллельно генерируем все заголовки
    tasks = [
        generate_single_label(topic_id, topic_words, semaphore) 
        for topic_id, topic_words in topics_dict.items()
    ]
    labels = await asyncio.gather(*tasks)
    
    # Обрабатываем результаты и проверяем уникальность
    processed_labels = []
    label_counts = defaultdict(int)
    
    for i, label in enumerate(labels):
        if label and label not in ["Ошибка генерации", "Модель не ответила", "Разные темы (нет общего)"]:
            # Очистка и форматирование
            cleaned = clean_label(label)
            
            # Проверка уникальности
            if cleaned in label_counts:
                label_counts[cleaned] += 1
                cleaned = f"{cleaned} ({label_counts[cleaned]})"
            else:
                label_counts[cleaned] = 1
            
            processed_labels.append(cleaned)
        else:
            processed_labels.append("Разные темы (нет общего)")
    
    return processed_labels


def validate_label(label: str, key_words: str) -> bool:
    """Валидация качества сгенерированного заголовка"""
    if not label or len(label.strip()) == 0:
        return False
    
    # Проверка длины (не меньше 5 символов, не больше 100)
    if len(label) < 5 or len(label) > 100:
        return False
    
    # Проверка количества слов (2-10)
    words = label.split()
    if len(words) < 2 or len(words) > 10:
        return False
    
    # Проверка на "мусорные" ответы
    bad_patterns = [
        "ключевые слова",
        "нет данных",
        "ошибка",
        "не удалось",
        "повторите",
        "заголовок:",
        "тема:"
    ]
    
    label_lower = label.lower()
    if any(pattern in label_lower for pattern in bad_patterns):
        return False
    
    # Проверка, что заголовок не является просто перечислением ключевых слов
    keywords_list = [kw.strip() for kw in key_words.split('|')][:5]
    matches = sum(1 for kw in keywords_list if kw.lower() in label_lower)
    
    # Если в заголовке просто перечислены все ключевые слова - плохой заголовок
    if matches >= len(keywords_list) and len(keywords_list) > 2:
        return False
    
    return True

async def generate_topic_label_optimized(
    key_words: str, 
    keywords_with_weights: List[str] = None,
    topic_id: int = None,
    used_labels: set = None,
    max_retries: int = 2
) -> str:
    """Оптимизированная генерация заголовков с контролем качества"""
    
    # Проверка входных данных
    if not key_words or len(key_words.strip()) == 0:
        return "Разные темы (нет общего)"
    
    clean_keywords = key_words.replace('|', '').replace(' ', '').strip()
    if len(clean_keywords) == 0:
        return "Разные темы (нет общего)"
    
    from mlops.gateway import GatewayError, achat
    from mlops.lock import prompt_id as lock_prompt_id
    from mlops.prompts import render_prompt

    try:
        system_prompt = render_prompt(lock_prompt_id("llm_topic_label", "topic_label_v1"))
    except Exception:
        system_prompt = "Ты эксперт по созданию коротких заголовков тематик. Отвечай ТОЛЬКО заголовком на русском, 3-6 слов."

    # Формируем контекст с весами для лучшего понимания
    context = f"Ключевые слова: {key_words}"
    if keywords_with_weights:
        context += f"\n\nСлова с важностью: {', '.join(keywords_with_weights[:10])}"
    
    user_content = f"""{context}

Создай краткий заголовок (3-6 слов), который точно отражает тему:"""

    for attempt in range(max_retries + 1):
        try:
            result = await achat(
                provider="vllm",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.5,
                max_tokens=50,
                timeout=30,
                extra={"top_p": 0.85, "chat_template_kwargs": {"enable_thinking": False}},
            )
            label = result.content.strip()
            if validate_label(label, key_words):
                return label
            if attempt < max_retries:
                await asyncio.sleep(0.2)
                continue
            return "Разные темы (нет общего)"
        except GatewayError:
            if attempt < max_retries:
                await asyncio.sleep(0.2)
                continue
            return "Разные темы (нет общего)"
        except Exception:
            if attempt < max_retries:
                await asyncio.sleep(0.2)
                continue
            return "Разные темы (нет общего)"

    return "Разные темы (нет общего)"

async def save_visualizations_async(topic_model: BERTopic, valid_data: Dict, 
                                  file_location: str, index_name: str, 
                                  current_time: str, umap_model: UMAP):
    """Асинхронное сохранение визуализаций"""
    loop = asyncio.get_event_loop()
    
    def _save_visualizations():
        try:
            # Создание UMAP эмбеддингов для визуализации
            valid_embeddings_umap = umap_model.fit_transform(valid_data['embeddings'])
            
            # BERTopic визуализация
            new_filename = f"{index_name}_{current_time}.html"
            fig = topic_model.visualize_documents(
                valid_data['texts'], 
                reduced_embeddings=valid_embeddings_umap,
                hide_annotations=True, 
                hide_document_hover=False,
                custom_labels=True, 
                title='Документы и тематики'
            )
            
            fig.write_html(os.path.join(file_location, new_filename))
            
            # Datamapplot визуализация
            if len(valid_embeddings_umap) > 3:
                try:
                    plot = datamapplot.create_interactive_plot(
                        valid_embeddings_umap,
                        valid_data['texts'],
                        font_family="Playfair Display SC",
                        hover_text=valid_data['urls'],
                        on_click="window.open(`{hover_text}`)",
                        enable_search=True,
                    )
                    
                    filename = f'datamapplot_{new_filename}'
                    plot.save(os.path.join(file_location, filename))
                except Exception as e:
                    print(f"Ошибка при создании datamapplot: {e}")
        
        except Exception as e:
            print(f"Ошибка при сохранении визуализаций: {e}")
    
    await loop.run_in_executor(executor, _save_visualizations)

async def save_model_and_labels_async(topic_model: BERTopic, topics: List, 
                                    file_location: str, index_name: str, current_time: str):
    """Асинхронное сохранение модели и меток"""
    loop = asyncio.get_event_loop()
    
    def _save_model_and_labels():
        try:
            # Сохранение модели
            filename = f'topic_model_{index_name}_{current_time}'
            empty_embedding_model = get_or_create_model("deepvk/USER2-base")
            
            os.chdir(file_location)
            topic_model.save(
                filename, 
                serialization="safetensors", 
                save_ctfidf=True, 
                save_embedding_model=empty_embedding_model
            )
            
            # Сохранение topic labels
            loaded_model = BERTopic.load(filename)
            df_topic = loaded_model.get_topic_info()[['Topic', 'CustomName']]
            dct_df_topic = dict(zip(df_topic['Topic'], df_topic['CustomName']))
            text_labels = [dct_df_topic[label] for label in topics]
            
            label_filename = f'topic_names_{index_name}_{current_time}.pkl'
            with open(os.path.join(file_location, label_filename), 'wb') as file:
                pickle.dump(text_labels, file)
                
        except Exception as e:
            print(f"Ошибка при сохранении модели: {e}")
    
    await loop.run_in_executor(executor, _save_model_and_labels)

async def update_user_data(
    task_data: dict, 
    index_name: str, 
    current_time: str, 
    start_time: float, 
    total_texts: int, 
    unique_total: int,
    pkl_filename: str = None,
    csv_filename: str = None  # ✅ Добавляем параметр
):
    """
    Обновление данных пользователя с информацией о созданных файлах
    """
    try:
        elapsed_time = time.time() - start_time
        total_seconds = int(elapsed_time)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        execution_all_time = f"{hours} ч. {minutes} мин. {seconds} сек."

        user_data = await redis_db.execute_command('HGETALL', task_data['user_id'])
        user_data = {key.decode('utf-8'): value.decode('utf-8') for key, value in user_data.items()}

        creation_date = datetime.strptime(current_time, "%Y%m%d_%H%M%S")
        
        # ✅ Используем переданное имя файла или формируем его
        if not csv_filename:
            csv_filename = f'result_graph_{index_name}_{current_time}.csv'
        
        file_info = {
            "html-file": f"{index_name}_{current_time}.html",
            "model-file": f'topic_model_{index_name}_{current_time}',
            "pkl-file": pkl_filename,
            "csv-file": csv_filename,
            "creation_date": str(creation_date.strftime("%Y-%m-%d %H:%M:%S")),
            "execution_all_time": execution_all_time,
            "min_data": task_data.get('min_data', ''),
            "max_data": task_data.get('max_data', ''),
            "index_number": int(task_data['index']),
            "task_id": task_data['task_id'],
            "query_str": task_data.get('query_str', ''),
            "count_texts": total_texts,
            "unique_texts": unique_total,
            "promt_question": task_data.get('promt_question', ''),
            "status": "completed"
        }

        # ✅ Обновляем bertopic_files_directory
        if user_data and "bertopic_files_directory" in user_data:
            user_folders = json.loads(user_data["bertopic_files_directory"])
        else:
            user_folders = {}

        folder_name = task_data['folder_name']
        if folder_name in user_folders:
            user_folders[folder_name].append(file_info)
        else:
            user_folders[folder_name] = [file_info]

        serialized_folders = json.dumps(user_folders)
        await redis_db.hset(task_data["user_id"], "bertopic_files_directory", serialized_folders)

        # ✅ НОВОЕ: Обновляем csv_files_directory
        file_location = f'/home/dev/tellscope_app/tellscope_backend/data/{task_data["user_id"]}/bertopic_files_directory/{folder_name}/{index_name}/'
        csv_full_path = os.path.join(file_location, csv_filename)
        
        # Получаем существующие CSV файлы
        if user_data and "csv_files_directory" in user_data:
            csv_folders = json.loads(user_data["csv_files_directory"])
        else:
            csv_folders = {}
        
        # Ключ папки (относительный путь)
        csv_folder_key = f"{folder_name}/{index_name}"
        
        # Создаем информацию о CSV файле
        csv_file_info = {
            "file": csv_filename,
            "full_path": csv_full_path,
            "relative_path": f"{csv_folder_key}/{csv_filename}",
        }
        
        # Добавляем размер файла
        try:
            if os.path.exists(csv_full_path):
                csv_file_info["size"] = os.path.getsize(csv_full_path)
            else:
                csv_file_info["size"] = 0
        except:
            csv_file_info["size"] = 0
        
        # Добавляем в структуру
        if csv_folder_key in csv_folders:
            # Проверяем, нет ли уже такого файла
            existing_files = [f["file"] for f in csv_folders[csv_folder_key]]
            if csv_filename not in existing_files:
                csv_folders[csv_folder_key].append(csv_file_info)
        else:
            csv_folders[csv_folder_key] = [csv_file_info]
        
        # Сохраняем в Redis
        serialized_csv_folders = json.dumps(csv_folders)
        await redis_db.hset(task_data["user_id"], "csv_files_directory", serialized_csv_folders)
        
        logging.info(f"✅ Данные пользователя обновлены: {file_info}")
        logging.info(f"✅ CSV файл зарегистрирован: {csv_file_info}")
        
    except Exception as e:
        logging.error(f"❌ Ошибка при обновлении данных пользователя: {e}")
        raise

async def cleanup_resources(task_data: dict, indexes: dict, current_time: str):
    """Очистка ресурсов и финальное обновление статуса"""
    try:
        # Очистка GPU памяти
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Финальное обновление статуса
        await redis_db.hset(f"task:{task_data['task_id']}", mapping={
            "final_status": "done",
            "html-file": f"{indexes[int(task_data['index'])]}_{current_time}.html",
            "folder_name": task_data['folder_name']
        })

        await reset_gpu_status()
        logging.info(f"GPU статус сброшен. Задача {task_data['task_id']} завершена.")
        
    except Exception as e:
        logging.error(f"Ошибка при очистке ресурсов: {e}")

# Дополнительные оптимизации

async def reset_all_gpu_processes():
    """Асинхронный сброс GPU процессов"""
    import subprocess
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        executor, 
        subprocess.call, 
        "nvidia-smi | awk '/[0-9]+/ {print $5}' | xargs -r kill -9", 
        True  # shell=True
    )

# Закрытие пула потоков при завершении приложения
def cleanup_executor():
    executor.shutdown(wait=True)


import uuid
import asyncio 
# import redis  # redis-py >= 4.x (или 5.x)
import traceback
import redis.asyncio

# @app.on_event("startup")
# async def startup_event():
#     try:
#         await redis_db.ping()
#         logging.info("Redis подключен!")
#         # Инициализируем статус GPU при старте
#         existing_status = await redis_db.get("gpu:status")
#         if not existing_status:
#             logging.info("Инициализация статуса GPU как 'idle'.")
#             await redis_db.set("gpu:status", "idle")
#     except Exception as e:
#         logging.error(f"Ошибка подключения к Redis: {e}")
#         raise RuntimeError("Не удалось подключиться к Redis")


# @app.on_event("shutdown")
# async def shutdown_event():
#     await redis_db.close()


@app.get("/is_gpu_busy", tags=['metrics'])
async def is_gpu_busy() -> bool:
    try:
        status = await redis_db.get("gpu:status")
        if status is None:
            logging.warning("Ключ gpu:status отсутствует в Redis.")
        else:
            logging.info(f"Текущий статус GPU из Redis: {status}")
        return status == b"busy"
    except Exception as e:
        logging.error(f"Ошибка при проверке статуса GPU: {e}")
        return False


# Установка статуса GPU
async def set_gpu_status(status: str):
    logging.info(f"Устанавливается статус GPU: {status}")
    await redis_db.set("gpu:status", status)


# Сброс статуса GPU
async def reset_gpu_status():
    await set_gpu_status("idle")


# Обработка LLM задач
@app.post("/llm-run/", tags=['ai analytics'])
async def llm_run(
    analysis_request: AnalysisRequest,
    background_tasks: BackgroundTasks
):
    try:
        task_id = str(uuid.uuid4())
        task_data = {
            "task_id": task_id,
            "user_id": str(analysis_request.user_id),
            "folder_name": str(analysis_request.folder_name),
            "index": str(analysis_request.index),
            "query_str": analysis_request.query_str or "all", 
            "min_data": str(analysis_request.min_date),
            "max_data": str(analysis_request.max_date),
            "system_prompt": analysis_request.system_prompt or "",
            "promt_question": analysis_request.promt_question or "",
            "status": "pending",
            "total_texts": "0",
            "completed_texts": "0",
            "progress": "0",
            "bad_request": "0",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }

        try:
            from mlops.runtime import GpuBusy, assert_can_start, register_llm_run
            assert_can_start("llm-run")
        except GpuBusy as exc:
            holder = (exc.holders or [{}])[0]
            return JSONResponse(
                status_code=409,
                content={"error": f"GPU занят задачей {holder.get('product')} {holder.get('job_id')}"},
            )
        await redis_db.hset(f"task:{task_id}", mapping=task_data)
        await redis_db.rpush("queue:tasks", task_id)
        try:
            from mlops.runtime import register_llm_run
            register_llm_run(task_data, status="pending")
        except Exception:
            pass

        # Проверка на количество активных задач
        # active_tasks = await redis_db.get("active_tasks_count") or 0
        # if int(active_tasks) < 2:
        #     # Увеличиваем количество активных задач здесь, если текущих задач меньше допустимых
        #     await redis_db.incr("active_tasks_count")
        #     background_tasks.add_task(process_task, task_id, task_data, background_tasks)

        background_tasks.add_task(process_task, task_id, task_data, background_tasks)

        return JSONResponse(
            content={
                "task_id": task_id,
                "status": "pending",
                "message": "Task has been added to the queue."
            },
            status_code=202
        )
    
    except Exception as e:
        logging.error(f"Error in llm_run: {e}")
        return JSONResponse(
            content={
                "error": str(e)
            },
            status_code=500
        )


async def process_task(task_id: str, task_data: dict, background_tasks: BackgroundTasks):
    try:
        # Получаем данные задачи из Redis
        task_data = await redis_db.hgetall(f"task:{task_id}")
        if not task_data:
            raise Exception(f"Задача {task_id}: данные не найдены в Redis!")

        # Декодируем данные задачи
        task_data = {k.decode("utf-8"): v.decode("utf-8") for k, v in task_data.items()}
        # Создаем новый словарь с изменением ключей
        renamed_data = {}
        if 'min_date' in task_data:
            for key, value in task_data.items():
                if key == 'min_date':
                    new_key = 'min_data'
                elif key == 'max_date':
                    new_key = 'max_data'
                else:
                    new_key = key
                renamed_data[new_key] = value
                task_data = renamed_data

        task_data["min_data"] = int(task_data["min_data"])
        task_data["max_data"] = int(task_data["max_data"])

        # Устанавливаем блокировку на задачу
        if await redis_db.set(f"lock:task:{task_id}", "1", nx=True, ex=300):
            try:
                # Обновляем статус задачи
                await redis_db.hset(
                    f"task:{task_id}",
                    mapping={"status": "in_progress", "updated_at": datetime.now().isoformat()},
                )
                try:
                    from mlops.runtime import register_llm_run
                    register_llm_run({**task_data, "updated_at": datetime.now().isoformat()}, status="in_progress")
                except Exception:
                    pass

                # Выполнение обработки
                await run_llm_query(task_data)

                # Отмечаем задачу как завершенную
                await redis_db.hset(
                    f"task:{task_id}",
                    mapping={"status": "done", "updated_at": datetime.now().isoformat()},
                )
                try:
                    from mlops.runtime import register_llm_run
                    register_llm_run({**task_data, "status": "done", "updated_at": datetime.now().isoformat()}, status="done")
                except Exception:
                    pass
            finally:
                # Удаляем блокировку
                await redis_db.delete(f"lock:task:{task_id}")
        else:
            logging.info(f"Задача {task_id} уже обрабатывается, пропускаем.")
    except Exception as e:
        logging.error(f"Ошибка при обработке задачи {task_id}: {e}")
        traceback.print_exc()

        # Обновляем статус в случае ошибки
        await redis_db.hset(f"task:{task_id}", mapping={"status": "failed", "error": str(e)})
        try:
            from mlops.runtime import register_llm_run
            register_llm_run({**task_data, "status": "failed"}, status="failed")
        except Exception:
            pass

    finally:
        # Сбрасываем статус GPU
        await reset_gpu_status()
        logging.info(f"GPU статус сброшен. Задача {task_id} завершена.")


@app.post("/reset-queue/", tags=['ai analytics'])
async def reset_queue():
    try:
        # Очищаем очередь задач из Redis
        await redis_db.delete("queue:tasks")

        # Получаем все ID задач, находящихся в состоянии "in_progress"
        in_progress_task_ids = await redis_db.keys("task:*")  # Находим все задачи
        for task_id in in_progress_task_ids:
            # Обновляем статус каждой задачи на "pending"
            await redis_db.hset(task_id.decode(), "status", "pending")

        # Сбрасываем счетчик активных задач
        await redis_db.set("active_tasks_count", 0)

        # Логируем действие об успешном сбросе
        logger.info("Очередь LLM-задач успешно сброшена, счетчик активных задач обновлен на 0.")

        # Возвращаем успешный ответ клиенту
        return JSONResponse(
            content={
                "message": "Очередь LLM-задач сброшена."
            },
            status_code=200
        )
    except Exception as e:
        # Логируем ошибку и возвращаем ответ с кодом 500
        logger.error(f"Ошибка при сбросе очереди LLM-задач: {e}")
        return JSONResponse(
            content={
                "error": "Не удалось сбросить очередь LLM-задач. Пожалуйста, попробуйте снова позже."
            },
            status_code=500
        )


@app.get("/status/{task_id}", tags=['ai analytics'])
async def get_task_status(task_id: str):
    # Ожидаем асинхронный вызов метода hgetall
    task_data = await redis_db.hgetall(f"task:{task_id}")

    # print(f"task_data type: {type(task_data)}, content: {task_data}")

    # Проверяем, существует ли задача
    if not task_data:
        raise HTTPException(status_code=404, detail="Задача не найдена")

    # Декодируем ключи и значения 
    decoded_task_data = {key.decode("utf-8"): value.decode("utf-8") for key, value in task_data.items()}

    # Убираем символы новой строки из значений
    cleaned_task_data = {k: v.replace("\n", "") for k, v in decoded_task_data.items()}

    return cleaned_task_data

@app.get("/task-status/{task_id}", tags=['tasks'])
async def get_task_status(task_id: str):
    task_info = await redis_db.hgetall(f"task:{task_id}")
    
    if not task_info:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    
    # Преобразование байтов в строки
    result = {k.decode("utf-8"): v.decode("utf-8") for k, v in task_info.items()}
    
    return result


# Настраиваем логгер
logger = logging.getLogger("uvicorn.error")  # Используем логгер Uvicorn для ошибок

@app.post("/reset-queue/", tags=['ai analytics'])
async def reset_queue():
    try:
        # Очищаем очередь задач из Redis
        await redis_db.delete("queue:tasks")
        
        # Получаем все текущие ID задач, находящихся в состоянии "in_progress"
        in_progress_task_ids = await redis_db.lrange("queue:tasks", 0, -1)
        
        # Обновляем статус каждой задачи на "pending"
        for task_id in in_progress_task_ids:
            await redis_db.hset(f"task:{task_id.decode()}", "status", "pending")

        # Логируем действие об успешном сбросе
        logger.info("Очередь LLM-задач успешно сброшена.")

        # Возвращаем успешный ответ клиенту
        return JSONResponse(
            content={
                "message": "Очередь LLM-задач сброшена."
            },
            status_code=200
        )
    except Exception as e:
        # Логируем ошибку и возвращаем ответ с кодом 500
        logger.error(f"Ошибка при сбросе очереди LLM-задач: {e}")
        return JSONResponse(
            content={
                "error": "Не удалось сбросить очередь LLM-задач. Пожалуйста, попробуйте снова позже."
            },
            status_code=500
        )
    

@app.get("/llm-analyze", tags=['ai analytics'])
async def llm_analyze(user_id: int, folder_name: str, file_name: str):

    print(f'llm_analyze: folder_name: {folder_name}, file_name: {file_name}')

    global full_data_store, aggregated_data_store

    user_data = await redis_db.hgetall(str(user_id))
    user_data = {key.decode('utf-8'): value.decode('utf-8') for key, value in user_data.items()}
    
    # Декодируем JSON-значения в словари
    for key, value in user_data.items():
        try:
            user_data[key] = json.loads(value)
        except json.JSONDecodeError:
            print(f"Ошибка декодирования JSON для ключа {key}: {value}")

    if user_data is None:
        raise HTTPException(status_code=404, detail="User not found")

    # Находим нужный HTML-файл
    html_files = user_data["bertopic_files_directory"].get(folder_name, [])
    html_file_path = None

    info_html = {}
    for file_info in html_files:
        if file_info["html-file"] == file_name:
            info_html = file_info
            break

    if not info_html:
        raise HTTPException(status_code=404, detail="File info not found")

    # Извлекаем название индекса из file_name
    # Убираем расширение .html и убираем дату и время в конце
    base_name = file_name.replace('.html', '')
    # Находим последнее вхождение даты (формат _YYYYMMDD_HHMMSS)
    import re
    date_pattern = r'_\d{8}_\d{6}$'
    index_name = re.sub(date_pattern, '', base_name)
    
    # Путь к HTML файлу с учетом новой структуры папок
    html_file_path = os.path.join("/home/dev/tellscope_app/tellscope_backend/data", str(user_id), 
                                   "bertopic_files_directory", folder_name, index_name, file_name)

    if not os.path.exists(html_file_path):
        raise HTTPException(status_code=404, detail="HTML file not found")

    # Определяем базовое имя модели без расширения
    model_file_name_base = file_name.replace('.html', '').split('_')[-1]

    # Находим нужный модельный файл
    model_folder_name = None
    for file_info in html_files:
        if model_file_name_base in file_info["model-file"]:
            model_folder_name = folder_name
            break

    if model_folder_name is None:
        raise HTTPException(status_code=404, detail="Model folder not found")

    # Создаем путь к модели с учетом новой структуры папок
    model_path = os.path.join("/home/dev/tellscope_app/tellscope_backend/data", str(user_id), 
                               "bertopic_files_directory", model_folder_name, index_name,
                               next(file_info["model-file"] for file_info in html_files if model_file_name_base in file_info["model-file"]))

    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model file not found")

    # Модель BERTopic
    topic_model = BERTopic.load(model_path)

    # Путь к файлу с темами 
    file_path = '/home/dev/tellscope_app/tellscope_backend/data/indexes.pkl'
    indexes = load_dict_from_pickle(file_path)
    
    # ЗАГРУЖАЕМ СОХРАНЕННЫЕ ДАННЫЕ ИЗ ФАЙЛА my_list_llm_ans_*.pkl
    texts_path = os.path.join("/home/dev/tellscope_app/tellscope_backend/data", str(user_id), 
                                "bertopic_files_directory", model_folder_name, index_name)
    
    # Находим файл с метками LLM
    files = os.listdir(texts_path)
    llm_file = None
    for file in files:
        if 'my_list_llm_ans_' in file and file_name.replace('.html', '') in file and file.endswith('.pkl'):
            llm_file = file
            break
    
    if llm_file is None:
        raise HTTPException(status_code=404, detail="LLM labels file not found")
    
    llm_file_path = os.path.join(texts_path, llm_file)
    
    # Загружаем сохраненные данные (хэши и метки)
    try:
        with open(llm_file_path, 'rb') as f:
            saved_data = pickle.load(f)
            
        saved_hashes, saved_labels = unpack_saved_llm_labels(saved_data)
            
    except Exception as e:
        print(f"Ошибка при загрузке файла меток: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error loading labels file: {str(e)}")

    # Если есть хэши, используем их для точной фильтрации
    if saved_hashes:
        # Получаем все данные из Elasticsearch
        if info_html.get('query_str') is None:
            info_html['query_str'] = 'all'

        if 'min_data' not in info_html:
            all_data = elastic_query(theme_index=indexes[info_html['index_number']], query_str=info_html['query_str'], 
                                min_date=info_html['min_date'], max_date=info_html['max_date'])
        else:
            all_data = elastic_query(theme_index=indexes[info_html['index_number']], query_str=info_html['query_str'], 
                                 min_date=info_html['min_data'], max_date=info_html['max_data'])
        
        # Фильтруем данные по сохраненным хэшам, сохраняя выравнивание меток
        filtered_data = []
        aligned_labels = []
        hash_to_data = {item['hash']: item for item in all_data}
        
        for hash_val, lab in zip(saved_hashes, saved_labels):
            if hash_val in hash_to_data:
                filtered_data.append(hash_to_data[hash_val])
                aligned_labels.append(lab)
        
        data = pd.DataFrame(filtered_data)
        thematics = aligned_labels
        
    else:
        # Старый способ - используем весь запрос
        if info_html.get('query_str') is None:
            info_html['query_str'] = 'all'

        if 'min_data' not in info_html:
            data_list = elastic_query(theme_index=indexes[info_html['index_number']], query_str=info_html['query_str'], 
                                min_date=info_html['min_date'], max_date=info_html['max_date'])
        else:
            data_list = elastic_query(theme_index=indexes[info_html['index_number']], query_str=info_html['query_str'], 
                                 min_date=info_html['min_data'], max_date=info_html['max_data'])
        
        data = pd.DataFrame(data_list[:len(saved_labels)])  # Обрезаем до количества меток
        thematics = saved_labels

    # Проверяем соответствие длин
    if len(data) != len(thematics):
        print(f"Предупреждение: длина данных ({len(data)}) не совпадает с длиной меток ({len(thematics)})")
        # Обрезаем до минимальной длины
        min_len = min(len(data), len(thematics))
        data = data.iloc[:min_len]
        thematics = thematics[:min_len]

    # Обработка тематики из модели
    df_topic = topic_model.get_topic_info()[['CustomName', 'Topic', 'Count']]
    dct_df_topic = dict(zip(df_topic['Topic'], df_topic['CustomName']))
    
    # Получаем имена кластеров из модели BERTopic
    if hasattr(topic_model, 'topics_') and len(topic_model.topics_) > 0:
        # Используем топики из модели для получения имен кластеров
        cluster_names = [dct_df_topic.get(x, 'Неизвестная тема') for x in topic_model.topics_[:len(thematics)]]
    else:
        # Если нет topic_model.topics_, используем LLM метки как имена кластеров
        cluster_names = thematics

    # Обработка данных в зависимости от состава полей
    limited_fields = ['id', 'text', 'timeCreate', 'hub', 'city', 'audienceCount', 'url']
    is_limited_data = all(col in data.columns for col in limited_fields) and len(data.columns) <= len(limited_fields) + 3

    if is_limited_data:
        data['text_url'] = data.get('url', '')
        data['author_url'] = ''
        data['fullname'] = data.get('hub', '')
        data['author_type'] = ''
        data['sex'] = ''
        data['age'] = ''
        data['hubtype'] = 'Онлайн-СМИ'
        data['commentsCount'] = 0
        data['repostsCount'] = 0
        data['likesCount'] = 0
        data['er'] = 0
        data['viewsCount'] = 0
        data['toneMark'] = 0
        data['country'] = ''
        data['region'] = data.get('city', '')
    else:
        if 'authorObject' in data.columns:
            data.rename(columns={'url': 'text_url'}, inplace=True)
            data = data.join(pd.DataFrame(list(data['authorObject'].values)))
            data.rename(columns={'url': 'author_url'}, inplace=True)
        else:
            if 'url' in data.columns:
                data.rename(columns={'url': 'text_url'}, inplace=True)
            else:
                data['text_url'] = ''
            data['author_url'] = ''
            data['fullname'] = data.get('hub', '')
            data['author_type'] = ''
            data['sex'] = ''
            data['age'] = ''

    # Убедимся, что все необходимые колонки существуют
    required_columns = ['timeCreate', 'hub', 'author_url', 'fullname', 'text_url', 'author_type', 'sex', 'age',
                        'hubtype', 'commentsCount', 'audienceCount', 'repostsCount', 'likesCount', 
                        'er', 'viewsCount', 'toneMark', 'country', 'region']
    
    for column in required_columns:
        if column not in data.columns:
            if column == 'hubtype':
                data[column] = 'Онлайн-СМИ'
            elif column == 'toneMark':
                data[column] = 0
            elif column == 'audienceCount' and 'citeIndex' in data.columns:
                data[column] = data['citeIndex']
            elif column == 'region' and 'city' in data.columns:
                data[column] = data['city']
            else:
                data[column] = ''
    
    # Выбираем только необходимые колонки
    data = data[required_columns]

    # Создаем DataFrame с правильным разделением данных
    df_join = pd.DataFrame({
        'Имя кластера': cluster_names,  # Имена кластеров из BERTopic
        'Тематика текста': thematics    # LLM обработка
    }).join(data, how='inner')

    # Остальные столбцы добавляем как обычно
    df_join.columns = ['Имя кластера', 'Тематика текста', 'Время', 'Источник', 'Ссылка на автора', 'Автор', 'Ссылка на текст', 'Тип автора', 'Пол', 'Возраст',
                    'Тип источника', 'Комментариев', 'Аудитория', 'Репостов', 'Лайков', 'Вовлеченность', 'Просмотров',
                    'Тональность', 'Страна', 'Регион']

    # Добавляем ID
    df_join.reset_index(drop=True, inplace=True)
    df_join.insert(0, 'ID', df_join.index)

    # Финальные столбцы
    df_join.columns = ['ID', 'Имя кластера', 'Тематика текста', 'Время', 'Источник', 'Ссылка на автора', 'Автор', 'Ссылка на текст', 
                    'Тип автора', 'Пол', 'Возраст', 'Тип источника', 'Комментариев', 'Аудитория', 
                    'Репостов', 'Лайков', 'Вовлеченность', 'Просмотров', 'Тональность', 
                    'Страна', 'Регион']

    # Преобразуем значения тональности
    df_join['Тональность'] = df_join['Тональность'].map({0: 'Нейтральная', -1: 'Негатив', 1: 'Позитив'})
    
    # Функция для безопасного преобразования значений в числа
    def safe_to_numeric(value):
        try:
            if pd.isna(value) or value == '':
                return 0
            return pd.to_numeric(value, errors='coerce')
        except:
            return 0

    # Получение агрегированной таблицы
    df_group = df_join[['Имя кластера', 'Комментариев', 'Аудитория', 'Репостов', 'Лайков', 'Вовлеченность', 'Просмотров']].copy()
    
    numerical_columns = ['Комментариев', 'Аудитория', 'Репостов', 'Лайков', 'Вовлеченность', 'Просмотров']
    
    for column in numerical_columns:
        df_group[column] = df_group[column].apply(safe_to_numeric)
        df_group[column] = df_group[column].fillna(0).astype(int)
    
    # Сначала считаем количество записей в каждом кластере ДО группировки
    theme_count = df_group['Имя кластера'].value_counts()

    # Затем группируем и суммируем остальные показатели
    result = df_group.groupby('Имя кластера').sum().reset_index()

    # Добавляем количество записей
    result['Количество'] = result['Имя кластера'].map(theme_count)

    result.sort_values(by='Количество', ascending=False, inplace=True)
    result = result[['Имя кластера', 'Количество', 'Аудитория', 'Комментариев', 'Репостов', 'Лайков', 'Вовлеченность', 'Просмотров']]

    result = result.where(pd.notnull(result), None)

    # Сохранение данных в Redis
    await redis_db.hset(str(user_id), "full_data", json.dumps(df_join.where(pd.notnull(df_join), None).to_dict(orient='records')))
    await redis_db.hset(str(user_id), "aggregated_data", json.dumps(result.where(pd.notnull(result), None).to_dict(orient='records')))

    # Возвращение HTML файлов
    with open(html_file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()
    
    if html_content is None or not os.path.exists(html_file_path):
        raise HTTPException(status_code=404, detail="HTML file not found")

    # HTML файл dataplot с учетом новой структуры папок
    html_file_path_dataplot = os.path.join("/home/dev/tellscope_app/tellscope_backend/data", str(user_id), 
                                "bertopic_files_directory", folder_name, index_name, 'datamapplot_' + file_name)

    with open(html_file_path_dataplot, 'r', encoding='utf-8') as file:
        html_file_path_dataplot = file.read()
    
    if html_file_path_dataplot is None:
        raise HTTPException(status_code=404, detail="HTML file dataplot not found")
    
    return {
        "html_content": html_content,
        "html_content_dataplot": html_file_path_dataplot, 
        "full_data": json.loads(df_join.replace([np.nan, np.inf, -np.inf], None).to_json(orient='records')),
        "aggregated_data": json.loads(result.replace([np.nan, np.inf, -np.inf], None).to_json(orient='records'))
    }


@app.delete("/delete-theme-files", tags=['ai analytics'])
async def delete_theme_files(user_id: int, folder_name: str, file_name: str):
    import logging
    logging.warning(f"Delete requested: {user_id=} {folder_name=} {file_name=}")

    # Извлекаем тему из имени файла (без даты и расширения)
    # Пример: beyond_taylor_10.06.2025-16.06.2025_20250824_152603.html
    match = re.match(r"(.+)_(\d{8}_\d{6})\.html$", file_name)
    if not match:
        raise HTTPException(status_code=400, detail=f"Некорректное название файла: {file_name}")
    
    theme_prefix = match.group(1)  # например: "beyond_taylor_10.06.2025-16.06.2025"
    datetime_suffix = match.group(2)  # например: "20250824_152603"

    logging.warning(f"Extracted theme_prefix: {theme_prefix}, datetime_suffix: {datetime_suffix}")

    # Базовый путь к папке с темой (подпапка внутри folder_name)
    # folder_name - это родительская папка (например, "test")
    # theme_prefix - это название подпапки темы
    base_path = f"/home/dev/tellscope_app/tellscope_backend/data/{user_id}/bertopic_files_directory/{folder_name}/{theme_prefix}"
    
    logging.warning(f"Base path: {base_path}")

    # Список файлов для удаления
    targets = [
        f"{theme_prefix}_{datetime_suffix}.html",
        f"datamapplot_{theme_prefix}_{datetime_suffix}.html",
        f"topic_model_{theme_prefix}_{datetime_suffix}",
        f"my_list_llm_ans_{theme_prefix}_{datetime_suffix}.pkl",
        f"topic_names_{theme_prefix}_{datetime_suffix}.pkl"
    ]

    deleted = []
    errors = []
    not_found = []

    # Проверяем существование базовой папки
    if not os.path.exists(base_path):
        logging.error(f"Base path does not exist: {base_path}")
        raise HTTPException(status_code=404, detail=f"Папка темы не найдена: {base_path}")

    # Удаляем файлы
    for obj in targets:
        obj_path = os.path.join(base_path, obj)
        logging.warning(f"Trying to delete: {obj_path}")
        
        if os.path.isdir(obj_path):
            try:
                shutil.rmtree(obj_path)
                deleted.append(obj)
                logging.warning(f"✓ Deleted directory: {obj_path}")
            except Exception as e:
                error_msg = f"Ошибка при удалении директории {obj}: {e}"
                errors.append(error_msg)
                logging.error(error_msg)
        elif os.path.isfile(obj_path):
            try:
                os.remove(obj_path)
                deleted.append(obj)
                logging.warning(f"✓ Deleted file: {obj_path}")
            except Exception as e:
                error_msg = f"Ошибка при удалении файла {obj}: {e}"
                errors.append(error_msg)
                logging.error(error_msg)
        else:
            not_found.append(obj)
            logging.warning(f"✗ Not found: {obj_path}")

    # Проверяем, пуста ли папка темы после удаления
    try:
        if os.path.exists(base_path) and not os.listdir(base_path):
            shutil.rmtree(base_path)
            deleted.append(f"[folder] {theme_prefix}")
            logging.warning(f"✓ Deleted empty theme folder: {base_path}")
    except Exception as e:
        error_msg = f"Ошибка при удалении пустой папки темы {base_path}: {e}"
        errors.append(error_msg)
        logging.error(error_msg)

    # Если ничего не удалено и есть ошибки
    if not deleted and errors:
        raise HTTPException(
            status_code=500, 
            detail=f"Не удалось удалить ни один файл. Ошибки: {'; '.join(errors)}"
        )
    
    # Если ничего не найдено для удаления
    if not deleted and not errors:
        raise HTTPException(
            status_code=404, 
            detail=f"Ни одного элемента для удаления не найдено. Ожидаемые файлы: {', '.join(targets)}"
        )

    # Обновляем данные в Redis
    try:
        user_data = await redis_db.hgetall(str(user_id))
        user_data_decoded = {key.decode('utf-8'): value.decode('utf-8') for key, value in user_data.items()}
        
        if "bertopic_files_directory" in user_data_decoded:
            bertopic_data = json.loads(user_data_decoded["bertopic_files_directory"])
            
            # Удаляем файл из структуры данных
            if folder_name in bertopic_data:
                original_count = len(bertopic_data[folder_name])
                bertopic_data[folder_name] = [
                    file_info for file_info in bertopic_data[folder_name]
                    if file_info.get("html-file") != file_name
                ]
                new_count = len(bertopic_data[folder_name])
                logging.warning(f"Redis update: removed {original_count - new_count} entries from {folder_name}")
                
                # Если в папке не осталось файлов, удаляем папку из структуры
                if not bertopic_data[folder_name]:
                    del bertopic_data[folder_name]
                    logging.warning(f"Redis update: removed empty folder {folder_name}")
            
            # Сохраняем обратно в Redis
            await redis_db.hset(
                str(user_id), 
                "bertopic_files_directory", 
                json.dumps(bertopic_data)
            )
            logging.warning("✓ Redis data updated successfully")
            
    except Exception as e:
        error_msg = f"Ошибка при обновлении Redis: {e}"
        logging.error(error_msg)
        errors.append(error_msg)

    return {
        "deleted": deleted,
        "not_found": not_found if not_found else None,
        "errors": errors if errors else None,
        "message": f"Успешно удалено файлов: {len(deleted)}"
    }


from sqlalchemy.ext.asyncio import AsyncSession
# Функция для получения сессии базы данных
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_maker() as session:
        yield session


from sqlalchemy.future import select

# JWT token scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

from jwt.exceptions import ExpiredSignatureError

from sqlalchemy.orm import selectinload

async def get_current_user(token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        # Добавьте options для отключения проверки audience
        payload = jwt.decode(
            token, 
            SECRET_KEY, 
            algorithms=[ALGORITHM],
            options={"verify_aud": False}  # Отключаем проверку audience
        )
    except ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError:
        raise credentials_exception

    user_id: str = payload.get("sub")
    
    if user_id is None:
        raise credentials_exception

    return user_id
 

class ResponseModel(BaseModel):
    id: int

    class Config:
        orm_mode = True

# # Route to retrieve the current user profile details
@app.get('/user-id', tags=['user'])
async def get_user_profile(current_user: int = Depends(get_current_user)):
    return current_user

def get_user_profile(current_user: User = Depends(get_current_user)):
    return current_user


################################################ new token ################################################

ACCESS_TOKEN_EXPIRE_MINUTES = 60  # Время жизни основного токена
REFRESH_TOKEN_EXPIRE_DAYS = 30      # Время жизни refresh-токена

# Модели данных
class User(BaseModel):
    username: str

class Token(BaseModel):
    access_token: str
    token_type: str
    refresh_token: str

# Функция для создания токена
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def create_refresh_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from fastapi_users import FastAPIUsers
from auth.manager import UserManager

# Инициализация FastAPIUsers (добавьте в ваш основной файл, например, main.py)
fastapi_users = FastAPIUsers[User, int](get_user_manager, [auth_backend])

@app.post("/auth/jwt/login")
async def login(
    credentials: OAuth2PasswordRequestForm = Depends(),
    user_manager: UserManager = Depends(get_user_manager),
):
    user = await user_manager.authenticate(credentials)
    if not user:
        raise HTTPException(status_code=400, detail="Invalid credentials")

    access_token = create_access_token({"sub": str(user.id)})
    refresh_token = create_refresh_token({"sub": str(user.id)})
    
    # Сохраняем refresh-токен в Redis
    await redis_db.setex(f"refresh:{user.id}", 2592000, refresh_token)
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
    }

from fastapi import Depends, HTTPException
from fastapi_users import FastAPIUsers
from auth.manager import UserManager
import jwt

@app.post("/auth/refresh")
async def refresh_token(
    refresh_token: str,
    user_manager: UserManager = Depends(get_user_manager),
):
    try:
        payload = jwt.decode(refresh_token, SECRET, algorithms=["HS256"])
        user_id = payload.get("sub")
        
        # Проверяем, что refresh-токен сохранён в Redis
        stored_token = await redis_db.get(f"refresh:{user_id}")
        if not stored_token or stored_token != refresh_token:
            raise HTTPException(status_code=401, detail="Invalid refresh token") 

        # Получаем пользователя
        user = await user_manager.get(user_id)
        
        # Генерируем новый access-токен через JWTStrategy
        new_access_token = await auth_backend.get_strategy().write_token(user)
        
        return {"access_token": new_access_token, "token_type": "bearer"}
    
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Refresh token expired")

# # Эндпоинт для выхода пользователя (удаление refresh-токена)
# @app.post("/logout")
# async def logout(user: User):
#     redis_db.delete(f"refresh_token:{user.username}")
#     return {"msg": "Successfully logged out"}

 
# Dependency to get the current user based on the provided token
# async def get_current_user(token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)):
#     credentials_exception = HTTPException(
#         status_code=status.HTTP_401_UNAUTHORIZED,
#         detail="Invalid authentication credentials",
#         headers={"WWW-Authenticate": "Bearer"},
#     )
#     payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

#     user_id: str = payload.get("sub")
    
#     # Получаем объект User из базы данных по user_id
#     query = select(User).where(User.id == int(user_id))
#     result = await db.execute(query)
#     user = result.scalar_one_or_none()
    
#     if user is None:
#         raise credentials_exception
#     return user_id

################################################ new token ################################################

@app.get("/history_llm_search/{user_id}", tags=['data & folders'])
async def history_search(user_id: int):

    os.chdir('/home/dev/tellscope_app/tellscope_backend/data')
    
    # Загрузка словаря истории запросов пользователей
    try:
        with open('llm_history_progress.pickle', 'rb') as file:  # 'rb' - читать в бинарном формате
            search_history = pickle.load(file)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Файл истории запросов не найден.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при загрузке файла: {str(e)}")

    # Поиск по user_id
    user_requests = next((item for item in search_history if item['user_id'] == user_id), None)

    if user_requests:
        # Извлечение необходимой информации
        date = user_requests['values']['date']
        llm_queries = user_requests['values']['llm_queries']

        # Формирование ответа
        response = {
            "date": date,
            "llm_queries": llm_queries
        }
        return response
    else:
        raise HTTPException(status_code=404, detail="Запросы для данного пользователя не найдены.")

############################## Хранение данных о файлах и папках пользователей в Redis ####################

# Добавление папки
@app.get("/add-folder/{user_id}/{folder_name}", tags=['data & folders'])
async def add_folder(user_id: str, folder_name: str):
    print(f'user_id: {user_id}, folder_name: {folder_name}')
    
    json_files_directory = f"/home/dev/tellscope_app/tellscope_backend/data/{user_id}/json_files_directory"
    storage_path = f"{json_files_directory}/{folder_name}"

    if not os.path.exists(json_files_directory):
        os.makedirs(json_files_directory)

    if os.path.exists(storage_path):
        raise HTTPException(status_code=400, detail="Папка с таким именем уже существует.")

    os.makedirs(storage_path)

    # ✅ ИСПРАВЛЕНИЕ: Правильная работа с кириллицей
    user_data = await redis_db.hget(user_id, "json_files_directory")
    if user_data is None:
        user_folders = {}
    else:
        # Декодируем байты в UTF-8 строку
        decoded_data = user_data.decode('utf-8') if isinstance(user_data, bytes) else user_data
        user_folders = json.loads(decoded_data)

    if folder_name not in user_folders:
        user_folders[folder_name] = []

    # ✅ Сохраняем с ensure_ascii=False и кодируем в байты
    await redis_db.hset(
        user_id, 
        "json_files_directory", 
        json.dumps(user_folders, ensure_ascii=False).encode('utf-8')
    )

    return f"Папка {folder_name} у пользователя {user_id} создана!"


# from files_MLG_KRIBRUM import load_medialogia_excel
# from files_MLG_KRIBRUM import load_file_to_elastic

from celery import Celery
import os
import pandas as pd
# from models import tasks, processing_results
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


async def sync_files_with_redis(user_id: str, folder_name: str):
    base_path = f'/home/dev/tellscope_app/tellscope_backend/data/{user_id}/json_files_directory/{folder_name}'
    
    try:
        fs_files = []
        if os.path.exists(base_path):
            fs_files = [f for f in os.listdir(base_path) if f.endswith('.json')]
        
        # ✅ ИСПРАВЛЕНИЕ: Правильная декодировка
        user_folders_data = await redis_db.hget(user_id, "json_files_directory")
        if user_folders_data:
            decoded_data = user_folders_data.decode("utf-8") if isinstance(user_folders_data, bytes) else user_folders_data
            user_folders = json.loads(decoded_data)
        else:
            user_folders = {}
            
        redis_files = user_folders.get(folder_name, [])
        
        if set(fs_files) != set(redis_files):
            user_folders[folder_name] = fs_files
            # ✅ Сохраняем с ensure_ascii=False
            await redis_db.hset(
                user_id, 
                "json_files_directory", 
                json.dumps(user_folders, ensure_ascii=False).encode('utf-8')
            )
            return True
        
        return False
    except Exception as e:
        logger.error(f"Ошибка синхронизации файлов: {str(e)}")
        return False

from tasks import process_file_task

@app.post("/add-file/{user_id}/{folder_name}", tags=["data & folders"])
async def add_file(
    user_id: str,
    folder_name: str,
    uploaded_file: UploadFile = File(..., max_size=50*1024*1024*1024),
    methods=["POST"],
    user: User = Depends(current_user),
):
    _folder_guard(user_id, folder_name, user, need_write=True)
    if not folder_name:
        raise HTTPException(status_code=400, detail="Необходимо указать имя папки")

    # Путь к индексу файлов и получение индексов
    file_path = '/home/dev/tellscope_app/tellscope_backend/data/indexes.pkl'
    indexes = load_dict_from_pickle(file_path)
    
    # ИСПРАВЛЕНИЕ: Проверяем, что indexes не None
    if indexes is None:
        indexes = {}
        logger.warning("Файл indexes.pkl не найден или поврежден, создаем новый словарь")

    original_filename = uploaded_file.filename.lower()
    file_extension = os.path.splitext(original_filename)[1]
    json_filename = original_filename
    if file_extension in ('.xlsx', '.xls'):
        json_filename = original_filename.replace(file_extension, '.json')

    next_key = max(indexes.keys()) + 1 if indexes else 1
    formatted_value = json_filename.replace('.json', '').lower()
    indexes[next_key] = formatted_value
    save_dict_to_pickle(file_path, indexes)

    max_file_size_admin = 50 * 1024 * 1024 * 1024 # 50 GB
    max_file_size_non_admin = 500 * 1024 * 1024 # 500 MB
    size = uploaded_file.size if hasattr(uploaded_file, 'size') else 0

    if user_id in ('1', '3', '13'):
        if size > max_file_size_admin:
            raise HTTPException(status_code=400, detail="Размер файла превышает 50 ГБ")
    else:
        if size > max_file_size_non_admin:
            raise HTTPException(status_code=400, detail="Размер файла превышает 500 МБ")

    # Директория для файла
    file_location = f'/home/dev/tellscope_app/tellscope_backend/data/{user_id}/json_files_directory/{folder_name}/{json_filename}'
    os.makedirs(os.path.dirname(file_location), exist_ok=True)
    with open(file_location, "wb+") as f:
        uploaded_file.file.seek(0)
        shutil.copyfileobj(uploaded_file.file, f)

    task_id = str(uuid.uuid4())
    await redis_db.hset(
        f"task:{task_id}",
        mapping={
            "status": "pending",
            "progress": "0",
            "user_id": user_id,
            "folder_name": folder_name,
            "filename": json_filename,
            "original_filename": original_filename,
            "file_extension": file_extension,
            "created_at": datetime.now().isoformat(),
            "next_key": str(next_key)
        }
    )

    # Celery task_id, user_id, folder_name, json_filename, file_location, file_extension, next_key
    print('-------------1----------------')
    process_file_task.apply_async(
        kwargs={ 
            "task_id": task_id,
            "user_id": user_id,
            "folder_name": folder_name,
            "json_filename": json_filename,
            "file_location": file_location,
            "file_extension": file_extension,
            "next_key": next_key
        }
    )

    # Для интерфейса – обновляем Redis (и список файлов в папке)
    # ✅ ИСПРАВЛЕНИЕ: Правильная декодировка
    user_folders_data = await redis_db.hget(user_id, "json_files_directory")
    user_folders = {}
    if user_folders_data:
        try:
            # Декодируем байты в строку UTF-8, затем парсим JSON
            decoded_data = user_folders_data.decode("utf-8") if isinstance(user_folders_data, bytes) else user_folders_data
            user_folders = json.loads(decoded_data)
        except Exception as e:
            logger.error(f"Ошибка при загрузке данных из Redis: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Ошибка при загрузке данных из Redis: {str(e)}")

    current_files = user_folders.get(folder_name, [])
    if json_filename in current_files:
        current_files = [f for f in current_files if f != json_filename]
    current_files.append(json_filename)
    user_folders[folder_name] = current_files
    
    # ✅ Сохраняем с ensure_ascii=False
    await redis_db.hset(
        user_id, 
        "json_files_directory", 
        json.dumps(user_folders, ensure_ascii=False).encode('utf-8')  # Кодируем в байты
    )
        
    try:
        # После успешной загрузки файла
        await sync_files_with_redis(user_id, folder_name)
        
        # Проверяем статус обработки через 2 секунды
        await asyncio.sleep(2)
        task_status = await redis_db.hgetall(f"task:{task_id}")
        
        return {
            "message": f"Файл {uploaded_file.filename} загружен",
            "task_id": task_id,
            "status": task_status.get(b"status", b"unknown").decode()
        }
    except Exception as e:
        logger.error(f"Ошибка при загрузке файла: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/check-task-status/{task_id}")
async def check_task_status(task_id: str):
    task_info = await redis_db.hgetall(f"task:{task_id}")
    if not task_info:
        raise HTTPException(status_code=404, detail="Task not found")

    response_data = {
        "status": task_info.get(b"status", b"unknown").decode(),
        "progress": task_info.get(b"progress", b"0").decode(),
        "error": task_info.get(b"error", b"").decode(),
        "completed": task_info.get(b"completed", b"0").decode(),
        "total": task_info.get(b"total", b"0").decode(),
        "filename": task_info.get(b"original_filename", b"").decode(),
        "stage": task_info.get(b"stage", b"").decode(),
        "stage_details": task_info.get(b"stage_details", b"").decode()
    }

    return response_data

@app.get("/check-files/{user_id}/{folder_name}", tags=["data & folders"])
async def check_files(user_id: str, folder_name: str):
    try:
        synced = await sync_files_with_redis(user_id, folder_name)
        user_folders_data = await redis_db.hget(user_id, "json_files_directory")
        user_folders = json.loads(user_folders_data.decode("utf-8")) if user_folders_data else {}
        files = user_folders.get(folder_name, [])
        
        return {
            "synced": synced,
            "files": files,
            "count": len(files)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Удаление папки
@app.delete("/delete-folder/{user_id}/{directory_type}/{folder_name}", tags=['data & folders'])
async def delete_folder(user_id: str, directory_type: str, folder_name: str, user: User = Depends(current_user)):
    _folder_guard(user_id, folder_name, user, need_write=True)
    # Получаем текущее содержимое для указанного пользователя
    json_folders = await redis_db.hget(user_id, directory_type)
    
    # Если данных для данного user_id нет, возвращаем ошибку
    if json_folders is None:
        raise HTTPException(status_code=404, detail="Директории не найдены для данного пользователя.")

    # Декодируем JSON данные в словарь
    folders_dict = json.loads(json_folders)

    # Проверяем наличие запрашиваемой папки
    if not folder_name or not isinstance(folder_name, str):
        raise HTTPException(status_code=400, detail="Имя папки должно быть строкой")

    # Получаем список файлов, относящихся к этой папке
    files_in_directory = folders_dict[folder_name]

    # Удаляем папку из Redis
    del folders_dict[folder_name]  # Удаляем папку из словаря
    await redis_db.hset(user_id, directory_type, json.dumps(folders_dict))  # Обновляем данные в Redis

    # Получаем список всех индексов для удаления из Elasticsearch
    es_indexes = [index for index in es.indices.get(index='*')]  # список всех индексов elastic
    
    # Удаляем данные из Elasticsearch
    if files_in_directory and directory_type == 'json_files_directory':
        for file in files_in_directory:
            # Индекс, который нужно удалить
            index_to_delete = file.replace('.json', '')

            # Проверка существования индекса и его удаление
            if index_to_delete in es_indexes:
                es.indices.delete(index=index_to_delete)
                print(f"Индекс '{index_to_delete}' успешно удалён.")
            else:
                print(f"Индекс '{index_to_delete}' не найден.")

    # Формируем путь к удаляемой папке в файловой системе
    folder_path = f"/home/dev/tellscope_app/tellscope_backend/data/{user_id}/{directory_type}/{folder_name}"

    try:
        # Проверяем, существует ли папка
        if os.path.exists(folder_path):
            # Рекурсивно удаляем папку и все ее содержимое
            for root, dirs, files in os.walk(folder_path, topdown=False):
                for file in files:
                    file_path = os.path.join(root, file)
                    os.remove(file_path)
                for dir in dirs:
                    dir_path = os.path.join(root, dir) 
                    os.rmdir(dir_path)
            os.rmdir(folder_path)

            return {"message": f"Папка '{folder_name}' пользователя '{user_id}' и все ее содержимое успешно удалены."}
        else:
            raise HTTPException(status_code=404, detail="Папка не найдена.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Удаление файла
@app.delete("/delete-file/{user_id}/{directory_type}/{directory_name}/{file_name}", tags=['data & folders'])
async def delete_file(user_id: str, directory_type: str, directory_name: str, file_name: str, user: User = Depends(current_user)):
    _folder_guard(user_id, directory_name, user, need_write=True)
    # Получаем директории для указанного user_id
    folders = await redis_db.hgetall(user_id)
    # Преобразуем байтовые строки в обычные строки и десериализуем JSON
    folders = {key.decode('utf-8'): json.loads(value.decode('utf-8')) for key, value in folders.items()}

    # Проверяем, есть ли директории для данного пользователя
    if not folders:
        raise HTTPException(status_code=404, detail="Директории не найдены для данного пользователя.")

    # Определяем путь к директории файлов на диске
    folder_path = f"/home/dev/tellscope_app/tellscope_backend/data/{user_id}/{directory_type}/{directory_name}"

    # Удаляем файл из json_files_directory
    if directory_type == "json_files_directory":

        try:
            # Удаляем соответствующий словарь
            if directory_name in folders.get("json_files_directory", {}):
                schools_data = folders["json_files_directory"]
                # Ищем и удаляем словарь с необходимими файлами
                updated_schools = [item for item in schools_data[directory_name] if item != file_name + '.json']
                schools_data[directory_name] = updated_schools
                await redis_db.hset(user_id, "json_files_directory", json.dumps(schools_data))

            # Удаляем файл из файловой системы
            print(111)
            print(os.path.join(folder_path, file_name + '.json'))
            os.remove(os.path.join(folder_path, file_name + '.json'))
            print(222)

            return {"message": f"Файл {file_name + '.json'} из директории {directory_name} был успешно удалён!"}
        except Exception as e:
            print(333)
            raise HTTPException(status_code=500, detail=f"Ошибка при удалении файлов: {str(e)}")


    # Удаляем файлы из bertopic_files_directory
    elif directory_type == "bertopic_files_directory":
        try:
            search_string = file_name.replace('topic_model_', '').replace('.html', '')
            # Удаляем соответствующий словарь
            print(folders.get("bertopic_files_directory", {}))
            if directory_name in folders.get("bertopic_files_directory", {}):
                schools_data = folders["bertopic_files_directory"]
                # Ищем и удаляем словарь с необходимым файлом
                updated_schools = [item for item in schools_data[directory_name] if item.get("html-file") != file_name]
                schools_data[directory_name] = updated_schools
                await redis_db.hset(user_id, "bertopic_files_directory", json.dumps(schools_data))

            # Удаляем файлы
            file_pattern = os.path.join(folder_path, f"*{search_string}*")
            for f in glob.glob(file_pattern):
                if os.path.isdir(f):
                    shutil.rmtree(f)
                else:
                    os.remove(f)

            return {"message": f"Все файлы, содержащие {search_string}, из директории {directory_name} были успешно удалены!"}

        except Exception as e:
            print(f"Ошибка при удалении файлов: {e}")
            raise HTTPException(status_code=500, detail=f"Ошибка при удалении файлов: {str(e)}")

    # Удаляем файлы из projector_files_directory
    elif directory_type == "projector_files_directory":
        try:
            search_string = file_name.replace('.txt', '').replace('.tsv', '')
            # Удаляем соответствующий словарь
            if directory_name in folders.get("projector_files_directory", {}):
                schools_data = folders["projector_files_directory"]
                # Ищем и удаляем словарь с необходимими файлами
                updated_schools = [
                    entry for entry in schools_data[directory_name]
                    if not (search_string in entry.get('tsv-file', '') or 
                            search_string in entry.get('txt-file', ''))
                ]
                schools_data[directory_name] = updated_schools
                await redis_db.hset(user_id, "projector_files_directory", json.dumps(schools_data))

            # Удаляем файлы (tsv + txt) из папки projector
            def extract_search_string(base_file_path):
                """
                Извлекает search_string из полного имени файла.
                Например:
                    Вход: '/home/dev/tellscope_app/tellscope_backend/data/123/projector/folder/geekbrains_08.12.2024-07.01.2025_authors_point_2025-01-10_09-09-48.tsv'
                    Выход: '2025-01-10_09-09-48'
                """
                # Берем только имя файла без пути
                file_name = os.path.basename(base_file_path)
                
                # Ищем search_string с помощью регулярного выражения
                match = re.search(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}', file_name)
                if match:
                    return match.group(0)
                return None


            def remove_related_files(base_file_path):
                """
                Удаляет файлы, которые содержат тот же search_string в имени, что и базовый файл.
                """
                # Извлекаем путь папки, где лежит файл
                folder_path = os.path.dirname(base_file_path)
                
                # Извлекаем search_string из имени базового файла
                search_string = extract_search_string(base_file_path)
                if not search_string:
                    print("Не удалось извлечь search_string из пути:", base_file_path)
                    return
                
                # Шаблон для поиска файлов с теми же датами
                file_pattern = os.path.join(folder_path, f"*{search_string}*")
                
                # Удаляем файлы с совпадающим search_string
                for f in glob.glob(file_pattern):
                    try:
                        if os.path.isdir(f):
                            shutil.rmtree(f)
                        else:
                            os.remove(f)
                            print(f"Удален файл: {f}")
                    except Exception as e:
                        print(f"Ошибка при удалении {f}: {e}")

            remove_related_files(folder_path + '/' + file_name)

            return {"message": f"Все файлы, содержащие {search_string}, из директории {directory_name} были успешно удалены!"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Ошибка при удалении файлов: {str(e)}")

    else:
        raise HTTPException(status_code=400, detail="Некорректный тип директории.")


# # Переименование папки
# @app.put("/rename-folder/{user_id}/{old_folder_name}/{new_folder_name}")
# async def rename_folder(user_id: str, old_folder_name: str, new_folder_name: str):
#     # Путь до директории json_files
#     json_files_directory = f"/home/dev/tellscope_app/tellscope_backend/data/{user_id}/json_files_directory"
#     old_storage_path = f"{json_files_directory}/{old_folder_name}"
#     new_storage_path = f"{json_files_directory}/{new_folder_name}"

#     # Проверяем, существует ли старая папка
#     if not os.path.exists(old_storage_path):
#         raise HTTPException(status_code=404, detail="Старая папка не существует.")

#     # Проверяем, существует ли уже новая папка
#     if os.path.exists(new_storage_path):
#         raise HTTPException(status_code=400, detail="Папка с таким именем уже существует.")

#     # Переименовываем папку на файловой системе
#     os.rename(old_storage_path, new_storage_path)

#     # Обновляем информацию о папках в Redis
#     user_data = redis_db.hget(user_id, "json_folders")
#     if user_data is None:
#         raise HTTPException(status_code=404, detail="Данные пользователя не найдены.")

#     user_folders = json.loads(user_data)

#     # Переименовываем папку в структуре
#     if old_folder_name in user_folders:
#         user_folders[new_folder_name] = user_folders.pop(old_folder_name)
#     else:
#         raise HTTPException(status_code=404, detail="Старая папка не найдена в данных пользователя.")

#     # Сохраняем обновленную структуру в Redis
#     redis_db.hset(user_id, "json_folders", json.dumps(user_folders))

#     return f"Папка '{old_folder_name}' переименована в '{new_folder_name}' у пользователя {user_id}!"


# # Переименование файла
# @app.put("/rename-file/{user_id}/{folder_name}/{old_file_name}/{new_file_name}")
# async def rename_file(user_id: str, folder_name: str, old_file_name: str, new_file_name: str):
#     # Устанавливаем путь к директории файла
#     file_directory = f'/home/dev/tellscope_app/tellscope_backend/data/{user_id}/json_files_directory/{folder_name}'
#     old_file_path = f'{file_directory}/{old_file_name}'
#     new_file_path = f'{file_directory}/{new_file_name}'

#     # Проверяем, существует ли старая версия файла
#     if not os.path.exists(old_file_path):
#         raise HTTPException(status_code=404, detail="Старый файл не существует.")

#     # Проверяем, существует ли уже новая версия файла
#     if os.path.exists(new_file_path):
#         raise HTTPException(status_code=400, detail="Файл с таким именем уже существует в папке.")

#     # Переименовываем файл на файловой системе
#     os.rename(old_file_path, new_file_path)

#     # Обновляем информацию о файлах в Redis
#     user_folders_data = redis_db.hget(user_id, "json_folders")
#     if user_folders_data is None:
#         raise HTTPException(status_code=404, detail="Данные пользователя не найдены.")

#     user_folders = json.loads(user_folders_data)

#     # Проверка существования папки в Redis
#     if folder_name not in user_folders:
#         raise HTTPException(status_code=404, detail="Папка не найдена в данных пользователя.")

#     # Переименование файла в структуре
#     if old_file_name in user_folders[folder_name]:
#         user_folders[folder_name].remove(old_file_name)
#         user_folders[folder_name].append(new_file_name)
#     else:
#         raise HTTPException(status_code=404, detail="Старый файл не найден в папке.")

#     # Сохраняем обновленный список в Redis
#     redis_db.hset(user_id, "json_folders", json.dumps(user_folders))

#     return f"Файл '{old_file_name}' переименован в '{new_file_name}' в папке '{folder_name}' у пользователя {user_id}!"


# Создайте функцию для получения es
def get_elasticsearch():
    return es

@app.get("/user-folders/{user_id}", tags=['data & folders'])
async def get_user_folders(
    user_id: str, 
    es: Elasticsearch = Depends(get_elasticsearch)
):
    import os
    from pathlib import Path
    
    user = get_user_profile(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    file_path = '/home/dev/tellscope_app/tellscope_backend/data/indexes.pkl'
    indexes = load_dict_from_pickle(file_path)
    
    folders = await redis_db.hgetall(user_id)
    if not folders:
        return {
            "user_id": user_id, 
            "json_files_directory": {}, 
            "bertopic_files_directory": {},
            "csv_files_directory": {}
        }

    # ✅ ИСПРАВЛЕНИЕ: Правильная декодировка кириллицы из Redis
    formatted_folders = {}
    for folder_key, files_value in folders.items():
        folder_name = folder_key.decode('utf-8')  # Декодируем ключ
        try:
            # Загружаем JSON с ensure_ascii=False для корректной обработки кириллицы
            files_data = json.loads(files_value.decode('utf-8'))
            formatted_folders[folder_name] = files_data
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.error(f"Ошибка декодирования данных для папки {folder_name}: {e}")
            formatted_folders[folder_name] = {}

    # Получение данных из Elasticsearch с обработкой ошибок
    try:
        es_indexes = list(es.indices.get(index='*').keys())
    except Exception as e:
        logger.error(f"Ошибка подключения к Elasticsearch: {e}")
        es_indexes = []

    query = {
        "aggs": {
            "max_timeCreate": {
                "max": {
                    "field": "timeCreate"
                }
            },
            "min_timeCreate": {
                "min": {
                    "field": "timeCreate"
                }
            }
        },
    }

    json_folders = {}

    # ✅ Проверяем наличие ключа json_files_directory
    json_files_dir = formatted_folders.get('json_files_directory', {})
    
    # Инициализация папок
    for folder_name in json_files_dir.keys():
        json_folders[folder_name] = []

    # Обработка файлов
    for folder_name, files in json_files_dir.items():
        for file_name in files:
            file_name_stripped = file_name.replace('.json', '').lower()

            if file_name_stripped in es_indexes:
                try:
                    date_period_query = es.search(index=file_name_stripped, body=query)['aggregations']
                    
                    index_numbers = [i for i in indexes if indexes[i] == file_name_stripped]
                    index_number = index_numbers[0] if index_numbers else None
                    
                    file_info = {
                        "file": file_name_stripped,
                        "min_data": date_period_query['min_timeCreate']['value'],
                        "max_data": date_period_query['max_timeCreate']['value'],
                    }
                    
                    if index_number is not None:
                        file_info["index_number"] = index_number
                        
                    json_folders[folder_name].append(file_info)
                except Exception as e:
                    print(f"Error processing file {file_name_stripped}: {str(e)}")
                    continue

    # Обработка bertopic_files_directory
    bertopic_folders = formatted_folders.get('bertopic_files_directory', {})
    
    # Обработка projector_files_directory
    projector_folders = formatted_folders.get('projector_files_directory', {})
        
    # CSV файлы
    csv_files_directory = formatted_folders.get('csv_files_directory', {})
    
    # Если CSV данных нет, сканируем файловую систему
    if not csv_files_directory:
        bertopic_base_path = f'/home/dev/tellscope_app/tellscope_backend/data/{user_id}/bertopic_files_directory'
        
        if os.path.exists(bertopic_base_path):
            for root, dirs, files in os.walk(bertopic_base_path):
                csv_files = [f for f in files if f.startswith('result_graph_') and f.endswith('.csv')]
                
                if csv_files:
                    relative_path = os.path.relpath(root, bertopic_base_path)
                    
                    csv_info_list = []
                    for csv_file in csv_files:
                        full_path = os.path.join(root, csv_file)
                        
                        csv_info = {
                            "file": csv_file,
                            "full_path": full_path,
                            "relative_path": f"{relative_path}/{csv_file}".replace('\\', '/'),
                        }
                        
                        try:
                            file_size = os.path.getsize(full_path)
                            csv_info["size"] = file_size
                        except:
                            csv_info["size"] = 0
                        
                        csv_info_list.append(csv_info)
                    
                    folder_key = relative_path if relative_path != '.' else 'root'
                    csv_files_directory[folder_key] = csv_info_list

        # ✅ Сохраняем с ensure_ascii=False
        await redis_db.hset(
            user_id, 
            "csv_files_directory", 
            json.dumps(csv_files_directory, ensure_ascii=False)
        )

    return {
        "user_id": user_id,
        "json_files_directory": json_folders,
        "bertopic_files_directory": bertopic_folders,
        "projector_files_directory": projector_folders,
        "csv_files_directory": csv_files_directory
    }


###########################################################################################################
import aiohttp

class SingleTextRequest(BaseModel):
    user_id: int
    # folder_name: str
    text: str
    system_prompt: Optional[str] = None 
    prompt_question: str

    @validator('text', 'system_prompt', 'prompt_question', pre=True)
    def clean_strings(cls, v):
        if v is None:
            return v 
        # Удаляем двойные кавычки
        v = v.replace('"', '')
        # Удаляем все символы, кроме букв, цифр, пробелов и основных знаков препинания
        v = re.sub(r'[^a-zA-Zа-яА-Я0-9\s.,?!;:]', '', v)
        # Заменяем последовательности пробелов на один пробел
        v = re.sub(r'\s+', ' ', v)
        return v
    

class MultipleTextRequest(BaseModel):
    user_id: int
    texts: List[str]
    system_prompt: Optional[str] = None
    prompt_question: str

async def process_text(text: str, question: str, system_prompt: Optional[str]) -> str:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(VLLM_CHAT_COMPLETIONS_URL, json={
                "text": text,
                "question": question,
                "system_prompt": system_prompt
            }) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("answer", "")
                else:
                    logging.error(f"Error calling LLM API: {response.status}")
                    return ""
    except Exception as e:
        logging.error(f"Error processing text: {str(e)}", exc_info=True)
        return ""


# =========================
# Конфигурация (модель: get_vllm_model_id(), URL — VLLM_CHAT_COMPLETIONS_URL в начале файла)
# =========================

# Параллелизм и батчи
BATCH_SIZE = 32
MAX_CONCURRENCY = 64
SAVE_THRESHOLD = 50
PROGRESS_THROTTLE_SEC = 1.0

# HTTP
CONNECT_TIMEOUT = 15
TOTAL_TIMEOUT = 180
TCP_LIMIT = 512
TCP_LIMIT_PER_HOST = 256
KEEPALIVE_SEC = 300

# Ретраи
MAX_RETRIES = 2
BASE_BACKOFF = 0.5

# Усечение входа
MAX_INPUT_CHARS = 4000

# Параметры модели
TEMPERATURE = 0.1
TOP_P = 0.7
MAX_NEW_TOKENS = 4000
USE_STREAMING = False

# Фильтры "думания"
THINK_TAGS = [
    (r"<think>.*?</think>", re.DOTALL),
    (r"<reflection>.*?</reflection>", re.DOTALL),
    (r"<reasoning>.*?</reasoning>", re.DOTALL),
]
THINK_PREFIXES = [
    "think:", "thinking:", "размышления:", "мысли:", "internal:", "internal thoughts:",
    "план:", "plan:", "analysis:", "анализ:", "chain of thought:", "cot:", "coT:"
]


@app.post("/llm-run-multiple/", tags=['ai analytics'])
async def llm_run_multiple(
    analysis_request: MultipleTextRequest,
    background_tasks: BackgroundTasks
):
    logging.info(f"[MULTI] got request with {len(analysis_request.texts)} texts")
    try:
        task_id = str(uuid.uuid4())

        # Инициализируем статус задачи на "0"
        await redis_db.hset(f"task:{task_id}", mapping={
            "status": "0",
            "completed_texts": "0",
            "progress": "0"
        })
        
        # Запуск обработки текстов в фоновом режиме
        background_tasks.add_task(process_multiple_texts_task, task_id, analysis_request.dict())
        
        return JSONResponse({
            "task_id": task_id,
            "status": "processing"
        })
    except Exception as e:
        logging.error(f"Error processing request: {str(e)}", exc_info=True)
        return JSONResponse(content={"error": "Something went wrong"}, status_code=500)

MAX_CONCURRENCY = 32  # для веб‑сервиса можно поменьше, чем в оффлайн‑скрипте

session: aiohttp.ClientSession | None = None
llm_semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

# =========================
# Вспомогательные функции
# =========================

def _cache_key(text: str, question: str, system_prompt: Optional[str]) -> str:
    h = hashlib.sha256()
    safe_text = text[:2048] if text else ""
    h.update(safe_text.encode("utf-8", errors="ignore"))
    h.update(b"\x00")
    h.update((question or "").encode("utf-8", errors="ignore"))
    h.update(b"\x00")
    if system_prompt:
        h.update(system_prompt.encode("utf-8", errors="ignore"))
    return h.hexdigest()


def parse_openai_stream_line(raw: str) -> Optional[str]:
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None

    if "choices" not in data or not data["choices"]:
        return None

    ch = data["choices"][0]
    delta = None
    if isinstance(ch, dict):
        if "delta" in ch and isinstance(ch["delta"], dict):
            delta = ch["delta"].get("content")
        if delta is None and "message" in ch and isinstance(ch["message"], dict):
            delta = ch["message"].get("content")
    return delta

async def read_streaming_response(resp: aiohttp.ClientResponse) -> str:
    full_parts: List[str] = []
    async for chunk, _ in resp.content.iter_chunks():
        if not chunk:
            continue
        text = chunk.decode("utf-8", errors="ignore")
        for line in text.splitlines():
            raw = line.strip()
            if not raw or raw.startswith(":"):
                continue
            if raw.startswith("data:"):
                raw = raw[5:].strip()
            if raw == "[DONE]":
                continue
            delta = parse_openai_stream_line(raw)
            if delta:
                full_parts.append(delta)
    combined = "".join(full_parts).strip()
    return combined

async def read_non_stream_response(resp: aiohttp.ClientResponse) -> str:
    data = await resp.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    return content or ""


# =========================
# HTTP сессия
# =========================

async def create_optimized_session() -> aiohttp.ClientSession:
    connector = aiohttp.TCPConnector(
        limit=TCP_LIMIT,
        limit_per_host=TCP_LIMIT_PER_HOST,
        keepalive_timeout=KEEPALIVE_SEC,
        enable_cleanup_closed=True,
        ttl_dns_cache=300,
        use_dns_cache=True,
    )
    timeout = aiohttp.ClientTimeout(
        total=TOTAL_TIMEOUT,
        connect=CONNECT_TIMEOUT,
        sock_read=120
    )
    return aiohttp.ClientSession(
        connector=connector,
        timeout=timeout,
        headers={"Connection": "keep-alive"}
    )

def strip_thinking(raw: str) -> str:
    if not raw:
        return raw
    s = raw
    for pattern, flags in THINK_TAGS:
        s = re.sub(pattern, "", s, flags=flags)
    s_lines = s.splitlines()
    cleaned_lines = []
    for line in s_lines:
        ln = line.strip()
        lowered = ln.lower()
        if any(lowered.startswith(pfx) for pfx in THINK_PREFIXES):
            continue
        cleaned_lines.append(line)
    s = "\n".join(cleaned_lines)
    return s.strip()

def normalize_answer(answer: str) -> str:
    if not answer:
        return ""
    s = answer.strip()
    s = strip_thinking(s)
    s = s.strip()
    # убираем только одну хвостовую точку, не ломая JSON
    if s.endswith(".") and not s.endswith(".."):
        stripped = s.lstrip()
        if not (stripped.startswith("{") or stripped.startswith("[")):
            s = s[:-1].strip()
    if not s:
        return "Модель не ответила"
    return s

# =========================
# Единственная реализация generate_answer
# =========================

async def generate_answer_single(
    session: aiohttp.ClientSession,
    system_line: str,
    question_line: str,
    text: str
) -> str:
    if not text or len(text) < 8:
        return "Короткий текст"
    if len(text) > 25000:
        return "Длинный текст"

    txt = cached_truncate_text(text, MAX_INPUT_CHARS)
    user_content = f"{question_line}\n\nТекст:\n{txt}"

    messages = []
    if system_line:
        messages.append({"role": "system", "content": system_line})
    messages.append({"role": "user", "content": user_content})

    from mlops.gateway import GatewayError, achat

    extra = {"top_p": TOP_P, "stream": False}
    if USE_STREAMING:
        extra["stream"] = True

    for attempt in range(MAX_RETRIES + 1):
        try:
            result = await achat(
                provider="vllm",
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_NEW_TOKENS,
                timeout=TOTAL_TIMEOUT,
                extra=extra,
            )
            normalized = normalize_answer(result.content)
            return normalized if normalized else "Модель не ответила"
        except GatewayError as e:
            if e.status_code == 0 and "timeout" in str(e).lower():
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(BASE_BACKOFF * (2 ** attempt))
                    continue
                return "Timeout ошибка"
            if attempt < MAX_RETRIES:
                await asyncio.sleep(BASE_BACKOFF * (2 ** attempt))
                continue
            return f"Ошибка API: {e.status_code or str(e)}"
        except Exception as e:
            if attempt < MAX_RETRIES:
                await asyncio.sleep(BASE_BACKOFF * (2 ** attempt))
                continue
            return f"Ошибка: {str(e)}"

    return "Модель не ответила"
    

async def get_llm_session() -> aiohttp.ClientSession:
    global session
    if session is None or session.closed:
        timeout = aiohttp.ClientTimeout(total=180, connect=15, sock_read=120)
        connector = aiohttp.TCPConnector(
            limit=256,
            limit_per_host=128,
            keepalive_timeout=300,
            enable_cleanup_closed=True,
            ttl_dns_cache=300,
            use_dns_cache=True,
        )
        session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={"Connection": "keep-alive"}
        )
    return session


async def generate_answer(text: str, question: str, system_prompt: str | None = None) -> str:
    if not text or len(text) < 8:
        return "Короткий текст"

    system_line = system_prompt.strip() if system_prompt else ""
    # максимально близко к скрипту: question_line + "Текст:\n..."
    question_line = question.strip()

    sess = await get_llm_session()

    async with llm_semaphore:
        return await generate_answer_single(
            sess,
            system_line=system_line,
            question_line=question_line,
            text=text,
        )

BATCH_SIZE = 16  # как в скрипте, можно уменьшить до 16, если боитесь нагрузки

async def process_multiple_texts_task(task_id: str, task_data: dict):
    try:
        texts = task_data['texts']
        prompt_question = task_data['prompt_question']
        system_prompt = task_data.get('system_prompt')

        total_texts = len(texts)
        results = [""] * total_texts

        for start in range(0, total_texts, BATCH_SIZE):
            end = min(start + BATCH_SIZE, total_texts)
            batch = texts[start:end]

            tasks = [
                asyncio.create_task(
                    generate_answer(text, prompt_question, system_prompt=system_prompt)
                )
                for text in batch
            ]

            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, res in enumerate(batch_results):
                idx = start + i
                if isinstance(res, Exception):
                    logging.error(f"Error in batch idx={idx}: {res}", exc_info=True)
                    res = f"Ошибка: {res}"
                results[idx] = res

            progress = int(((end) / total_texts) * 100)
            await redis_db.hset(
                f"task:{task_id}",
                mapping={
                    "completed_texts": str(end),
                    "progress": str(progress),
                    "status": "processing"
                }
            )

        json_results = json.dumps(results, ensure_ascii=False)
        json_texts = json.dumps(texts, ensure_ascii=False)

        await redis_db.hset(f"task:{task_id}", mapping={
            'texts': json_texts,
            'results': json_results,
            "progress": "100",
            "status": "done"
        })

    except Exception as e:
        logging.error(f"Error processing task {task_id}: {str(e)}", exc_info=True)
        await redis_db.hset(f"task:{task_id}", "status", f"failed: {str(e)}")


# async def generate_answer(text, question, system_prompt=None, session=None, max_retries=2):
#     url = "http://localhost:8000/v1/chat/completions"
#     headers = {"Content-Type": "application/json"}
    
#     system_line = (
#         system_prompt.strip() if system_prompt else
#         "Ты отвечаешь только на поставленный вопрос. Только факт из текста, не повторяй формулировки вопроса."
#     )
    
#     # user_content = f"Текст: {text.strip()}\n\nВопрос: {question.strip()}\n\nОтвет (строго кратко, только факт, без разъяснений):"

#     user_content = f"Текст: {text.strip()}\nВопрос: {question.strip()}\nОтвет:"

#     payload = {
#         "model": "Qwen/Qwen3-32B-FP8",
#         "messages": [
#             {"role": "system", "content": system_line},
#             {"role": "user", "content": user_content}
#         ],
#         "temperature": 0.7,
#         "top_p": 0.8,
#         "chat_template_kwargs": {"enable_thinking": False}
#     }

#     for attempt in range(max_retries + 1):
#         try:
#             # Используем переданную сессию или создаем новую если нет
#             if session is None:
#                 async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as temp_session:
#                     return await _make_request(temp_session, url, headers, payload, attempt, max_retries)
#             else:
#                 return await _make_request(session, url, headers, payload, attempt, max_retries)
#         except Exception as e:
#             logging.error(f"Error in generate_answer, attempt {attempt + 1}: {str(e)}", exc_info=True)
#             if attempt < max_retries:
#                 await asyncio.sleep(1)
#                 continue
#             return f"Ошибка соединения с сервером генерации ответов: {str(e)}"
    
#     return "Модель не ответила"

# async def _make_request(session, url, headers, payload, attempt, max_retries):
#     async with session.post(url, json=payload, headers=headers) as response:
#         if response.status == 200:
#             data = await response.json()
#             try:
#                 generated = data["choices"][0]["message"]["content"]
#             except Exception as e:
#                 logging.error(f"Не удалось извлечь текст ответа: {e}, data={data}")
#                 generated = ""
#             answer = generated.strip().rstrip('.').strip()
#             if answer:
#                 return answer
#             else:
#                 logging.warning(f"Empty answer on attempt {attempt + 1}")
#                 if attempt < max_retries:
#                     payload["temperature"] = min(0.7, payload["temperature"] + 0.2)
#                     await asyncio.sleep(1)
#                     raise Exception("Empty answer")  # Для повторной попытки
#                 return "Модель не ответила"
#         else:
#             error_text = await response.text()
#             logging.error(f"LLM API error {response.status}: {error_text}")
#             if attempt < max_retries:
#                 await asyncio.sleep(2)
#                 raise Exception(f"API error {response.status}")  # Для повторной попытки
#             return f"Ошибка генерации ответа (код {response.status}): {error_text}"


# async def generate_answer(
#     text: str, 
#     prompt_question: str, 
#     system_prompt: Optional[str] = None,
#     max_tokens: Optional[int] = None,
#     temperature: float = 0.95,  # Увеличиваем температуру
#     top_p: float = 0.95
# ):
#     url = "http://tellscope40.headsmade.com:8000/v1/completions"
    
#     request_id = str(uuid.uuid4())[:8]
#     full_prompt = f"[ID: {request_id}] {prompt_question} Текст для анализа: {text}\nОтвет:"
    
#     payload = {
#         "prompt": full_prompt,
#         "temperature": temperature,
#         "top_p": top_p,
#         "max_tokens": max_tokens or 1000,
#         # "stop": ["\n", ".", "ID:", "[ID", "Текст"]
#     }
#     if system_prompt:
#         payload["system_prompt"] = system_prompt
    
#     try:
#         async with aiohttp.ClientSession() as session:
#             async with session.post(url, json=payload, headers={"Content-Type": "application/json"}) as response:
#                 if response.status == 200:
#                     response_json = await response.json()
#                     result = response_json.get("choices", [{}])[0].get("text", "").strip()
#                     logging.info(f"Full payload being sent to LLM:\n{payload}, answer: {result}")
                    
#                     # Более мягкая очистка ответа
#                     result = result.replace("Ответ:", "").strip()
                    
#                     return result if result else "Ошибка: пустой ответ от модели"
#                 else:
#                     error_text = await response.text()
#                     return f"Ошибка генерации ответа (код {response.status}): {error_text}"
#     except Exception as e:
#         logging.error(f"Error in generate_answer: {str(e)}", exc_info=True)
#         return f"Ошибка соединения с сервером генерации ответов: {str(e)}"


########################################### Monitoring ###############################################

@app.get("/gpu_metrics", tags=['metrics'])
async def get_gpu_metrics(): 
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.free', '--format=csv,noheader,nounits'], 
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = result.stdout.decode('utf-8').strip().split('\n')
        gpu_data = [line.split(', ') for line in output]
        return {"gpu_metrics": gpu_data}
    except Exception as e:
        return {"error": str(e)}

@app.get("/server_metrics", tags=['metrics'])
async def get_metrics():
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()
    return {
        "cpu_usage": cpu_usage,
        "memory_usage": memory_info.percent,
        "total_memory": memory_info.total,
        "available_memory": memory_info.available,
    }

########################################### Monitoring  End ###############################################

from sqlalchemy import text
from sklearn.cluster import DBSCAN
# from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import silhouette_score
from typing import List, Dict
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN


# Функция для получения текстов
async def get_texts(user_id: int, folder_name: str, file_name: str, session: AsyncSession) -> list:
    file_name = f"my_list_llm_ans_{file_name}".replace('.html', '.pkl')
    file_path = f"/home/dev/tellscope_app/tellscope_backend/data/{user_id}/bertopic_files_directory/{folder_name}/{file_name}"
    print(f"Loading texts from: {file_path}")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found.")
    with open(file_path, 'rb') as f:
        texts = pickle.load(f)  # Файл предполагается, что содержит список текстов
    _, texts = unpack_saved_llm_labels(texts)
    return texts

def cosine_similarity_vectors(vec1: np.ndarray, norm1: float,
                                vec2: np.ndarray, norm2: float) -> float:
    """Вычисляет косинусное сходство между двумя векторами."""
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)


# Определяем базовый класс для моделей
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String, JSON, select
Base = declarative_base()

# Определяем модель для хранения эмбеддингов
class Embedding(Base):
    __tablename__ = 'embedding'
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)
    filename = Column(String, nullable=False) 
    # Например, поле для хранения эмбеддингов
    vectors = Column(JSON, nullable=False)


# Функция для получения эмбеддинга по user_id и filename
async def get_embedding(session: AsyncSession, user_id: int, file_name: str):
    stmt = select(Embedding).where(
        Embedding.user_id == user_id,
        Embedding.filename == file_name
    )
    result = await session.execute(stmt)
    return result.scalars().first()


@app.get("/text_clusters/", tags=['ai analytics'])
async def get_text_clusters(user_id: int, folder_name: str, file_name: str,
                            session: AsyncSession = Depends(get_db),
                            threshold: float = 0.8):
    if user_id < 1:
        raise HTTPException(status_code=400, detail="user_id must be a positive integer.")

    texts = await get_texts(user_id, folder_name, file_name, session)
    texts = texts[:10]
    if not texts:
        raise HTTPException(status_code=404, detail="No texts found for clustering.")
    
    # Получение эмбеддингов из базы данных
    embedding = await get_embedding(session, user_id, file_name)
    if embedding is None:
        raise HTTPException(status_code=404, detail="Embeddings not found for the specified user and file.")

    vectors = embedding.vectors
    if not vectors:
        raise HTTPException(status_code=404, detail="No vectors found in embedding.")

    clusters = []

    def calculate_distance(vec1, vec2):
        """Calculate the Euclidean distance between two vectors."""
        return np.sqrt(np.sum((np.array(vec1) - np.array(vec2)) ** 2))
    
    for idx, vec in enumerate(vectors):
        found_cluster = False
        
        for cluster in clusters:
            # Находим расстояние между вектором и центром кластера
            distance = calculate_distance(cluster['center'], vec)  # Функция для вычисления расстояния
            
            if distance < threshold:
                new_count = cluster['count'] + 1
                # Обновляем центр кластера
                cluster['center'] = [(cluster['center'][i] * cluster['count'] + vec[i]) / new_count for i in range(len(cluster['center']))]
                cluster['count'] = new_count
                cluster['texts'].append(texts[idx])
                found_cluster = True
                break
        
        if not found_cluster:
            clusters.append({
                'center': vec,
                'count': 1,
                'texts': [texts[idx]]
            })

    # Формируем список результатов, где каждому тексту сопоставлен номер кластера
    results = []
    for cluster_id, cluster in enumerate(clusters):
        for txt in cluster['texts']:
            results.append((cluster_id, txt))

    # Далее – получение пользовательских данных из Redis
    user_data = await redis_db.hgetall(str(user_id))
    # Декодирование данных пользователя
    user_data = {key.decode('utf-8'): value.decode('utf-8') for key, value in user_data.items()}
    for key, value in user_data.items():
        try:
            user_data[key] = json.loads(value)
        except json.JSONDecodeError:
            print(f"Ошибка декодирования JSON для ключа {key}: {value}")

    if user_data is None:
        raise HTTPException(status_code=404, detail="User not found")

    # Поиск нужного HTML‑файла в данных пользователя
    html_files = user_data.get("bertopic_files_directory", {}).get(folder_name, [])
    html_file_path = None
    info_html = {}  # для использования далее в elasticsearch

    for file_info in html_files:
        if file_info.get("html-file") == file_name:
            info_html = file_info
            html_file_path = os.path.join("/home/dev/tellscope_app/tellscope_backend/data", str(user_id),
                                          "bertopic_files_directory", folder_name, file_name)
            break

    if html_file_path is None or not os.path.exists(html_file_path):
        raise HTTPException(status_code=404, detail="HTML file not found")

    # Выполнение запроса в elasticsearch за указанный диапазон дат и с нужной строкой поиска
    file_path_indexes = '/home/dev/tellscope_app/tellscope_backend/data/indexes.pkl'
    indexes = load_dict_from_pickle(file_path_indexes)
    
    if info_html.get('query_str') is None:
        info_html['query_str'] = 'all'
    
    data = elastic_query(theme_index=indexes[info_html['index_number']],
                         query_str=info_html['query_str'],
                         min_date=info_html['min_date'],
                         max_date=info_html['max_date'])
    data = pd.DataFrame(data)

    # Объединение LLM с метаданными
    data.rename(columns={'url': 'text_url'}, inplace=True)
    data = data.join(pd.DataFrame(list(data['authorObject'].values)))
    data.rename(columns={'url': 'author_url'}, inplace=True)
    data = data[['timeCreate', 'hub', 'author_url', 'fullname', 'text_url', 'author_type',
                 'sex', 'age', 'hubtype', 'commentsCount', 'audienceCount',
                 'repostsCount', 'likesCount', 'er', 'viewsCount',
                 'toneMark', 'country', 'region']]

    # Объединение результатов кластеризации с данными из elasticsearch
    df_results = pd.DataFrame(results, columns=['Кластер', 'Тематика текста'])
    df_join = df_results.join(data, how='inner', lsuffix='_df1', rsuffix='_df2')
    df_join.columns = ['Кластер', 'Тематика текста', 'Время', 'Источник', 'Ссылка на автора',
                       'Автор', 'Ссылка на текст', 'Тип автора', 'Пол', 'Возраст',
                       'Тип источника', 'Комментариев', 'Аудитория', 'Репостов', 'Лайков',
                       'Вовлеченность', 'Просмотров', 'Тональность', 'Страна', 'Регион']

    df_join.to_excel('/home/dev/tellscope_app/tellscope_backend/data/1/cluster_fobii.xlsx', index=False, engine='openpyxl')

    return {
        "cluster_data": df_join.where(pd.notnull(df_join), None).to_dict(orient='records')
    }

################################################### RAG ########################################################
from fastapi import APIRouter, HTTPException, Depends
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, Table, MetaData
import aiohttp

from sqlalchemy import Column, Integer, String, JSON, Table, MetaData, Text
from sqlalchemy.future import select
from sqlalchemy import insert

from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Embedding(Base):
    __tablename__ = 'embeddings_pg'
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)
    filename = Column(String(255), nullable=False)
    folder_name = Column(String(255), nullable=False)
    vectors = Column(JSON, nullable=False)


# Загрузка модели SentenceTransformer для создания эмбеддингов
# embedding_model = SentenceTransformer("/home/dev/tellscope_app/tellscope_backend/data/embed_files/embed_files/DeepPavlov/rubert-base-cased-sentence")

# async def generate_answers(client, prompt):
#     url = "http://localhost:11434/api/generate"
#     payload = {
#         "model": "erwan2/DeepSeek-R1-Distill-Qwen-14B",  # Vikhr_Q3
#         "prompt": prompt,
#         "stream": False
#     }
#     async with aiohttp.ClientSession() as session:
#         async with session.post(url, json=payload) as response:
#             if response.status == 200:
#                 response_json = await response.json()
#                 return response_json.get("response", "")
#             else:
#                 print(f"Ошибка при запросе к Ollama: {response.status}")
#                 return None

# class QueryRequest(BaseModel):
#     query: str
#     user_id: int
#     filename: str
#     folder_name: str
#     num_results: int = 5
#     generate_answer: bool = True

# from ollama import AsyncClient
# # Создаём клиент один раз
# client = AsyncClient(host='http://localhost:11434')

# @app.post("/rag", tags=['ai analytics'])
# async def rag_query(request: QueryRequest, session: AsyncSession = Depends(get_db)):
#     try:
#         user_query = request.query
#         user_id = request.user_id
#         filename = request.filename
#         folder_name = request.folder_name
#         num_results = request.num_results
#         generate_answer = request.generate_answer

#         # Получение информации из Redis
#         user_data = await redis_db.hgetall(user_id)
#         user_data = {key.decode('utf-8'): value.decode('utf-8') for key, value in user_data.items()}
#         # Декодируем JSON-значения в словари
#         for key, value in user_data.items():
#             try:
#                 user_data[key] = json.loads(value)
#             except json.JSONDecodeError:
#                 print(f"Ошибка декодирования JSON для ключа {key}: {value}")

#         def extract_relevant_part(filename):
#             # Разделяем строку на части по символу '_'
#             parts = filename.split('_')
#             # Объединяем все части до последнего подчеркивания
#             relevant_part = '_'.join(parts[:-2])  # исключаем последние две части
#             return relevant_part
        
#         # Поиск нужной информации в bertopic_files_directory
#         theme_index = None
#         min_date = None
#         max_date = None
#         query_str = None
#         for item in user_data["bertopic_files_directory"][folder_name]:
#             print(111555)
#             if item["html-file"] == filename:
#                 print(item)
#                 theme_index = extract_relevant_part(filename)
#                 print(555999777)
#                 print(theme_index)
#                 if "min_date" in item:
#                     min_date = item["min_date"]
#                     max_date = item["max_date"]
#                 else:
#                     min_date = item["min_data"]
#                     max_date = item["max_data"]
#                 query_str = item["query_str"]
#                 break
        
#         if theme_index is None:
#             raise HTTPException(status_code=404, detail="Файл не найден")
        
#         # Получение текстов из Elasticsearch
#         data = elastic_query(theme_index=theme_index, min_date=min_date, max_date=max_date, query_str=query_str)
#         texts = [x['text'] for x in data]

#         # Создание эмбеддинга для запроса пользователя
#         query_embedding = embedding_model.encode(user_query, show_progress_bar=False)

#         # Извлечение эмбеддингов из базы данных с учетом user_id, filename и folder_name
#         query = select(Embedding).where(
#             Embedding.user_id == user_id,
#             Embedding.filename == filename,
#             Embedding.folder_name == folder_name
#         )
#         result = await session.execute(query)
#         embeddings = result.scalars().all()

#         if not embeddings:
#             raise HTTPException(status_code=404, detail="Эмбеддинги не найдены")
        
#         # Расчет косинусного сходства между запросом и эмбеддингами
#         query_embedding = list(query_embedding)  # Преобразование в одномерный список
#         user_embeddings = [emb.vectors for emb in embeddings][0]  # Преобразование каждого вектора в одномерный список

#         print(f'len_user_embeddings: {len(user_embeddings)}')

#         # similarities = cosine_similarity([query_embedding], user_embeddings)[0]

#         query_embedding_reshaped = np.array(query_embedding).reshape(1, -1)  # Преобразование в двумерный массив для одного запроса
#         user_embeddings_reshaped = np.array(user_embeddings)  # Двумерный массив эмбеддингов пользователей

#         similarities = cosine_similarity(query_embedding_reshaped, user_embeddings_reshaped)[0]
#         # print(similarities)

#         # Получение индексов наиболее релевантных эмбеддингов
#         # top_indices = similarities.argsort()[-num_results:][::-1]
#         top_indices = np.argpartition(similarities, -num_results)[-num_results:]
#         print(top_indices)
#         # print(similarities.argsort())
#         # print(f'top_indices: {top_indices}')

#         # Получение наиболее релевантных текстов
#         top_texts = [texts[i] for i in top_indices]
#         # print(top_texts)

#         if generate_answer:
#             # Генерация ответа с использованием модели генерации текста
#             prompt = f"Query: {user_query}\nContext: {' '.join([texts[i] for i in top_indices])}\nAnswer:"  # Здесь берем тексты по индексам
#             answer = await generate_answers(client=client, prompt=prompt)
            
#             return {"answer": answer, "top_texts": top_texts}
#         else:
#             return {"top_texts": top_texts}

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


from fastapi import APIRouter, HTTPException, Query

@app.get("/llm-analyze-excel", tags=['files'])
async def llm_analyze_excel(user_id: int, folder_name: str, file_name: str, all_table: bool = Query(True)):
    global full_data_store, aggregated_data_store

    user_data = await redis_db.hgetall(str(user_id))  # Получаем данные пользователя из Redis

    user_data = {key.decode('utf-8'): value.decode('utf-8') for key, value in user_data.items()}
    # Декодируем JSON-значения в словари
    for key, value in user_data.items():
        try:
            user_data[key] = json.loads(value)
        except json.JSONDecodeError:
            print(f"Ошибка декодирования JSON для ключа {key}: {value}")

    if user_data is None:
        raise HTTPException(status_code=404, detail="User not found")

    # Находим нужный HTML-файл
    html_files = user_data["bertopic_files_directory"].get(folder_name, [])
    html_file_path = None

    info_html = {}  # для использования далее в elasticsearch
    for file_info in html_files:
        if file_info["html-file"] == file_name:
            info_html = file_info
            html_file_path = os.path.join("/home/dev/tellscope_app/tellscope_backend/data", str(user_id), 
                                           "bertopic_files_directory", folder_name, file_name)
            break

    if html_file_path is None or not os.path.exists(html_file_path):
        raise HTTPException(status_code=404, detail="HTML file not found")

    # Определяем базовое имя модели без расширения
    model_file_name_base = file_name.replace('.html', '').split('_')[-1]

    # Теперь ищем нужный модельный файл
    model_folder_name = None
    for file_info in html_files:
        if model_file_name_base in file_info["model-file"]:
            model_folder_name = folder_name
            break

    if model_folder_name is None:
        raise HTTPException(status_code=404, detail="Model folder not found")

    # Создаем путь к модели
    model_path = os.path.join("/home/dev/tellscope_app/tellscope_backend/data", str(user_id), 
                               "bertopic_files_directory", model_folder_name, 
                               next(file_info["model-file"] for file_info in html_files if model_file_name_base in file_info["model-file"]))

    # Проверяем, существует ли путь к модели
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model file not found")

    # Модель BERTopic
    topic_model = BERTopic.load(model_path)

    # Поиск в elastic за те же даты и строку поиска
    file_path = '/home/dev/tellscope_app/tellscope_backend/data/indexes.pkl'
    indexes = load_dict_from_pickle(file_path)
    
    if info_html['query_str'] is None:
        info_html['query_str'] = 'all'

    if 'min_data' not in info_html:
        data = elastic_query(theme_index=indexes[info_html['index_number']], query_str=info_html['query_str'], 
                             min_date=info_html['min_date'], max_date=info_html['max_date'])
    else:
        data = elastic_query(theme_index=indexes[info_html['index_number']], query_str=info_html['query_str'], 
                             min_date=info_html['min_data'], max_date=info_html['max_data'])
    data = pd.DataFrame(data)

    # Обработка тематики
    df_topic = topic_model.get_topic_info()[['CustomName', 'Topic']]
    dct_df_topic = dict(zip(df_topic['Topic'], df_topic['CustomName']))
    thematics = [dct_df_topic[x] for x in topic_model.topics_] 

    data.rename(columns={'url': 'text_url'}, inplace=True)
    data = data.join(pd.DataFrame(list(data['authorObject'].values)))
    data.rename(columns={'url': 'author_url'}, inplace=True)
    data = data[['timeCreate', 'hub', 'author_url', 'fullname', 'text_url', 'author_type', 'sex', 'age',
                   'hubtype', 'commentsCount', 'audienceCount',
                   'repostsCount', 'likesCount', 'er', 'viewsCount',
                   'massMediaAudience', 'toneMark', 'country', 'region']]

    df_join = pd.DataFrame(thematics).join(data, how='inner', lsuffix='_df1', rsuffix='_df2')
    df_join.columns = ['Имя кластера', 'Время', 'Источник', 'Ссылка на автора', 'Автор', 'Ссылка на текст', 
                       'Тип автора', 'Пол', 'Возраст', 'Тип источника', 'Комментариев', 'Аудитория', 
                       'Репостов', 'Лайков', 'Вовлеченность', 'Просмотров',
                       'Аудитория СМИ', 'Тональность', 'Страна', 'Регион']
    
    df_join.reset_index(drop=True, inplace=True)  
    df_join.insert(0, 'id', df_join.index)  
    
    df_join.columns = ['ID', 'Имя кластера', 'Время', 'Источник', 'Ссылка на автора', 'Автор', 'Ссылка на текст', 
                       'Тип автора', 'Пол', 'Возраст', 'Тип источника', 'Комментариев', 'Аудитория', 
                       'Репостов', 'Лайков', 'Вовлеченность', 'Просмотров', 'Аудитория СМИ', 'Тональность', 
                       'Страна', 'Регион'] 

    df_join.drop('Аудитория СМИ', axis=1, inplace=True)
    df_join['Тональность'] = df_join['Тональность'].map({0: 'Нейтральная', -1: 'Негатив', 1: 'Позитив'})

    df_group = df_join[['Имя кластера', 'Комментариев', 'Аудитория', 'Репостов', 'Лайков', 'Вовлеченность', 'Просмотров']].copy()
    
    numerical_columns = ['Комментариев', 'Аудитория', 'Репостов', 'Лайков', 'Вовлеченность', 'Просмотров']
    
    for column in numerical_columns:
        df_group[column] = pd.to_numeric(df_group[column], errors='coerce')
        df_group[column] = df_group[column].fillna(0).astype(int)

    # Сначала считаем количество записей в каждом кластере ДО группировки
    theme_count = df_group['Имя кластера'].value_counts()

    # Затем группируем и суммируем остальные показатели
    result = df_group.groupby('Имя кластера').sum().reset_index()

    # Добавляем количество записей
    result['Количество'] = result['Имя кластера'].map(theme_count)

    result.sort_values(by='Количество', ascending=False, inplace=True)
    result = result[['Имя кластера', 'Количество', 'Аудитория', 'Комментариев', 'Репостов', 'Лайков', 'Вовлеченность', 'Просмотров']]

    result = result.where(pd.notnull(result), None)

    texts_path = os.path.join("/home/dev/tellscope_app/tellscope_backend/data", str(user_id), 
                                "bertopic_files_directory", model_folder_name)
    files = os.listdir(texts_path)

    file = [file for file in files if file_name.replace('.html', '') in file][0]
    thematics_path = texts_path + '/' + 'my_list_llm_ans_' + file.replace('.html', '.pkl').replace('topic_names_', '')
    
    with open(thematics_path.replace('_datamapplot', ''), 'rb') as f:
        texts_thematics = pickle.load(f)
    _, texts_thematics = unpack_saved_llm_labels(texts_thematics)
    df_join.insert(1, 'Тематика текста', texts_thematics)

    output_path = os.path.join('/home/dev/tellscope_app/tellscope_backend/data/files', (file_name.capitalize() + 'aggregated_table.xlsx').replace('.html', '_') 
                               if not all_table else (file_name.capitalize() + 'all_table.xlsx').replace('.html', '_'))

    # В зависимости от параметра all_table сохраняем соответствующую таблицу
    if all_table:
        # Удаление столбца 'ID'
        if 'ID' in df_join.columns:
            df_join = df_join.drop(columns=['ID'])
        df_join.to_excel(output_path, index=False)
    else:
        # Удаление столбца 'ID'
        if 'ID' in result.columns:
            result = result.drop(columns=['ID'])
        result.to_excel(output_path, index=False)

    # Возврат файла
    return FileResponse(output_path, media_type='application/octet-stream', filename=os.path.basename(output_path))




class TextInput(BaseModel):
    texts: List[str]

# @app.post("/text-clusters-similarity/", tags=['ai analytics'])
# async def get_text_clusters(
#     user_id: int,
#     folder_name: str,
#     file_name: str,
#     text_input: TextInput,  # Изменяем параметр на text_input
#     session: AsyncSession = Depends(get_db),
#     threshold: float = 0.8):

#     # Получаем эмбеддинги из базы данных
#     embedding = await get_embedding(session, user_id, file_name)
#     if embedding is None:
#         raise HTTPException(status_code=404, detail="Embeddings not found for the specified user and file.")
    
#     if user_id < 0:
#         raise HTTPException(status_code=400, detail="Invalid user ID.")
        
#     vectors = embedding.vectors
#     if not vectors:
#         raise HTTPException(status_code=404, detail="No vectors found in embedding.")
        
#     print(len(vectors))
#     gc.collect()
#     torch.cuda.empty_cache()

#     # Инициализируем модель эмбеддингов
#     embedding_model = SentenceTransformer("/home/dev/tellscope_app/tellscope_backend/data/embed_files/DeepPavlov/rubert-base-cased-sentence")

#     # Получаем эмбеддинги для текстов
#     embedding_texts = embedding_model.encode(text_input.texts)  # Изменяем на text_input.texts

#     # Итоговый словарь для хранения результатов
#     results = {text: [] for text in text_input.texts}  # Инициализируем результат по каждому тексту
#     other_indices = []  # Список для индексов, которые не соответствуют ни одному тексту

#     # Сравнение текстовых эмбеддингов с векторами
#     for text_idx, text_vector in enumerate(embedding_texts):
#         text_vector = np.array(text_vector)  # Приводим текстовый вектор к numpy массиву
#         has_match = False  # Переменная для отслеживания наличия совпадений

#         for idx, vector in enumerate(vectors):
#             vector = np.array(vector)  # Приводим вектор к numpy массиву
#             # Вычисляем косинусное сходство
#             cosine_similarity = np.dot(text_vector, vector) / (np.linalg.norm(text_vector) * np.linalg.norm(vector))
#             print(cosine_similarity)
#             # Проверяем, превышает ли сходство указанный порог
#             if cosine_similarity >= threshold:
#                 results[text_input.texts[text_idx]].append(idx)  # Добавляем индекс к соответствующему тексту
#                 has_match = True  # Найдено совпадение

#         # Если совпадений не найдено, добавляем индекс в 'other'
#         if not has_match:
#             other_indices.append(text_idx)

#     # Добавляем индексы, которые ни с чем не совпадают, в специальный ключ 'other'
#     results['other'] = other_indices

#     # Возвращаем итоговый словарь результатов
#     return json.dumps(results, ensure_ascii=False)


from sklearn.metrics.pairwise import cosine_distances

# @app.post("/text-clusters-files/", tags=['ai analytics'])
# async def get_text_clusters(
#     user_id: int,
#     folder_name: str,
#     file_name: str,
#     text_input: TextInput,  # Изменяем параметр на text_input
#     session: AsyncSession = Depends(get_db),
#     threshold: float = 0.8):

#     # Получаем эмбеддинги из базы данных
#     embedding = await get_embedding(session, user_id, file_name)
#     if embedding is None:
#         raise HTTPException(status_code=404, detail="Embeddings not found for the specified user and file.")
    
#     if user_id < 0:
#         raise HTTPException(status_code=400, detail="Invalid user ID.")
        
#     text_embeddings = embedding.vectors
#     if not text_embeddings:
#         raise HTTPException(status_code=404, detail="No vectors found in embedding.")
        
#     print(len(text_embeddings))

#     gc.collect()
#     torch.cuda.empty_cache()

#     # texts = df[1].values[:100]
#     # print(texts[:3])
#     themes_texts = text_input.texts
#     # print(themes_texts[:3])

#     # Инициализируем модель эмбеддингов
#     embedding_model = SentenceTransformer("/home/dev/tellscope_app/tellscope_backend/data/embed_files/DeepPavlov/rubert-base-cased-sentence")

#     # Получаем эмбеддинги для текстов themes_texts
#     embedding_themes = embedding_model.encode(themes_texts, show_progress_bar=False)
#     print(f'len(embedding_themes): {len(embedding_themes)}')

#     # Максимальная длина токенов
#     max_length = 512

#     # Функция для нахождения близких эмбеддингов
#     def find_similar_embeddings(theme_embedding, text_embeddings):

#         # Вычисляем косинусные расстояния между theme_embedding и text_embeddings
#         distances = cosine_distances([theme_embedding], text_embeddings).flatten()
        
#         # Находим индексы эмбеддингов, которые близки к theme_embedding
#         similar_idx = np.where((1-distances) > threshold)[0]
        
#         return similar_idx

#     result = {}
#     for i in range(len(embedding_themes)):

#         indexes = find_similar_embeddings(embedding_themes[i], text_embeddings)
#         print(indexes)
#         print("+++!!!+++")
#         result[themes_texts[i]] = [str(j) for j in indexes]

#     # Возвращаем итоговый словарь результатов в формате JSON
#     return json.dumps(result, ensure_ascii=False)


### запрос на поиск близости для нескольких текстов
@app.post("/text-clusters-embed/", tags=['ai analytics'])
async def get_text_clusters(
    user_id: int,
    folder_name: str,
    file_name: str,
    text_input: TextInput,  # Изменяем параметр на text_input
    session: AsyncSession = Depends(get_db),
    threshold: float = 0.8):

    # Получаем эмбеддинги из базы данных
    embedding = await get_embedding(session, user_id, file_name)
    if embedding is None:
        raise HTTPException(status_code=404, detail="Embeddings not found for the specified user and file.")
    
    file_path = '/home/dev/tellscope_app/tellscope_backend/data/indexes.pkl'
    indexes = load_dict_from_pickle(file_path)

    def remove_timestamp_from_filename(filename):
        """
        Очищает название файла, удаляя дату, время и расширение.
        
        Args:
            filename (str): Название файла.
            
        Returns:
            str: Очищенное название файла.
        """
        pattern = r'_\d{8}_\d{6}\.html$'
        return re.sub(pattern, '', filename)
    
    user_data = await redis_db.hgetall(str(user_id))  # Получаем данные пользователя из Redis
    user_data = {key.decode('utf-8'): value.decode('utf-8') for key, value in user_data.items()}
    # Декодируем JSON-значения в словари
    for key, value in user_data.items():
        try:
            user_data[key] = json.loads(value)
        except json.JSONDecodeError:
            print(f"Ошибка декодирования JSON для ключа {key}: {value}")

    if user_data is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Находим нужный HTML-файл
    html_files = user_data["bertopic_files_directory"].get(folder_name, [])
    html_file_path = None

    info_html = {}  # для использования далее в elasticsearch
    # Ищем файл по указанному имени
    for file_info in html_files:
        if file_info["html-file"] == file_name:
            info_html = file_info
            html_file_path = os.path.join("/home/dev/tellscope_app/tellscope_backend/data", str(user_id), 
                                           "bertopic_files_directory", folder_name, file_name)
            break

    if html_file_path is None or not os.path.exists(html_file_path):
        raise HTTPException(status_code=404, detail="HTML file not found")

    index_name = remove_timestamp_from_filename(file_name)
    data = elastic_query(theme_index=index_name, min_date=info_html['min_data'], max_date=info_html['max_data'], 
                         query_str=info_html['query_str'])
    
    theme_texts = [x['text'] for x in data]
    
    if user_id < 0:
        raise HTTPException(status_code=400, detail="Invalid user ID.")
        
    text_embeddings = embedding.vectors
    if not text_embeddings:
        raise HTTPException(status_code=404, detail="No vectors found in embedding.")
        
    print(len(text_embeddings))

    gc.collect()
    torch.cuda.empty_cache()

    themes_texts = text_input.texts

    # Инициализируем модель эмбеддингов
    # embedding_model = SentenceTransformer("/home/dev/tellscope_app/tellscope_backend/data/embed_files/DeepPavlov/rubert-base-cased-sentence")

    # Получаем эмбеддинги для текстов themes_texts
    embedding_themes = embedding_model.encode(themes_texts, show_progress_bar=False)
    print(f'len(embedding_themes): {len(embedding_themes)}')

    # Максимальная длина токенов
    max_length = 512

    # Функция для нахождения близких эмбеддингов
    def find_similar_embeddings(theme_embeddings, text_embeddings):
        similar_indexes = []
        for theme_embedding in theme_embeddings:
            # Вычисляем косинусные расстояния между theme_embedding и text_embeddings
            distances = cosine_distances([theme_embedding], text_embeddings).flatten()
            print(distances)
            
            # Находим индексы эмбеддингов, которые близки к theme_embedding
            similar_idx = np.where((1-distances) > threshold)[0]
            similar_indexes.extend(similar_idx)
        
        return similar_indexes

    result = {}
    similar_indexes = find_similar_embeddings(embedding_themes, text_embeddings)
    result["theme"] = [theme_texts[int(j)] for j in similar_indexes]

    print(len(result["theme"]))
    print(777)
    print(similar_indexes)
    # print(theme_texts[int(result["theme"][0])])
    # print([x for x in theme_texts if 'Здравствуйте! Забрали обратную связь. Спасибо большое за отзыв!' in x])
    # Возвращаем итоговый словарь результатов в формате JSON
    return json.dumps(result, ensure_ascii=False)


@app.post("/ai-question", tags=['data analytics'])
def ai_question():

    return f'Да, пришел запрос, вот мой ответ!'


from mlops.gateway import GatewayChatClient
from mlops.lock import external_cfg

client = GatewayChatClient(provider="aitunnel", profile="dashboard_qa")
ai_model = external_cfg("dashboard_qa")["model"]


def _dashboard_prompt(lock_name: str, default_id: str) -> str:
    try:
        from mlops.lock import prompt_id as lock_prompt_id
        from mlops.prompts import render_prompt
        return render_prompt(lock_prompt_id(lock_name, default_id))
    except Exception:
        return "Ты аналитик социальных медиа. Отвечай только по входным данным. Не выдумывай цифры."

# Вариант 2: Если вы не знаете структуру данных заранее 
@app.post("/ai-question-raw", tags=['data analytics'])
async def ai_question_raw(request: Request):
    from mlops.dashboard_qa import handle_question_raw
    return await handle_question_raw(request)
    # Получаем тело запроса в виде байтов
    body_bytes = await request.body()
    
    # Пытаемся преобразовать в JSON
    try:
        body_json = json.loads(body_bytes)
        
        # Обработка данных в зависимости от current_tab
        processed_data = process_data_by_tab(body_json)

        system_prompt = _dashboard_prompt("dashboard_qa_raw", "dashboard_qa_raw_v1")

        question = processed_data["question"]
        data = processed_data["data"]

        print('texts_examples-texts_examples-texts_examples-texts_examples-texts_examples')
        print("===============++++++++++++++===============")
        texts_examples = None

        if processed_data["current_tab"] == "Тональность авторов":
            
            # Безопасный доступ к данным
            if "data" in processed_data and "similar_texts" in processed_data["data"]:
                texts_examples = processed_data["data"]["similar_texts"]
            elif "similar_texts" in processed_data:
                texts_examples = processed_data["similar_texts"]
            else:
                # Предоставляем пустой список или значение по умолчанию, если данные отсутствуют
                texts_examples = []
                print("Предупреждение: similar_texts не найдены в processed_data")

            
            print(processed_data)
            # Отладочный вывод структуры данных
            
            system_prompt = _dashboard_prompt("dashboard_qa_tonality", "dashboard_qa_tonality_v1")

            chat_result = client.chat.completions.create(
                messages=[{"role": "user", "content": f"{system_prompt}, {question}: {data} Примеры текстов: {texts_examples}"}],
                model=ai_model,
                max_tokens=10000,
            )

            return chat_result.choices[0].message
        
        chat_result = client.chat.completions.create(
            messages=[{"role": "user", "content": f"{system_prompt}, {question}: {data}"}],
            model=ai_model,
            max_tokens=30000, # Старайтесь указывать для более точного расчёта цены
        )

        return chat_result.choices[0].message
    
    except json.JSONDecodeError:
        body_json = {"error": "Не удалось разобрать JSON"}
        print(f"Получены неструктурированные данные: {body_bytes.decode()}")
    
        return {
            "message": "Да, пришел запрос, но произошла ошибка!",
            # "received_data": processed_data
        }
    
from qdrant_client.http import models
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny
  

def process_data_by_tab(data):

    if "question" not in data:
        return data
    
    try:
        current_tab = data.get("current_tab", "")
        processed_data = {
            "question": data.get("question", ""),
            "current_tab": current_tab,
            "data": {}
        }
        
        # Подключение к Elasticsearch
        es = Elasticsearch(
            hosts=["http://localhost:9200"],
            basic_auth=("elastic", "biz8z5i1w0nLPmEweKgP"),
            verify_certs=False,  # Если используется HTTP без SSL
            headers={"Accept": "application/vnd.elasticsearch+json; compatible-with=9"}  # Явно указываем версию 7
        )
        
        if current_tab == "Негативные упоминания":
            # Обработка для негативных упоминаний
            if "tonality_values" in data.get("data", {}):
                processed_data["data"]["Количество негатива на источниках"] = {
                    "negative_count": data["data"]["tonality_values"].get("negative_count", 0)
                }
            
            if "tonality_hubs_values" in data.get("data", {}):
                negative_hubs = []
                
                for hub in data["data"]["tonality_hubs_values"].get("negative_hubs", []):
                    negative_hubs.append({
                        "Название источника": hub.get("name", ""),
                        "Количество сообщений": hub.get("values", 0),
                        "Количество комментариев на источнике": hub.get("comments_sum", 0),
                        "Количество лайков на источнике": hub.get("likes_sum", 0),
                        "Количество просмотров на источнике": hub.get("views_sum", 0),
                        "Суммарная аудитория": hub.get("audience_sum", 0)
                    })
                
                processed_data["data"]["Тональные источники"] = {
                    "Негативные источники": negative_hubs
                }
                
        elif current_tab == "Позитивные упоминания":
            # Обработка для позитивных упоминаний 
            if "tonality_values" in data.get("data", {}):
                processed_data["data"]["Количество позитива на источниках"] = {
                    "positive_count": data["data"]["tonality_values"].get("positive_count", 0)
                }
            
            if "tonality_hubs_values" in data.get("data", {}):
                positive_hubs = []
                
                for hub in data["data"]["tonality_hubs_values"].get("positive_hubs", []):
                    positive_hubs.append({
                        "Название источника": hub.get("name", ""),
                        "Количество сообщений": hub.get("values", 0),
                        "Количество комментариев на источнике": hub.get("comments_sum", 0),
                        "Количество лайков на источнике": hub.get("likes_sum", 0),
                        "Количество просмотров на источнике": hub.get("views_sum", 0),
                        "Суммарная аудитория": hub.get("audience_sum", 0)
                    })
                
                processed_data["data"]["Тональные источники"] = {
                    "Позитивные источники": positive_hubs
                }
                
        elif current_tab == "Тональность авторов":

            question = data.get("question", "")
            
            # Получаем embedding для вопроса
            embeddings = client.embeddings.create(
                input=question,
                model="text-embedding-3-small"
            )
            embedding = embeddings.data[0].embedding
            
            # Получаем имя коллекции/индекса
            indexes = load_dict_from_pickle('/home/dev/tellscope_app/tellscope_backend/data/indexes.pkl')
            collection_name = indexes.get(data.get("index", 0), "")
            
            # Получаем elastic_ids из данных
            elastic_ids = []
            for author_type in ["negative_authors_values", "positive_authors_values"]:
                if author_type in data.get("data", {}):
                    for author in data["data"][author_type]:
                        for author_data in author.get("author_data", []):
                            for text in author_data.get("texts", []):
                                if "elastic_id" in text:
                                    elastic_ids.append(text["elastic_id"])

            # Удаляем дубликаты и пустые значения
            elastic_ids = list(set([int(eid) for eid in elastic_ids if eid]))

            print("-----------++++++++++++-----------------")
            print(f'elastic_ids: {elastic_ids}')
            
            # Получаем тексты из Elasticsearch
            texts_from_elastic = []
            if elastic_ids and collection_name:
                # Формируем запрос к Elasticsearch
                query = {
                    "query": {
                        "terms": {
                            "_id": elastic_ids
                        }
                    },
                    "_source": ["text", "title", "hub", "url", "authorObject", "toneMark"]
                }
                
                # Выполняем поиск в Elasticsearch
                response = es.search(
                    index=collection_name,
                    body=query,
                    size=len(elastic_ids)
                )

                # Обрабатываем результаты
                print('Yes-yes')
                for hit in response.get('hits', {}).get('hits', []):
                    source = hit.get('_source', {})
                    texts_from_elastic.append({
                        "text": source.get("text", ""),
                        "title": source.get("title", ""),
                        "source": {
                            "hub": source.get("hub", ""),
                            "url": source.get("url", "")
                        },
                        "author": source.get("authorObject", {}),
                        "elastic_id": hit.get("_id")  # <-- Вот здесь получаем _id из метаданных, а не из _source
                    })
            
                print('Yes-yes-yes')
                print(f'texts_from_elastic: {texts_from_elastic}')

            # Добавляем результаты в processed_data
            processed_data["data"]["similar_texts"] = texts_from_elastic[:50]
            # processed_data["similar_texts"] = texts_from_elastic[:50]
            
            # Дополнительная проверка соответствия ID
            # found_ids = [text["elastic_id"] for text in [str(x) for x in texts_from_elastic]]
            found_ids = [text["elastic_id"] for text in texts_from_elastic]
            missing_ids = set(elastic_ids) - set(found_ids)
            
            if missing_ids:
                print(f"Не найдены документы с ID: {missing_ids}")
            
        else:
            # Если current_tab не соответствует ни одному из условий, возвращаем исходные данные
            return data
        
        print(888999)
        print(f'processed_data: {processed_data}')
        return processed_data
    except Exception as e:
        print(f"Ошибка при обработке данных: {e}")
        return data

# Подключение к Qdrant (если используется локальный сервер)
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny, MatchValue
qdrant_client = QdrantClient(
    url="http://localhost:6333",
    timeout=300,  # 5 минут вместо стандартных 60 секунд
    prefer_grpc=False
)

@app.post("/ai-question-information-graph", tags=['data analytics'])
async def ai_question_information_graph(request: Request):
    from mlops.dashboard_qa import handle_question_information
    return await handle_question_information(request)
    # Получаем тело запроса в виде байтов
    body_bytes = await request.body()
    
    # Пытаемся преобразовать в JSON
    body_json = json.loads(body_bytes)

    # Извлекаем нужные данные
    question = body_json.get('question', '')
    data = body_json.get('data', {})
    filters = body_json.get('filters', {})
    
    # Формируем структурированное сообщение для LLM
    system_prompt = _dashboard_prompt("dashboard_qa_graph", "dashboard_qa_graph_v1")
    
    # Обработка данных
    filtered_data = data.get('values', [])
    
    # Создаем человекочитаемое представление фильтров
    filter_description = f"""
    Применены следующие фильтры:
    - Размер аудитории: от {filters.get('audienceRange', [0, 0])[0]} до {filters.get('audienceRange', [0, 0])[1]}
    - Количество репостов: от {filters.get('repostsRange', [0, 0])[0]} до {filters.get('repostsRange', [0, 0])[1]}
    - Показатель вовлеченности (ER): от {filters.get('erRange', [0, 0])[0]} до {filters.get('erRange', [0, 0])[1]}
    - Количество просмотров: от {filters.get('viewsCountRange', [0, 0])[0]} до {filters.get('viewsCountRange', [0, 0])[1]}
    
    Общее количество сообщений: {data.get('num_messages', 0)}
    Количество уникальных авторов: {data.get('num_unique_authors', 0)}
    """
    
    # Добавляем базовую статистику
    platforms = {}
    author_types = {}
    sexes = {}
    
    for item in filtered_data:
        author = item.get('author', {})
        hub = author.get('hub', 'unknown')
        author_type = author.get('author_type', 'unknown')
        sex = author.get('sex', 'unknown')
        
        platforms[hub] = platforms.get(hub, 0) + 1
        author_types[author_type] = author_types.get(author_type, 0) + 1
        sexes[sex] = sexes.get(sex, 0) + 1
    
    stats = f"""
    Базовая статистика по данным:
    
    Распределение по платформам:
    {', '.join([f"{platform}: {count}" for platform, count in platforms.items()])}
    
    Распределение по типам авторов:
    {', '.join([f"{author_type}: {count}" for author_type, count in author_types.items()])}
    
    Распределение по полу (если известно):
    {', '.join([f"{sex}: {count}" for sex, count in sexes.items()])}
    """
    
    # Ограничиваем количество записей для отправки в LLM
    max_items = 50  # Ограничиваем количество записей для экономии токенов
    data_sample = filtered_data[:max_items]
    
    # Формируем структурированный user_message
    user_message = f"""
    Запрос пользователя: {question}
    
    {filter_description}
    
    {stats}
    
    Данные для анализа (первые {min(max_items, len(filtered_data))} из {len(filtered_data)} записей):
    {json.dumps(data_sample, ensure_ascii=False, indent=2)}
    
    Пожалуйста, проведи анализ на основе этих данных и ответь на вопрос пользователя.
    """
    
    # если применен поиск по текстам
    def collect_unique_es_ids(data):
        """
        Собирает уникальные es_id из данных.
        """
        es_ids_set = set()
        
        # Получаем список значений из data -> values
        values = data.get('data', {}).get('values', [])

        for item in values:
            # Извлекаем elastic_id или es_id из автора
            if isinstance(item.get('author'), dict):
                if 'elastic_id' in item['author']:
                    es_ids_set.add(item['author']['elastic_id'])
                elif 'es_id' in item['author']:
                    es_ids_set.add(item['author']['es_id'])

            # Извлекаем elastic_id или es_id из репостов
            if isinstance(item.get('reposts'), list):
                for repost in item['reposts']:
                    if isinstance(repost, dict):
                        if 'elastic_id' in repost:
                            es_ids_set.add(repost['elastic_id'])
                        elif 'es_id' in repost:
                            es_ids_set.add(repost['es_id'])

        return list(es_ids_set)
    
    semantic_texts_output = ""  # для информации о найденных релевантных текстах

    if body_json.get('searchInTexts', False):

        es_ids = collect_unique_es_ids(body_json)
        
        # Получаем embedding для вопроса пользователя
        question = body_json.get("question", "")

        embeddings = client.embeddings.create(
            input=question,
            model="text-embedding-3-small"
        )
        embedding = embeddings.data[0].embedding
        
        # Получаем имя коллекции/индекса
        indexes = load_dict_from_pickle('/home/dev/tellscope_app/tellscope_backend/data/indexes.pkl')
        collection_name = indexes.get(body_json.get("index", 0), "")
        
        # Ищем близкие векторы в Qdrant, используя фильтр по es_ids
        search_result = []
        if es_ids and collection_name:
            try:
                filter = {
                    "must": [
                        {
                            "key": "metadata._id",
                            "match": { "any": es_ids }
                        }
                    ]
                }
                search_result = qdrant_client.search(
                    collection_name=collection_name,
                    query_vector=embedding,
                    limit=50,
                    with_payload=True,
                    query_filter=filter
                )
                payloads = [point.payload for point in search_result]
            except Exception as e:
                print(f"Ошибка при запросе к Qdrant: {e}")
                print(f"Тип ошибки: {type(e)}")
        
        # Извлекаем elastic_ids из результатов поиска в Qdrant
        elastic_ids = [point.payload.get("_id") for point in search_result if "_id" in point.payload]
        
        # Получаем тексты из Elasticsearch
        elastic_ids = [str(x) for x in elastic_ids]
        es_ids = [str(x) for x in es_ids]

        texts_from_elastic = []
        if collection_name:
            query = {
                "size": len(es_ids),  # чтобы получить все совпадения
                "query": {
                    "ids": {
                        "values": es_ids
                    }
                }
            }
            response = es.search(
                index=collection_name,
                body=query
            )
            for hit in response.get('hits', {}).get('hits', []):
                source = hit.get('_source', {})
                texts_from_elastic.append({
                    "text": source.get("text", ""),
                    "title": source.get("title", ""),
                    "source": {
                        "hub": source.get("hub", ""),
                        "url": source.get("url", "")
                    },
                    "author": source.get("authorObject", {}),
                })

        texts_from_elastic = texts_from_elastic[:10]
        # Формируем расширенный user_message, включающий найденные тексты
        if texts_from_elastic:
            semantic_texts_output += f"""## Семантически релевантные тексты
            
    Ниже представлены наиболее релевантные сообщения, найденные по смысловой близости к запросу пользователя. Используй содержимое этих сообщений для обоснования своих выводов и для поиска дополнительных инсайтов:
    """
            for i, text_item in enumerate(texts_from_elastic, 1):
                text_preview = text_item.get("text", "")[:300] + "..." if len(text_item.get("text", "")) > 300 else text_item.get("text", "")
                semantic_texts_output += f"""
    ### Документ {i}
    - **Источник**: {text_item.get("source", {}).get("hub", "неизвестно")}
    - **Заголовок**: {text_item.get("title", "без заголовка")}
    - **Текст**: {text_preview}
    - **URL**: {text_item.get("source", {}).get("url", "")}
    """
            semantic_texts_output += """
    ---

    **В анализе обязательно учитывай все приведённые выше тексты! Обобщай их содержание, выделяй однотипные мнения, противоречия и необычные находки.**
    """

    # Формируем итоговое задание для LLM — всегда включаем фильтры, статистику и sample данных, а также найденные тексты (если они есть).
    user_message = f"""
    Запрос пользователя: {question}

    {filter_description}

    {stats}

    Данные для анализа (первые {min(max_items, len(filtered_data))} из {len(filtered_data)} записей):
    {json.dumps(data_sample, ensure_ascii=False, indent=2)}

    {semantic_texts_output}

    Проведи анализ на основе всех представленных выше материалов. Если приведены семантически релевантные тексты — обязательно используй их в вычислениях и рассуждениях.
    """

    print('=====================user_message==========================')
    # print(f'user_message: {user_message}')

    # Отправляем запрос к LLM
    try:
        chat_result = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            model=ai_model,
            max_tokens=10000,
        )
        
        # Исправленная строка - обращаемся к content как к строке, а не как к словарю
        return chat_result.choices[0].message
    except Exception as e:
        return {"error": str(e)}



@app.post("/ai-question-media-rating", tags=['data analytics'])
async def ai_question_media_rating(request: Request):
    from mlops.dashboard_qa import handle_question_media
    return await handle_question_media(request)
    # Получаем тело запроса в виде байтов
    body_bytes = await request.body()
    
    # Пытаемся преобразовать в JSON
    body_json = json.loads(body_bytes)

    # Извлекаем нужные данные
    question = body_json.get('question', '')
    first_graph = body_json.get('data', {}).get('first_graph', {})
    second_graph = body_json.get('data', {}).get('second_graph', {})
    filters = body_json.get('filters', {})

    # Формируем структурированное сообщение для LLM
    system_prompt = _dashboard_prompt("dashboard_qa_media", "dashboard_qa_media_v1")

    # Формируем user_message с анализом данных из графов
    first_graph_stats = f"""## Статистика по негативным и позитивным упоминаниям
    **Негативные ссылки:**
    - {', '.join([f"{item['name']} (индекс: {item['index']}, сообщения: {item['message_count']})" for item in first_graph.get('negative_smi', [])])}

    **Позитивные ссылки:**
    - {', '.join([f"{item['name']} (индекс: {item['index']}, сообщения: {item['message_count']})" for item in first_graph.get('positive_smi', [])])}
    """
    
    # Обработка данных из второго графа
    second_graph_links = "\n".join([f"- [{item['name']}]({item['url']})" for item in second_graph])

    second_graph_summary = f"""## Упоминания в источниках
    Выше представлены ссылки на ресурсы:
    {second_graph_links}
    """

    # Составляем финальное сообщение для модели
    user_message = f"""
    Запрос пользователя: {question}

    {first_graph_stats}

    {second_graph_summary}

    Пожалуйста, проведи анализ на основе всех представленных выше материалов. 
    """

    print('=====================user_message==========================')
    print(f'user_message: {user_message}')

    # Отправляем запрос к LLM
    try:
        chat_result = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            # Добавьте необходимые параметры для модели
            model=ai_model,
            max_tokens=1000,
        )
        
        # Возвращаем ответ LLM, корректно обращаясь к атрибуту объекта
        return chat_result.choices[0].message.content
    except Exception as e:
        return {"error": str(e)}

# Используйте это:
from fastapi.responses import PlainTextResponse

def format_percent(n, total):
    if total == 0:
        return "0% (0)"
    percent = round(100 * n / total, 1)
    return f"{percent}% ({n})"

@app.post("/ai-question-voice", tags=['data analytics'])
async def ai_question_voice(request: Request):
    try:
        body_bytes = await request.body()
        body_json = json.loads(body_bytes)

        print(body_json)
        print(len(body_json['data']['values'][0]['tonality']))

        question = body_json.get('question', '')
        values = body_json.get('data', {}).get('values', [{}])[0]
        tonality_data = values.get('tonality', [])
        sunkey_data = values.get('sunkey_data', [])
        index = body_json.get('index', 0)
        min_date = body_json.get('min_date', 0)
        max_date = body_json.get('max_date', 0)
        current_tab = body_json.get('current_tab', '')

        period = f"{datetime.fromtimestamp(min_date).strftime('%d.%m.%Y')} — {datetime.fromtimestamp(max_date).strftime('%d.%m.%Y')}"

        # ======= Markdown по аналитике (оставляем как было) =======
        table_md = "| Источник | Всего | Негатив (%) | Нейтрал (%) | Позитив (%) |\n"
        table_md += "|:---------|------:|------------:|------------:|------------:|\n"

        sources_sorted = sorted(tonality_data, key=lambda x: -sum([x.get('Нейтрал',0), x.get('Позитив',0), x.get('Негатив',0)]))
        for source in sources_sorted:
            name = source['source']
            n_neg = source.get('Негатив', 0)
            n_neu = source.get('Нейтрал', 0)
            n_pos = source.get('Позитив', 0)
            total = n_neg + n_neu + n_pos
            table_md += f"| {name} | {total} | {format_percent(n_neg, total)} | {format_percent(n_neu, total)} | {format_percent(n_pos, total)} |\n"

        # ======= Основные источники =======
        main_sources = sources_sorted[:5]
        sum_main = sum(sum([src.get('Нейтрал',0), src.get('Позитив',0), src.get('Негатив',0)]) for src in main_sources)
        sum_total = sum(sum([src.get('Нейтрал',0), src.get('Позитив',0), src.get('Негатив',0)]) for src in sources_sorted)
        others = sum_total - sum_main

        top_sources_md = "| Источник | Кол-во сообщений | Доля |\n|:---------|-----------------:|------:|\n"
        for src in main_sources:
            name = src['source']
            cnt = sum([src.get('Нейтрал',0), src.get('Позитив',0), src.get('Негатив',0)])
            share = f"{round(cnt/sum_total*100, 1)}%"
            top_sources_md += f"| {name} | {cnt} | {share} |\n"
        if others:
            top_sources_md += f"| Остальные | {others} | {round(others/sum_total*100, 1)}% |\n"

        # ======= Вовлеченность =======
        engagement_by_hub = {}
        for post in sunkey_data:
            hub = post['hub']
            if hub not in engagement_by_hub:
                engagement_by_hub[hub] = {
                    'posts': 0,
                    'comments': 0,
                    'audience': 0,
                    'engagement': 0,
                }
            engagement_by_hub[hub]['posts'] += 1
            engagement_by_hub[hub]['comments'] += post.get('commentsCount', 0)
            engagement_by_hub[hub]['audience'] += post.get('audienceCount', 0)
            engagement_by_hub[hub]['engagement'] += post.get('commentsCount', 0) + post.get('repostsCount', 0)

        if engagement_by_hub:
            engagement_md = "| Источник | Посты | Комменты | Аудитория | Вовлеченность |\n"
            engagement_md += "|:---------|------:|---------:|----------:|--------------:|\n"
            for hub, stats in sorted(engagement_by_hub.items(), key=lambda x: -x[1]['audience']):
                engagement_md += f"| {hub} | {stats['posts']} | {stats['comments']} | {stats['audience']:,} | {stats['engagement']} |\n"
        else:
            engagement_md = "> Нет данных по вовлечённости."

        # ======== Новый блок: поиск по текстам, если активирован ========
        semantic_texts_output = ""  # Markdown с релевантными текстами

        def collect_elastic_ids_by_tab(data, current_tab):
            es_ids_set = set()
            values = data.get('data', {}).get('values', [])
            
            if current_tab == 'sources':
                # Собираем elastic_id из tonality
                for item in values:
                    tonality = item.get('tonality', [])
                    for t in tonality:
                        elastic_ids = t.get('elastic_id')
                        if isinstance(elastic_ids, list):
                            es_ids_set.update(elastic_ids)
                        elif isinstance(elastic_ids, str):
                            es_ids_set.add(elastic_ids)
            elif current_tab == 'mention_types':
                # Собираем elastic_id из sunkey_data
                for item in values:
                    sunkey_data = item.get('sunkey_data', [])
                    for s in sunkey_data:
                        elastic_ids = s.get('elastic_id')
                        if isinstance(elastic_ids, list):
                            es_ids_set.update(elastic_ids)
                        elif isinstance(elastic_ids, str):
                            es_ids_set.add(elastic_ids)

            return list(es_ids_set)

        # if body_json.get('searchInTexts', False):
        es_ids = collect_elastic_ids_by_tab(body_json, body_json.get("current_tab", ""))

        question = body_json.get("question", "") 

        embeddings = client.embeddings.create(
            input=question,
            model="text-embedding-3-small"
        )
        embedding = embeddings.data[0].embedding

        indexes = load_dict_from_pickle('/home/dev/tellscope_app/tellscope_backend/data/indexes.pkl')
        collection_name = indexes.get(body_json.get("index", 0), "")

        search_result = []
        if es_ids and collection_name:
            try:
                filter = {
                    "must": [
                        {
                            "key": "metadata._id",  # Правильный путь к полю
                            "match": {"any": es_ids}
                        }
                    ]
                }
                search_result = qdrant_client.search(
                    collection_name=collection_name,
                    query_vector=embedding,
                    limit=50,
                    with_payload=True,
                    query_filter=filter
                )
                # payloads = [point.payload for point in search_result]
            except Exception as e:
                print(f"Ошибка при запросе к Qdrant: {e}")
                print(f"Тип ошибки: {type(e)}")


        # Извлекаем ids из Qdrant
        # elastic_ids = [point.payload.get("_id") for point in search_result if "_id" in point.payload]
        # Для извлечения результатов
        elastic_ids = [point.payload["metadata"]["_id"] for point in search_result if "metadata" in point.payload and "_id" in point.payload["metadata"]]
        elastic_ids = [str(x) for x in elastic_ids]
        es_ids = [str(x) for x in es_ids]


        texts_from_elastic = []
        if collection_name and elastic_ids:
            query = {
                "size": len(elastic_ids),
                "query": {
                    "ids": {
                        "values": elastic_ids
                    }
                }
            }
            response = es.search(
                index=collection_name,
                body=query
            )
            for hit in response.get('hits', {}).get('hits', []):
                source = hit.get('_source', {})
                texts_from_elastic.append({
                    "text": source.get("text", ""),
                    "title": source.get("title", ""),
                    "source": {
                        "hub": source.get("hub", ""),
                        "url": source.get("url", "")
                    },
                    "author": source.get("authorObject", {}),
                })

        texts_from_elastic = texts_from_elastic[:10]

        if texts_from_elastic:
            semantic_texts_output += f"""## Семантически релевантные тексты

Ниже представлены наиболее релевантные сообщения, найденные по смысловой близости к запросу пользователя. Используй содержимое этих сообщений для обоснования своих выводов и для поиска дополнительных инсайтов:
"""
            for i, text_item in enumerate(texts_from_elastic, 1):
                text_preview = text_item.get("text", "")
                if len(text_preview) > 300:
                    text_preview = text_preview[:300] + "..."
                semantic_texts_output += f"""
### Документ {i}
- **Источник**: {text_item.get("source", {}).get("hub", "неизвестно")}
- **Заголовок**: {text_item.get("title", "без заголовка")}
- **Текст**: {text_preview}
- **URL**: {text_item.get("source", {}).get("url", "")}
"""
            semantic_texts_output += """
---

**В анализе обязательно учитывай все приведённые выше тексты! Обобщай их содержание, выделяй однотипные мнения, противоречия и необычные находки.**
"""

        # ======= Формируем финальный prompt для LLM =======
        user_message = f"""\
        **Запрос:** {question}

        ### 📅 Период анализа: {period}
        ### 📈 Индекс активности: **{index}**

        ---

        ## 🟦 Распределение тональности по источникам
        {table_md}

        ---

        ## 🔑 Ключевые источники обсуждений
        {top_sources_md}

        ---

        ## 💬 Активные платформы и вовлечённость
        {engagement_md}

        {semantic_texts_output}

        ---

        **Проведи детальный анализ данных:** выдели неожиданные инсайты, неочевидные тренды, дай рекомендации по работе с негативом. 
        Если представлены релевантные тексты — обязательно используй их при анализе и выводах.
        """

        system_prompt = _dashboard_prompt("dashboard_qa_voice", "dashboard_qa_voice_v1")

        chat_result = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            model=ai_model,
            max_tokens=2000,
            temperature=0.7
        )

        return chat_result.choices[0].message.content

    except json.JSONDecodeError:
        return {"status": "error", "message": "Invalid JSON format"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    
import math
import hashlib

def medialogia_record_to_export(row, idx):
    def safe_int(v, default=0):
        try:
            if pd.isna(v): return default
            v = str(v).replace(' ', '').replace(',', '.')
            return int(float(v))
        except: return default

    def safe_float(v, default=0.0):
        try:
            if pd.isna(v): return default
            v = str(v).replace(' ', '').replace(',', '.')
            return float(v)
        except: return default

    def safe_str(v):
        if v is None or v != v or (isinstance(v, float) and math.isnan(v)): return ""
        return str(v).strip()

    def unix_time(dt):
        import pandas as pd
        import math
        if isinstance(dt, int):
            return dt
        if dt is None or dt == '' or (isinstance(dt, float) and math.isnan(dt)) or pd.isna(dt):
            return 0
        if isinstance(dt, float):
            # Возможно это unix timestamp, например, 1714752000.0
            if math.isnan(dt):
                return 0
            return int(dt)
        if isinstance(dt, str):
            try:
                val = dt.strip()
                if len(val) <= 10:  # например, "01.01.2022"
                    dtt = pd.to_datetime(val, format='%d.%m.%Y', errors='coerce')
                else:
                    dtt = pd.to_datetime(val, errors='coerce')
                # Если конвертировалось успешно — рекурсивно обработаем
                return unix_time(dtt)
            except Exception:
                return 0
        # Если это datetime
        if hasattr(dt, 'timestamp'):
            return int(dt.timestamp())
        # Неизвестный тип
        return 0

    idExternal = safe_str(row.get("idExternal") or row.get("url") or row.get("URL") or row.get("№") or row.get("_id") or idx + 1)
    timeCreate = unix_time(row.get("timeCreate") or row.get("Дата") or row.get("date") or row.get('time') or row.get('Время публикации'))
    hash_str = f"{idExternal}{timeCreate}"
    hash_id = hashlib.md5(hash_str.encode('utf-8')).hexdigest() + "20250505"

    authorObject = {
        "fullname": safe_str(row.get("author") or row.get("authorFullName") or row.get("Автор")) or row.get("Кто пишет"),
        "url": safe_str(row.get("author_url") or row.get("Ссылка на автора")),
        "author_type": safe_str(row.get("author_type") or row.get("Тип автора")),
        "sex": safe_str(row.get("author_sex") or row.get("Пол")),
        "age": safe_str(row.get("author_age") or row.get("Возраст")),
    }
    authorObject = {k: v for k, v in authorObject.items() if v}

    tone_map = {
        'негативная': -1,
        'нейтральная': 0,
        'позитивная': 1
    }

    tone_label = row.get('Тональность')
    toneMark = row.get("toneMark")
    if toneMark is not None and toneMark != "":
        toneMark_val = safe_int(toneMark)
    elif tone_label is not None and tone_label.lower() in tone_map:
        toneMark_val = safe_int(tone_map[tone_label.lower()])
    else:
        toneMark_val = 0

    return {
        "id": idx + 1,
        "hash": hash_id,
        "idExternal": idExternal,
        "timeCreate": timeCreate,
        "title": safe_str(row.get("title") or row.get("Заголовок")),
        "text": safe_str(row.get("text") or row.get("Текст сообщения") or row.get("Заголовок")),
        "hub": safe_str(row.get("hub") or row.get("СМИ") or row.get("platform")),
        "url": safe_str(row.get("url") or row.get('URL статьи') or row.get('Ссылка на сообщение')),
        "hubtype": safe_str(row.get("hubtype") or row.get("Тип площадки") or row.get("Тип")),
        "type": safe_str(row.get("type") or row.get("Тип")),
        "authorObject": authorObject,
        "commentsCount": safe_int(row.get("commentsCount") or row.get("комментарии") or row.get("comments") or row.get("Комментарии")),
        "audienceCount": safe_int(row.get("audienceCount") or row.get("Аудитория блога") or row.get("Аудитория автора") or row.get("Охват (из открытых источников)")),
        "citeIndex": safe_str(row.get("citeIndex") or row.get('СМ Индекс')),
        "repostsCount": safe_int(row.get("repostsCount") or row.get("shares") or row.get("Репосты")),
        "likesCount": safe_int(row.get("likesCount") or row.get("likes") or row.get("лайки")),
        "er": safe_float(row.get("er") or row.get("engagement") or row.get("Вовлеченность")),
        "viewsCount": safe_int(row.get("viewsCount") or row.get("Просмотры")),
        "review_rating": safe_str(row.get("review_rating") or row.get('Оценка от 1 до 5')),
        "duplicateCount": safe_int(row.get("duplicateCount"), 1),
        "massMediaAudience": safe_int(row.get("massMediaAudience")),
        "toneMark": toneMark_val,
        "role": safe_str(row.get("role")),
        "aggression": safe_str(row.get("aggression")),
        "country": safe_str(row.get("country") or row.get("Страна")),
        "region": safe_str(row.get("region") or row.get("Регион")),
        "city": safe_str(row.get("city") or row.get("Город") or row.get("Город")),
        "language": safe_str(row.get("language") or "Русский"),
        "aspects": [],
        "wom": safe_str(row.get("wom") or row.get('WOM')),
        "processed": safe_str(row.get("processed") or "Нет"),
        "story": safe_str(row.get("story")),
        "geoObject": [],
    }

def load_medialogia_excel(file_path):
    try:
        df = pd.read_excel(file_path, keep_default_na=False)
    except Exception as e:
        print(f"Ошибка при чтении Excel файла: {e}")
        return []

    df = df.replace(['-', '—', '–', 'nan', 'NaN', '', ' '], None, regex=False)
    df = df.where(pd.notna(df), None)

    # Поиск строки с заголовками
    header_row = None
    for i in range(min(20, df.shape[0])):
        vals = [str(val).strip().lower() for val in df.iloc[i].tolist() if pd.notna(val) and str(val).strip()]
        if 'заголовок' in vals and 'дата' in vals:
            header_row = i
            break
        if "время публикации" in ' '.join(vals) or "площадка" in ' '.join(vals):
            header_row = i
            break
    if header_row is not None:
        headers = df.iloc[header_row].apply(lambda x: str(x).strip() if pd.notna(x) else '')
        df = df.iloc[header_row + 1:].reset_index(drop=True)
        df.columns = headers
        df = df.replace(['-', '—', '–', 'nan', 'NaN', '', ' '], None, regex=False)
        df = df.where(pd.notna(df), None)
    else:
        # Первая строка — заголовки
        df.columns = [str(x).strip() for x in df.iloc[0]]
        df = df.iloc[1:].reset_index(drop=True)

    # ==== Маппим колонки по двум логикам ====
    manual_map = {
      '№': '_id',
      'Заголовок': 'text',         # Мэпим на text, а не title
      'Дата': 'timeCreate',
      'СМИ': 'hub',
      'Город': 'city',
      'Охват (из открытых источников)': 'audienceCount',
      'URL статьи': 'url'
    }

    # Автоматическая эвристика
    auto_map = {}
    for col in df.columns:
        if not col: continue
        col_lower = str(col).lower()
        if 'время публикации' in col_lower:
            auto_map[col] = 'time'
        elif 'площадка' in col_lower:
            auto_map[col] = 'platform'
        elif any(word in col_lower for word in ['охват', 'просмотры', 'views']):
            auto_map[col] = 'views'
        elif any(word in col_lower for word in ['вовлеченность', 'engagement']):
            auto_map[col] = 'engagement'
        elif any(word in col_lower for word in ['лайки', 'likes']):
            auto_map[col] = 'likes'
        elif any(word in col_lower for word in ['комментарии', 'comments']):
            auto_map[col] = 'comments'
        elif any(word in col_lower for word in ['репосты', 'shares']):
            auto_map[col] = 'shares'

    # Объединяем оба маппинга, приоритет у ручного
    column_mapping = {}
    column_mapping.update(auto_map)
    column_mapping.update(manual_map)
    # manual_map должен "переехать" в column_mapping поверх автогенерации
    # (т.е. если ключ из manual_map встречается — он заменяет автоматическую).

    # Ренеймим колонки по column_mapping
    for k, v in column_mapping.items():
        if k in df.columns: df = df.rename(columns={k: v})

    # Приводим timeCreate к unixtime, если колонка есть
    if 'timeCreate' in df.columns:
        df['timeCreate'] = pd.to_datetime(df['timeCreate'], errors='coerce')
        df['timeCreate'] = df['timeCreate'].apply(lambda x: int(x.timestamp()) if pd.notna(x) else None)
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        df['time'] = df['time'].apply(lambda x: int(x.timestamp()) if pd.notna(x) else None)

    records = df.to_dict('records')
    result = [medialogia_record_to_export(rec, idx) for idx, rec in enumerate(records) if any(rec.values())]
    return result

from fastapi.responses import StreamingResponse
import pandas as pd
import traceback
from io import BytesIO
from transliterate import translit, detect_language

@app.post("/convert-file-mlg")
async def convert_file_mlg(file: UploadFile = File(...)):
    try:
        def safe_filename(filename):
            # Транслитерация кириллицы (если строка на русском)
            if detect_language(filename) == 'ru':
                filename = translit(filename, 'ru', reversed=True)
            # Удаляем все запрещенные символы Elasticsearch
            filename = re.sub(r'[/,|><?*" \\]', '_', filename)
            # Заменяем пробелы на подчеркивания
            filename = filename.replace(' ', '_')
            # Приводим к нижнему регистру
            filename = filename.lower()
            # Удаляем возможные двойные подчеркивания
            filename = re.sub(r'_+', '_', filename)
            # Удаляем подчеркивания в начале и конце
            filename = filename.strip('_')
            return filename

        # Применяем safe_filename ко всему имени файла (включая расширение)
        original_filename = file.filename
        safe_name = safe_filename(original_filename.replace('.xlsx', ''))
        safe_output_filename = f"converted_{safe_name}.json"

        contents = await file.read()
        temp_file_path = f"/home/dev/tellscope_app/tellscope_backend/data/temp/{original_filename}"

        with open(temp_file_path, "wb") as f:
            f.write(contents)

        result = load_medialogia_excel(temp_file_path)
        if result == 'error':
            return JSONResponse(status_code=400, content={'error': 'Не удалось обработать файл'})

        json_result = json.dumps(result, ensure_ascii=False, indent=2)
        json_bytes = BytesIO(json_result.encode('utf-8'))
        
        response = StreamingResponse(
            json_bytes,
            media_type="application/json",
            headers={
                "Content-Disposition": f"attachment; filename={safe_output_filename}"
            }
        )
        return response
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={'error': str(e)})


@app.post("/ai-question-analysis", tags=['data analytics'])
async def ai_question_analysis(request: Request):
    from mlops.ai_bot_rag import handle_question_analysis
    return await handle_question_analysis(
        request,
        es=es,
        qdrant_client=qdrant_client,
        model_manager=model_manager,
        models=models,
        client=client,
        ai_model=ai_model,
        logger=logger,
        load_indexes=lambda: load_dict_from_pickle('/home/dev/tellscope_app/tellscope_backend/data/indexes.pkl'),
        system_prompt=_dashboard_prompt("dashboard_qa_bot", "dashboard_qa_bot_v1"),
    )


@app.post("/ai-bot/corpus-summary", tags=['data analytics'])
async def ai_bot_corpus_summary(request: Request):
    from mlops.ai_bot_rag import handle_corpus_summary
    return await handle_corpus_summary(
        request,
        es=es,
        load_indexes=lambda: load_dict_from_pickle('/home/dev/tellscope_app/tellscope_backend/data/indexes.pkl'),
        logger=logger,
    )


@app.post("/ai-bot/deep-brief", tags=['data analytics'])
async def ai_bot_deep_brief(request: Request):
    from mlops.ai_bot_rag import handle_deep_brief
    return await handle_deep_brief(
        request,
        es=es,
        load_indexes=lambda: load_dict_from_pickle('/home/dev/tellscope_app/tellscope_backend/data/indexes.pkl'),
        logger=logger,
    )


@app.get("/ai-bot/deep-brief", tags=['data analytics'])
async def ai_bot_deep_brief_status(request: Request):
    from mlops.ai_bot_rag import handle_deep_brief_status
    return await handle_deep_brief_status(request)

# Инициализация клиента через mlops.gateway (ключи только из .env)
client = GatewayChatClient(provider="aitunnel", profile="dashboard_qa")

# Модель запроса
class ChatRequest(BaseModel):
    message: str
    model: Optional[str] = "deepseek-chat-v3.1"
    max_tokens: Optional[int] = 50000

# Модель ответа
class ChatResponse(BaseModel):
    response: str
    model: str

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        chat_result = client.chat.completions.create(
            messages=[{"role": "user", "content": request.message}],
            model=request.model,
            max_tokens=request.max_tokens,
        )
        
        return ChatResponse(
            response=chat_result.choices[0].message.content,
            model=request.model
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def get_models():
    """Возвращает список доступных моделей"""
    return {
        "models": [
            "deepseek-chat-v3.1",
            "gpt-5.1-chat",
            "gpt-4o-mini",
            "claude-sonnet-4.5", 
            "gpt-5.1-codex-max",
            "gemini-2.5-pro",
            "claude-sonnet-4.6",
        ]
    }

@app.get("/test-collection/{collection_name}")
async def test_collection(collection_name: str):
    """Диагностика коллекции"""
    try:
        # 1. Инфо о коллекции
        info = qdrant_client.get_collection(collection_name)
        
        # 2. Sample векторов
        sample = qdrant_client.scroll(
            collection_name=collection_name,
            limit=100,
            with_vectors=True
        )[0]
        
        norms = [float(np.linalg.norm(p.vector)) for p in sample]
        
        # 3. Тестовый поиск
        test_vector = sample[0].vector
        test_search = qdrant_client.search(
            collection_name=collection_name,
            query_vector=test_vector,
            limit=5
        )
        
        return {
            "collection": collection_name,
            "points_count": int(info.points_count),
            "vector_size": int(info.config.params.vectors.size),
            "distance": str(info.config.params.vectors.distance),
            "hnsw_m": int(info.config.hnsw_config.m),
            "hnsw_ef_construct": int(info.config.hnsw_config.ef_construct),
            "sample_vectors": {
                "count": len(sample),
                "avg_norm": float(np.mean(norms)),
                "std_norm": float(np.std(norms)),
                "min_norm": float(min(norms)),
                "max_norm": float(max(norms)),
                "normalized": bool(abs(np.mean(norms) - 1.0) < 0.05)  # Преобразуем в bool
            },
            "test_search": {
                "found": len(test_search),
                "best_score": float(test_search[0].score) if test_search else 0.0
            }
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/qdrant/collections", tags=['qdrant'])
async def get_qdrant_collections():
    """Получить список всех коллекций из Qdrant"""
    try:
        collections = qdrant_client.get_collections()
        
        collection_list = []
        for collection in collections.collections:
            try:
                collection_info = qdrant_client.get_collection(collection.name)
                collection_list.append({
                    "name": collection.name,
                    "points_count": collection_info.points_count,
                    "vector_size": collection_info.config.params.vectors.size,
                })
            except Exception as e:
                logger.warning(f"Не удалось получить информацию о коллекции {collection.name}: {e}")
                collection_list.append({
                    "name": collection.name,
                    "points_count": 0,
                    "vector_size": 0,
                })
        
        return JSONResponse(
            status_code=200,
            content={"collections": collection_list}
        )
    except Exception as e:
        logger.error(f"Ошибка получения коллекций: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.delete("/qdrant/collections/{collection_name}", tags=['qdrant'])
async def delete_qdrant_collection(collection_name: str):
    """Удалить коллекцию из Qdrant"""
    try:
        # Проверяем существование коллекции
        collections = qdrant_client.get_collections()
        collection_exists = any(c.name == collection_name for c in collections.collections)
        
        if not collection_exists:
            return JSONResponse(
                status_code=404,
                content={"error": f"Коллекция {collection_name} не найдена"}
            )
        
        # Удаляем коллекцию
        qdrant_client.delete_collection(collection_name=collection_name)
        
        # Также удаляем из Elasticsearch (если используется)
        try:
            if es.indices.exists(index=collection_name):
                es.indices.delete(index=collection_name)
                logger.info(f"Удален индекс Elasticsearch: {collection_name}")
        except Exception as es_error:
            logger.warning(f"Ошибка удаления индекса Elasticsearch: {es_error}")
        
        # Обновляем файл indexes.pkl
        try:
            indexes = load_dict_from_pickle('/home/dev/tellscope_app/tellscope_backend/data/indexes.pkl')
            # Удаляем коллекцию из словаря
            indexes_to_remove = [key for key, value in indexes.items() if value == collection_name]
            for key in indexes_to_remove:
                del indexes[key]
            
            # Сохраняем обновленный словарь
            save_dict_to_pickle(indexes, '/home/dev/tellscope_app/tellscope_backend/data/indexes.pkl')
            logger.info(f"Обновлен файл indexes.pkl")
        except Exception as pickle_error:
            logger.warning(f"Ошибка обновления indexes.pkl: {pickle_error}")
        
        logger.info(f"✅ Успешно удалена коллекция: {collection_name}")
        
        return JSONResponse(
            status_code=200,
            content={"message": f"Коллекция {collection_name} успешно удалена"}
        )
    except Exception as e:
        logger.error(f"Ошибка удаления коллекции: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/csv-files")
async def list_csv_files():
    """Список доступных CSV файлов"""
    base_dir = "/home/dev/tellscope_app/tellscope_backend/data/1/bertopic_files_directory/test"
    
    try:
        files = []
        for root, dirs, filenames in os.walk(base_dir):
            for filename in filenames:
                if filename.endswith('.csv') and 'result_graph' in filename:
                    full_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(full_path, base_dir)
                    
                    # Получаем размер файла
                    file_size = os.path.getsize(full_path)
                    
                    files.append({
                        'name': filename,
                        'path': full_path,
                        'relative_path': rel_path,
                        'size': file_size,
                        'size_mb': round(file_size / 1024 / 1024, 2)
                    })
        
        # Сортируем по дате модификации
        files.sort(key=lambda x: os.path.getmtime(x['path']), reverse=True)
        
        return {
            'files': files,
            'default': DEFAULT_CSV_PATH,
            'total': len(files)
        }
    except Exception as e:
        raise HTTPException(500, str(e))

import networkx as nx

# Дефолтный путь к CSV
DEFAULT_CSV_PATH = "/home/dev/tellscope_app/tellscope_backend/data/1/bertopic_files_directory/test/platon_31.10.2025-30.11.2025/result_graph_Beyond_Taylor_25.10.2025-24.11.2025.csv"

class GraphQuery(BaseModel):
    question: str
    graph_data: Any  # Принимаем любую структуру

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from collections import Counter
import re

class TopicAnalyzer:
    """Анализатор для группировки тематик по ключевым фразам"""
    
    def __init__(self, topics: list):
        self.topics = topics
        self.topic_groups = {}
        self.key_phrases = []
    
    # ✅ ИСПРАВЛЕНО: добавлен параметр min_word_length
    def extract_key_phrases(self, min_words=3, min_word_length=4, top_n=20):
        """Извлечение ключевых фраз из тематик (минимум 3 слова)
        
        Args:
            min_words: минимальное количество слов во фразе
            min_word_length: минимальная длина хотя бы одного слова
            top_n: количество топ-фраз для возврата
        """
        # Объединяем все тематики в текст
        all_text = " ".join(self.topics).lower()
        
        # Стоп-слова для фильтрации
        stop_words = {'тематика', 'текста', 'новости', 'сегодня', 'россия', 'текст'}
        
        # Извлекаем N-граммы (от min_words слов и больше)
        phrases = []
        topics_split = [topic.lower().split() for topic in self.topics]
        
        # Создаем словарь: фраза -> список индексов тематик, где она встречается
        phrase_to_topics = {}
        
        for topic_idx, words_list in enumerate(topics_split):
            # Извлекаем фразы разной длины (от min_words до 5 слов)
            for n in range(min_words, min(6, len(words_list) + 1)):
                for i in range(len(words_list) - n + 1):
                    phrase_words = words_list[i:i+n]
                    
                    # Фильтруем стоп-слова
                    if any(word in stop_words for word in phrase_words):
                        continue
                    
                    # ✅ ИСПРАВЛЕНО: используем параметр min_word_length
                    # Проверяем минимальную длину хотя бы одного слова
                    if not any(len(word) >= min_word_length for word in phrase_words):
                        continue
                    
                    phrase = " ".join(phrase_words)
                    
                    if phrase not in phrase_to_topics:
                        phrase_to_topics[phrase] = set()
                    phrase_to_topics[phrase].add(topic_idx)
        
        # Преобразуем в список с подсчетом частоты
        phrase_list = [
            {
                'phrase': phrase,
                'count': len(topic_indices),
                'topic_indices': topic_indices,
                'word_count': len(phrase.split())
            }
            for phrase, topic_indices in phrase_to_topics.items()
            if len(topic_indices) >= 2  # Минимум 2 упоминания
        ]
        
        # Сортируем по частоте
        phrase_list.sort(key=lambda x: x['count'], reverse=True)
        
        # Удаляем дублирующиеся фразы (подфразы из тех же текстов)
        unique_phrases = self._remove_duplicate_phrases(phrase_list)
        
        # Возвращаем топ-N уникальных фраз
        return unique_phrases[:top_n]
    
    def _remove_duplicate_phrases(self, phrases):
        """Удаление фраз, которые являются подфразами других и встречаются в тех же текстах"""
        filtered = []
        used_topic_sets = []
        
        for phrase_info in phrases:
            current_phrase = phrase_info['phrase']
            current_topics = phrase_info['topic_indices']
            
            # Проверяем, не является ли эта фраза подфразой уже добавленной
            is_duplicate = False
            
            for existing_info in filtered:
                existing_phrase = existing_info['phrase']
                existing_topics = existing_info['topic_indices']
                
                # Если текущая фраза - подфраза существующей
                if current_phrase in existing_phrase or existing_phrase in current_phrase:
                    # И они встречаются в одних и тех же текстах (>80% пересечения)
                    intersection = len(current_topics & existing_topics)
                    union = len(current_topics | existing_topics)
                    
                    if union > 0 and intersection / union > 0.8:
                        # Оставляем более длинную фразу
                        if len(current_phrase.split()) > len(existing_phrase.split()):
                            # Заменяем старую на новую (более длинную)
                            filtered.remove(existing_info)
                            filtered.append(phrase_info)
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                filtered.append(phrase_info)
        
        return filtered
    
    def group_topics_by_phrases(self, key_phrases):
        """Группировка тематик по ключевым фразам"""
        groups = {}
        
        for phrase_info in key_phrases:
            phrase = phrase_info['phrase']
            matching_topics = []
            
            for topic in self.topics:
                if phrase.lower() in topic.lower():
                    matching_topics.append(topic)
            
            if matching_topics:
                groups[phrase] = {
                    'phrase': phrase,
                    'count': phrase_info['count'],
                    'word_count': phrase_info['word_count'],
                    'topics': matching_topics,
                    'topics_count': len(matching_topics)
                }
        
        return groups

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.sparse import csr_matrix

class GraphBuilder:
    def __init__(self, 
        df: pd.DataFrame, 
        max_nodes: int = 5000,
        similarity_threshold: float = None,  # ✅ Теперь None по умолчанию
        max_features: int = 500,
        ngram_range: tuple = (1, 2)):
        
        self.df = df
        self.G = nx.Graph()
        self.max_nodes = max_nodes
        
        # ✅ АВТОМАТИЧЕСКИЙ РАСЧЕТ ПОРОГА
        if similarity_threshold is None:
            # Для малых графов (<50 узлов) - низкий порог
            # Для больших (>500) - высокий порог
            num_nodes = len(df['fullname'].unique()) if 'fullname' in df.columns else len(df)
            
            if num_nodes < 50:
                self.similarity_threshold = 0.05  # Очень низкий
            elif num_nodes < 200:
                self.similarity_threshold = 0.10  # Низкий
            elif num_nodes < 500:
                self.similarity_threshold = 0.15  # Средний
            else:
                self.similarity_threshold = 0.20  # Высокий
            
            print(f"📊 Auto-threshold: {self.similarity_threshold} (nodes: {num_nodes})")
        else:
            self.similarity_threshold = similarity_threshold
        
        self._topic_vectors = None
        self._vectorizer = None
    
    def _compute_topic_similarity(self, topics_list: list) -> np.ndarray:
        """Вычисление матрицы схожести с улучшенной обработкой"""
        if not topics_list or len(topics_list) < 2:
            return np.array([[]])
        
        # Расширенные русские стоп-слова
        russian_stop_words = {
            'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а',
            'то', 'все', 'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же',
            'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне', 'было', 'вот', 'от',
            'меня', 'о', 'из', 'ему', 'теперь', 'когда', 'даже', 'ну', 'вдруг',
            'ли', 'если', 'уже', 'или', 'ни', 'быть', 'был', 'него', 'до', 'вас',
            'нибудь', 'опять', 'уж', 'вам', 'сказал', 'себя', 'ей', 'они', 'тут',
            'где', 'надо', 'ней', 'для', 'мы', 'тебя', 'их', 'чем', 'была', 'сам',
            'чтоб', 'без', 'будто', 'чего', 'раз', 'тоже', 'себе', 'под', 'будет',
            'ж', 'тогда', 'кто', 'этот', 'того', 'какой', 'совсем', 'ним', 'здесь',
            'этом', 'один', 'почти', 'мой', 'тем', 'чтобы', 'нее', 'сейчас', 'были',
            'куда', 'зачем', 'всех', 'никогда', 'можно', 'при', 'наконец', 'два',
            'об', 'другой', 'хоть', 'после', 'над', 'больше', 'тот', 'через', 'эти',
            'нас', 'про', 'всего', 'них', 'какая', 'много', 'разве', 'три', 'эту',
            'моя', 'впрочем', 'хорошо', 'свою', 'этой', 'перед', 'иногда', 'лучше',
            'чуть', 'том', 'нельзя', 'такой', 'им', 'более', 'всегда', 'конечно',
            'всю', 'между', 'текст', 'тематика', 'новость', 'сегодня', 'вчера'
        }
        
        # ✅ КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: более агрессивные параметры
        self._vectorizer = TfidfVectorizer(
            max_features=2000,  # ↑ Увеличиваем размерность (было 1000)
            stop_words=russian_stop_words,
            ngram_range=(1, 4),  # ↑ Добавляем 4-граммы (было 1,3)
            min_df=1,  # ↓ Убираем минимум (было 2) - учитываем редкие слова
            max_df=0.85,  # ↑ Менее строгая фильтрация частых слов (было 0.7)
            norm='l2',
            sublinear_tf=True,
            use_idf=True,  # ✅ Включаем IDF
            smooth_idf=True,  # ✅ Сглаживание IDF
            token_pattern=r'(?u)\b\w{3,}\w*\b'  # ✅ Минимум 3 символа в слове
        )
        
        try:
            tfidf_matrix = self._vectorizer.fit_transform(topics_list)
            
            # ✅ ВАЖНО: используем dense_output=True для малых графов
            # Для больших (>500 узлов) используйте False
            if len(topics_list) < 500:
                similarity_matrix = cosine_similarity(tfidf_matrix, dense_output=True)
            else:
                similarity_matrix = cosine_similarity(tfidf_matrix, dense_output=False)
            
            print(f"📊 TF-IDF matrix: {tfidf_matrix.shape}, features: {len(self._vectorizer.get_feature_names_out())}")
            
            return similarity_matrix
        except Exception as e:
            print(f"⚠️ Error in similarity computation: {e}")
            return np.array([[]])
    
    def _find_similar_nodes(self, node_topics: dict, threshold: float = None) -> list:
        """Поиск похожих узлов с СМЯГЧЕННОЙ фильтрацией"""
        if threshold is None:
            threshold = self.similarity_threshold
        
        node_ids = list(node_topics.keys())
        topics_texts = [node_topics[nid] for nid in node_ids]
        
        print(f"🔍 Computing similarity for {len(node_ids)} nodes...")
        
        similarity_matrix = self._compute_topic_similarity(topics_texts)
        
        if similarity_matrix.size == 0:
            return []
        
        links = []
        
        # Для разреженной матрицы
        if hasattr(similarity_matrix, 'tocoo'):
            sim_coo = similarity_matrix.tocoo()
            
            for i, j, sim in zip(sim_coo.row, sim_coo.col, sim_coo.data):
                if i >= j:
                    continue
                
                # ✅ СМЯГЧЕННАЯ ФИЛЬТРАЦИЯ
                if sim < threshold:
                    continue
                
                text1, text2 = topics_texts[i], topics_texts[j]
                
                # ✅ Убираем проверку минимальной длины (было >= 3)
                # if len(text1.split()) < 3 or len(text2.split()) < 3:
                #     continue
                
                # ✅ СМЯГЧАЕМ требование к общим словам (было >= 2)
                words1 = set(text1.split())
                words2 = set(text2.split())
                common_words = words1.intersection(words2)
                
                # Минимум 1 общее слово ИЛИ высокая схожесть
                if len(common_words) < 1 and sim < threshold * 1.5:
                    continue
                
                source = node_ids[i]
                target = node_ids[j]
                links.append((source, target, float(sim)))
        
        # Для плотной матрицы
        else:
            for i in range(len(node_ids)):
                for j in range(i + 1, len(node_ids)):
                    sim = similarity_matrix[i, j]
                    
                    if sim < threshold:
                        continue
                    
                    text1, text2 = topics_texts[i], topics_texts[j]
                    
                    words1 = set(text1.split())
                    words2 = set(text2.split())
                    common_words = words1.intersection(words2)
                    
                    if len(common_words) < 1 and sim < threshold * 1.5:
                        continue
                    
                    source = node_ids[i]
                    target = node_ids[j]
                    links.append((source, target, float(sim)))
        
        print(f"✅ Found {len(links)} similarity links (threshold={threshold})")
        
        # ✅ ДИАГНОСТИКА: показываем распределение весов
        if links:
            weights = [link[2] for link in links]
            print(f"📊 Weight distribution: min={min(weights):.3f}, max={max(weights):.3f}, avg={sum(weights)/len(weights):.3f}")
        
        return links

    def extract_topic_phrases(self):
        """Извлечение ключевых фраз из всех тематик"""
        # Собираем все уникальные тематики
        all_topics = []
        for topics_list in self.df['labels'].dropna():
            if isinstance(topics_list, str):
                all_topics.append(topics_list)
        
        all_topics = list(set(all_topics))
        
        if not all_topics:
            return {'phrases': [], 'groups': {}}
        
        # Анализируем тематики
        analyzer = TopicAnalyzer(all_topics)
        key_phrases = analyzer.extract_key_phrases(min_word_length=4, top_n=30)
        topic_groups = analyzer.group_topics_by_phrases(key_phrases)
        
        return {
            'phrases': key_phrases,
            'groups': topic_groups,
            'total_topics': len(all_topics)
        }
        
    def filter_top_nodes(self, nodes, metric='audience'):
        """Фильтрация топ-N узлов по метрике"""
        if len(nodes) <= self.max_nodes:
            return nodes
        
        # Сортируем по важности
        nodes_sorted = sorted(
            nodes, 
            key=lambda x: x.get(metric, 0) + x.get('count', 0), 
            reverse=True
        )
        
        return nodes_sorted[:self.max_nodes]
    
    def build_author_graph(self):
        """Граф связей между авторами с поиском похожих тематик"""
        graph_data = {'nodes': [], 'links': []}
        
        # Группируем по авторам
        author_topics = self.df.groupby('fullname')['labels'].apply(list).to_dict()
        author_urls = self.df.groupby('fullname')['url'].apply(list).to_dict()
        author_stats = self.df.groupby('fullname').agg({
            'audienceCount': 'sum',
            'url': 'count'
        }).to_dict('index')
        
        # Создаем узлы
        node_topics_text = {}  # Для поиска схожести
        
        for author, topics in author_topics.items():
            if not author or pd.isna(author):
                continue
                
            author_data = self.df[self.df['fullname'] == author].iloc[0]
            stats = author_stats.get(author, {})
            
            topics_with_urls = []
            author_posts = self.df[self.df['fullname'] == author]
            
            for idx, row in author_posts.iterrows():
                if pd.notna(row['labels']) and pd.notna(row['url']):
                    topics_with_urls.append({
                        'text': str(row['labels']),
                        'url': str(row['url'])
                    })
            
            topics_with_urls = topics_with_urls[:5]
            
            node = {
                'id': str(author),
                'label': str(author),
                'type': str(author_data.get('author_type', 'unknown')),
                'audience': int(stats.get('audienceCount', 0)),
                'posts_count': int(stats.get('url', 0)),
                'topics': topics_with_urls,
            }
            graph_data['nodes'].append(node)
            
            # Сохраняем объединенный текст для поиска схожести
            all_topics_text = " ".join([t['text'] for t in topics_with_urls])
            node_topics_text[str(author)] = all_topics_text
        
        # Фильтруем топ-узлы
        graph_data['nodes'] = self.filter_top_nodes(graph_data['nodes'])
        
        # Обновляем словарь после фильтрации
        filtered_node_ids = {node['id'] for node in graph_data['nodes']}
        node_topics_text = {
            nid: text for nid, text in node_topics_text.items() 
            if nid in filtered_node_ids
        }
        
        # 1. Создаем связи по точному совпадению тематик (быстро)
        exact_links = set()
        for i, node1 in enumerate(graph_data['nodes']):
            for node2 in graph_data['nodes'][i+1:]:
                topics1 = set(t['text'] for t in node1['topics'])
                topics2 = set(t['text'] for t in node2['topics'])
                common = topics1.intersection(topics2)
                
                if len(common) > 0:
                    exact_links.add((node1['id'], node2['id']))
                    graph_data['links'].append({
                        'source': node1['id'],
                        'target': node2['id'],
                        'weight': len(common),
                        'type': 'exact'  # Метка типа связи
                    })
        
        # 2. Ищем похожие тематики (для узлов без точных совпадений)
        print(f"🔗 Found {len(exact_links)} exact links")
        print(f"🔍 Searching for similar topics...")
        
        similarity_links = self._find_similar_nodes(node_topics_text, self.similarity_threshold)
        
        # Добавляем только новые связи (не дублирующие точные)
        added_similar = 0
        for source, target, weight in similarity_links:
            # Проверяем, нет ли уже точной связи
            if (source, target) not in exact_links and (target, source) not in exact_links:
                graph_data['links'].append({
                    'source': source,
                    'target': target,
                    'weight': weight * 5,  # Масштабируем для визуализации
                    'type': 'similar'  # Метка типа связи
                })
                added_similar += 1
        
        print(f"✅ Added {added_similar} similarity links")
        print(f"📊 Total links: {len(graph_data['links'])}")
        
        return graph_data
    
    def build_topic_graph(self):
        """Граф связей между тематиками с поиском похожих"""
        graph_data = {'nodes': [], 'links': []}
        
        # Собираем все тематики
        topic_urls = {}
        topic_authors = {}
        
        for _, row in self.df.iterrows():
            if pd.isna(row['labels']):
                continue
                
            topic = str(row['labels'])
            author = str(row['fullname']) if pd.notna(row['fullname']) else 'unknown'
            url = str(row['url']) if pd.notna(row['url']) else None
            
            if topic not in topic_urls:
                topic_urls[topic] = []
                topic_authors[topic] = set()
            
            if url:
                topic_urls[topic].append(url)
            topic_authors[topic].add(author)
        
        # Создаем узлы
        node_topics_text = {}
        for topic, urls in topic_urls.items():
            node = {
                'id': topic,
                'label': topic[:50] + '...' if len(topic) > 50 else topic,
                'type': 'topic',
                'count': len(urls),
                'authors_count': len(topic_authors[topic]),
                'urls': urls[:5]
            }
            graph_data['nodes'].append(node)
            node_topics_text[topic] = topic  # Для тематик используем сам текст
        
        # Фильтруем топ-узлы
        graph_data['nodes'] = self.filter_top_nodes(graph_data['nodes'], metric='count')
        
        # Обновляем словарь после фильтрации
        filtered_node_ids = {node['id'] for node in graph_data['nodes']}
        node_topics_text = {
            nid: text for nid, text in node_topics_text.items() 
            if nid in filtered_node_ids
        }
        
        # 1. Создаем связи по общим авторам
        exact_links = set()
        for i, node1 in enumerate(graph_data['nodes']):
            authors1 = topic_authors[node1['id']]
            
            for node2 in graph_data['nodes'][i+1:]:
                authors2 = topic_authors[node2['id']]
                common_authors = authors1.intersection(authors2)
                
                if len(common_authors) > 0:
                    exact_links.add((node1['id'], node2['id']))
                    graph_data['links'].append({
                        'source': node1['id'],
                        'target': node2['id'],
                        'weight': len(common_authors),
                        'type': 'exact'
                    })
        
        # 2. Ищем похожие тематики
        print(f"🔗 Found {len(exact_links)} exact links (common authors)")
        print(f"🔍 Searching for similar topics...")
        
        similarity_links = self._find_similar_nodes(node_topics_text, self.similarity_threshold)
        
        added_similar = 0
        for source, target, weight in similarity_links:
            if (source, target) not in exact_links and (target, source) not in exact_links:
                graph_data['links'].append({
                    'source': source,
                    'target': target,
                    'weight': weight * 3,
                    'type': 'similar'
                })
                added_similar += 1
        
        print(f"✅ Added {added_similar} similarity links")
        print(f"📊 Total links: {len(graph_data['links'])}")
        
        return graph_data
    
    def get_graph_statistics(self):
        """Статистика графа для AI анализа"""
        if len(self.G.nodes()) == 0:
            return {
                'nodes_count': 0,
                'edges_count': 0,
                'density': 0,
                'avg_degree': 0,
                'top_nodes': []
            }
            
        stats = {
            'nodes_count': self.G.number_of_nodes(),
            'edges_count': self.G.number_of_edges(),
            'density': nx.density(self.G),
            'avg_degree': sum(dict(self.G.degree()).values()) / self.G.number_of_nodes() if self.G.number_of_nodes() > 0 else 0,
        }
        
        if nx.is_connected(self.G):
            stats['diameter'] = nx.diameter(self.G)
            stats['avg_shortest_path'] = nx.average_shortest_path_length(self.G)
        
        # Топ узлы по центральности
        if len(self.G.nodes()) > 0:
            degree_centrality = nx.degree_centrality(self.G)
            stats['top_nodes'] = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        else:
            stats['top_nodes'] = []
        
        return stats

# tellscope:author-graph-enhance
try:
    from mlops.author_graph import patch_graph_builder
    patch_graph_builder(GraphBuilder)
except Exception as exc:
    print('author_graph patch skipped:', exc)

def load_csv_data(file_path: str) -> pd.DataFrame:
    """Загрузка CSV с обработкой ошибок"""
    try:
        df = pd.read_csv(file_path)
        
        # Приводим timeCreate к datetime если есть
        if 'timeCreate' in df.columns:
            df['timeCreate'] = pd.to_datetime(df['timeCreate'], errors='coerce')
        
        # Заполняем пропуски
        if 'audienceCount' in df.columns:
            df['audienceCount'] = df['audienceCount'].fillna(0)
        if 'viewsCount' in df.columns:
            df['viewsCount'] = df['viewsCount'].fillna(0)
            
        return df
    except Exception as e:
        raise HTTPException(500, f"Ошибка чтения CSV: {str(e)}")

@app.get("/csv-files")
async def list_csv_files():
    """Список доступных CSV файлов на сервере"""
    base_dir = "/home/dev/tellscope_app/tellscope_backend/data/1/bertopic_files_directory/test"
    
    try:
        files = []
        for root, dirs, filenames in os.walk(base_dir):
            for filename in filenames:
                if filename.endswith('.csv') and 'result_graph' in filename:
                    full_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(full_path, base_dir)
                    files.append({
                        'name': filename,
                        'path': full_path,
                        'relative_path': rel_path
                    })
        
        return {'files': files, 'default': DEFAULT_CSV_PATH}
    except Exception as e:
        raise HTTPException(500, str(e))

def _compute_topic_similarity_hybrid(self, topics_list: list) -> np.ndarray:
    """Гибридный метод: TF-IDF + Jaccard"""
    if not topics_list or len(topics_list) < 2:
        return np.array([[]])
    
    # 1. TF-IDF схожесть
    tfidf_sim = self._compute_topic_similarity(topics_list)
    
    # 2. Jaccard схожесть по биграммам
    def get_bigrams(text):
        words = text.lower().split()
        return set(zip(words[:-1], words[1:]))
    
    n = len(topics_list)
    jaccard_sim = np.zeros((n, n))
    
    for i in range(n):
        bigrams_i = get_bigrams(topics_list[i])
        for j in range(i + 1, n):
            bigrams_j = get_bigrams(topics_list[j])
            
            if len(bigrams_i) == 0 or len(bigrams_j) == 0:
                jaccard_sim[i, j] = 0
            else:
                intersection = len(bigrams_i & bigrams_j)
                union = len(bigrams_i | bigrams_j)
                jaccard_sim[i, j] = intersection / union if union > 0 else 0
            
            jaccard_sim[j, i] = jaccard_sim[i, j]
    
    # 3. Комбинируем (70% TF-IDF + 30% Jaccard)
    if isinstance(tfidf_sim, np.ndarray):
        hybrid_sim = 0.7 * tfidf_sim + 0.3 * jaccard_sim
    else:
        # Для разреженных матриц
        hybrid_sim = tfidf_sim.multiply(0.7).toarray() + 0.3 * jaccard_sim
    
    print(f"🔀 Hybrid similarity: TF-IDF + Jaccard")
    
    return hybrid_sim


@app.post("/build-from-csv")
async def build_graph_from_csv(
    graph_type: str = Form('author'),
    csv_path: Optional[str] = Form(None),  # ✅ Теперь полный путь
    file: Optional[UploadFile] = File(None),
    similarity_threshold: Optional[float] = Form(None),
    similarity_method: str = Form('tfidf'),
    min_common_words: int = Form(1)
):
    try:
        # Загружаем данные
        if file:
            df = pd.read_csv(file.file)
        elif csv_path:
            # ✅ Проверяем, что файл существует и принадлежит пользователю
            if not os.path.exists(csv_path):
                raise HTTPException(404, f"CSV file not found: {csv_path}")
            
            # ✅ Дополнительная проверка безопасности
            base_data_dir = '/home/dev/tellscope_app/tellscope_backend/data/'
            if not os.path.abspath(csv_path).startswith(base_data_dir):
                raise HTTPException(403, "Access denied")
            
            df = pd.read_csv(csv_path)
        else:
            raise HTTPException(400, "No CSV file provided")
        
        print(f"📂 Loaded {len(df)} records from CSV: {csv_path}")
        
        if 'timeCreate' in df.columns:
            df['timeCreate'] = pd.to_datetime(df['timeCreate'], errors='coerce')
        
        # ✅ Создаем builder с автоматическим порогом
        builder = GraphBuilder(df, similarity_threshold=similarity_threshold)
        
        # ✅ Выбираем метод схожести
        if similarity_method == 'hybrid':
            builder._compute_topic_similarity = builder._compute_topic_similarity_hybrid
        
        # Строим граф
        if graph_type == 'author':
            graph_data = builder.build_author_graph()
        elif graph_type == 'topic':
            graph_data = builder.build_topic_graph()
        else:
            raise HTTPException(400, f"Unknown graph type: {graph_type}")
        
        # ✅ ДИАГНОСТИКА связей
        exact_links = [l for l in graph_data['links'] if l.get('type') == 'exact']
        similar_links = [l for l in graph_data['links'] if l.get('type') == 'similar']
        
        print(f"📊 Graph stats:")
        print(f"  - Nodes: {len(graph_data['nodes'])}")
        print(f"  - Exact links: {len(exact_links)}")
        print(f"  - Similar links: {len(similar_links)}")
        print(f"  - Threshold used: {builder.similarity_threshold}")
        
        statistics = builder.get_graph_statistics()
        topic_analysis = builder.extract_topic_phrases()
        
        # Обработка дат
        date_range = {}
        if 'timeCreate' in df.columns and not df.empty:
            min_date = df['timeCreate'].min()
            max_date = df['timeCreate'].max()
            date_range = {
                'start': min_date.isoformat() if pd.notna(min_date) else None,
                'end': max_date.isoformat() if pd.notna(max_date) else None
            }
        else:
            date_range = {'start': None, 'end': None}
        
        result = {
            'graph': graph_data,
            'statistics': statistics,
            'topic_analysis': topic_analysis,
            'metadata': {
                'total_records': len(df),
                'date_range': date_range,
                'graph_type': graph_type,
                'similarity_threshold': builder.similarity_threshold,
                'similarity_method': similarity_method,
                'exact_links_count': len(exact_links),
                'similar_links_count': len(similar_links)
            }
        }
        
        return result
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, str(e))
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, str(e))

@app.post("/build")
async def build_graph(data: dict):
    """Построение графа из данных (старый эндпойнт)"""
    try:
        df = pd.DataFrame(data['records'])
        graph_type = data.get('type', 'author')
        
        builder = GraphBuilder(df)
        
        if graph_type == 'author':
            graph_data = builder.build_author_graph()
        elif graph_type == 'topic':
            graph_data = builder.build_topic_graph()
        elif graph_type == 'geo':
            graph_data = builder.build_geo_graph()
        else:
            raise HTTPException(400, "Unknown graph type")
        
        statistics = builder.get_graph_statistics()
        
        return {
            'graph': graph_data,
            'statistics': statistics,
            'metadata': {
                'total_records': len(df),
                'date_range': {
                    'start': df['timeCreate'].min().isoformat() if 'timeCreate' in df.columns and not df.empty else None,
                    'end': df['timeCreate'].max().isoformat() if 'timeCreate' in df.columns and not df.empty else None
                }
            }
        }
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/analyze")
async def analyze_graph(query: GraphQuery):
    from mlops.dashboard_qa import handle_analyze_graph
    return await handle_analyze_graph(query.question, query.graph_data)
    """AI анализ графа"""
    print(f"=== ANALYZE REQUEST ===")
    print(f"Question: {query.question[:100]}...")
    print(f"Graph data type: {type(query.graph_data)}")
    print(f"Graph data keys: {query.graph_data.keys() if isinstance(query.graph_data, dict) else 'not a dict'}")
    
    try:
        # ✅ Безопасное извлечение данных
        if isinstance(query.graph_data, dict):
            graph = query.graph_data.get('graph', query.graph_data)
            statistics = query.graph_data.get('statistics', {})
            metadata = query.graph_data.get('metadata', {})
        else:
            raise HTTPException(400, "graph_data must be a dictionary")
        
        # ✅ Проверяем наличие узлов
        if 'nodes' not in graph or len(graph['nodes']) == 0:
            raise HTTPException(400, "Graph has no nodes")
        
        print(f"Graph info: {len(graph['nodes'])} nodes, {len(graph.get('links', []))} links")
        
        # Формируем контекст
        context = f"""
Ты - аналитик социальных сетей. У тебя есть граф с следующими характеристиками:

СТАТИСТИКА ГРАФА:
- Количество узлов: {statistics.get('nodes_count', len(graph['nodes']))}
- Количество связей: {statistics.get('edges_count', len(graph.get('links', [])))}
- Плотность графа: {statistics.get('density', 0):.4f}
- Средняя степень: {statistics.get('avg_degree', 0):.2f}

МЕТАДАННЫЕ:
{json.dumps(metadata, ensure_ascii=False, indent=2)}

ТОП УЗЛЫ ПО ЦЕНТРАЛЬНОСТИ:
{json.dumps(statistics.get('top_nodes', [])[:5], ensure_ascii=False, indent=2)}

ПРИМЕРЫ УЗЛОВ:
{json.dumps(graph['nodes'][:10], ensure_ascii=False, indent=2)}

ПРИМЕРЫ СВЯЗЕЙ:
{json.dumps(graph.get('links', [])[:10], ensure_ascii=False, indent=2)}

ВОПРОС ПОЛЬЗОВАТЕЛЯ: {query.question}

Проанализируй граф и дай развернутый ответ на вопрос. Используй конкретные данные из графа.
Форматируй ответ с использованием Markdown для лучшей читаемости.
"""

        response = client.chat.completions.create(
            messages=[{"role": "user", "content": context}],
            model="gpt-5-mini",
            max_tokens=2000,
        )
        
        answer = response.choices[0].message.content
        
        return {
            'answer': answer,
            'context_used': {
                'nodes_analyzed': min(10, len(graph['nodes'])),
                'links_analyzed': min(10, len(graph.get('links', [])))
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR in analyze: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Analysis error: {str(e)}")

from llm_agent import SocialMediaAgent

class SmartAgentRequest(BaseModel):
    user_query: str
    index: int
    min_date: Optional[str] = None
    max_date: Optional[str] = None
    user_id: Optional[str] = None
    reports_dir: Optional[str] = None

active_tasks: Dict[str, dict] = {}
websocket_ready_events: Dict[str, Event] = {}

# --- ИСПРАВЛЕНИЕ: УЛУЧШЕННЫЙ CONNECTION MANAGER ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, task_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[task_id] = websocket
        print(f"WebSocket accepted and stored for task {task_id}")

    def disconnect(self, task_id: str):
        if task_id in self.active_connections:
            del self.active_connections[task_id]
            print(f"WebSocket disconnected and removed for task {task_id}")
        if task_id in websocket_ready_events:
            del websocket_ready_events[task_id]

    async def send_message(self, task_id: str, message: dict):
        if task_id in self.active_connections:
            websocket = self.active_connections[task_id]
            if websocket.client_state.name == "CONNECTED":
                try:
                    await websocket.send_json(message)
                except Exception as e:
                    print(f"Error sending message to {task_id}: {e}. Disconnecting.")
                    self.disconnect(task_id)
            else:
                print(f"WebSocket for task {task_id} is not connected. Removing.")
                self.disconnect(task_id)

manager = ConnectionManager()

class AgentProgressLogger:
    """Класс для отправки прогресса через WebSocket"""
    def __init__(self, task_id: str):
        self.task_id = task_id
    
    async def log(self, message: str, msg_type: str = "status", **kwargs):
        await manager.send_message(self.task_id, {
            "type": msg_type,
            "message": message,
            **kwargs
        })

from llm_agent import ReportBuilder

async def run_agent_task(task_id: str, user_query: str, input_file: str, user_id: str = None, folder_name: str = None):
    """
    Эта функция теперь выступает как простой диспетчер.
    Она создает агента и запускает его основной метод `run_task`,
    который выполняет всю работу и возвращает путь к готовому отчету.
    """
    logger = AgentProgressLogger(task_id)
    try:
        # 1. Ожидание WebSocket соединения (логика не меняется)
        if task_id in websocket_ready_events:
            try:
                await asyncio.wait_for(websocket_ready_events[task_id].wait(), timeout=15.0)
                print(f"WebSocket is ready for task {task_id}. Starting agent work.")
            except asyncio.TimeoutError:
                error_msg = "Client failed to connect via WebSocket within 15 seconds."
                await logger.log(error_msg, "error")
                active_tasks[task_id].update({"status": "failed", "error": error_msg})
                try:
                    from mlops.runtime import register_smart_agent
                    register_smart_agent(task_id, user_query, "failed", user_id=str(user_id or ""))
                except Exception:
                    pass
                return

        await logger.log("Инициализация интеллектуального агента...", "status")
        
        # 2. Создание экземпляра агента с передачей callback для логирования прогресса
        agent = SocialMediaAgent(progress_callback=logger.log)

        # 3. Определение пути для сохранения отчета
        if user_id:
            reports_dir = Path(f"/home/dev/tellscope_app/tellscope_backend/data/{user_id}/users_reports/{folder_name or ''}")
        else:
            reports_dir = Path("/home/dev/tellscope_app/tellscope_backend/reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_save_path = reports_dir / f"SmartReport_{task_id}.docx"

        # 4. Запуск основной задачи агента
        # Агент сам выполнит планирование, анализ, генерацию и сохранение отчета.
        report_path = await agent.run_task(
            user_query=user_query,
            input_file=input_file,
            report_save_path=str(report_save_path)
        )

        # 5. Обновление статуса задачи по завершении
        active_tasks[task_id]["report_path"] = report_path
        active_tasks[task_id]["status"] = "completed"
        try:
            from mlops.runtime import finish_smart_agent
            finish_smart_agent(
                task_id,
                user_query,
                "done",
                user_id=str(user_id or ""),
                artifact_path=str(active_tasks[task_id].get("report_path") or ""),
            )
        except Exception:
            pass
        await logger.log("Отчет успешно создан!", "complete", report_url=f"/api/download-report/{task_id}")
        await asyncio.sleep(1) # Небольшая задержка для отправки сообщения

    except Exception as e:
        error_msg = f"Произошла критическая ошибка в работе агента: {str(e)}"
        print(f"Error in task {task_id}: {error_msg}")
        traceback.print_exc()
        await logger.log(error_msg, "error")
        active_tasks[task_id].update({"status": "failed", "error": str(e)})
        try:
            from mlops.runtime import register_smart_agent
            register_smart_agent(task_id, user_query, "failed", user_id=str(user_id or ""))
        except Exception:
            pass
        await asyncio.sleep(1)


@app.post("/run-smart-agent", tags=['smart agent'])
async def run_smart_agent(
    background_tasks: BackgroundTasks,
    request: SmartAgentRequest
):
    """Запуск умного агента для анализа данных"""
    try:
        from mlops.runtime import GpuBusy, assert_can_start
        assert_can_start("smart-agent")
    except GpuBusy as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception:
        pass
    
    task_id = str(uuid.uuid4())
    
    websocket_ready_events[task_id] = Event()
    
    active_tasks[task_id] = {
        "status": "initializing",
        "user_query": request.user_query,
        "input_file": None,
        "report_path": None,
        "error": None,
        "created_at": time.time(),
        "user_id": request.user_id  # <-- сохраняем user_id
    }
    
    try:
        file_path = '/home/dev/tellscope_app/tellscope_backend/data/indexes.pkl'
        indexes = load_dict_from_pickle(file_path)
        
        data = elastic_query(
            theme_index=indexes[request.index], 
            min_date=request.min_date, 
            max_date=request.max_date, 
            query_str='all'
        )
        
        temp_dir = Path("/home/dev/tellscope_app/tellscope_backend/temp")
        temp_dir.mkdir(exist_ok=True)
        
        input_file = temp_dir / f"data_{task_id}.json"
        with open(input_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        active_tasks[task_id]["input_file"] = str(input_file)
        active_tasks[task_id]["status"] = "pending"
        try:
            from mlops.runtime import register_smart_agent
            register_smart_agent(task_id, request.user_query, "pending", user_id=str(request.user_id or ""))
        except Exception:
            pass
        
        # Передаем user_id в фоновую задачу
        background_tasks.add_task(
            run_agent_task, 
            task_id, 
            request.user_query, 
            str(input_file),
            request.user_id  # <-- добавили
        )
        
        return {"task_id": task_id, "status": "started"}
        
    except Exception as e:
        active_tasks[task_id]["status"] = "failed"
        active_tasks[task_id]["error"] = str(e)
        if task_id in websocket_ready_events:
            del websocket_ready_events[task_id]
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/agent/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    """
    WebSocket для получения прогресса выполнения агента.
    Эта версия использует неблокирующий цикл для отправки обновлений и keep-alive сообщений.
    """
    print(f"WebSocket connection attempt for task_id: {task_id}")

    # 1. Ждем, пока задача будет создана в active_tasks
    # Увеличиваем время ожидания, чтобы решить race condition
    max_wait_time = 20  # секунд
    start_time = time.time()
    task_found = False
    while time.time() - start_time < max_wait_time:
        if task_id in active_tasks:
            task_found = True
            break
        await asyncio.sleep(0.1)

    if not task_found:
        print(f"Task {task_id} not found after {max_wait_time}s. Closing connection.")
        await websocket.close(code=1008, reason="Task not found or timed out.")
        return

    # 2. Принимаем и регистрируем соединение
    try:
        await manager.connect(task_id, websocket)
        # Сообщаем фоновой задаче, что WebSocket готов
        if task_id in websocket_ready_events:
            websocket_ready_events[task_id].set()
            print(f"Signaled 'ready' event for task {task_id}")
    except Exception as e:
        print(f"Failed to accept/connect WebSocket for {task_id}: {e}")
        return # Просто выходим, если не удалось принять соединение

    # 3. Основной цикл работы WebSocket
    try:
        # Этот цикл будет работать, пока задача не завершится или клиент не отключится
        while True:
            # Проверяем статус задачи в словаре
            # Если задачи больше нет (например, старая и удалена), выходим
            if task_id not in active_tasks:
                print(f"Task {task_id} was removed. Closing WebSocket.")
                break

            task_status = active_tasks[task_id].get("status")
            if task_status in ["completed", "failed"]:
                print(f"Task {task_id} finished with status: {task_status}. Closing WebSocket.")
                # Финальная задержка перед закрытием, чтобы клиент успел получить последнее сообщение
                await asyncio.sleep(2)
                break

            # Неблокирующее ожидание сообщения от клиента (для ping/pong)
            # с таймаутом, чтобы цикл не зависал.
            try:
                # Ждем сообщение от клиента не более 20 секунд
                client_data = await asyncio.wait_for(websocket.receive_json(), timeout=20.0)
                if client_data.get("type") == "ping":
                    await manager.send_message(task_id, {"type": "pong"})
            except asyncio.TimeoutError:
                # Если клиент молчит 20 секунд, отправляем ему keepalive,
                # чтобы прокси (Nginx) не закрыли соединение.
                await manager.send_message(task_id, {"type": "keepalive"})
            except WebSocketDisconnect:
                # Если клиент сам отключился
                print(f"Client disconnected for task {task_id}.")
                break
            except Exception as e:
                # Другие ошибки при получении данных
                print(f"Error while receiving data for task {task_id}: {e}")
                break

    except Exception as e:
        print(f"An unexpected error occurred in WebSocket loop for task {task_id}: {e}")
        traceback.print_exc()
    finally:
        # 4. Очистка при любом выходе из цикла
        await manager.send_message(task_id, {"type": "status", "message": "Соединение закрыто."})
        manager.disconnect(task_id)
        if websocket.client_state.name != "DISCONNECTED":
            await websocket.close(code=1000)
        print(f"WebSocket cleanup complete for {task_id}")


@app.get("/download-report/{task_id}", tags=['smart agent'])
async def download_report(task_id: str):
    """Скачивание готового отчета"""
    if task_id not in active_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = active_tasks[task_id]
    report_path = task.get("report_path")
    
    if not report_path or not Path(report_path).exists():
        raise HTTPException(status_code=404, detail="Report not found")
    
    return FileResponse(
        path=report_path,
        filename=Path(report_path).name,
        media_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    )

@app.get("/agent-status/{task_id}", tags=['smart agent'])
async def get_agent_status(task_id: str):
    """Получить статус задачи агента"""
    if task_id not in active_tasks:
        return {"error": "Task not found"}
    
    task = active_tasks[task_id]
    return {
        "status": task["status"],
        "error": task.get("error"),
        "has_report": task.get("report_path") is not None
    }
    

from mosinform_api import router as mosinform_router
app.include_router(mosinform_router)
from mlops_api import router as mlops_router
app.include_router(mlops_router)
from ba_api import router as ba_router
app.include_router(ba_router)
from information_summary import router as information_summary_router
app.include_router(information_summary_router)

@app.post("/graph-analysis/cluster-summary", tags=['data analytics'])
async def graph_cluster_summary(request: Request):
    from mlops.author_graph import handle_cluster_summary
    return await handle_cluster_summary(request)

@app.get("/graph-analysis/cluster-summary/status", tags=['data analytics'])
async def graph_cluster_summary_status(request: Request):
    from mlops.author_graph import handle_cluster_summary_status
    return await handle_cluster_summary_status(request)


from auth.database import User as AuthUser
# ================= Admin & Access (P1 multiuser) =================
import redis as _redis_sync
_redis_s = _redis_sync.Redis(host='localhost', port=6379, db=0, decode_responses=True)
_SHARES_FILE = '/home/dev/tellscope_app/tellscope_backend/data/shares.json'
_SHARES_LOCK = None

def _shares_lock():
    global _SHARES_LOCK
    import threading
    if _SHARES_LOCK is None:
        _SHARES_LOCK = threading.Lock()
    return _SHARES_LOCK

def _shares_db():
    import psycopg2
    from config import DB_HOST, DB_NAME, DB_PASS, DB_PORT, DB_USER
    return psycopg2.connect(host=DB_HOST, port=DB_PORT or 5432, dbname=DB_NAME, user=DB_USER, password=DB_PASS, connect_timeout=5)

def _shares_init(cur):
    cur.execute("CREATE TABLE IF NOT EXISTS tellscope_shares (id BIGSERIAL PRIMARY KEY, owner_user_id INTEGER NOT NULL, folder TEXT NOT NULL, user_id INTEGER NOT NULL, access TEXT NOT NULL DEFAULT 'read', created TEXT, updated TEXT)")
    cur.execute('CREATE INDEX IF NOT EXISTS ix_shares_user ON tellscope_shares(user_id)')

def _load_shares():
    import os
    with _shares_lock():
        try:
            conn = _shares_db()
            cur = conn.cursor()
            _shares_init(cur)
            conn.commit()
            cur.execute("SELECT owner_user_id, folder, user_id, access, COALESCE(created, ''), COALESCE(updated, '') FROM tellscope_shares ORDER BY id")
            rows = [{'owner_user_id': r[0], 'folder': r[1], 'user_id': r[2], 'access': r[3], 'created': r[4], 'updated': r[5]} for r in cur.fetchall()]
            cur.close()
            conn.close()
            return rows
        except Exception as e:
            print('shares: PG load unavailable, fallback file:', e)
        if os.path.exists(_SHARES_FILE):
            try:
                with open(_SHARES_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return []
        return []

def _save_shares(items):
    import os
    with _shares_lock():
        os.makedirs(os.path.dirname(_SHARES_FILE), exist_ok=True)
        try:
            conn = _shares_db()
            cur = conn.cursor()
            _shares_init(cur)
            conn.commit()
            cur.execute('DELETE FROM tellscope_shares')
            for it in items:
                cur.execute('INSERT INTO tellscope_shares (owner_user_id, folder, user_id, access, created, updated) VALUES (%s, %s, %s, %s, %s, %s)', (int(it['owner_user_id']), it['folder'], int(it['user_id']), it.get('access', 'read'), it.get('created') or '', it.get('updated') or ''))
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            print('shares: PG save unavailable, keep file only:', e)
        with open(_SHARES_FILE, 'w', encoding='utf-8') as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
class AdminShareBody(BaseModel):
    owner_user_id: int
    folder: str
    user_id: int
    access: str = "read"


def _folder_allowed(owner_user_id, folder, user, need_write=False):
    """Owner / superuser всегда имеют доступ; иначе — запись в реестре shares."""
    try:
        if str(user.id) == str(owner_user_id):
            return True
        if getattr(user, "is_superuser", False):
            return True
    except Exception:
        return False
    if not folder:
        return False
    for it in _load_shares():
        if (int(it.get("owner_user_id", -1)) == int(owner_user_id)
                and it.get("folder") == folder
                and int(it.get("user_id", -1)) == int(user.id)):
            if need_write and it.get("access", "read") != "write":
                return False
            return True
    return False


def _folder_guard(owner_user_id, folder, user, need_write=False):
    if not _folder_allowed(owner_user_id, folder, user, need_write=need_write):
        raise HTTPException(status_code=403, detail="Нет доступа к этой папке пользователя")

current_superuser = fastapi_users.current_user(active=True, superuser=True)

@app.get("/admin/users")
async def admin_users(admin: User = Depends(current_superuser)):
    async with async_session_maker() as session:
        res = await session.execute(select(AuthUser).order_by(AuthUser.id))
        users = res.scalars().all()
    return [{
        "id": u.id, "email": u.email, "username": u.username,
        "role_id": u.role_id, "is_superuser": bool(u.is_superuser),
        "is_active": bool(u.is_active), "is_verified": bool(u.is_verified),
    } for u in users]

@app.post("/admin/users")
async def admin_create_user(body: UserCreate, admin: User = Depends(current_superuser)):
    from auth.manager import UserManager
    from auth.database import SQLAlchemyUserDatabase
    async with async_session_maker() as session:
        udb = SQLAlchemyUserDatabase(session, AuthUser)
        mgr = UserManager(udb)
        try:
            user = await mgr.create(body, safe=False, request=None)
        except Exception as exc:
            return JSONResponse(status_code=400, content={"detail": str(exc)})
        return {"id": user.id, "email": user.email, "username": user.username,
                "is_superuser": bool(user.is_superuser), "role_id": body.role_id}

@app.get("/admin/folders/{owner_id}")
async def admin_owner_folders(owner_id: int, admin: User = Depends(current_superuser)):
    raw = _redis_s.hget(str(owner_id), "json_files_directory")
    if not raw:
        return {"owner_id": owner_id, "folders": []}
    try:
        folders = json.loads(raw)
    except Exception:
        folders = {}
    return {"owner_id": owner_id, "folders": [{"name": k, "files": len(v or [])} for k, v in folders.items()]}

@app.post("/admin/shares")
async def admin_add_share(body: AdminShareBody, admin: User = Depends(current_superuser)):
    if body.access not in ("read", "write"):
        raise HTTPException(400, "access должен быть read или write")
    items = _load_shares()
    for it in items:
        if it["owner_user_id"] == body.owner_user_id and it["folder"] == body.folder and it["user_id"] == body.user_id:
            it["access"] = body.access
            it["updated"] = datetime.now().isoformat()
            _save_shares(items)
            return {"status": "updated", **body.__dict__}
    items.append({"owner_user_id": body.owner_user_id, "folder": body.folder, "user_id": body.user_id,
                  "access": body.access, "created": datetime.now().isoformat()})
    _save_shares(items)
    return {"status": "ok", **body.__dict__}

@app.delete("/admin/shares")
async def admin_remove_share(body: AdminShareBody, admin: User = Depends(current_superuser)):
    items = _load_shares()
    n = len(items)
    items = [it for it in items if not (it["owner_user_id"] == body.owner_user_id and it["folder"] == body.folder and it["user_id"] == body.user_id)]
    _save_shares(items)
    return {"removed": n - len(items)}

@app.get("/admin/shares")
async def admin_list_shares(admin: User = Depends(current_superuser)):
    return {"shares": _load_shares()}

@app.get("/access")
async def my_access(user: User = Depends(current_user)):
    items = [it for it in _load_shares() if it["user_id"] == user.id]
    return {"shares": items}



@app.get("/my-datasets")
async def my_datasets(user: User = Depends(current_user)):
    """Свои папки + папки, расшаренные мне (read/write)."""
    def folders_of(uid):
        raw = _redis_s.hget(str(uid), "json_files_directory")
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except Exception:
            return {}
    own = [{"name": k, "files": v or []} for k, v in folders_of(user.id).items()]
    shares = [it for it in _load_shares() if it["user_id"] == user.id]
    shared = []
    for it in shares:
        fo = folders_of(it["owner_user_id"])
        files = fo.get(it["folder"], [])
        shared.append({"owner_user_id": it["owner_user_id"], "folder": it["folder"],
                       "access": it.get("access", "read"), "files": files or []})
    return {"own": own, "shared": shared}


class AdminUserPatch(BaseModel):
    is_active: Optional[bool] = None
    is_superuser: Optional[bool] = None
    role_id: Optional[int] = None
    password: Optional[str] = None


@app.get("/me")
async def whoami(user: User = Depends(current_user)):
    return {"id": user.id, "email": user.email, "username": user.username,
            "role_id": user.role_id, "is_active": bool(user.is_active),
            "is_superuser": bool(user.is_superuser), "is_verified": bool(user.is_verified)}


@app.patch("/admin/users/{user_id}")
async def admin_patch_user(user_id: int, body: AdminUserPatch, admin: User = Depends(current_superuser)):
    from auth.manager import UserManager
    from auth.database import SQLAlchemyUserDatabase
    async with async_session_maker() as session:
        user = await session.get(AuthUser, user_id)
        if user is None:
            raise HTTPException(status_code=404, detail="Пользователь не найден")
        if admin.id == user_id:
            if body.is_active is False:
                raise HTTPException(status_code=400, detail="Нельзя деактивировать собственную учётную запись")
        if body.is_active is not None:
            user.is_active = bool(body.is_active)
        if body.is_superuser is not None and user_id != admin.id:
            user.is_superuser = bool(body.is_superuser)
        if body.role_id is not None:
            user.role_id = body.role_id
        if body.password:
            if len(body.password) < 6:
                raise HTTPException(status_code=400, detail="Пароль слишком короткий (минимум 6 символов)")
            udb = SQLAlchemyUserDatabase(session, AuthUser)
            mgr = UserManager(udb)
            user.hashed_password = mgr.password_helper.hash(body.password)
        await session.commit()
        await session.refresh(user)
        return {"id": user.id, "email": user.email, "username": user.username,
                "role_id": user.role_id, "is_active": bool(user.is_active),
                "is_superuser": bool(user.is_superuser)}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
