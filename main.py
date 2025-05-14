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
from typing import List, Optional, Union, Dict
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

from fastapi import BackgroundTasks, FastAPI, File, Request, UploadFile, WebSocket, logger, status, Depends
from fastapi.encoders import jsonable_encoder
# from fastapi.exceptions import ValidationError
from fastapi.responses import JSONResponse
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

from auth.auth import auth_backend
from auth.auth import get_jwt_strategy, get_refresh_strategy, SECRET
from auth.database import User
from auth.manager import get_user_manager
from auth.schemas import UserRead, UserCreate
from fastapi.middleware.cors import CORSMiddleware 
from elasticsearch import Elasticsearch, helpers
import sys, json, os
from load_data_elastic import load_file_to_elstic
from search_data_elastic import elastic_query
from operator import itemgetter
from transformers import AutoTokenizer, pipeline
import torch

import tensorflow_hub as hub
import tensorflow_text
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware

import jwt
from sqlalchemy.orm import Session 
from fastapi import HTTPException, status
from fastapi import FastAPI, Depends
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

from umap import UMAP
from hdbscan import HDBSCAN
import gc
import torch, os, json
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, TextGeneration

from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
from pathlib import Path
from PIL import Image
import joblib  # import pickle
import tensorflow as tf
from prometheus_fastapi_instrumentator import Instrumentator


# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:50"
os.environ["SUNO_USE_SMALL_MODELS"] = "True"
 
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
    ['localhost'],
    port=9200
)

path_json_files = '/home/dev/fastapi/fastapi_app/data/json_files'

app = FastAPI(
    title="Analytics App"
)
Instrumentator().instrument(app).expose(app)

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# Настройка CORS
origins = [ 
    "http://localhost",
    "http://localhost:5000",
    "http://localhost:5173",
    "http://194.146.113.123:5000",  # Добавьте ваш IP адрес
    "http://localhost:5174",
    "http://194.146.113.123",
    "https://194.146.113.123",
    "http://194.146.113.123:8000"
    "http://194.146.113.123:3000",
    "http://194.146.113.123:4000",
    "http://194.146.113.123:5173",
    "http://194.146.113.123:5175",
    "https://localhost:4000",
    "http://localhost:4000",
    "http://localhost:5175",
    "https://194.146.113.123:4000",
    "http://194.146.113.123:8080"
    "https://tellscope.headsmade.com",
    "https://tsdoc.headsmade.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

# from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
# app.add_middleware(HTTPSRedirectMiddleware)

# db

torch.cuda.empty_cache() 
gc.collect()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

# load LLM
# os.chdir('/home/dev/fastapi/fastapi_app/data/LLM_models')

# model = "gemma-2b-it"
# tokenizer = AutoTokenizer.from_pretrained(model)
# pipeline = pipeline(
#     "text-generation",
#     model=model,
#     model_kwargs={"torch_dtype": torch.bfloat16},
#     device="cuda",
# ) 
 

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

# ModelAiAnalytics
class ModelAiAnalyticsItem(BaseModel):
    id: int
    timeCreate: int
    text: str
    hub: str
    audienceCount: Optional[Union[int, str]] = None
    commentsCount: Optional[Union[int, str]] = None
    er: Optional[float] = None  # Предположим, что это число с плавающей точкой
    url: str

class ModelAiAnalytics(BaseModel):
    data: List[ModelAiAnalyticsItem]


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
def save_dict_to_pickle(file_path, data_dict):
    """
    Сохраняет словарь в файл с использованием Pickle.
    :param file_path: Путь к файлу, куда нужно сохранить словарь (str).
    :param data_dict: Словарь, который нужно сохранить (dict).
    """
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(data_dict, f)
        print(f"Словарь успешно сохранен в {file_path}.")
    except Exception as e:
        print(f"Произошла ошибка при сохранении файла: {e}")

# загрузка словаря с темами
def load_dict_from_pickle(file_name):
    """
    Загружает словарь из файла Pickle.
    :param file_name: Имя файла (str), из которого нужно загрузить словарь.
    :return: Загруженный словарь (dict) или None, если загрузка не удалась.
    """
    try:
        with open(file_name, 'rb') as f:
            your_dict = pickle.load(f)
        return your_dict
    except Exception as e:
        print(f"Произошла ошибка при загрузке файла: {e}")
        return None


@app.get("/tonality_landscape", tags=['data analytics'])
async def tonality_landscape(
    index: int = None,
    min_date: Optional[int] = None,
    max_date: Optional[int] = None
) -> Model_TonalityLandscape:
    file_path = '/home/dev/fastapi/analytics_app/data/indexes.pkl'
    indexes = load_dict_from_pickle(file_path)
    data = elastic_query(theme_index=indexes[index], min_date=min_date, max_date=max_date, query_str='all')

    print(555777999)
    print(data[:2])

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
            
            # Преобразуем потенциальные строковые значения в числа
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

    neg_authors = [entry for entry in neg if 'authorObject' in entry]
    pos_authors = [entry for entry in pos if 'authorObject' in entry]

    neg_hub_metrics = aggregate_metrics(neg + neg_authors)
    pos_hub_metrics = aggregate_metrics(pos + pos_authors)

    neg_hub_response = prepare_hub_response(neg_hub_metrics)
    pos_hub_response = prepare_hub_response(pos_hub_metrics)

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
        # Обработка полей для TextData
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
        for entry in entries:
            author_obj = process_author_object(entry)
            author_id = (author_obj['fullname'], author_obj['url'])
            groups[author_id].append(entry)

        # Теперь собираем ModeAuthorValues
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
        # Теперь отдаем в ModeAuthorValues. Строго: один элемент = одна группа author_data.
        # Т.е. на выходе: [{author_data: [1]}, {author_data: [1]}, ...]
        # (А не один с большим списком).
        for author_data in author_data_list:
            res.append(ModeAuthorValues(author_data=[author_data]))
        return res

    negative_authors_values = build_authors_groups(neg + neg_authors)
    positive_authors_values = build_authors_groups(pos + pos_authors)

    values = Model_TonalityLandscape(
        tonality_values=TonalityValues(
            negative_count=len(neg),
            positive_count=len(pos)
        ),
        tonality_hubs_values=ModelAuthorsTonalityLandscape(
            negative_hubs=neg_hub_response,
            positive_hubs=pos_hub_response
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
    file_path = '/home/dev/fastapi/analytics_app/data/indexes.pkl'
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

    print(555777999)
    print(data)
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

    # создаем словарь похожих текстов вида {12: [11, 13], 44: [190], ...}
    fin_dict = {}
    threashhold = 0.8

    # выявляем список строк с похожими текстам
    for i in range(dff.shape[0]):
        if list(np.where(dff.loc[i].values >= threashhold)[0]) != []:
            if i not in [item for sublist in list(fin_dict.values()) for item in sublist]:
                fin_dict[i] = list(
                    np.where(dff.loc[i].values >= threashhold)[0])
            else:
                fin_dict[i] = []

    df_meta.fillna('', inplace=True)

    # Проверяем и создаем недостающие столбцы для модели Pydantic
    required_columns = ['id', 'fullname', 'url', 'author_type', 'hub', 'sex', 'age',
                       'audienceCount', 'er', 'viewsCount', 'timeCreate', '_id']

    # Добавляем столбец 'id', если его нет в df_meta
    if 'id' not in df_meta.columns:
        df_meta['id'] = ''

    for col in required_columns:
        if col not in df_meta.columns:
            if col in ['audienceCount', 'er', 'viewsCount']:
                df_meta[col] = 0
            else:
                df_meta[col] = ''

    # оставляем необходимую мету
    df_meta = df_meta[required_columns]

    # получение итогового массива данных с последовательностями авторов распространения информации и репостами (похожими текстами)
    data = []

    for key, val in fin_dict.items():
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
            "es_id": author_data['_id']
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
                    "es_id": repost_data['_id']
                }

                author_dct['reposts'].append(repost_struct)

        data.append(author_dct)

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
    df['audienceCount'] = [int(x) if x != '' and x != '-' else 0 for x in df['audienceCount'].values]
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
                # Учитываем авторов в комментариях
                if 'comments' in item and isinstance(item['comments'], list):
                    for comment in item['comments']:
                        if isinstance(comment, dict) and 'author' in comment and isinstance(comment['author'], dict) and 'fullname' in comment['author']:
                            authors.add(comment['author']['fullname'])
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


# @app.get("/themes")
# async def themes_analize(user: User = Depends(current_user), index: int =None, 
#                              min_date=None, max_date=None) -> ThemesModel:
#     # Путь к файлу с темами 
#     file_path = '/home/dev/fastapi/analytics_app/data/indexes.pkl'
#     # Загрузка словаря с темами
#     indexes = load_dict_from_pickle(file_path)

#     os.chdir('/home/dev/fastapi/analytics_app/files')
#     # данные с описанием тематик
#     # filename = indexes[index] + '_LLM'
#     os.chdir('/home/dev/fastapi/analytics_app/files/Росбанк/')
#     filename = 'rosbank_01.04.2024-15.04.2024_LLM'
#     with open (filename, 'rb') as fp:
#         data = pickle.load(fp)


#     data = [x[0]['generated_text'].split('model\n')[1] if len(x) == 1 else x for x in data]
#     data = pd.DataFrame(data) 

#     # print(data)

#     query = {
#             "size": 10000,
#             "query": {
#                         "range": {
#                             "timeCreate": {      # skillfactory_zaprosy_na_obuchenie_15.01.2024-21.01.2024
#                                 "gte": min_date, # 1705329992
#                                 "lte": max_date, # 1705848392
#                                 "boost": 2.0
#                             }
#                         }
#                     }
#                 }
    
#     # данные с авторами, текстами и метаинформацией
#     # dict_train = es.search(index='skillfactory_15.01.2024-21.01.2024', body=query)
#     dict_train = es.search(index=indexes[index], body=query)
#     dict_train = dict_train['hits']['hits']
#     dict_train = [x['_source'] for x in dict_train]
    
#     # with codecs.open(indexes[index], "r", "utf_8_sig") as train_file:
#     #     dict_train = json.load(train_file)

#     columns = ['timeCreate', 'text', 'hub', 'url', 'hubtype',
#         'commentsCount', 'audienceCount',
#         'citeIndex', 'repostsCount', 'likesCount', 'er', 'viewsCount',
#         'toneMark', 'role',
#         'country', 'region', 'city', 'language', 'fullname',
#         'author_url', 'author_type', 'sex', 'age']

#     author_df = pd.DataFrame(list(pd.DataFrame(dict_train)['authorObject'].values))
#     author_df.columns=['fullname', 'author_url', 'author_type', 'sex', 'age']
#     df_res = pd.DataFrame(dict_train).join(author_df)
#     df_res = df_res[columns]
#     # df_res.columns = ['Время', 'Текст', 'Источник', 'Ссылка', 'Тип источника', 'Комментариев', 'Аудитория',
#     #        'Сайт-Индекс', 'Репостов', 'Лайков', 'Суммарная вовлеченность', 'Просмотров',
#     #        'Тональность', 'Роль', 'Страна',
#     #        'Регион', 'Город', 'Язык', 'Имя автора', 'Ссылка на автора', 'Тип автора',
#     #        'Пол', 'Возраст']

#     df_res = df_res.join(data)
#     df_res = df_res[(df_res['timeCreate'] >= int(min_date)) & (df_res['timeCreate'] <= int(max_date))]
#     df_res.reset_index(inplace=True)
#     df_res.drop('index', axis=1, inplace=True)

#     data = df_res[[0]]

#     # функция для удаления лишних символов в текстах
#     import re
#     regex = re.compile("[А-Яа-я:=!\)\()A-z\_\%/|]+")

#     def words_only(text, regex=regex):
#         try:
#             return " ".join(regex.findall(text))
#         except:
#             return ""

#     # удаляем лишние символы, оставляем слова
#     data[0] = data[0].apply(words_only)

#     # получение векторов текстов и сравнение
#     count_vectorizer = CountVectorizer()
#     vector_matrix = count_vectorizer.fit_transform(
#         data[0].values)

#     cosine_similarity_matrix = cosine_similarity(vector_matrix)
#     dff = pd.DataFrame(cosine_similarity_matrix)
#     # dff = dff.round(5)
#     # dff = dff.replace([1.000], 0)

#     val_dff = dff.values
#     # заменяем значения по главной диагонали на 0
#     for i in range(len(val_dff)):
#         val_dff[i][i] = 0
        
#     dff = pd.DataFrame(val_dff)

#     # создаем словарь похожих текстов вида {11: [12, 132],  44: [190], ...}
#     fin_dict = {}
#     threashhold = 0.70

#     # print('threashhold')

#     # выявляем список строк с похожими текстам
#     for i in range(dff.shape[0]):
#         if list(np.where(dff.loc[i].values >= threashhold)[0]) != []:
#             if i not in [item for sublist in list(fin_dict.values()) for item in sublist]:

#                 fin_dict[i] = list(
#                     np.where(dff.loc[i].values >= threashhold)[0])
                
#         else:
#             fin_dict[i] = []
            
#     len_val = [len(x) for x in fin_dict.values()]
#     dct_len_val = dict(zip(list(fin_dict.keys()), len_val))
#     # dct_len_val = dict(sorted(dct_len_val.items(), key=itemgetter(1), reverse=True))

#     # добавление текстов и метаданных в итоговый словарь
#     fin_data = []
#     texts = []
#     texts_list = data.loc[list(fin_dict.keys())][0].values # список текстов с описанием, берется первое описание по первому тексту-ключу
#     list_len = list(dct_len_val.values()) # список с количеством текстов по тематике
#     # [{'description': 'Тема текста связана с ..', 'count': 152, 'texts': [...]},
#     #  {'description': 'Тема текста связана с ..', 'count': 141, 'texts': [...]}, ..]

#     for i in range(len(fin_dict.keys())):
        
#         if fin_dict[list(fin_dict.keys())[i]] != []:

#             a = {}
#             a['description'] = texts_list[i] # описание тематики
#             a['count'] = list_len[i] # количество текстов по тематике
#             a['audience'] = str(np.sum([x['audienceCount'] for x in df_res.iloc[fin_dict[list(fin_dict.keys())[i]]].to_dict(orient='records') if x['audienceCount'] != ''])) # количество аудитории в тематике
#             a['er'] = str(np.sum([x['er'] for x in df_res.iloc[fin_dict[list(fin_dict.keys())[i]]].to_dict(orient='records') if x['er'] != ''])) # количество вовлеченности в тематику
#             a['viewsCount'] = str(np.sum([x['viewsCount'] for x in df_res.iloc[fin_dict[list(fin_dict.keys())[i]]].to_dict(orient='records') if x['viewsCount'] != '']))# количество просмотров в тематике
#             a['texts'] = 'texts' 
#             # texts.append(df_res[df_res.index.isin(fin_dict[list(fin_dict.keys())[i]])].to_dict(orient='records'))
#             fin_data.append(a)
            
#         else:
            
#             a = {}
#             a['description'] = texts_list[i] # описание тематики
#             a['count'] = list_len[i] # количество текстов по тематике
#             a['audience'] = str(np.sum([x['audienceCount'] for x in df_res.iloc[fin_dict[list(fin_dict.keys())[i]]].to_dict(orient='records') if x['audienceCount'] != ''])) # количество аудитории в тематике
#             a['er'] = str(np.sum([x['er'] for x in df_res.iloc[fin_dict[list(fin_dict.keys())[i]]].to_dict(orient='records') if x['er'] != ''])) # количество вовлеченности в тематику
#             a['viewsCount'] = str(np.sum([x['viewsCount'] for x in df_res.iloc[fin_dict[list(fin_dict.keys())[i]]].to_dict(orient='records') if x['viewsCount'] != '']))# количество просмотров в тематике
#             a['texts'] = 'texts'
#             # texts.append(df_res.iloc[[list(fin_dict.keys())[i]]].to_dict(orient='records'))
#             fin_data.append(a)
  
#     return ThemesModel(values=fin_data)


@app.get("/voice", tags=['data analytics'])
async def voice_analize(
    index: int = None,
    min_date: int = None,
    max_date: int = None,
    query_str: str = None
) -> ModelVoice:
    file_path = '/home/dev/fastapi/analytics_app/data/indexes.pkl'
    indexes = load_dict_from_pickle(file_path)
    search = query_str.split(',')
    topn = 20
    values = []

    for i in range(len(search)):
        data = elastic_query(theme_index=indexes[index], query_str=search[i])
        # Оставляем только нужные по timeCreate
        data = [x for x in data if x.get('timeCreate') is not None and min_date <= x['timeCreate'] <= max_date]

        # Заполнение недостающих полей
        for item in data:
            if 'toneMark' not in item:
                item['toneMark'] = 0
            if 'type' not in item:
                item['type'] = "other"

        search_name = search[i].strip()

        # --- Для tonality: собираем elastic_id для каждого источника и тональности
        # source_ids_by_tonality: {source: {Тональность: [id, id, ...]}}
        source_ids_by_tonality = defaultdict(lambda: defaultdict(list))
        for x in data:
            source = x['hub']
            tonality = str(x['toneMark']).replace('0', 'Нейтрал').replace('-1', 'Негатив').replace('1', 'Позитив')
            _id = x.get('_id')
            source_ids_by_tonality[source][tonality].append(_id)

        dcts = []
        for source in source_ids_by_tonality:
            dct = {
                'source': source,
                'Нейтрал': len(source_ids_by_tonality[source].get('Нейтрал', [])),
                'Позитив': len(source_ids_by_tonality[source].get('Позитив', [])),
                'Негатив': len(source_ids_by_tonality[source].get('Негатив', [])),
                'elastic_id': (
                    source_ids_by_tonality[source].get('Нейтрал', []) +
                    source_ids_by_tonality[source].get('Позитив', []) +
                    source_ids_by_tonality[source].get('Негатив', [])
                )
            }
            dcts.append(dct)
        # заполняем отсутствующие тональности и elastic_id для всех известных хабов
        all_sources = set(x['hub'] for x in data)
        for dct in dcts:
            for key in ('Нейтрал', 'Позитив', 'Негатив'):
                if key not in dct:
                    dct[key] = 0
            if 'elastic_id' not in dct:
                dct['elastic_id'] = []

        # ---- Для sunkey_data: собираем elastic_id ровно по hub/type/тональность
        # map: (hub, type, tonality) -> [id, ...]
        sunkey_map = defaultdict(list)
        for x in data:
            hub = x['hub']
            typ = x['type']
            tonality = str(x['toneMark']).replace('0', 'Нейтрал').replace('-1', 'Негатив').replace('1', 'Позитив')
            _id = x.get('_id')
            sunkey_map[(hub, typ, tonality)].append(_id)

        # Далее идет подсчет всех нужных metrics как раньше, только надо elastic_id добавить:
        hubs = Counter([x['hub'] for x in data])
        hubs = dict(sorted(hubs.items(), key=lambda x: x[1], reverse=True)[:topn])
        list_topn_hubs = list(hubs.keys())

        message_tonality_type = [
            [
                x['hub'],
                x['type'],
                str(x['toneMark']).replace('0', 'Нейтрал').replace('-1', 'Негатив').replace('1', 'Позитив'),
                x.get('commentsCount', 0),
                x.get('audienceCount', 0),
                x.get('repostsCount', 0),
                x.get('viewsCount', 0) if x.get('viewsCount') is not None and x.get('viewsCount') != '' else 0
            ]
            for x in data if x['hub'] in list_topn_hubs
        ]

        dct_tonality_hubs = Counter([', '.join(x[:3]) for x in message_tonality_type])
        hub_tonality_type_list = []
        for x in list(dct_tonality_hubs.items()):
            split_hub = x[0].split(',')
            if len(split_hub) >= 3:
                hub_val, type_val, tonality_val = [v.strip() for v in split_hub[:3]]
                # Поиск metrics
                comments_count = 0
                audience_count = 0
                reposts_count = 0
                views_count = 0
                for msg in message_tonality_type:
                    if msg[0] == hub_val and msg[1] == type_val and msg[2] == tonality_val:
                        comments_count = msg[3]
                        audience_count = msg[4]
                        reposts_count = msg[5]
                        views_count = msg[6]
                        break
                # Соберём elastic_id для этой тройки:
                elastic_ids = sunkey_map.get((hub_val, type_val, tonality_val), [])
                # Если сообщений несколько, можно возвращать или список, или один элемент. Вернуть список:
                hub_tonality_type_list.append({
                    "hub": hub_val,
                    "type": type_val,
                    "tonality": tonality_val,
                    "count": x[1],
                    "search": search_name,
                    "commentsCount": comments_count,
                    "audienceCount": audience_count,
                    "repostsCount": reposts_count,
                    "viewsCount": views_count,
                    "elastic_id": elastic_ids if len(elastic_ids) != 1 else elastic_ids[0]
                })
            else:
                print(f"Недостаточно значений для элемента: {x[0]}")

        hub_tonality_type_list = sorted(hub_tonality_type_list, key=lambda x: x['count'], reverse=True)

        # --- Собираем финальный вид
        values_search = {
            'name': search_name,
            'tonality': dcts,
            'sunkey_data': hub_tonality_type_list,
        }
        values.append(values_search)


    return ModelVoice(values=values)


@app.get("/media-rating", tags=["data analytics"])
def media_rating(index: int = None, min_date: int = None, max_date: int = None) -> MediaRatingModel:
    # 1. Загружаем словарь с темами
    file_path = "/home/dev/fastapi/analytics_app/data/indexes.pkl"
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
            smi_df = smi[[
                "timeCreate", "hub", "toneMark", "audienceCount",
                "url", "er", "hubtype", "text", "citeIndex", "_id"
            ]].copy()
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
        bobble.append({
            "name": row["hub"],
            "time": times[i],
            "index": int(row.get("citeIndex", 0)),
            "url": row["url"],
            "color": color,
            "elastic_id": id
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
async def ai_analytics_get(index: int = None, min_date: int = None, # (user: User = Depends(current_user),
                           max_date: int = None, query_str: str = None) -> ModelAiAnalytics:
    # Путь к файлу с темами 
    file_path = '/home/dev/fastapi/analytics_app/data/indexes.pkl'
    # Загрузка словаря с темами
    indexes = load_dict_from_pickle(file_path)
    
    # делаем запрос на текстовый поиск
    data = elastic_query(theme_index=indexes[index], query_str=query_str)


    # отфильтровываем по необходимой дате из календаря
    data = [x for x in data if x['timeCreate'] is not None and min_date <= x['timeCreate'] <= max_date]
    keys = ['id', 'timeCreate', 'text', 'hub', 'audienceCount', 'commentsCount', 'er', 'url']  # ключи для отображения в первой таблице
    data = [{k: y.get(k, None) for k in keys} for y in data]  # данные для первой таблицы
    ranges = list(np.arange(0, len(data)))
    [x.update({'id': y.item()}) for x, y in zip(data, ranges)]  # меняем значение id на 0,1,2...для передачи далее при выборе на LLM

    return ModelAiAnalytics(data=data)


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
    file_path = '/home/dev/fastapi/analytics_app/data/indexes.pkl'
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


@app.get("/create-data-projector/{user_id}/{folder_name}/{file_name}")
async def create_data_projector(user_id: str, folder_name: str, file_name: str, user: User = Depends(current_user)):
    # Путь к файлу с темами 
    file_path = '/home/dev/fastapi/analytics_app/data/indexes.pkl'
    indexes = load_dict_from_pickle(file_path)

    # Отключаем использование GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    embed = hub.load("/home/dev/fastapi/analytics_app/data/embed_files/universal-sentence-encoder-multilingual_3")

    # Полный путь к файлу
    file_path = f'/home/dev/fastapi/analytics_app/data/{user_id}/json_files_directory/{folder_name}/{file_name}' + '.json'

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
    
    regex = re.compile("[А-Яа-я:=!\)\()A-z\_\%/|]+")

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
        text = re.sub('((www\[^\s]+)|(https?://[^\s]+))', 'URL', text)
        text = re.sub('@[^\s]+', 'USER', text)
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
    df_text['text'] = df_text['text'].apply(remove_stopwords)
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
    project_files_dir = f'/home/dev/fastapi/analytics_app/data/{user_id}/projector_files_directory/{folder_name}/'
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
    # Логируем параметры запроса для отладки
    print(f"Received request with parameters: user_id={user_id}, file_type={file_type}, folder_name={folder_name}, file_name={file_name}")

    BASE_DIR = '/home/dev/fastapi/analytics_app/data'
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
    os.chdir('/home/dev/fastapi/analytics_app/data')
    
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
HISTORY_FILE = '/home/dev/fastapi/analytics_app/data/llm_history_progress.pickle'

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
from run_llm_query_new import run_llm_query
# from test_interactive_embed import run_llm_query

import uuid
import asyncio 
# import redis  # redis-py >= 4.x (или 5.x)
import traceback
import redis.asyncio

@app.on_event("startup")
async def startup_event():
    try:
        await redis_db.ping()
        logging.info("Redis подключен!")
        # Инициализируем статус GPU при старте
        existing_status = await redis_db.get("gpu:status")
        if not existing_status:
            logging.info("Инициализация статуса GPU как 'idle'.")
            await redis_db.set("gpu:status", "idle")
    except Exception as e:
        logging.error(f"Ошибка подключения к Redis: {e}")
        raise RuntimeError("Не удалось подключиться к Redis")


@app.on_event("shutdown")
async def shutdown_event():
    await redis_db.close()


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
            "bad_request": "0"
        }

        await redis_db.hset(f"task:{task_id}", mapping=task_data)
        await redis_db.rpush("queue:tasks", task_id)

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
                await redis_db.hset(f"task:{task_id}", "status", "in_progress")

                # Выполнение обработки
                await run_llm_query(task_data)

                # Отмечаем задачу как завершенную
                await redis_db.hset(f"task:{task_id}", "status", "done")
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

    print(f"task_data type: {type(task_data)}, content: {task_data}")

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
    # Ищем файл по указанному имени
    for file_info in html_files:
        if file_info["html-file"] == file_name:
            info_html = file_info
            html_file_path = os.path.join("/home/dev/fastapi/analytics_app/data", str(user_id), 
                                           "bertopic_files_directory", folder_name, file_name)
            break

    if html_file_path is None or not os.path.exists(html_file_path):
        raise HTTPException(status_code=404, detail="HTML file not found")

    # Определяем базовое имя модели без расширения
    model_file_name_base = file_name.replace('.html', '').split('_')[-1]  # Извлекаем идентификатор из имени

    # Теперь ищем нужный модельный файл
    model_folder_name = None
    for file_info in html_files:
        if model_file_name_base in file_info["model-file"]:
            model_folder_name = folder_name  # Используем текущее имя папки
            break

    if model_folder_name is None:
        raise HTTPException(status_code=404, detail="Model folder not found")

    # Создаем путь к модели
    model_path = os.path.join("/home/dev/fastapi/analytics_app/data", str(user_id), 
                               "bertopic_files_directory", model_folder_name, 
                               next(file_info["model-file"] for file_info in html_files if model_file_name_base in file_info["model-file"]))

    # Проверяем, существует ли путь к модели
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model file not found")

    # Модель BERTopic
    topic_model = BERTopic.load(model_path)

    # Поиск в elastic за те же даты и строку поиска
    # Путь к файлу с темами 
    file_path = '/home/dev/fastapi/analytics_app/data/indexes.pkl'
    # Загрузка словаря с темами
    indexes = load_dict_from_pickle(file_path)
    
    # делаем запрос на текстовый поиск
    if info_html['query_str'] is None:
        info_html['query_str'] = 'all'

    if 'min_data' not in info_html:
        data = elastic_query(theme_index=indexes[info_html['index_number']], query_str=info_html['query_str'], 
                            min_date=info_html['min_date'], max_date=info_html['max_date'])
    else:
        data = elastic_query(theme_index=indexes[info_html['index_number']], query_str=info_html['query_str'], 
                             min_date=info_html['min_data'], max_date=info_html['max_data'])
    data = pd.DataFrame(data)

    print(555777999)
    print(data.head())

    # Обработка тематики
    df_topic = topic_model.get_topic_info()[['CustomName', 'Topic', 'Count']]
    dct_df_topic = dict(zip(df_topic['Topic'], df_topic['CustomName']))
    thematics = [dct_df_topic[x] for x in topic_model.topics_]
    thematics_count = dict(zip(df_topic['CustomName'], df_topic['Count']))

    # Обработка в зависимости от состава данных
    # Проверяем, содержит ли data только ограниченный набор полей
    limited_fields = ['id', 'text', 'timeCreate', 'hub', 'city', 'audienceCount', 'url']
    is_limited_data = all(col in data.columns for col in limited_fields) and len(data.columns) <= len(limited_fields) + 3  # +3 с запасом на технические поля
    
    if is_limited_data:
        # Создаем необходимые поля с значениями по умолчанию
        data['text_url'] = data.get('url', '')
        data['author_url'] = ''
        data['fullname'] = data.get('hub', '')  # hub в качестве имени автора
        data['author_type'] = ''
        data['sex'] = ''
        data['age'] = ''
        data['hubtype'] = 'Онлайн-СМИ'
        data['commentsCount'] = 0
        data['repostsCount'] = 0
        data['likesCount'] = 0
        data['er'] = 0
        data['viewsCount'] = 0
        data['toneMark'] = 0  # Нейтральная тональность
        data['country'] = ''
        data['region'] = data.get('city', '')  # city в качестве региона
    else:
        # Оригинальная логика обработки
        if 'authorObject' in data.columns:
            data.rename(columns={'url': 'text_url'}, inplace=True)
            data = data.join(pd.DataFrame(list(data['authorObject'].values)))
            data.rename(columns={'url': 'author_url'}, inplace=True)
        else:
            # Если authorObject отсутствует, создаем пустые поля
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
                data[column] = 0  # Нейтральная тональность
            elif column == 'audienceCount' and 'citeIndex' in data.columns:
                data[column] = data['citeIndex']
            elif column == 'region' and 'city' in data.columns:
                data[column] = data['city']
            else:
                data[column] = ''
    
    # Выбираем только необходимые колонки в нужном порядке
    data = data[['timeCreate', 'hub', 'author_url', 'fullname', 'text_url', 'author_type', 'sex', 'age',
                 'hubtype', 'commentsCount', 'audienceCount', 'repostsCount', 'likesCount', 
                 'er', 'viewsCount', 'toneMark', 'country', 'region']]

    print(333555777)
    print(data.columns)

    # Получение полной таблицы
    df_join = pd.DataFrame(thematics).join(data, how='inner', lsuffix='_df1', rsuffix='_df2')
    df_join.columns = ['Имя кластера', 'Время', 'Источник', 'Ссылка на автора', 'Автор', 'Ссылка на текст', 'Тип автора', 'Пол', 'Возраст',
                       'Тип источника', 'Комментариев', 'Аудитория', 'Репостов', 'Лайков', 'Вовлеченность', 'Просмотров',
                       'Тональность', 'Страна', 'Регион']
    


    # Добавление уникального идентификатора
    df_join.reset_index(drop=True, inplace=True)  # Сбрасываем индекс, чтобы не было старых индексов
    df_join.insert(0, 'id', df_join.index)  # Добавляем новый столбец в качестве идентификатора

    df_join.columns = ['ID', 'Имя кластера', 'Время', 'Источник', 'Ссылка на автора', 'Автор', 'Ссылка на текст', 
                    'Тип автора', 'Пол', 'Возраст', 'Тип источника', 'Комментариев', 'Аудитория', 
                    'Репостов', 'Лайков', 'Вовлеченность', 'Просмотров', 'Тональность', 
                    'Страна', 'Регион']
    


    # Преобразуем значения тональности
    df_join['Тональность'] = df_join['Тональность'].map({0: 'Нейтральная', -1: 'Негатив', 1: 'Позитив'})

    print(777555333)
    print(df_join.columns)
    print(df_join.head())
    
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
    
    # Группировка по 'Тематика' и суммирование
    result = df_group.groupby('Имя кластера').sum().reset_index()

    # Подсчет количества тем
    theme_count = result['Имя кластера'].value_counts()
    result['Количество'] = result['Имя кластера'].map(thematics_count)
    result.sort_values(by='Количество', ascending=False, inplace=True)
    result = result[['Имя кластера', 'Количество', 'Аудитория', 'Комментариев', 'Репостов', 'Лайков', 'Вовлеченность', 'Просмотров']]

    # Замена NaN на None в итоговых данных
    result = result.where(pd.notnull(result), None)

    # Подгружаем данные с тематикой по каждому тексту
    texts_path = os.path.join("/home/dev/fastapi/analytics_app/data", str(user_id), 
                                "bertopic_files_directory", model_folder_name)
    # Получаем список файлов в директории
    files = os.listdir(texts_path) 

    # Находим файл, в имени которых содержится выбранный html
    file = [file for file in files if file_name.replace('.html', '') in file]
    file = file[0].replace('topic_model_', 'my_list_llm_ans_')

    thematics_path = texts_path + '/' + 'my_list_llm_ans_' + file.replace('.html', '.pkl').replace('topic_names_', '')

    try:
        with open(thematics_path.replace('_datamapplot', '').replace('my_list_llm_ans_my_list_llm_ans_', 'my_list_llm_ans_'), 'rb') as f:
            texts_thematics = pickle.load(f)
        df_join.insert(1, 'Тематика текста', texts_thematics)
    except:
        try:
            with open(thematics_path.replace('_datamapplot', '').replace('my_list_llm_ans_my_list_llm_ans_', 'my_list_llm_ans_') + '.pkl', 'rb') as f:
                texts_thematics = pickle.load(f)
            df_join.insert(1, 'Тематика текста', texts_thematics)
        except:
            print("Не удалось загрузить тематики текстов, добавляем пустые значения")
            df_join.insert(1, 'Тематика текста', [''] * len(df_join))

    # Полный путь к выходному файлу
    output_path = os.path.join('/home/dev/fastapi/analytics_app/files/', 'df_join.xlsx')

    # Сохранение DataFrame в Excel
    df_join.to_excel(output_path, index=False)  # index=False, если не нужно сохранять индексы

    # Сохранение полных данных и агрегированных данных в Redis
    await redis_db.hset(str(user_id), "full_data", json.dumps(df_join.where(pd.notnull(df_join), None).to_dict(orient='records')))
    await redis_db.hset(str(user_id), "aggregated_data", json.dumps(result.where(pd.notnull(result), None).to_dict(orient='records')))

    # Возвращение HTML файла и таблиц
    with open(html_file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()
    print(111)
    if html_content is None or not os.path.exists(html_file_path):
        raise HTTPException(status_code=404, detail="HTML file not found")
    print(333)
    # Возвращение HTML файла dataplot
    html_file_path_dataplot = os.path.join("/home/dev/fastapi/analytics_app/data", str(user_id), 
                                "bertopic_files_directory", folder_name, 'datamapplot_' + file_name)
    print(555)
    with open(html_file_path_dataplot, 'r', encoding='utf-8') as file:
        html_file_path_dataplot = file.read()
    print(777)
    if html_file_path_dataplot is None:
        raise HTTPException(status_code=404, detail="HTML file dataplot not found")
    print(df_join.values[:2])
    
    return {
        "html_content": html_content,
        "html_content_dataplot": html_file_path_dataplot, 
        "full_data": json.loads(df_join.replace([np.nan, np.inf, -np.inf], None).to_json(orient='records')),  # Замена NaN на None
        "aggregated_data": json.loads(result.replace([np.nan, np.inf, -np.inf], None).to_json(orient='records'))  # Замена NaN на None
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
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id: str = payload.get("sub")

    # Здесь мы предполагаем, что определена ORM модель User
    # query = select(User).where(User.id == int(user_id)).options(selectinload(User.role))
    # result = await db.execute(query)
    # user_record = result.scalars().first()  # Используйте first() для получения единственного объекта 
    
    # if user_record is None:
    #     raise credentials_exception

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

    # Генерация access-токена через JWTStrategy
    access_token = await auth_backend.get_strategy().write_token(user)
    
    # Генерация refresh-токена через свою стратегию
    refresh_strategy = get_refresh_strategy()
    refresh_token = await refresh_strategy.write_token(user)
    
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

    os.chdir('/home/dev/fastapi/analytics_app/data')
    
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
    # Путь до директории json_files
    json_files_directory = f"/home/dev/fastapi/analytics_app/data/{user_id}/json_files_directory"
    # Путь, где будет создана папка
    storage_path = f"{json_files_directory}/{folder_name}"

    # Проверяем, существует ли директория json_files_directory, если нет - создаем её
    if not os.path.exists(json_files_directory):
        os.makedirs(json_files_directory)

    # Проверяем, существует ли уже папка
    if os.path.exists(storage_path):
        raise HTTPException(status_code=400, detail="Папка с таким именем уже существует.")

    # Создаём папку
    os.makedirs(storage_path)

    # Получаем текущее состояние папок в Redis
    user_data = await redis_db.hget(user_id, "json_files_directory")
    if user_data is None:
        user_folders = {}
    else:
        user_folders = json.loads(user_data)

    # Добавляем новую папку в структуру
    if folder_name not in user_folders:
        user_folders[folder_name] = []

    # Сохраняем обновлённую структуру в Redis
    await redis_db.hset(user_id, "json_files_directory", json.dumps(user_folders))

    return f"Папка {folder_name} у пользователя {user_id} создана!"


# from files_MLG_KRIBRUM import load_medialogia_excel
# from files_MLG_KRIBRUM import load_file_to_elastic

from celery import Celery
import os
import pandas as pd
from models import tasks, processing_results

# Настройка Celery
celery_app = Celery('tasks', broker='redis://localhost:6379/0')

from tasks import process_file_task


@app.post("/add-file/{user_id}/{folder_name}", tags=["data & folders"])
async def add_file(user_id: str, folder_name: str, uploaded_file: UploadFile = File(...)):
    if not folder_name:
        raise HTTPException(status_code=400, detail="Необходимо указать имя папки")

    # Путь к индексу файлов и получение индексов
    file_path = '/home/dev/fastapi/analytics_app/data/indexes.pkl'
    indexes = load_dict_from_pickle(file_path)

    original_filename = uploaded_file.filename.lower()
    file_extension = os.path.splitext(original_filename)[1]
    json_filename = original_filename
    if file_extension in ('.xlsx', '.xls'):
        json_filename = original_filename.replace(file_extension, '.json')

    next_key = max(indexes.keys()) + 1 if indexes else 1
    formatted_value = json_filename.replace('.json', '').lower()
    indexes[next_key] = formatted_value
    save_dict_to_pickle(file_path, indexes)

    max_file_size_admin = 10 * 1024 * 1024 * 1024 # 10 GB
    max_file_size_non_admin = 100 * 1024 * 1024 # 100 MB
    size = uploaded_file.size if hasattr(uploaded_file, 'size') else 0

    if user_id in ('1', '13'):
        if size > max_file_size_admin:
            raise HTTPException(status_code=400, detail="Размер файла превышает 10 ГБ")
    else:
        if size > max_file_size_non_admin:
            raise HTTPException(status_code=400, detail="Размер файла превышает 100 МБ")

    # Директория для файла
    file_location = f'/home/dev/fastapi/analytics_app/data/{user_id}/json_files_directory/{folder_name}/{json_filename}'
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
    user_folders_data = await redis_db.hget(user_id, "json_files_directory")
    user_folders = {}
    if user_folders_data:
        try:
            user_folders = json.loads(user_folders_data.decode("utf-8"))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Ошибка при загрузке данных из Redis: {str(e)}")

    current_files = user_folders.get(folder_name, [])
    if json_filename in current_files:
        current_files = [f for f in current_files if f != json_filename]
    current_files.append(json_filename)
    user_folders[folder_name] = current_files
    await redis_db.hset(user_id, "json_files_directory", json.dumps(user_folders))
        
    return f"Файл {uploaded_file.filename} загружен в папку {folder_name} пользователя - {user_id}!"
    #     {
    #     "message": f"Файл {original_filename} добавлен в очередь обработки. Статус обработки можно отслеживать по task_id.",
    #     "task_id": task_id,
    #     "status": "processing"
    # }

# from celery.result import AsyncResult
# from tasks import app as celery_app

# Удаление папки
@app.delete("/delete-folder/{user_id}/{directory_type}/{folder_name}", tags=['data & folders'])
async def delete_folder(user_id: str, directory_type: str, folder_name: str):
    # Получаем текущее содержимое для указанного пользователя
    json_folders = await redis_db.hget(user_id, directory_type)
    
    # Если данных для данного user_id нет, возвращаем ошибку
    if json_folders is None:
        raise HTTPException(status_code=404, detail="Директории не найдены для данного пользователя.")

    # Декодируем JSON данные в словарь
    folders_dict = json.loads(json_folders)

    # Проверяем наличие запрашиваемой папки
    if folder_name not in folders_dict:
        raise HTTPException(status_code=404, detail="Запрашиваемая папка не найдена.")

    # Получаем список файлов, относящихся к этой папке
    files_in_directory = folders_dict[folder_name]

    # Удаляем папку из Redis
    del folders_dict[folder_name]  # Удаляем папку из словаря
    await redis_db.hset(user_id, directory_type, json.dumps(folders_dict))  # Обновляем данные в Redis

    # Получаем список всех индексов для удаления из Elasticsearch
    es_indexes = [index for index in es.indices.get('*')]
    
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
    folder_path = f"/home/dev/fastapi/analytics_app/data/{user_id}/{directory_type}/{folder_name}"

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
async def delete_file(user_id: str, directory_type: str, directory_name: str, file_name: str):
    # Получаем директории для указанного user_id
    folders = await redis_db.hgetall(user_id)
    # Преобразуем байтовые строки в обычные строки и десериализуем JSON
    folders = {key.decode('utf-8'): json.loads(value.decode('utf-8')) for key, value in folders.items()}

    # Проверяем, есть ли директории для данного пользователя
    if not folders:
        raise HTTPException(status_code=404, detail="Директории не найдены для данного пользователя.")

    # Определяем путь к директории файлов на диске
    folder_path = f"/home/dev/fastapi/analytics_app/data/{user_id}/{directory_type}/{directory_name}"

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
                    Вход: '/home/dev/fastapi/analytics_app/data/123/projector/folder/geekbrains_08.12.2024-07.01.2025_authors_point_2025-01-10_09-09-48.tsv'
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
#     json_files_directory = f"/home/dev/fastapi/analytics_app/data/{user_id}/json_files_directory"
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
#     file_directory = f'/home/dev/fastapi/analytics_app/data/{user_id}/json_files_directory/{folder_name}'
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


# Эндпойнт получения папок и файлов для пользователя с данными из Elasticsearch
@app.get("/user-folders/{user_id}", tags=['data & folders'])
async def get_user_folders(user_id: str):
    # Проверяем, существует ли пользователь в БД
    user = get_user_profile(user_id)
    
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Путь к файлу с темами 
    file_path = '/home/dev/fastapi/analytics_app/data/indexes.pkl'
    # Загрузка словаря с темами
    indexes = load_dict_from_pickle(file_path)
    
    # Получаем папки пользователя из Redis
    folders = await redis_db.hgetall(user_id)  # Используем await для асинхронного вызова

    if not folders:
        return {"user_id": user_id, "json_files_directory": {}, "bertopic_files_directory": {}}
    
    # Преобразуем данные из Redis в формат JSON
    formatted_folders = {folder.decode('utf-8'): json.loads(files) for folder, files in folders.items()}

    # Получение данных из Elasticsearch
    es_indexes = [index for index in es.indices.get('*')]  # список всех индексов elastic
    es_indexes = [x.strip() for x in es_indexes]

    # Запрос для поиска мин и макс дат в данных/файлах
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

    # Создаем новый формат результата
    json_folders = {}

    # Проходим по всем именам папок в formatted_folders
    for folder_name in formatted_folders['json_files_directory'].keys():
        # Инициализируем ключ с пустым списком для каждой папки
        json_folders[folder_name] = []

    # Теперь обрабатываем файлы для каждой папки
    for folder_name, files in formatted_folders['json_files_directory'].items():
        for file_name in files:
            file_name = file_name.replace('.json', '').lower()

            # Проверяем, существует ли индекс для файла
            if file_name in es_indexes:
                date_period_query = es.search(index=file_name, body=query)['aggregations']

                json_folders[folder_name].append(
                    {
                        "file": file_name,
                        "min_data": date_period_query['min_timeCreate']['value'],
                        "max_data": date_period_query['max_timeCreate']['value'],
                        "index_number": list({i for i in indexes if indexes[i] == file_name})[0]
                    }
                )

    # Получаем папки пользователя из Redis для bertopic
    bertopic_folders = await redis_db.hget(user_id, "bertopic_files_directory")  # Добавлено await
    
    # Если данные существуют и не пустые, обрабатываем их
    if bertopic_folders is not None:
        # Преобразуем данные из Redis в формат JSON
        try:
            # Поскольку redis_db.hget возвращает строку, нужно загрузить ее как JSON
            bertopic_folders = json.loads(bertopic_folders)
            # Преобразование в словарь, если требуется
            bertopic_folders = {folder: files for folder, files in bertopic_folders.items()}
        except json.JSONDecodeError:
            # Обработка случая, когда данные не валидные JSON
            bertopic_folders = {}

    else:
        bertopic_folders = {}

    # Получаем папки пользователя из Redis для projector
    projector_folders = await redis_db.hget(user_id, "projector_files_directory")  # Добавлено await
    
    # Если данные существуют и не пустые, обрабатываем их
    if projector_folders is not None:
        # Преобразуем данные из Redis в формат JSON
        try:
            # Поскольку redis_db.hget возвращает строку, нужно загрузить ее как JSON
            projector_folders = json.loads(projector_folders)
            # Преобразование в словарь, если требуется
            projector_folders = {folder: files for folder, files in projector_folders.items()}
        except json.JSONDecodeError:
            # Обработка случая, когда данные не валидные JSON
            projector_folders = {}

    else:
        projector_folders = {}

    return {
        "user_id": user_id,
        "json_files_directory": json_folders,
        "bertopic_files_directory": bertopic_folders,
        "projector_files_directory": projector_folders,
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
    
# # Обработка LLM задачи для одного текста
# @app.post("/llm-run-single/", tags=['ai analytics'])
# async def llm_run(
#     analysis_request: SingleTextRequest,
#     background_tasks: BackgroundTasks
# ):
#     try:
#         task_id = str(uuid.uuid4())
#         # Создаем task_data на основе входной модели `AnalysisRequest`
#         # Преобразуем данные в строковый формат перед сохранением
#         task_data = {
#             "task_id": task_id,
#             "user_id": str(analysis_request.user_id),
#             # "folder_name": str(analysis_request.folder_name),
#             "text": str(analysis_request.text),
#             "system_prompt": str(analysis_request.system_prompt) if analysis_request.system_prompt else "",
#             "prompt_question": str(analysis_request.prompt_question),
#             "status": "pending"
#         }

#         # Сохраняем задачу в Redis для логирования
#         await redis_db.hset(f"task:{task_id}", mapping=task_data)

#         # Добавляем задачу в очередь на выполнение
#         background_tasks.add_task(process_single_text_task, task_id, task_data)

#         return JSONResponse(
#             content={
#                 "task_id": task_id,
#                 "status": "pending",
#                 "message": "Task has been added to the queue."
#             },
#             status_code=202
#         )
#     except Exception as e:
#         logging.error(f"Error in llm_run_single: {e}")
#         return JSONResponse(
#             content={
#                 "error": str(e)
#             },
#             status_code=500
#         )
        

async def process_single_text_task(task_id: str, task_data: dict):
    try:
        # Получаем текст запроса
        text = task_data['text']
        prompt_question = task_data['prompt_question']

        # Формируем запрос к LLM
        result = await generate_answers(text, prompt_question)

        # Сохраняем результат обработки LLM в Redis
        await redis_db.hset(f"task:{task_id}", "result", result)

        # Если необходимо, обновляем статус задачи как завершенную
        await redis_db.hset(f"task:{task_id}", "status", "done")
    except Exception as e:
        logging.error(f"Ошибка при обработке задачи {task_id}: {e}")
        await redis_db.hset(f"task:{task_id}", "status", f"failed: {str(e)}")


async def generate_answers(text: str, prompt_question: str):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "erwan2/DeepSeek-R1-Distill-Qwen-14B",
        "prompt": f"{text}\n\n{prompt_question}",
        "stream": False
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as response:
            if response.status == 200:
                response_json = await response.json()
                # Обработка ответа от LLM
                result = response_json.get("response", "")
                print(f"Ответ LLM: {result}")
                result = result.split('</think>')[1].replace('\n\n', '').replace('\n', '')
                return result
            else:
                print(f"Ошибка при запросе к LLM: {response.status}")


class MultipleTextRequest(BaseModel):
    user_id: int
    texts: List[str]
    system_prompt: Optional[str] = None
    prompt_question: str

async def process_text(text: str, question: str, system_prompt: Optional[str]) -> str:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post("URL_Ваша_LLM_API", json={
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

@app.post("/llm-run-multiple/", tags=['ai analytics'])
async def llm_run_multiple(
    analysis_request: MultipleTextRequest,
    background_tasks: BackgroundTasks
):
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


async def process_multiple_texts_task(task_id: str, task_data: dict):
    try:
        texts = task_data['texts']
        prompt_question = task_data['prompt_question']
        system_prompt = task_data.get('system_prompt')
        
        total_texts = len(texts)
        results = []
        
        for i, text in enumerate(texts):
            # Обработка текста
            result = await generate_answer(text, prompt_question, system_prompt)
            results.append(result)
            
            # Обновление процента выполнения
            progress = int(((i + 1) / total_texts) * 100)
            await redis_db.hset(f"task:{task_id}", mapping={
                "completed_texts": str(i + 1),
                "progress": str(progress),
                "status": "processing"  # Задача всё ещё в процессе
            })
        
        # Сериализуем и сохраняем результаты
        json_results = json.dumps(results, ensure_ascii=False)
        json_texts = json.dumps(texts, ensure_ascii=False)
        
        # Обновляем статус как завершенный
        await redis_db.hset(f"task:{task_id}", mapping={
            'texts': json_texts,
            'results': json_results,
            "progress": "100",
            "status": "done"
        })
        
    except Exception as e:
        logging.error(f"Error processing task {task_id}: {str(e)}", exc_info=True)
        await redis_db.hset(f"task:{task_id}", "status", f"failed: {str(e)}")


async def generate_answer(text: str, prompt_question: str, system_prompt: Optional[str]):
    url = "http://localhost:11434/api/generate"
    payload = {
        # "model": "erwan2/DeepSeek-R1-Distill-Qwen-14B",
        "model": "Vikhr_Q3",
        "prompt": f"{system_prompt}\n\n{text}\n\n{prompt_question}" if system_prompt else f"{text}\n\n{prompt_question}",
        "stream": False
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as response:
            if response.status == 200:
                response_json = await response.json()
                result = response_json.get("response", "")
                logging.info(f"LLM response: {result}")
                # result = result.split('</think>')[1].replace('\n\n', '').replace('\n', '') # DeepSeek-R1-Distill-Qwen-14B
                return result
            else:
                logging.error(f"Error calling LLM API: {response.status}")
                return ""


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
    file_path = f"/home/dev/fastapi/analytics_app/data/{user_id}/bertopic_files_directory/{folder_name}/{file_name}"
    print(f"Loading texts from: {file_path}")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found.")
    with open(file_path, 'rb') as f:
        texts = pickle.load(f)  # Файл предполагается, что содержит список текстов
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
            html_file_path = os.path.join("/home/dev/fastapi/analytics_app/data", str(user_id),
                                          "bertopic_files_directory", folder_name, file_name)
            break

    if html_file_path is None or not os.path.exists(html_file_path):
        raise HTTPException(status_code=404, detail="HTML file not found")

    # Выполнение запроса в elasticsearch за указанный диапазон дат и с нужной строкой поиска
    file_path_indexes = '/home/dev/fastapi/analytics_app/data/indexes.pkl'
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

    df_join.to_excel('/home/dev/fastapi/analytics_app/data/1/cluster_fobii.xlsx', index=False, engine='openpyxl')

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
embedding_model = SentenceTransformer("/home/dev/fastapi/analytics_app/data/embed_files/DeepPavlov/rubert-base-cased-sentence")

async def generate_answers(client, prompt):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "erwan2/DeepSeek-R1-Distill-Qwen-14B",  # Vikhr_Q3
        "prompt": prompt,
        "stream": False
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as response:
            if response.status == 200:
                response_json = await response.json()
                return response_json.get("response", "")
            else:
                print(f"Ошибка при запросе к Ollama: {response.status}")
                return None

class QueryRequest(BaseModel):
    query: str
    user_id: int
    filename: str
    folder_name: str
    num_results: int = 5
    generate_answer: bool = True

from ollama import AsyncClient
# Создаём клиент один раз
client = AsyncClient(host='http://localhost:11434')

@app.post("/rag", tags=['ai analytics'])
async def rag_query(request: QueryRequest, session: AsyncSession = Depends(get_db)):
    try:
        user_query = request.query
        user_id = request.user_id
        filename = request.filename
        folder_name = request.folder_name
        num_results = request.num_results
        generate_answer = request.generate_answer

        # Получение информации из Redis
        user_data = await redis_db.hgetall(user_id)
        user_data = {key.decode('utf-8'): value.decode('utf-8') for key, value in user_data.items()}
        # Декодируем JSON-значения в словари
        for key, value in user_data.items():
            try:
                user_data[key] = json.loads(value)
            except json.JSONDecodeError:
                print(f"Ошибка декодирования JSON для ключа {key}: {value}")

        def extract_relevant_part(filename):
            # Разделяем строку на части по символу '_'
            parts = filename.split('_')
            # Объединяем все части до последнего подчеркивания
            relevant_part = '_'.join(parts[:-2])  # исключаем последние две части
            return relevant_part
        
        # Поиск нужной информации в bertopic_files_directory
        theme_index = None
        min_date = None
        max_date = None
        query_str = None
        for item in user_data["bertopic_files_directory"][folder_name]:
            print(111555)
            if item["html-file"] == filename:
                print(item)
                theme_index = extract_relevant_part(filename)
                print(555999777)
                print(theme_index)
                if "min_date" in item:
                    min_date = item["min_date"]
                    max_date = item["max_date"]
                else:
                    min_date = item["min_data"]
                    max_date = item["max_data"]
                query_str = item["query_str"]
                break
        
        if theme_index is None:
            raise HTTPException(status_code=404, detail="Файл не найден")
        
        # Получение текстов из Elasticsearch
        data = elastic_query(theme_index=theme_index, min_date=min_date, max_date=max_date, query_str=query_str)
        texts = [x['text'] for x in data]

        # Создание эмбеддинга для запроса пользователя
        query_embedding = embedding_model.encode(user_query, show_progress_bar=False)

        # Извлечение эмбеддингов из базы данных с учетом user_id, filename и folder_name
        query = select(Embedding).where(
            Embedding.user_id == user_id,
            Embedding.filename == filename,
            Embedding.folder_name == folder_name
        )
        result = await session.execute(query)
        embeddings = result.scalars().all()

        if not embeddings:
            raise HTTPException(status_code=404, detail="Эмбеддинги не найдены")
        
        # Расчет косинусного сходства между запросом и эмбеддингами
        query_embedding = list(query_embedding)  # Преобразование в одномерный список
        user_embeddings = [emb.vectors for emb in embeddings][0]  # Преобразование каждого вектора в одномерный список

        print(f'len_user_embeddings: {len(user_embeddings)}')

        # similarities = cosine_similarity([query_embedding], user_embeddings)[0]

        query_embedding_reshaped = np.array(query_embedding).reshape(1, -1)  # Преобразование в двумерный массив для одного запроса
        user_embeddings_reshaped = np.array(user_embeddings)  # Двумерный массив эмбеддингов пользователей

        similarities = cosine_similarity(query_embedding_reshaped, user_embeddings_reshaped)[0]
        # print(similarities)

        # Получение индексов наиболее релевантных эмбеддингов
        # top_indices = similarities.argsort()[-num_results:][::-1]
        top_indices = np.argpartition(similarities, -num_results)[-num_results:]
        print(top_indices)
        # print(similarities.argsort())
        # print(f'top_indices: {top_indices}')

        # Получение наиболее релевантных текстов
        top_texts = [texts[i] for i in top_indices]
        # print(top_texts)

        if generate_answer:
            # Генерация ответа с использованием модели генерации текста
            prompt = f"Query: {user_query}\nContext: {' '.join([texts[i] for i in top_indices])}\nAnswer:"  # Здесь берем тексты по индексам
            answer = await generate_answers(client=client, prompt=prompt)
            
            return {"answer": answer, "top_texts": top_texts}
        else:
            return {"top_texts": top_texts}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
            html_file_path = os.path.join("/home/dev/fastapi/analytics_app/data", str(user_id), 
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
    model_path = os.path.join("/home/dev/fastapi/analytics_app/data", str(user_id), 
                               "bertopic_files_directory", model_folder_name, 
                               next(file_info["model-file"] for file_info in html_files if model_file_name_base in file_info["model-file"]))

    # Проверяем, существует ли путь к модели
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model file not found")

    # Модель BERTopic
    topic_model = BERTopic.load(model_path)

    # Поиск в elastic за те же даты и строку поиска
    file_path = '/home/dev/fastapi/analytics_app/data/indexes.pkl'
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

    result = df_group.groupby('Имя кластера').sum().reset_index()

    theme_count = result['Имя кластера'].value_counts()
    result['Количество'] = result['Имя кластера'].map(theme_count)
    result.sort_values(by='Количество', ascending=False, inplace=True)
    result = result[['Имя кластера', 'Количество', 'Аудитория', 'Комментариев', 'Репостов', 'Лайков', 'Вовлеченность', 'Просмотров']]

    result = result.where(pd.notnull(result), None)

    texts_path = os.path.join("/home/dev/fastapi/analytics_app/data", str(user_id), 
                                "bertopic_files_directory", model_folder_name)
    files = os.listdir(texts_path)

    file = [file for file in files if file_name.replace('.html', '') in file][0]
    thematics_path = texts_path + '/' + 'my_list_llm_ans_' + file.replace('.html', '.pkl').replace('topic_names_', '')
    
    with open(thematics_path.replace('_datamapplot', ''), 'rb') as f:
        texts_thematics = pickle.load(f)
    df_join.insert(1, 'Тематика текста', texts_thematics)

    output_path = os.path.join('/home/dev/fastapi/analytics_app/files/', (file_name.capitalize() + 'aggregated_table.xlsx').replace('.html', '_') 
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
#     embedding_model = SentenceTransformer("/home/dev/fastapi/analytics_app/data/embed_files/DeepPavlov/rubert-base-cased-sentence")

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
#     embedding_model = SentenceTransformer("/home/dev/fastapi/analytics_app/data/embed_files/DeepPavlov/rubert-base-cased-sentence")

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
    
    file_path = '/home/dev/fastapi/analytics_app/data/indexes.pkl'
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
            html_file_path = os.path.join("/home/dev/fastapi/analytics_app/data", str(user_id), 
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
    embedding_model = SentenceTransformer("/home/dev/fastapi/analytics_app/data/embed_files/DeepPavlov/rubert-base-cased-sentence")

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


from openai import OpenAI

client = OpenAI(
    api_key="sk-aitunnel-PrKMg8fNFewHciI2DvmAHGaD8g7cSyjD", # Ключ из нашего сервиса
    base_url="https://api.aitunnel.ru/v1/",
)
# ai_model="deepseek-chat"
ai_model = "gpt-4.1-mini"

# Вариант 2: Если вы не знаете структуру данных заранее 
@app.post("/ai-question-raw", tags=['data analytics'])
async def ai_question_raw(request: Request):
    # Получаем тело запроса в виде байтов
    body_bytes = await request.body()
    
    # Пытаемся преобразовать в JSON
    try:
        body_json = json.loads(body_bytes)
        
        # Обработка данных в зависимости от current_tab
        processed_data = process_data_by_tab(body_json)

        system_prompt = """
            Ты — аналитик социальных медиа. Форматируй ответы в источниках строго по следующим правилам:

            ### **1. Структура ответа**
            ---
            ## 📊 [Заголовок анализа]  
            [Используй иконки: 🔍 для выводов, ⚠️ для особенностей, 📌 для ключевых точек]  

            ### **2. Обязательные блоки**  
            - **Источники негатива или позитива** (таблица с метриками)  
            - **График/визуализация** (если есть данные)  
            - **Рекомендации** (маркированный список)  

            ### **3. Требования к оформлению**  
            ```markdown
            ### 🔍 Концентрация негатива или позитива 
            {текст}  

            | Метрика       | Значение | На сообщение |  
            |---------------|----------|--------------|  
            | Сообщения     | X        | —            |  

            ⚠️ **Особенности:**  
            - {пункт}  
            - {пункт}  
        """

        question = processed_data["question"]
        data = processed_data["data"]

        print('texts_examples-texts_examples-texts_examples-texts_examples-texts_examples')
        print("===============++++++++++++++===============")
        texts_examples = None

        if processed_data["current_tab"] == "Тональность авторов":

            print(77777)
            print(f'processed_data: {processed_data}')
            
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
            
            system_prompt = """
            Ты — senior-аналитик социальных медиа. Проводишь комплексный анализ тональности авторов с привязкой к контексту платформы.
            Помоги ответить на поставленный вопрос на основе сообщени йиз базы знаний, приведи ссылки на сообщения в своем ответе.
            """

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
        es = Elasticsearch(["http://localhost:9200"])
        
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
            indexes = load_dict_from_pickle('/home/dev/fastapi/analytics_app/data/indexes.pkl')
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
qdrant_client = QdrantClient(host="localhost", port=6333)

@app.post("/ai-question-information-graph", tags=['data analytics'])
async def ai_question_information_graph(request: Request):
    # Получаем тело запроса в виде байтов
    body_bytes = await request.body()
    
    # Пытаемся преобразовать в JSON
    body_json = json.loads(body_bytes)

    # Извлекаем нужные данные
    question = body_json.get('question', '')
    data = body_json.get('data', {})
    filters = body_json.get('filters', {})
    
    # Формируем структурированное сообщение для LLM
    system_prompt = """
    Ты — senior-аналитик социальных медиа с опытом работы более 10 лет. 
    Ты анализируешь данные из социальных сетей и предоставляешь детальный анализ.
    Твоя задача - дать глубокий анализ предоставленных данных, выделить ключевые тренды, 
    паттерны и инсайты, соответствующие запросу пользователя.
    
    Структура данных:
    - author: информация об авторе сообщения
      - fullname: имя автора
      - url: ссылка на сообщение
      - author_type: тип автора (Личный профиль, Сообщество и т.д.)
      - hub: источник сообщения (telegram.org, vk.com и т.д.)
      - sex: пол автора
      - age: возраст автора (если известен)
      - audienceCount: количество подписчиков/размер аудитории
      - er: показатель вовлеченности
      - viewsCount: количество просмотров
      - timeCreate: время создания сообщения (Unix timestamp)
    - reposts: массив репостов данного сообщения
    
    Используй эти данные для формирования своего анализа.

    Всегда структурируй ответы следующим образом:
    - Используй Markdown для форматирования
    - Разделяй длинный текст на параграфы (не более 3-4 предложений)
    - Используй заголовки второго уровня (##) для основных разделов
    - Используй заголовки третьего уровня (###) для подразделов
    - Для списков используй маркированные списки (-)
    - Выделяй важные моменты **жирным шрифтом**
    """
    
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
    max_items = 20  # Ограничиваем количество записей для экономии токенов
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
        indexes = load_dict_from_pickle('/home/dev/fastapi/analytics_app/data/indexes.pkl')
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
    system_prompt = """
    Ты — senior-аналитик социальных медиа с опытом работы более 10 лет. 
    Ты анализируешь данные из социальных сетей и предоставляешь детальный анализ.
    Твоя задача - дать глубокий анализ предоставленных данных, выделить ключевые тренды, 
    паттерны и инсайты, соответствующие запросу пользователя.

    Структура данных:
    - В первой графе собраны данные по негативной и позитивной активности в СМ.
    Каждая запись включает имя ресурса, индекс и количество сообщений:
        - negative_smi: массив с негативными ссылками
        - positive_smi: массив с позитивными ссылками
    - Во второй графе находятся уникальные ссылки на ресурсы, упомянутые в данных.
    
    Используй эти данные для формирования своего анализа.

    Всегда структурируй ответы следующим образом:
    - Используй Markdown для форматирования
    - Разделяй длинный текст на параграфы (не более 3-4 предложений)
    - Используй заголовки второго уровня (##) для основных разделов
    - Используй заголовки третьего уровня (###) для подразделов
    - Для списков используй маркированные списки (-)
    - Выделяй важные моменты **жирным шрифтом**
    """

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

        indexes = load_dict_from_pickle('/home/dev/fastapi/analytics_app/data/indexes.pkl')
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

        system_prompt = (
            "Ты — senior-аналитик социальных медиа. Даёшь структурированный, лаконичный и информативный вывод по анализу обсуждений. "
            "Выделяй важное **жирным**. Не пиши лишнего, но добавляй краткие пояснения по вопросам пользователя."
        )

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

@app.post("/convert-file-mlg")
async def convert_file_mlg(file: UploadFile = File(...)):
    try:

        def safe_filename(filename):
            return re.sub(r'[^\x00-\x7F]', '_', filename)

        safe_name = safe_filename(file.filename.replace('.xlsx', ''))


        contents = await file.read()
        temp_file_path = f"/home/dev/fastapi/analytics_app/data/temp/{file.filename}"

        with open(temp_file_path, "wb") as f:
            f.write(contents)

        print(111333)
        result = load_medialogia_excel(temp_file_path)
        print(222333)
        if result == 'error':
            return JSONResponse(status_code=400, content={'error': 'Не удалось обработать файл'})
        print(333333)
        json_result = json.dumps(result, ensure_ascii=False, indent=2)
        print(444333)
        from io import BytesIO
        json_bytes = BytesIO(json_result.encode('utf-8'))
        print(555333)
        response = StreamingResponse(
            json_bytes,
            media_type="application/json",
            headers={
                    "Content-Disposition": f"attachment; filename=converted_{safe_name}.json"
                }
        )
        return response
    except Exception as e:
        # Вывести traceback в консоль!
        traceback.print_exc()
        # Вернуть саму ошибку в ответе
        return JSONResponse(status_code=500, content={'error': str(e)})
    

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5001, reload=True)
