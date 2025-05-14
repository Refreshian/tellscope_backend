import ast
import asyncio
import subprocess
from datetime import datetime
from enum import Enum
import gc
import glob
import itertools
import re
import shutil
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
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Или укажите конкретные методы
    allow_headers=["*"],  # Или укажите конкретные заголовки
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

class PositiveHub(BaseModel):
    name: str
    values: int

class ModelAuthorsTonalityLandscape(BaseModel):
    negative_hubs: List[NegativeHub]
    positive_hubs: List[PositiveHub]

class Text(BaseModel):
    text: str
    hub: str
    url: str
    er: Optional[int]
    viewsCount: Optional[Union[int, str]]
    region: Optional[str]

class AuthorDatum(BaseModel):
    fullname: Optional[str]
    url: Optional[str]
    author_type: Optional[str]
    sex: Optional[str]
    age: Optional[str]
    count_texts: Optional[int]
    texts: List[Text]

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
    sex: str
    age: str
    audienceCount: int
    er: int
    viewsCount: Union[int, str]
    timeCreate: str

    @validator("timeCreate", pre=True)
    def convert_time_create(cls, value):
        # если приходит int, приводим к строке
        if isinstance(value, int):
            return str(value)
        return value


class RepostInfGraph(BaseModel):
    fullname: str
    url: str
    author_type: str
    sex: str
    age: str
    audienceCount: int
    er: int
    viewsCount: str
    timeCreate: str


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

# Customer Voice Model
class TonalityVoice(BaseModel):
    source: str
    Нейтрал: int
    Позитив: int
    Негатив: int


class SunkeyDatum(BaseModel):
    hub: str
    type: str
    tonality: str
    count: int
    search: str


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


class PositiveSmiMediaRating(BaseModel):
    name: str
    index: int
    message_count: int


class FirstGraphMediaRating(BaseModel):
    negative_smi: List[NegativeSmiMediaRating]
    positive_smi: List[PositiveSmiMediaRating]


class SecondGraphItemMediaRating(BaseModel):
    name: str
    time: int
    index: int
    url: str
    color: str


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
    audienceCount: Optional[int] = None
    commentsCount: Optional[int] = None
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

    # Обработка данных: заменяем значения в 'hub', если они соответствуют конкретным условиям
    for entry in data:
        if 'hub' in entry:
            hub = entry['hub']
            if hub == 'telegram.org':
                entry['hub'] = 'telegram.me'
            elif hub == 'maps.yandex.ru':
                entry['hub'] = 'yandex.ru'
            elif hub == 'tinkoff.ru':
                entry['hub'] = 'tbank.ru'

    # Подсчет позитивных и негативных тональностей
    pos = [entry for entry in data if entry.get('toneMark') == 1]
    neg = [entry for entry in data if entry.get('toneMark') == -1]

    # Подсчет источников (hub)
    neg_hub = [entry['hub'] for entry in data if entry.get('toneMark') == -1]
    dct_neg_hub = dict(Counter(neg_hub))
    dct_neg_hub = dict(sorted(dct_neg_hub.items(), key=lambda x: x[1], reverse=True))

    pos_hub = [entry['hub'] for entry in data if entry.get('toneMark') == 1]
    dct_pos_hub = dict(Counter(pos_hub))
    dct_pos_hub = dict(sorted(dct_pos_hub.items(), key=lambda x: x[1], reverse=True))

    # Обработка авторов (негатив)
    neg_authors = [entry for entry in data if entry.get('toneMark') == -1]
    neg_authors_hub = []
    for key in dct_neg_hub.keys():
        neg_authors_hub.append([(entry['authorObject'], [{"text": entry['text'], "hub": entry['hub'], "url": entry['url'], "er": entry['er'],
                                                         "viewsCount": entry['viewsCount'], "region": entry['region']}])
                                for entry in neg_authors if entry['hub'] == key])

    a = process_authors_data(neg_authors_hub) 

    # Обработка авторов (позитив)
    pos_authors = [entry for entry in data if entry.get('toneMark') == 1]
    pos_authors_hub = []
    for key in dct_pos_hub.keys():
        pos_authors_hub.append([(entry['authorObject'], [{"text": entry['text'], "hub": entry['hub'], "url": entry['url'], "er": entry['er'],
                                                         "viewsCount": entry['viewsCount'], "region": entry['region']}])
                                for entry in pos_authors if entry['hub'] == key])

    d = process_authors_data(pos_authors_hub)

    # Преобразование словарей для hubs в список объектов
    dct_pos_hub = [{"name": key, "values": value} for key, value in dct_pos_hub.items()]
    dct_neg_hub = [{"name": key, "values": value} for key, value in dct_neg_hub.items()]

    # Формирование итоговых данных
    values = Model_TonalityLandscape(
        tonality_values=TonalityValues(
            negative_count=len(neg),
            positive_count=len(pos)
        ),
        tonality_hubs_values=ModelAuthorsTonalityLandscape(
            negative_hubs=dct_neg_hub,
            positive_hubs=dct_pos_hub
        ),
        negative_authors_values=a,
        positive_authors_values=d
    )
    return values


def process_authors_data(authors_hub):
    authors_list = []
    for author_group in authors_hub:
        name_unique_author = [author[0].get('fullname', author[0].get('hub', '')) for author in author_group]
        dct_non_unique_author = dict(Counter(name_unique_author))
        
        list_non_unique_authors = list(set([key for key, val in dct_non_unique_author.items() if val > 1]))
        list_unique_authors = list(set([key for key, val in dct_non_unique_author.items() if val == 1]))

        for author_name in list_non_unique_authors + list_unique_authors:
            author_data = []
            for author in author_group:
                if author[0].get('fullname', author[0].get('hub', '')) == author_name:
                    texts = []
                    for text_item in author[1]:
                        try:
                            text = Text(
                                text=text_item.get('text', ''),
                                hub=text_item.get('hub', ''),
                                url=text_item.get('url', ''),
                                er=text_item.get('er'),
                                viewsCount=text_item.get('viewsCount'),
                                region=text_item.get('region')
                            )
                            texts.append(text)
                        except Exception as e:
                            print(f"Ошибка обработки текста: {e}")

                    author_obj = author[0]
                    author_obj['count_texts'] = len(texts)
                    author_obj['texts'] = texts

                    # Заполняем отсутствующие поля
                    author_obj['url'] = texts[0].url if texts else ''
                    author_obj['fullname'] = author_obj.get('fullname', '')
                    author_obj['sex'] = author_obj.get('sex', '')
                    author_obj['age'] = author_obj.get('age', '')

                    author_data.append(author_obj)

            authors_list.append({'author_data': author_data})

    return authors_list


@app.get('/information_graph', tags=['data analytics'])
async def information_graph(index: int=None, 
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
    # data = es.search(index='skillfactory_zaprosy_na_obuchenie_15.01.2024-21.01.2024', query_str='data')

    # отфильтровываем по необходимой дате из календаря
    data = [x for x in data if min_date <= x['timeCreate'] <= max_date]
    num_messages = len(data)

    # предобработка данных
    df_meta = pd.DataFrame(data)
    # del data

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

    df_meta = df_meta.join(pd.DataFrame(list(df_meta['authorObject'].values), columns=['fullname', 'text_url', 'author_type', 'sex', 'age']))
    # заменяем пустые fullname в СМИ на значения из hub
    df_meta['fullname'].fillna(df_meta['hub'], inplace=True)
    df = df_meta.copy()

    # создаем словарь похожих текстов вида {12: [11, 13],  44: [190], ...}
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
    # оставляем необходимую мету
    df_meta = df_meta[['fullname', 'url', 'author_type', 'hub', 'sex', 'age', 'audienceCount', 'er', 'viewsCount', 'timeCreate']]


    # получение итогового массива данных с последовательностями авторов распространения информации и репостами (похожими текстами)
    data = []

    for key, val in fin_dict.items():
        author_dct = {}
        author_dct['author'] = df_meta.loc[key].to_dict()
        
        # преобразование age в строку
        if isinstance(author_dct['author']['age'], int):
            author_dct['author']['age'] = str(author_dct['author']['age'])

        author_dct['reposts'] = []
        
        if len(val) > 0:
            for i in range(len(val)):
                repost = df_meta.loc[val[i]].to_dict()
                repost['viewsCount'] = str(repost['viewsCount'])
                repost['timeCreate'] = str(repost['timeCreate'])
                
                # преобразование age в строку
                if isinstance(repost['age'], int):
                    repost['age'] = str(repost['age'])

                author_dct['reposts'].append(RepostInfGraph(**repost))
        
        data.append(author_dct)

    ### данные для динамического графика
    def to_datetime(unixtime):
        return datetime.fromtimestamp(unixtime)
    
    df['timeCreate'] = df['timeCreate'].apply(to_datetime)
    df.sort_values(by='timeCreate', inplace=True)
    df.reset_index(inplace=True)
    df.drop('index', axis=1, inplace=True)

    bins = pd.date_range(np.min(df['timeCreate'].values), np.max(df['timeCreate'].values), freq='600T') # по 10 минут

    df['cut'] = pd.cut(df['timeCreate'], bins, right=False)
    df = df.astype(str)
    df['cut'] = [x.replace('nan', str(bins[-1])) if x == 'nan' else x for x in df['cut'].values]
    df['cut'] = [x.split(',')[0].replace("[", '') for x in df['cut'].values]
    # df.loc[0, 'timeCreate'] = df.loc[0, 'timeCreate'] + timedelta(minutes=9)
    # df.loc[df.shape[0]-1, 'timeCreate'] = df.loc[df.shape[0]-1, 'timeCreate'] - timedelta(minutes=9)

    # мержинг данных на 10 минутки
    df_bins = pd.DataFrame(bins, columns=['cut']).astype(str).set_index('cut')
    df_bins['cut'] = list(df_bins.index)

    df = df_bins.set_index('cut').join(df.set_index('cut'))
    df.fillna('', inplace=True)

    df['timeCreate'] = list(df.index)
    df.reset_index(inplace=True)
    df.reset_index(inplace=True)
    df.drop(['index', 'cut'], axis=1, inplace=True)
    df = df[['hub', 'timeCreate', 'audienceCount']]

    df['audienceCount'] = [int(x) if x != '' else x for x in df['audienceCount'].values]
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
    hub_dcts = [df[df['hub'] == x][['timeCreate', 'audienceCount']].set_index('timeCreate').to_dict() for x in listhubs]

    for i in range(len(hub_dcts)):
        hub_dcts[i][listhubs[i]] = hub_dcts[i].pop('audienceCount')

    dynamicdata_audience = []
    for i in range(len(hub_dcts)):
        hub_data = {}
        cumulative_audience = 0
        for key, val in hub_dcts[i][listhubs[i]].items():
            cumulative_audience += val
            hub_data[str(int(time.mktime(datetime.strptime(key, "%Y-%m-%d %H:%M:%S").timetuple())))] = str(cumulative_audience)
        dynamicdata_audience.append({listhubs[i]: hub_data})

    # мин и макс даты в выбранном интервале времени (10 мни, 20 мин..)
    mind, maxd = list(dynamicdata_audience[0][list(dynamicdata_audience[0].keys())[0]].keys())[0], list(dynamicdata_audience[0][list(dynamicdata_audience[0].keys())[0]].keys())[-1]

    dynamicdata_audience = dict(ChainMap(*dynamicdata_audience))

    # def sum_data(lst): # последовательно накапливает/суммирует кол-во по аудитории по столбцу..[1, 2, 4, 0, 2] -> [1, 3, 7, 7, 9..] 
    #     for i in range(len(lst)-1):
    #         lst[i+1] = lst[i] + lst[i+1]
    #     return lst

    # for key in dynamicdata_audience.keys():
    #     dynamicdata_audience[key] = dict(zip([int(x[0]) for x in dynamicdata_audience[key].items()], [str(x) for x in sum_data([int(x[1]) for x in dynamicdata_audience[key].items()])]))

    # Подсчет количества сообщений
    print(f"Количество сообщений: {num_messages}")

    def count_unique_authors(data):
        authors = set()
        try:
            for item in data:
                authors.add(item['author']['fullname'])  # Полное имя автора
                if 'reposts' in item and item['reposts']:
                    for repost in item['reposts']:
                        authors.add(repost.fullname)  # Полное имя из репостов
        except Exception as e:
            print(f"Произошла ошибка: {e}")
            print(item)
        
        return len(authors)  # Добавлено возвращаемое значение

    # В вашем основном обработчике
    num_unique_authors = count_unique_authors(data)  # Теперь будет корректно считать количество уникальных авторов

    # Проверка на корректность boolean значение
    repost_value = bool(repost) if repost is not None else False

    # Формирование результата
    values = ModelInfGraph(
        values=data, 
        post=post, 
        repost=repost_value,  # Убедитесь, что здесь передается boolean
        SMI=SMI,
        dynamicdata_audience=dynamicdata_audience, 
        num_messages=num_messages, 
        num_unique_authors=num_unique_authors  # Теперь должно быть корректное целое число
    )
    
    return values


@app.get("/themes")
async def themes_analize(user: User = Depends(current_user), index: int =None, 
                             min_date=None, max_date=None) -> ThemesModel:
    # Путь к файлу с темами 
    file_path = '/home/dev/fastapi/analytics_app/data/indexes.pkl'
    # Загрузка словаря с темами
    indexes = load_dict_from_pickle(file_path)

    os.chdir('/home/dev/fastapi/analytics_app/files')
    # данные с описанием тематик
    # filename = indexes[index] + '_LLM'
    os.chdir('/home/dev/fastapi/analytics_app/files/Росбанк/')
    filename = 'rosbank_01.04.2024-15.04.2024_LLM'
    with open (filename, 'rb') as fp:
        data = pickle.load(fp)


    data = [x[0]['generated_text'].split('model\n')[1] if len(x) == 1 else x for x in data]
    data = pd.DataFrame(data) 

    # print(data)

    query = {
            "size": 10000,
            "query": {
                        "range": {
                            "timeCreate": {      # skillfactory_zaprosy_na_obuchenie_15.01.2024-21.01.2024
                                "gte": min_date, # 1705329992
                                "lte": max_date, # 1705848392
                                "boost": 2.0
                            }
                        }
                    }
                }
    
    # данные с авторами, текстами и метаинформацией
    # dict_train = es.search(index='skillfactory_15.01.2024-21.01.2024', body=query)
    dict_train = es.search(index=indexes[index], body=query)
    dict_train = dict_train['hits']['hits']
    dict_train = [x['_source'] for x in dict_train]
    
    # with codecs.open(indexes[index], "r", "utf_8_sig") as train_file:
    #     dict_train = json.load(train_file)

    columns = ['timeCreate', 'text', 'hub', 'url', 'hubtype',
        'commentsCount', 'audienceCount',
        'citeIndex', 'repostsCount', 'likesCount', 'er', 'viewsCount',
        'toneMark', 'role',
        'country', 'region', 'city', 'language', 'fullname',
        'author_url', 'author_type', 'sex', 'age']

    author_df = pd.DataFrame(list(pd.DataFrame(dict_train)['authorObject'].values))
    author_df.columns=['fullname', 'author_url', 'author_type', 'sex', 'age']
    df_res = pd.DataFrame(dict_train).join(author_df)
    df_res = df_res[columns]
    # df_res.columns = ['Время', 'Текст', 'Источник', 'Ссылка', 'Тип источника', 'Комментариев', 'Аудитория',
    #        'Сайт-Индекс', 'Репостов', 'Лайков', 'Суммарная вовлеченность', 'Просмотров',
    #        'Тональность', 'Роль', 'Страна',
    #        'Регион', 'Город', 'Язык', 'Имя автора', 'Ссылка на автора', 'Тип автора',
    #        'Пол', 'Возраст']

    df_res = df_res.join(data)
    df_res = df_res[(df_res['timeCreate'] >= int(min_date)) & (df_res['timeCreate'] <= int(max_date))]
    df_res.reset_index(inplace=True)
    df_res.drop('index', axis=1, inplace=True)

    data = df_res[[0]]

    # функция для удаления лишних символов в текстах
    import re
    regex = re.compile("[А-Яа-я:=!\)\()A-z\_\%/|]+")

    def words_only(text, regex=regex):
        try:
            return " ".join(regex.findall(text))
        except:
            return ""

    # удаляем лишние символы, оставляем слова
    data[0] = data[0].apply(words_only)

    # получение векторов текстов и сравнение
    count_vectorizer = CountVectorizer()
    vector_matrix = count_vectorizer.fit_transform(
        data[0].values)

    cosine_similarity_matrix = cosine_similarity(vector_matrix)
    dff = pd.DataFrame(cosine_similarity_matrix)
    # dff = dff.round(5)
    # dff = dff.replace([1.000], 0)

    val_dff = dff.values
    # заменяем значения по главной диагонали на 0
    for i in range(len(val_dff)):
        val_dff[i][i] = 0
        
    dff = pd.DataFrame(val_dff)

    # создаем словарь похожих текстов вида {11: [12, 132],  44: [190], ...}
    fin_dict = {}
    threashhold = 0.70

    # print('threashhold')

    # выявляем список строк с похожими текстам
    for i in range(dff.shape[0]):
        if list(np.where(dff.loc[i].values >= threashhold)[0]) != []:
            if i not in [item for sublist in list(fin_dict.values()) for item in sublist]:

                fin_dict[i] = list(
                    np.where(dff.loc[i].values >= threashhold)[0])
                
        else:
            fin_dict[i] = []
            
    len_val = [len(x) for x in fin_dict.values()]
    dct_len_val = dict(zip(list(fin_dict.keys()), len_val))
    # dct_len_val = dict(sorted(dct_len_val.items(), key=itemgetter(1), reverse=True))

    # добавление текстов и метаданных в итоговый словарь
    fin_data = []
    texts = []
    texts_list = data.loc[list(fin_dict.keys())][0].values # список текстов с описанием, берется первое описание по первому тексту-ключу
    list_len = list(dct_len_val.values()) # список с количеством текстов по тематике
    # [{'description': 'Тема текста связана с ..', 'count': 152, 'texts': [...]},
    #  {'description': 'Тема текста связана с ..', 'count': 141, 'texts': [...]}, ..]

    for i in range(len(fin_dict.keys())):
        
        if fin_dict[list(fin_dict.keys())[i]] != []:

            a = {}
            a['description'] = texts_list[i] # описание тематики
            a['count'] = list_len[i] # количество текстов по тематике
            a['audience'] = str(np.sum([x['audienceCount'] for x in df_res.iloc[fin_dict[list(fin_dict.keys())[i]]].to_dict(orient='records') if x['audienceCount'] != ''])) # количество аудитории в тематике
            a['er'] = str(np.sum([x['er'] for x in df_res.iloc[fin_dict[list(fin_dict.keys())[i]]].to_dict(orient='records') if x['er'] != ''])) # количество вовлеченности в тематику
            a['viewsCount'] = str(np.sum([x['viewsCount'] for x in df_res.iloc[fin_dict[list(fin_dict.keys())[i]]].to_dict(orient='records') if x['viewsCount'] != '']))# количество просмотров в тематике
            a['texts'] = 'texts' 
            # texts.append(df_res[df_res.index.isin(fin_dict[list(fin_dict.keys())[i]])].to_dict(orient='records'))
            fin_data.append(a)
            
        else:
            
            a = {}
            a['description'] = texts_list[i] # описание тематики
            a['count'] = list_len[i] # количество текстов по тематике
            a['audience'] = str(np.sum([x['audienceCount'] for x in df_res.iloc[fin_dict[list(fin_dict.keys())[i]]].to_dict(orient='records') if x['audienceCount'] != ''])) # количество аудитории в тематике
            a['er'] = str(np.sum([x['er'] for x in df_res.iloc[fin_dict[list(fin_dict.keys())[i]]].to_dict(orient='records') if x['er'] != ''])) # количество вовлеченности в тематику
            a['viewsCount'] = str(np.sum([x['viewsCount'] for x in df_res.iloc[fin_dict[list(fin_dict.keys())[i]]].to_dict(orient='records') if x['viewsCount'] != '']))# количество просмотров в тематике
            a['texts'] = 'texts'
            # texts.append(df_res.iloc[[list(fin_dict.keys())[i]]].to_dict(orient='records'))
            fin_data.append(a)
  
    return ThemesModel(values=fin_data)


@app.get("/voice", tags=['data analytics'])
async def voice_analize(index: int = None, # user: User = Depends(current_user), 
                             min_date: int=None, max_date: int=None, query_str: str = None) -> ModelVoice:
    # Путь к файлу с темами 
    file_path = '/home/dev/fastapi/analytics_app/data/indexes.pkl'
    # Загрузка словаря с темами
    indexes = load_dict_from_pickle(file_path)

    search = query_str.split(',')
    topn = 20 # ТОП-источников, остальные пойдут в "Другие"
    values = []

    for i in range(len(search)):

        data = elastic_query(theme_index=indexes[index], query_str=search[i])

        # отфильтровываем по необходимой дате из календаря
        data = [x for x in data if min_date <= x['timeCreate'] <= max_date]
        
    #     data = elastiqsearc(search[i]) # данные из эластик
        search_name = search[i].strip()
        hubs_tonality = Counter([(x['hub'], str(x['toneMark']).replace('0', 'Нейтрал').replace('-1', 'Негатив').replace('1', 'Позитив')) for x in data])
        list_tonal_hubs = [[key[0], key[1], val] for key, val in hubs_tonality.items()]

        lst_dicts = [{x[0]: {x[1]: x[2]}} for x in list_tonal_hubs] # {'youtube.com': {'Нейтрал': 2}}, {'yaroslavl.bezformata.com': {'Нейтрал': 1}},
        keys_list = list(set([list(x.keys())[0] for x in lst_dicts]))

        hubs_tonality_dict = {} # финальный словарь по источникуам и тональности

        for j in range(len(keys_list)):
            list_same_dict = [x for x in lst_dicts if keys_list[j] in x]
            
            if len(list_same_dict) != 1:
            
                dict_hub_ton = {}
                dict_hub_ton[list(list_same_dict[0].keys())[0]] = {}

                for i in range(len(list_same_dict)):
                    dict_hub_ton[list(list_same_dict[0].keys())[0]].update(list(list_same_dict[i].values())[0])
                    
                hubs_tonality_dict.update(dict_hub_ton)
                    
            else:
                dict_hub_ton = {}
                dict_hub_ton[list(list_same_dict[0].keys())[0]] = {}
                dict_hub_ton[list(list_same_dict[0].keys())[0]].update(list(list_same_dict[0].values())[0])
                
                hubs_tonality_dict.update(dict_hub_ton)

        sort = Counter(dict(zip(list(hubs_tonality_dict.keys()), [np.sum(list(x.values())) for x in list(hubs_tonality_dict.values())]))).most_common()
        sort = [x[0] for x in sort]

        # финальная сортировка по количеству
        index_map = {v: i for i, v in enumerate(sort)}
        hubs_tonality_dict = sorted(hubs_tonality_dict.items(), key=lambda pair: index_map[pair[0]])

        hubs_tonality_dict = [{x[0]: x[1]} for x in hubs_tonality_dict]
        # hubs_tonality_dict = [{'source': x} for x in hubs_tonality_dict]
        dcts = [{'source': list(x.keys())[0]} for x in hubs_tonality_dict] # {'source': 'vk.com'}

        for i in range(len(dcts)):
            dcts[i].update([list(x.values())[0] for x in hubs_tonality_dict][i]) # {'source': 'vk.com', 'Нейтрал': 29, 'Негатив': 5}

        # [{'source': 'vk.com', 'Нейтрал': 29, 'Негатив': 5, 'Позитив': 0}, ...
        for i in range(len(dcts)):
            if 'Нейтрал' not in dcts[i]:
                dcts[i]['Нейтрал'] = 0
            if 'Позитив' not in dcts[i]:
                dcts[i]['Позитив'] = 0
            if 'Негатив' not in dcts[i]:
                dcts[i]['Негатив'] = 0


        ##### источники - тональность - тип сообщения
        hubs = Counter([x['hub'] for x in data])
        hubs = dict(sorted(hubs.items(), key=lambda x: x[1], reverse=True)[:topn])

        list_topn_hubs = list(hubs.keys())
        message_tonality = [[x['hub'], str(x['toneMark']).replace('0', 'Нейтрал').replace('-1', 'Негатив').replace('1', 'Позитив')] 
                            for x in data if x['hub'] in list_topn_hubs] 
 

        message_tonality_type = [[x['hub'], x['type'], str(x['toneMark']).replace('0', 'Нейтрал').replace('-1', 'Негатив').replace('1', 'Позитив')] 
                            for x in data if x['hub'] in list_topn_hubs]

        dct_tonality_hubs = Counter([', '.join(x) for x in message_tonality_type])

        hub_tonality_type_list = [[x[0].split(',')[0].strip(), x[0].split(',')[1].strip(), x[0].split(',')[2].strip(), 
                            x[1]] for x in list(dct_tonality_hubs.items())]
        hub_tonality_type_list = sorted(hub_tonality_type_list, key=itemgetter(3), reverse=True)
        
        for i in range(len(hub_tonality_type_list)):
            data = hub_tonality_type_list[i]
            data.append(search_name)
            hub_tonality_type_list[i] = dict(zip(["hub", "type", "tonality", "count", "search"], data))
        
        values_search = {}
        values_search['name'] = search_name
        values_search['tonality'] = dcts
        values_search['sunkey_data'] = hub_tonality_type_list

        values.append(values_search)

    return ModelVoice(values = values) 


@app.get("/media-rating", tags=['data analytics'])
def media_rating(index: int = None, min_date: int=None,  
                 max_date: int=None) -> MediaRatingModel: # user: User = Depends(current_user)
    
    # Путь к файлу с темами 
    file_path = '/home/dev/fastapi/analytics_app/data/indexes.pkl'
    # Загрузка словаря с темами
    indexes = load_dict_from_pickle(file_path)

    # Делаем запрос на текстовый поиск
    data = elastic_query(theme_index=indexes[index], query_str='all')
    # data = es.search(index='skillfactory_zaprosy_na_obuchenie_15.01.2024-21.01.2024', query_str='data')

    # Отфильтровываем по необходимой дате из календаря
    data = [x for x in data if min_date <= x['timeCreate'] <= max_date]
    data = data[:100]
    df = pd.DataFrame(data)

    # Если в данных есть столбец citeIndex, заменяем пустые строки на 0
    if 'citeIndex' in df.columns:
        df['citeIndex'] = df['citeIndex'].apply(lambda x: 0 if x == "" else x)

    # метаданные: разбивка и сборка соцмедиа и СМИ в один датафрэйм с данными
    df_meta = pd.DataFrame()

    # Случай выгрузки темы только по СМИ (нет столбца hubtype)
    if 'hubtype' not in df.columns:
        dff = df.copy()
        dff['timeCreate'] = [datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S') for x in dff['timeCreate'].values]
        df_meta_smi_only = dff[['timeCreate', 'hub', 'toneMark', 'audience', 'url', 'text', 'citeIndex']]
        # При необходимости можно переименовать столбцы:
        # df_meta_smi_only.columns = ['timeCreate', 'hub', 'toneMark', 'audienceCount', 'url', 'text', 'citeIndex']
        df_meta_smi_only['fullname'] = dff['hub']
        df_meta_smi_only['author_type'] = 'Онлайн-СМИ'
        df_meta_smi_only['hubtype'] = 'Онлайн-СМИ'
        df_meta_smi_only['type'] = 'Онлайн-СМИ'
        df_meta_smi_only['er'] = 0
        df_meta_smi_only.dropna(subset=['timeCreate'], inplace=True)
        df_meta_smi_only = df_meta_smi_only.set_index(['timeCreate'])
        df_meta_smi_only['date'] = [x[:10] for x in df_meta_smi_only.index]
        df_meta = df_meta_smi_only

    # Случай, когда присутствует столбец hubtype (выгрузка онлайн-СМИ или соцмедиа)
    if 'hubtype' in df.columns:
        for i in range(2):  # Онлайн-СМИ или соцмедиа
            if i == 0:
                dff = df[df['hubtype'] != 'Онлайн-СМИ']
                if dff.shape[0] != 0:
                    dff['timeCreate'] = [datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S') for x in dff['timeCreate'].values]
                    df_meta_socm = dff[['timeCreate', 'hub', 'toneMark', 'audienceCount', 'url', 'er', 'hubtype', 'text', 'type']]
                    df_meta_socm['fullname'] = pd.DataFrame.from_records(dff['authorObject'].values)['fullname'].values
                    df_meta_socm['author_type'] = pd.DataFrame.from_records(dff['authorObject'].values)['author_type'].values
                    df_meta_socm.dropna(subset=['timeCreate'], inplace=True)
                    df_meta_socm = df_meta_socm.set_index(['timeCreate'])
                    df_meta_socm['date'] = [x[:10] for x in df_meta_socm.index]
            if i == 1:
                dff = df[df['hubtype'] == 'Онлайн-СМИ']
                if dff.shape[0] != 0:
                    dff['timeCreate'] = [datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S') for x in dff['timeCreate'].values]
                    df_meta_smi = dff[['timeCreate', 'hub', 'toneMark', 'audienceCount', 'url', 'er', 'hubtype', 'text', 'citeIndex']]
                    df_meta_smi['fullname'] = dff['hub']
                    df_meta_smi['author_type'] = 'Онлайн-СМИ'
                    df_meta_smi['hubtype'] = 'Онлайн-СМИ'
                    df_meta_smi['type'] = 'Онлайн-СМИ'
                    df_meta_smi.dropna(subset=['timeCreate'], inplace=True)
                    df_meta_smi = df_meta_smi.set_index(['timeCreate'])
                    df_meta_smi['date'] = [x[:10] for x in df_meta_smi.index]

        if 'df_meta_smi' in locals() and 'df_meta_socm' in locals():
            df_meta = pd.concat([df_meta_socm, df_meta_smi])
        elif 'df_meta_smi' in locals():
            df_meta = df_meta_smi
        else:
            df_meta = df_meta_socm

    if set(df_meta['hub'].values) == {"telegram.org"}:
        df_meta = df_meta[(df_meta['hubtype'] == 'Мессенджеры каналы') & (df_meta['hub'] == "telegram.org")]

    # Negative smi для мессенджерных каналов
    df_hub_siteIndex = df_meta[(df_meta['hubtype'] == 'Мессенджеры каналы') & (df_meta['toneMark'] == -1)][['fullname', 'audienceCount']].values
    dict_neg = {}
    for i in range(len(df_hub_siteIndex)):
        if df_hub_siteIndex[i][0] not in dict_neg.keys():
            dict_neg[df_hub_siteIndex[i][0]] = []
            dict_neg[df_hub_siteIndex[i][0]].append(df_hub_siteIndex[i][1])
        else:
            dict_neg[df_hub_siteIndex[i][0]].append(df_hub_siteIndex[i][1])
    list_neg = [list(set(x)) for x in dict_neg.values()]
    list_neg = [[0] if x[0] == 'n/a' else x for x in list_neg if x != 'n/a']
    list_neg = [int(x[0]) if x[0] != '' else 0 for x in list_neg]
    for i in range(len(list_neg)):
        dict_neg[list(dict_neg.keys())[i]] = list_neg[i]
    dict_neg = dict(sorted(dict_neg.items(), key=lambda x: x[1], reverse=True))
    dict_neg_hubs_count = dict(Counter(list(df_meta[(df_meta['hubtype'] == 'Мессенджеры каналы') & (df_meta['toneMark'] == -1)]['fullname'])))
    fin_neg_dict = defaultdict(tuple)
    for d in (dict_neg, dict_neg_hubs_count):
        for key, value in d.items():
            fin_neg_dict[key] += (value,)
    list_neg_smi = list(fin_neg_dict.keys())
    list_neg_smi_index = [x[0] for x in fin_neg_dict.values()]
    list_neg_smi_massage_count = [x[1] for x in fin_neg_dict.values()]

    # Positive smi для мессенджерных каналов
    df_hub_siteIndex = df_meta[(df_meta['hubtype'] == 'Мессенджеры каналы') & (df_meta['toneMark'] == 1)][['fullname', 'audienceCount']].values
    dict_pos = {}
    for i in range(len(df_hub_siteIndex)):
        if df_hub_siteIndex[i][0] not in dict_pos.keys():
            dict_pos[df_hub_siteIndex[i][0]] = []
            dict_pos[df_hub_siteIndex[i][0]].append(df_hub_siteIndex[i][1])
        else:
            dict_pos[df_hub_siteIndex[i][0]].append(df_hub_siteIndex[i][1])
    list_pos = [list(set(x)) for x in dict_pos.values()]
    list_pos = [[0] if x[0] == 'n/a' else x for x in list_pos if x != 'n/a']
    list_pos = [int(x[0]) if x[0] != '' else 0 for x in list_pos]
    for i in range(len(list_pos)):
        dict_pos[list(dict_pos.keys())[i]] = list_pos[i]
    dict_pos = dict(sorted(dict_pos.items(), key=lambda x: x[1], reverse=True))
    dict_pos_hubs_count = dict(Counter(list(df_meta[(df_meta['hubtype'] == 'Мессенджеры каналы') & (df_meta['toneMark'] == 1)]['fullname'])))
    fin_pos_dict = defaultdict(tuple)
    for d in (dict_pos, dict_pos_hubs_count):
        for key, value in d.items():
            fin_pos_dict[key] += (value,)
    list_pos_smi = list(fin_pos_dict.keys())
    list_pos_smi_index = [x[0] for x in fin_pos_dict.values()]
    list_pos_smi_massage_count = [x[1] for x in fin_pos_dict.values()]

    # Приведение timeCreate к списку
    df_meta['timeCreate'] = list(df_meta.index)

    # Формирование данных для bobble graph (для мессенджерных каналов)
    bobble = []
    df_tonality = df_meta[(df_meta['hubtype'] == 'Мессенджеры каналы') & (df_meta['toneMark'] != 0)][['fullname', 'audienceCount', 'toneMark', 'url']].values
    index_ton = df_meta[(df_meta['hubtype'] == 'Мессенджеры каналы') & (df_meta['toneMark'] != 0)][['timeCreate']].values.tolist()
    date_ton = [x[0] for x in index_ton]
    date_ton = [int((datetime.strptime(x, '%Y-%m-%d %H:%M:%S') - datetime(1970, 1, 1)).total_seconds() * 1000) for x in date_ton]

    for i in range(len(df_tonality)):
        if df_tonality[i][2] == -1:
            bobble.append([date_ton[i], df_tonality[i][0], dict_neg[df_tonality[i][0]], -1, df_tonality[i][3]])
        elif df_tonality[i][2] == 1:
            bobble.append([date_ton[i], df_tonality[i][0], dict_pos[df_tonality[i][0]], 1, df_tonality[i][3]])
    for i in range(len(bobble)):
        if bobble[i][3] == 1:
            bobble[i][3] = "#32ff32"
        else:
            bobble[i][3] = "#FF3232"

    data = {
    "neg_smi_name": list_neg_smi,
    "neg_smi_count": list_neg_smi_massage_count,
    "neg_smi_rating": list_neg_smi_index,
    "pos_smi_name": list_pos_smi,
    "pos_smi_count": list_pos_smi_massage_count,
    "pos_smi_rating": list_pos_smi_index,
    "date_bobble": [x[0] for x in bobble],
    "name_bobble": [x[1] for x in bobble],
    "index_bobble": [x[2] for x in bobble],
    "z_index_bobble": [1] * len(bobble),
    "tonality_index_bobble": [x[3] for x in bobble],
    "tonality_url": [x[4] for x in bobble],
    }

    # Обработка данных для онлайн-СМИ
    df_meta = df_meta[df_meta['hubtype'] == 'Онлайн-СМИ']

    # Negative smi для онлайн-СМИ
    if 'citeIndex' in df_meta.columns:
        df_hub_siteIndex = df_meta[(df_meta['hubtype'] == 'Онлайн-СМИ') & (df_meta['toneMark'] == -1)][['hub', 'citeIndex']].values
    else:
        df_hub_siteIndex = df_meta[(df_meta['hubtype'] == 'Онлайн-СМИ') & (df_meta['toneMark'] == -1)][['hub']].values

    dict_neg = {}
    for i in range(len(df_hub_siteIndex)):
        if df_hub_siteIndex[i][0] not in dict_neg.keys():
            dict_neg[df_hub_siteIndex[i][0]] = []
            # Если значение citeIndex пустое (""), уже преобразовано в 0 выше
            dict_neg[df_hub_siteIndex[i][0]].append(df_hub_siteIndex[i][1])
        else:
            dict_neg[df_hub_siteIndex[i][0]].append(df_hub_siteIndex[i][1])
    list_neg = [list(set(x)) for x in dict_neg.values()]
    list_neg = [[0] if x[0] == 'n/a' else x for x in list_neg if x != 'n/a']
    list_neg = [int(x[0]) if x[0] != '' else 0 for x in list_neg]
    for i in range(len(list_neg)):
        dict_neg[list(dict_neg.keys())[i]] = list_neg[i]
    dict_neg = dict(sorted(dict_neg.items(), key=lambda x: x[1], reverse=True))
    dict_neg_hubs_count = dict(Counter(list(df_meta[(df_meta['hubtype'] == 'Онлайн-СМИ') & (df_meta['toneMark'] == -1)]['hub'])))
    fin_neg_dict = defaultdict(tuple)
    for d in (dict_neg, dict_neg_hubs_count):
        for key, value in d.items():
            fin_neg_dict[key] += (value,)
    list_neg_smi = list(fin_neg_dict.keys())
    list_neg_smi_index = [x[0] for x in fin_neg_dict.values()]
    list_neg_smi_massage_count = [x[1] for x in fin_neg_dict.values()]

    # Positive smi для онлайн-СМИ
    if 'citeIndex' in df_meta.columns:
        df_hub_siteIndex = df_meta[(df_meta['hubtype'] == 'Онлайн-СМИ') & (df_meta['toneMark'] == 1)][['hub', 'citeIndex']].values
    else:
        df_hub_siteIndex = df_meta[(df_meta['hubtype'] == 'Онлайн-СМИ') & (df_meta['toneMark'] == 1)][['hub']].values
    dict_pos = {}
    for i in range(len(df_hub_siteIndex)):
        if df_hub_siteIndex[i][0] not in dict_pos.keys():
            dict_pos[df_hub_siteIndex[i][0]] = []
            dict_pos[df_hub_siteIndex[i][0]].append(df_hub_siteIndex[i][1])
        else:
            dict_pos[df_hub_siteIndex[i][0]].append(df_hub_siteIndex[i][1])
    list_pos = [list(set(x)) for x in dict_pos.values()]
    list_pos = [[0] if x[0] == 'n/a' else x for x in list_pos if x != 'n/a']
    list_pos = [int(x[0]) if x[0] != '' else 0 for x in list_pos]
    for i in range(len(list_pos)):
        dict_pos[list(dict_pos.keys())[i]] = list_pos[i]
    dict_pos = dict(sorted(dict_pos.items(), key=lambda x: x[1], reverse=True))
    dict_pos_hubs_count = dict(Counter(list(df_meta[(df_meta['hubtype'] == 'Онлайн-СМИ') & (df_meta['toneMark'] == 1)]['hub'])))
    fin_pos_dict = defaultdict(tuple)
    for d in (dict_pos, dict_pos_hubs_count):
        for key, value in d.items():
            fin_pos_dict[key] += (value,)
    list_pos_smi = list(fin_pos_dict.keys())
    list_pos_smi_index = [x[0] for x in fin_pos_dict.values()]
    list_pos_smi_massage_count = [x[1] for x in fin_pos_dict.values()]

    # Приведение timeCreate к списку
    df_meta['timeCreate'] = list(df_meta.index)

    # Формирование данных для bobble graph (для онлайн-СМИ)
    bobble = []
    if 'citeIndex' in df_meta.columns:
        df_tonality = df_meta[(df_meta['hubtype'] == 'Онлайн-СМИ') & (df_meta['toneMark'] != 0)][['hub', 'citeIndex', 'toneMark', 'url']].values
        index_ton = df_meta[(df_meta['hubtype'] == 'Онлайн-СМИ') & (df_meta['toneMark'] != 0)][['timeCreate']].values.tolist()
        date_ton = [x[0] for x in index_ton]
        date_ton = [int((datetime.strptime(x, '%Y-%m-%d %H:%M:%S') - datetime(1970, 1, 1)).total_seconds() * 1000) for x in date_ton]
    else:
        df_tonality = df_meta[(df_meta['hubtype'] == 'Онлайн-СМИ') & (df_meta['toneMark'] != 0)][['hub', 'toneMark']].values
        index_ton = df_meta[(df_meta['hubtype'] == 'Онлайн-СМИ') & (df_meta['toneMark'] != 0)][['timeCreate']].values.tolist()
        date_ton = [x[0] for x in index_ton]
        date_ton = [int((datetime.strptime(x, '%Y-%m-%d %H:%M:%S') - datetime(1970, 1, 1)).total_seconds() * 1000) for x in date_ton]
        
    for i in range(len(df_tonality)):
        if df_tonality[i][2] == -1:
            bobble.append([date_ton[i], df_tonality[i][0], dict_neg[df_tonality[i][0]], -1, df_tonality[i][3]])
        elif df_tonality[i][2] == 1:
            bobble.append([date_ton[i], df_tonality[i][0], dict_pos[df_tonality[i][0]], 1, df_tonality[i][3]])
    for i in range(len(bobble)):
        if bobble[i][3] == 1:
            bobble[i][3] = "#32ff32"
        else:
            bobble[i][3] = "#FF3232"

    values = {}
    values['first_graph'] = {}
    values['first_graph']['negative_smi'] = [{'name': x, "index": y, "message_count": z} for (x, y, z) in zip(list_neg_smi, list_neg_smi_index, list_neg_smi_massage_count)]
    values['first_graph']['positive_smi'] = [{'name': x, "index": y, "message_count": z} for (x, y, z) in zip(list_pos_smi, list_pos_smi_index, list_pos_smi_massage_count)]

    values['second_graph'] = [{'name': x, 'time': y, 'index': z, 'url': u, 'color': t} for (x, y, z, u, t) in zip([b[1] for b in bobble], [b[0] for b in bobble], [b[2] for b in bobble], [b[4] for b in bobble], [b[3] for b in bobble])]

    return MediaRatingModel(first_graph=values['first_graph'], second_graph=values['second_graph'])


@app.get('/ai-analytics', tags=['ai analytics'])
async def ai_analytics_get(index: int = None, min_date: int = None, max_date: int = None) -> ModelAiAnalytics:
    # Путь к файлу с темами 
    file_path = '/home/dev/fastapi/analytics_app/data/indexes.pkl'
    # Загрузка словаря с темами
    indexes = load_dict_from_pickle(file_path)
    
    # делаем запрос на текстовый поиск
    data = elastic_query(theme_index=indexes[index], query_str='all')

    # отфильтровываем по необходимой дате из календаря
    data = [x for x in data if min_date <= x['timeCreate'] <= max_date]
    keys = ['id', 'timeCreate', 'text', 'hub', 'audienceCount', 'commentsCount', 'er', 'url']  # ключи для отображения в первой таблице
    data = [{k: y.get(k, None) for k in keys} for y in data[:100]]  # данные для первой таблицы
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
async def competitors(query: QueryCompetitors):
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

        # Замена audience на audienceCount
        ind_df = [{"audienceCount" if k == "audience" else k: v for k, v in x.items()} for x in data]

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
                        'rating': safe_to_int(row['citeIndex']),  # Изменено на безопасное преобразование
                        'url': row['url']
                    }
                    for _, row in neg_smi.iterrows()
                ],
                'pos': [
                    {
                        'hub': row['hub'],
                        'count': row['count'],
                        'rating': safe_to_int(row['citeIndex']),  # Изменено на безопасное преобразование
                        'url': row['url']
                    }
                    for _, row in pos_smi.iterrows()
                ],
            }
        })

        # Данные только по Соцмедиа (hubtype != 'Онлайн-СМИ')
        socmedia_data = df[df['hubtype'] != 'Онлайн-СМИ']

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
                    'rating': safe_to_int(row['audienceCount']),  # Изменено на безопасное преобразование
                    'url': row['url']
                }
                for _, row in neg_socmedia.iterrows()
            ],
            'pos': [
                {
                    'hub': row['hub'],
                    'count': row['count'],
                    'rating': safe_to_int(row['audienceCount']),  # Изменено на безопасное преобразование
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
            'rating': row['citeIndex'],
            'url': row['url']
        } for _, row in smi_data.iterrows()]

        # Socmedia данные
        df_socmedia = df[df['hubtype'] != 'Онлайн-СМИ']
        socmedia_data = df_socmedia.groupby('hub').agg(
            hub_count=('hub', 'size'),
            audienceCount=('audienceCount', 'first'),
            url=('url', 'first')
        ).reset_index()

        socmedia_results = [{
            'name': row['hub'],
            'count': row['hub_count'],
            'rating': row['audienceCount'],
            'url': row['url']
        } for _, row in socmedia_data.iterrows()]

        third_graph.append({
            'index_name': filename,
            'SMI': smi_results,
            'Socmedia': socmedia_results,
        })

    return {
        'first_graph': first_graph,
        'second_graph': second_graph,
        'third_graph': third_graph,
    }


@app.get("/create-data-projector/{user_id}/{folder_name}/{file_name}")
async def create_data_projector(user_id: str, folder_name: str, file_name: str):
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
            print(222)
            # Если ключа нет — создаём пустой словарь
            user_folders = {}

        # Проверяем существование папки, переданной в user_data['folder_name']
        if folder_name in user_folders:
            print(333)
            # Если папка существует, добавляем новый file_info в уже имеющийся список
            user_folders[folder_name].append(file_info)
        else:
            # Если папка не существует, создаём её и добавляем file_info в список
            print(444)
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


@app.get('/file-load/{user_id}/{file_type}/{folder_name}/{file_name}')
def load_file(user_id: str, file_type: str, folder_name: str, file_name: str):
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


from run_llm_query import run_llm_query
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
        # Создаем task_data на основе входной модели `AnalysisRequest`
        # Преобразуем данные в строковый формат перед сохранением
        task_data = {
            "task_id": task_id,
            "user_id": str(analysis_request.user_id),
            "folder_name": str(analysis_request.folder_name),
            "index": str(analysis_request.index),
            "query_str": analysis_request.query_str or "all",  # Используем "all", если `query_str` не передано
            "min_date": str(analysis_request.min_date),
            "max_date": str(analysis_request.max_date),
            "system_prompt": analysis_request.system_prompt or "",  # Пустая строка, если не указано
            "promt_question": analysis_request.promt_question or "",
            "status": "pending",
            "total_texts": "0",  # Значение "0" всегда строка
            "completed_texts": "0",  # Значение "0" всегда строка
            "progress": "0",  # Значение "0" всегда строка
            "bad_request": "0"
        }

        # Сохраняем задачу в Redis
        await redis_db.hset(f"task:{task_id}", mapping=task_data)
        await redis_db.rpush("queue:tasks", task_id)

        # Преобразуем данные задачи в строковый формат
        decoded_task = {key: value for key, value in task_data.items()}

        # Добавляем задачу в очередь на выполнение
        background_tasks.add_task(process_task, task_id, task_data, background_tasks)
        # background_tasks.add_task(process_task, next_task_id.decode(), next_task_data, background_tasks)

        return JSONResponse(
            content={
                "task_id": task_id,
                "status": "pending",
                "message": "Task has been added to the queue."
            },
            status_code=202
        )
    except Exception as e:
        logger.error(f"Error in llm_run: {e}")
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
        task_data["min_date"] = int(task_data["min_date"])
        task_data["max_date"] = int(task_data["max_date"])

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


@app.get("/status/{task_id}", tags=['ai analytics'])
async def get_task_status(task_id: str):
    # Ожидаем асинхронный вызов метода hgetall
    task_data = await redis_db.hgetall(f"task:{task_id}")

    # Проверяем, существует ли задача
    if not task_data:
        raise HTTPException(status_code=404, detail="Задача не найдена")

    # Функция возвращает данные в удобном формате
    return {key.decode("utf-8"): value.decode("utf-8") for key, value in task_data.items()}


# Эндпойнт для сброса очереди LLM-задач
@app.post("/reset-queue/", tags=['ai analytics'])
async def reset_queue():
    try:
        # Очищаем очередь задач
        await redis_db.delete("queue:tasks")

        # Делаем все задачи, находящиеся в состоянии "in_progress", в состояние "pending"
        task_ids = await redis_db.lrange("queue:tasks", 0, -1)
        for task_id in task_ids:
            await redis_db.hset(f"task:{task_id.decode()}", "status", "pending")

        return JSONResponse(
            content={
                "message": "Очередь LLM-задач сброшена."
            },
            status_code=200
        )
    except Exception as e:
        logger.error(f"Error in reset_queue: {e}")
        return JSONResponse(
            content={
                "error": str(e)
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
    # print(info_html['min_date'])
    # print(info_html['max_date'])

    data = elastic_query(theme_index=indexes[info_html['index_number']], query_str=info_html['query_str'], 
                         min_date=info_html['min_date'], max_date=info_html['max_date'])
    data = pd.DataFrame(data)

    # Обработка тематики
    df_topic = topic_model.get_topic_info()[['CustomName', 'Topic']]
    dct_df_topic = dict(zip(df_topic['Topic'], df_topic['CustomName']))
    thematics = [dct_df_topic[x] for x in topic_model.topics_] 

    # Объединяем LLM с метаданными
    data.rename(columns={'url': 'text_url'}, inplace=True)
    data = data.join(pd.DataFrame(list(data['authorObject'].values)))
    data.rename(columns={'url': 'author_url'}, inplace=True)
    data = data[['timeCreate', 'hub', 'author_url', 'fullname', 'text_url', 'author_type', 'sex', 'age',
                   'hubtype', 'commentsCount', 'audienceCount',
                   'repostsCount', 'likesCount', 'er', 'viewsCount',
                   'massMediaAudience', 'toneMark', 'country', 'region']]

    # Получение полной таблицы
    df_join = pd.DataFrame(thematics).join(data, how='inner', lsuffix='_df1', rsuffix='_df2')
    df_join.columns = ['Имя кластера', 'Время', 'Источник', 'Ссылка на автора', 'Автор', 'Ссылка на текст', 'Тип автора', 'Пол', 'Возраст',
                       'Тип источника', 'Комментариев', 'Аудитория', 'Репостов', 'Лайков', 'Вовлеченность', 'Просмотров',
                       'Аудитория СМИ', 'Тональность', 'Страна', 'Регион']

    df_join.drop('Аудитория СМИ', axis=1, inplace=True)
    df_join['Тональность'] = df_join['Тональность'].map({0: 'Нейтральная', -1: 'Негатив', 1: 'Позитив'})

    # Получение агрегированной таблицы
    df_group = df_join[['Имя кластера', 'Комментариев', 'Аудитория', 'Репостов', 'Лайков', 'Вовлеченность', 'Просмотров']].copy()
    
    numerical_columns = ['Комментариев', 'Аудитория', 'Репостов', 'Лайков', 'Вовлеченность', 'Просмотров']
    
    for column in numerical_columns:
        df_group[column] = pd.to_numeric(df_group[column], errors='coerce')
        df_group[column] = df_group[column].fillna(0).astype(int)

    # Группировка по 'Тематика' и суммирование
    result = df_group.groupby('Имя кластера').sum().reset_index()

    # Подсчет количества тем
    theme_count = result['Имя кластера'].value_counts()
    result['Количество'] = result['Имя кластера'].map(theme_count)
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

    thematics_path = texts_path + '/' + 'my_list_llm_ans_' + file.replace('.html', '.pkl')

    with open(thematics_path, 'rb') as f:
        texts_thematics = pickle.load(f)
    df_join.insert(1, 'Тематика текста', texts_thematics)

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
    
    return {
        "html_content": html_content,
        "full_data": df_join.where(pd.notnull(df_join), None).to_dict(orient='records'),  # Замена NaN на None
        "aggregated_data": result.where(pd.notnull(result), None).to_dict(orient='records')  # Замена NaN на None
    }

from sqlalchemy.ext.asyncio import AsyncSession
# Функция для получения сессии базы данных
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_maker() as session:
        yield session


from sqlalchemy.future import select

# JWT token scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")
 
# Dependency to get the current user based on the provided token
async def get_current_user(token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    print(555) 
    # print(payload)

    user_id: str = payload.get("sub")
    print(user_id)
    
    # Получаем объект User из базы данных по user_id
    query = select(User).where(User.id == int(user_id))
    result = await db.execute(query)
    user = result.scalar_one_or_none()
    
    if user is None:
        raise credentials_exception
    return user_id 
 

class ResponseModel(BaseModel):
    id: int

    class Config:
        orm_mode = True

# Route to retrieve the current user profile details
@app.get('/user-id', tags=['user'])
async def get_user_profile(current_user: int = Depends(get_current_user)):
    return current_user

def get_user_profile(current_user: User = Depends(get_current_user)):
    return current_user


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


# Добавление файла в папку
@app.post("/add-file/{user_id}/{folder_name}", tags=['data & folders'])
async def add_file(user_id: str, folder_name: str, uploaded_file: UploadFile = File(...)):
    # Проверка, что folder_name предоставлен
    if not folder_name:
        raise HTTPException(status_code=400, detail="Необходимо указать имя папки")
    
    # Путь к файлу с темами
    file_path = '/home/dev/fastapi/analytics_app/data/indexes.pkl'
    indexes = load_dict_from_pickle(file_path)

    # Новая строка для добавления
    new_value = uploaded_file.filename
    next_key = max(indexes.keys()) + 1
    formatted_value = new_value.replace('.json', '').lower()
    indexes[next_key] = formatted_value
    save_dict_to_pickle(file_path, indexes)

    # Устанавливаем путь к директории файла
    file_location = f'/home/dev/fastapi/analytics_app/data/{user_id}/json_files_directory/{folder_name}/{uploaded_file.filename.lower()}'

    max_file_size = 10 * 1024 * 1024 * 1024  # 10 GB
    if uploaded_file.size > max_file_size:
        raise HTTPException(
            status_code=400,
            detail="Размер файла превышает допустимый предел 10 ГБ"
        )

    os.makedirs(os.path.dirname(file_location), exist_ok=True)

    # Проверка существования файла в Redis
    user_folders_data = await redis_db.hget(user_id, "json_files_directory")

    if user_folders_data is None:
        user_folders = {}
    else:
        try:
            user_folders = json.loads(user_folders_data.decode("utf-8"))
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Ошибка при загрузке данных из Redis: {str(e)}"
            )

    # if uploaded_file.filename.lower() in user_folders.get(folder_name, []):
    #     print(555)
    #     print(user_folders.get(folder_name, []))
    #     return f"Файл с именем '{uploaded_file.filename}' уже существует в папке '{folder_name}'."

    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(uploaded_file.file, file_object)

    user_folders.setdefault(folder_name, []).append(uploaded_file.filename.lower())

    await redis_db.hset(user_id, "json_files_directory", json.dumps(user_folders))

    # Загрузка файла в Elasticsearch
    try:
        file_loc = f'/home/dev/fastapi/analytics_app/data/{user_id}/json_files_directory/{folder_name}/'
        load_file_to_elstic(uploaded_file, path=file_loc, next_key=str(next_key))
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail="Загрузите, пожалуйста, валидный json из темы мониторинга"
        )

    return f"Файл {uploaded_file.filename} загружен в папку {folder_name} пользователя - {user_id}!"


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
            os.remove(os.path.join(folder_path, file_name + '.json'))

            return {"message": f"Файл {file_name + '.json'} из директории {directory_name} был успешно удалён!"}
        except Exception as e:
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
    
# Обработка LLM задачи для одного текста
@app.post("/llm-run-single/", tags=['ai analytics'])
async def llm_run(
    analysis_request: SingleTextRequest,
    background_tasks: BackgroundTasks
):
    try:
        task_id = str(uuid.uuid4())
        # Создаем task_data на основе входной модели `AnalysisRequest`
        # Преобразуем данные в строковый формат перед сохранением
        task_data = {
            "task_id": task_id,
            "user_id": str(analysis_request.user_id),
            # "folder_name": str(analysis_request.folder_name),
            "text": str(analysis_request.text),
            "system_prompt": str(analysis_request.system_prompt) if analysis_request.system_prompt else "",
            "prompt_question": str(analysis_request.prompt_question),
            "status": "pending"
        }

        # Сохраняем задачу в Redis для логирования
        await redis_db.hset(f"task:{task_id}", mapping=task_data)

        # Добавляем задачу в очередь на выполнение
        background_tasks.add_task(process_single_text_task, task_id, task_data)

        return JSONResponse(
            content={
                "task_id": task_id,
                "status": "pending",
                "message": "Task has been added to the queue."
            },
            status_code=202
        )
    except Exception as e:
        logging.error(f"Error in llm_run_single: {e}")
        return JSONResponse(
            content={
                "error": str(e)
            },
            status_code=500
        )
        

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
        # Получаем список текстов и вопрос для каждого текста
        texts = task_data['texts']
        prompt_question = task_data['prompt_question']
        system_prompt = task_data.get('system_prompt')

        # Обработка каждого текста
        results = []
        for text in texts:
            result = await generate_answer(text, prompt_question, system_prompt)
            results.append(result)

        # Сериализуем результаты в JSON перед сохранением в Redis
        json_results = json.dumps(results)
        
        # Сохраняем результаты обработки LLM в Redis
        await redis_db.hset(f"task:{task_id}", "result", json_results)

        # Обновляем статус задачи как завершенную
        await redis_db.hset(f"task:{task_id}", "status", "done")
    except Exception as e:
        logging.error(f"Error processing task {task_id}: {str(e)}", exc_info=True)
        await redis_db.hset(f"task:{task_id}", "status", f"failed: {str(e)}")


async def generate_answer(text: str, prompt_question: str, system_prompt: Optional[str]):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "erwan2/DeepSeek-R1-Distill-Qwen-14B",
        "prompt": f"{system_prompt}\n\n{text}\n\n{prompt_question}" if system_prompt else f"{text}\n\n{prompt_question}",
        "stream": False
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as response:
            if response.status == 200:
                response_json = await response.json()
                result = response_json.get("response", "")
                logging.info(f"LLM response: {result}")
                result = result.split('</think>')[1].replace('\n\n', '').replace('\n', '')
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

@app.post("/query")
async def query(request: QueryRequest, session: AsyncSession = Depends(get_db)):
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
            if item["html-file"] == filename:
                theme_index = extract_relevant_part(filename)
                min_date = item["min_date"]
                max_date = item["max_date"]
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

        print(555999777)
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
        print(777)
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


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5001, reload=True)
