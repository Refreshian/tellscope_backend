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
from search_data_elastic import elastic_query
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

from sqlalchemy.orm import sessionmaker, declarative_base

# Определяем базовый класс для моделей
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

async def generate_answers(client, prompt):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "erwan2/DeepSeek-R1-Distill-Qwen-14B", # Vikhr_Q3
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


async def run_llm_query(task_data: dict):

    """Обрабатывает LLM-запрос с обновлением статуса задачи в Redis, с периодическим сохранением результатов."""
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    try:
        # Загружаем данные индекса
        file_path = '/home/dev/tellscope_app/tellscope_backend/data/indexes.pkl'
        indexes = load_dict_from_pickle(file_path)

        try:
            min_data = task_data['min_data']
            max_data = task_data['max_data']
        except:
            min_data = task_data['min_date']
            max_data = task_data['max_date']

        # Выполняем запрос к Elasticsearch
        data = []
        if task_data['query_str'] and task_data['query_str'] != 'all':
            search = task_data['query_str'].split(',')
            for query in search:
                data.extend(elastic_query(theme_index=indexes[int(task_data['index'])], query_str=query))
        else:
            data = elastic_query(
                theme_index=indexes[int(task_data['index'])],
                query_str='all',
                min_date=min_data,
                max_date=max_data
            )

        maxdata = 50000
        # Получаем тексты и ограничиваем их количество
        texts = [x['text'] for x in data]
        texts = texts[:maxdata]  # Ограничение – можно изменить срез
        total_texts = len(texts)
        urls = [x['url'] if 'url' in x else '' for x in data][:maxdata]
        print(f'Текстов для анализа: {total_texts}')
 
        await redis_db.hset(f"task:{task_data['task_id']}", mapping={
            "total_texts": str(total_texts), 
        })

        # Шаг 1: Дедупликация текстов
        unique_texts_dict: Dict[str, List[int]] = defaultdict(list)
        for idx, text in enumerate(texts):
            unique_texts_dict[text].append(idx)

        unique_texts = list(unique_texts_dict.keys())
        unique_total = len(unique_texts)

        # Инициализируем список меток для всех текстов
        llm_labels = [None] * len(texts)

        # Создаём клиент один раз
        client = AsyncClient(host='http://localhost:11434')
        semaphore = asyncio.Semaphore(5)  # Ограничение одновременных запросов
        et = time.time()

        # Путь для сохранения файла (используем абсолютный путь, избегая изменения CWD)
        file_location = f'/home/dev/tellscope_app/tellscope_backend/data/{task_data["user_id"]}/bertopic_files_directory/{task_data["folder_name"]}/'
        os.makedirs(file_location, exist_ok=True)
        # Имя файла (будет использоваться при каждом сохранении)
        file_name = f'my_list_llm_ans_{indexes[int(task_data["index"])]}_{current_time}.pkl'
        file_full_path = os.path.join(file_location, file_name)

        THRESHOLD = 100  # Сохраняем файл после накопления 100 новых обновлений
        
        # Функция для сохранения результатов в файл (вызывается через asyncio.to_thread)
        async def save_labels():
            await asyncio.to_thread(_save_labels)

        def _save_labels():
            # Полная перезапись файла с текущим состоянием llm_labels
            with open(file_full_path, 'wb') as file:
                pickle.dump(llm_labels, file)
            # print(f'Сохранено {file_full_path} в {datetime.now().strftime("%H:%M:%S")}')

        def clean_response(text: str) -> str:
            """Удаляет точку в начале строки и лишние пробелы"""
            text = text.strip()
            if text.startswith('.'):
                text = text[1:].strip()
            return text

        # Функция, которая обрабатывает один уникальный текст и возвращает результат вместе с самим текстом
        async def process_unique_text(text: str):
            if text is None or (isinstance(text, (str, list, tuple, dict)) and len(text) < 8):
                label = "Короткий текст"
            elif len(text) > 25000:
                label = "Длинный текст"
            else:
                payload = {
                    "prompt": f"У меня есть следующий текст:\n{text}\n\n{task_data['promt_question']}",
                    "max_tokens": 100,
                    "top_p": 0.9,
                    "temperature": 0.7
                }
                
                headers = {"Content-Type": "application/json"}
                
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            "http://tellscope40.headsmade.com:8000/v1/completions",
                            json=payload,
                            headers=headers,
                            timeout=60
                        ) as response:
                            if response.status == 200:
                                response_json = await response.json()
                                label = response_json.get("choices", [{}])[0].get("text", "").strip()
                                label = clean_response(label)  # Очистка ответа
                                if not label:
                                    label = "bad_request"
                            else:
                                label = "bad_request"
                                await redis_db.hincrby(f"task:{task_data['task_id']}", "bad_request", 1)
                except asyncio.TimeoutError:
                    label = "Таймаут выполнения"
                    await redis_db.hincrby(f"task:{task_data['task_id']}", "timeout", 1)
                except Exception as e:
                    label = "bad_request"
                    await redis_db.hincrby(f"task:{task_data['task_id']}", "bad_request", 1)
            
            count = len(unique_texts_dict[text])
            return text, count, label

        async def generate_with_semaphore(text: str):
            async with semaphore:
                return await process_unique_text(text)

        async def main():
            completed = 0
            new_count_since_save = 0
            tasks = [generate_with_semaphore(text) for text in unique_texts]

            # Обрабатываем результаты по мере их завершения
            for future in asyncio.as_completed(tasks):
                unique_text, count, label = await future
                indices = unique_texts_dict[unique_text]
                for idx in indices:
                    llm_labels[idx] = label

                completed += count
                new_count_since_save += count

                progress = round((completed / total_texts) * 100, 1)
                await redis_db.hset(f"task:{task_data['task_id']}", mapping={
                    "status": "in progress",
                    "completed_texts": completed,
                    "progress": progress
                })

                # Если накоплено THRESHOLD новых результатов, сохраняем файл
                if new_count_since_save >= THRESHOLD:
                    await save_labels()
                    new_count_since_save = 0

            # Если остались необсохранённые обновления – сохраняем файл
            if new_count_since_save > 0:
                await save_labels()

        # Запускаем основную задачу
        await main()

        # print(texts[:10])
        elapsed_time = time.time() - et
        print('Execution LLM time:', elapsed_time, 'seconds')

        # Обновляем статус задачи в Redis после завершения всех запросов
        await redis_db.hset(f"task:{task_data['task_id']}", mapping={
            "status": "done",
            "completed_texts": total_texts,
            "progress": 100
        })

        print('Текстов в llm_labels: {}'.format(len(llm_labels)))

        print(llm_labels[:10])
        elapsed_time = time.time() - et 
        total_seconds = int(elapsed_time)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60 
        seconds = total_seconds % 60
        execution_llm_time = f"{hours} ч. {minutes} мин. {seconds} сек."
        print('Execution LLM time:', execution_llm_time)

        gc.collect()
        torch.cuda.empty_cache()

        # Обработка эмбеддингов
        # embedding_model = SentenceTransformer("/home/dev/tellscope_app/tellscope_backend/data/embed_files/DeepPavlov/rubert-base-cased-sentence") # 768-hidden
        # embedding_model = SentenceTransformer("/home/dev/tellscope_app/tellscope_backend/data/embed_models/USER2-base") 
        embedding_model = SentenceTransformer("deepvk/USER2-base") 

        embeddings = []
        num_embeddings = len(llm_labels)

        # Обновляем статус перед началом процесса обработки эмбеддингов
        await redis_db.hset(f"task:{task_data['task_id']}", mapping={"embedding_status": "in_progress", "embedding_total": num_embeddings, "embedding_completed": 0, "embedding_progress": 0})

        # Генерация эмбеддингов с обновлением прогресса в Redis
        for i, label in enumerate(llm_labels):
            embedding = embedding_model.encode(label, show_progress_bar=False)
            embeddings.append(embedding)

            # Обновляем количество обработанных эмбеддингов и прогресс в Redis
            completed_embeddings = i + 1
            embedding_progress = round((completed_embeddings / num_embeddings) * 100, 1)
            await redis_db.hset(f"task:{task_data['task_id']}", mapping={
                "embedding_completed": completed_embeddings,
                "embedding_progress": embedding_progress
            })

        # Создаем уникальное имя для файла
        new_filename = f"{indexes[int(task_data['index'])]}_{current_time}.html"
        print(new_filename)

        # Проверяем, есть ли созданные эмбеддинги
        if len(embeddings) > 0:
            # Сохранение эмбеддингов в БД
            # async with AsyncSession(engine) as session:
            #     # Пример использования функции сохранения
            #     await save_embedding_to_pgvector(session, user_id=int(task_data["user_id"]), filename=new_filename, 
            #                                     folder_name=task_data["folder_name"], vectors=embeddings)
                    
            # Преобразование списка эмбеддингов в массив NumPy
            embeddings = np.array(embeddings)

            umap_model = UMAP(n_neighbors=20, n_components=min(len(embeddings), 2), min_dist=0.0, metric="cosine", random_state=42)
            embeddings_umap = umap_model.fit_transform(embeddings)

            # Обновляем статус после завершения обработки эмбеддингов
            await redis_db.hset(f"task:{task_data['task_id']}", mapping={"embedding_status": "done", "embedding_completed": num_embeddings, "embedding_progress": 100})
        else:
            print("Нет доступных эмбеддингов для обработки.")

        print(f'embeddings_umap.shape[0]: {embeddings_umap.shape[0]}')

        hdbscan_model = HDBSCAN(min_cluster_size=5, metric="euclidean", cluster_selection_method="eom", prediction_data=True)
        hdbscan_model.fit(embeddings_umap)

        labels = hdbscan_model.labels_
        unique_labels, counts = np.unique(labels, return_counts=True)
        num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        num_noise_points = counts[np.where(unique_labels == -1)[0][0]] if -1 in unique_labels else 0

        # Используем преобразованные эмбеддинги для topic_model
        representation_model = MaximalMarginalRelevance(diversity=0.8)

        topic_model = BERTopic(embedding_model=embedding_model, verbose=True, representation_model=representation_model)
        topics, probs = topic_model.fit_transform(llm_labels, embeddings)  # Теперь `embeddings` - это NumPy массив

        async def generate_topic_label(client, key_words):
            url = "http://tellscope40.headsmade.com:8000/v1/completions"
            payload = {
                "prompt": (
                    f"Используя ключевые слова: {key_words}, сгенерируй на русском языке короткий "
                    "(1 предложение) и понятный заголовок для данной темы, пиши только сам заголовок на русском языке."
                ),
                "max_tokens": 100,
                "top_p": 0.9,
                "temperature": 0.7
            }
            headers = {"Content-Type": "application/json"}
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=payload, headers=headers) as response:
                        if response.status == 200:
                            response_json = await response.json()
                            label = response_json.get("choices", [{}])[0].get("text", "").strip()
                            label = clean_response(label)  # Очистка ответа
                            return label
                        else:
                            print(f"Ошибка при запросе к API: {response.status}")
                            return None
            except Exception as e:
                print(f"Ошибка соединения: {str(e)}")
                return None

        topic_labels_llama3 = []
        for i, topic in enumerate(topic_model.get_topics().values()):
            key_words = " | ".join(token[0] for token in topic[:14])
            label = await generate_topic_label(client, key_words)
            if label:
                topic_labels_llama3.append(label)

        # for i, label in enumerate(topic_labels_llama3):
        #     print(f"Тема {i}: {label}")

        def shorten_by_words(text, max_words):
            words = text.split()
            if len(words) > max_words:
                return ' '.join(words[:max_words]) + '...'
            return text        


        topic_labels_llama3 = clean_texts(topic_labels_llama3)
        # topic_labels_llama3 = [topic.capitalize() for topic in topic_labels_llama3]
        topic_labels_llama3 = [shorten_by_words(topic, 10).capitalize() for topic in topic_labels_llama3]
        topic_model.set_topic_labels(topic_labels_llama3)
        
        ###################### save html visualize_documents ########################
        fig = topic_model.visualize_documents(llm_labels, reduced_embeddings=embeddings_umap, hide_annotations=True, 
                                        hide_document_hover=False, custom_labels=True, title='Документы и тематики')
        
        file_location = f'/home/dev/tellscope_app/tellscope_backend/data/{task_data["user_id"]}/bertopic_files_directory/{task_data["folder_name"]}/'
        os.makedirs(os.path.dirname(file_location), exist_ok=True)
        os.chdir(file_location)
        fig.write_html(file_location + new_filename)

        ###################### save html datamapplot ########################
        # Проверка на наличие достаточного количества точек данных
        if len(embeddings_umap) > 3:  # Минимум 4 точки для построения симплекса
            try:
                plot = datamapplot.create_interactive_plot(
                    embeddings_umap,
                    llm_labels,
                    font_family="Playfair Display SC",
                    hover_text=urls,
                    # title=f"{indexes[int(task_data['index'])]}",
                    on_click="window.open(`{hover_text}`)",
                    enable_search=True,
                    # cluster_boundary_polygons=True,
                    # cluster_boundary_line_width=5,
                )
            except ValueError as e:
                print(f"Ошибка при построении графика: {str(e)}")
                plot = None
        else:
            print("Недостаточно точек данных для построения графика.")
            plot = None

        if plot:
            file_location = f'/home/dev/tellscope_app/tellscope_backend/data/{task_data["user_id"]}/bertopic_files_directory/{task_data["folder_name"]}/'
            os.makedirs(os.path.dirname(file_location), exist_ok=True)
            os.chdir(file_location)
            filename = f'datamapplot_{new_filename}'
            plot.save(filename)
        else:
            print("Не удалось построить график.")


        filename = 'topic_model_' + new_filename.split('.html')[0]

        ###################### save topics over time ########################
        timestamps = [x['timeCreate'] for x in data][:maxdata]
        formatted_dates = [datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S') for ts in timestamps]
        
        topics_over_time = topic_model.topics_over_time(llm_labels[:maxdata], formatted_dates, nr_bins=20)
        topics_over_time_viz = visualize_topics_over_time(topic_model, topics_over_time, top_n_topics=20, title='Тематики во времени', 
                                                                      width=1250, height=550, custom_labels=True)
                
        topic_over_time_filename = f"{indexes[int(task_data['index'])]}_topic_over_time_{current_time}.html"
        # topics_over_time_viz.save(file_location + topic_over_time_filename)
        topics_over_time_viz.write_html(file_location + topic_over_time_filename)

        # with the original embeddings
        # topic_model.visualize_document_datamap(docs, embeddings=embeddings)

        # with the reduced embeddings

        ################################# visualize_document_datamap ######################################

        # vis_document_datamap = visualize_document_datamap(topic_model, 
        #                                                         llm_labels, embeddings=embeddings,
        #                                                         title = "Темы сообщений", 
        #                                                         enable_search=True, 
        #                                                         topic_prefix=False,
        #                                                         interactive=True) 
        # vis_document_datamap.savefig("/home/dev/tellscope_app/tellscope_backend/files/visualize_document_datamap_int.png", 
        #                                    bbox_inches="tight")

        elapsed_time = time.time() - et
        total_seconds = int(elapsed_time)

        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        execution_all_time = f"{hours} ч. {minutes} мин. {seconds} сек."
        print('Execution All time:', execution_all_time, 'seconds')

        try:
            os.chdir(file_location)
            topic_model.save(filename, serialization="safetensors", save_ctfidf=True, save_embedding_model=embedding_model)
            print(f"Модель успешно сохранена в: {file_location }")
        except Exception as e:
            print(f"Ошибка при сохранении модели: {e}")


        ###################################### save topic labels #################################
        # Путь для сохранения файла (используем абсолютный путь, избегая изменения CWD)
        file_location = f'/home/dev/tellscope_app/tellscope_backend/data/{task_data["user_id"]}/bertopic_files_directory/{task_data["folder_name"]}/'
        os.makedirs(file_location, exist_ok=True)
        topic_model = BERTopic.load(filename)
        # print(df_topic.columns)
        # print(df_topic[['Topic', 'Count', 'Representative_Docs']])
        df_topic = topic_model.get_topic_info()[['Topic', 'CustomName']]
        dct_df_topic = dict(zip(df_topic['Topic'], df_topic['CustomName']))
        # Замена меток на текстовые названия
        text_labels = [dct_df_topic[label] for label in topics]
        # Имя файла 
        file_name = f'topic_names_{indexes[int(task_data["index"])]}_{current_time}.pkl'
        file_full_path = os.path.join(file_location, file_name)
        
        with open(file_full_path, 'wb') as file:
            pickle.dump(text_labels, file)


        user_data = await redis_db.execute_command('HGETALL', task_data['user_id'])
        user_data = {key.decode('utf-8'): value.decode('utf-8') for key, value in user_data.items()}

        creation_date = datetime.strptime(current_time, "%Y%m%d_%H%M%S")

        file_info = {
            "html-file": f"{indexes[int(task_data['index'])]}_{current_time}.html",
            "model-file": filename,
            "creation_date": str(creation_date.strftime("%Y-%m-%d %H:%M:%S")),
            "execution_llm_time": execution_llm_time,
            "execution_all_time": execution_all_time, 
            "min_data": task_data['min_data'],
            "max_data": task_data['max_data'],
            "index_number": int(task_data['index']),
            "task_id": task_data['task_id'],
            "query_str": task_data['query_str'],
            "count_texts": total_texts,
            "unique_texts": unique_total,
            "promt_question": task_data['promt_question'],
        }

        if user_data:
            if "bertopic_files_directory" in user_data:
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
        else:
            raise Exception("User data does not exist.")

    except Exception as e:
        logging.error(f"Ошибка при обработке задачи {task_data['task_id']}: {e}")
        traceback.print_exc()
        await redis_db.hset(f"task:{task_data['task_id']}", mapping={"status": "failed", "error": str(e)})

    finally:

        await redis_db.hset(f"task:{task_data['task_id']}", mapping={
            "final_status": "done",
            "html-file": f"{indexes[int(task_data['index'])]}_{current_time}.html",
            "folder_name": task_data['folder_name']
        })

        await reset_gpu_status()
        logging.info(f"GPU статус сброшен. Задача {task_data['task_id']} завершена.")
        print(f"GPU статус сброшен. Задача {task_data['task_id']} завершена.")

        async def reset_all_gpu_processes():
            import subprocess
            subprocess.call("nvidia-smi | awk '/[0-9]+/ {print $5}' | xargs -r kill -9", shell=True)

        if not torch.cuda.is_available():  
            await reset_all_gpu_processes()