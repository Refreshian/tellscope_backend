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

from ollama import AsyncClient
from collections import OrderedDict
from typing import List, Dict

async def run_llm_query(task_data: dict):
    """Обрабатывает LLM-запрос с обновлением статуса задачи в Redis, с периодическим сохранением результатов."""
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    try:
        # Загружаем данные индекса
        file_path = '/home/dev/fastapi/analytics_app/data/indexes.pkl'
        indexes = load_dict_from_pickle(file_path)

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
                min_date=task_data['min_date'],
                max_date=task_data['max_date']
            )

        # Получаем тексты и ограничиваем их количество
        texts = [x['text'] for x in data]
        texts = texts[:100]  # Ограничение – можно изменить срез
        total_texts = len(texts)
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
        semaphore = asyncio.Semaphore(10)  # Ограничение одновременных запросов
        et = time.time()

        # Путь для сохранения файла (используем абсолютный путь, избегая изменения CWD)
        file_location = f'/home/dev/fastapi/analytics_app/data/{task_data["user_id"]}/bertopic_files_directory/{task_data["folder_name"]}/'
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

        # Функция, которая обрабатывает один уникальный текст и возвращает результат вместе с самим текстом
        async def process_unique_text(text: str):
            if len(text) < 8:
                label = "Короткий текст"
            elif len(text) > 25000:
                label = "Длинный текст"
            else:
                payload = {
                    "model": "erwan2/DeepSeek-R1-Distill-Qwen-14B",
                    "messages": [
                        {
                            "role": "user",
                            "content": f"У меня есть следующий текст:\n{text}\n\n{task_data['promt_question']}"
                        }
                    ]
                }
                try:
                    with torch.no_grad():
                        response = await client.chat(model='Vikhr_Q3', messages=payload['messages'])
                    if response and 'message' in response and 'content' in response['message']:
                        label = response['message']['content']
                    else:
                        label = "bad_request"
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

        pd.DataFrame(zip(texts, llm_labels)).to_excel('/home/dev/fastapi/analytics_app/files/Llm_labels_texts_2.xlsx', 
                                                      index=False)

        print(llm_labels[:10])
        elapsed_time = time.time() - et 
        total_seconds = int(elapsed_time)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60 
        seconds = total_seconds % 60
        execution_llm_time = f"{hours} ч. {minutes} мин. {seconds} сек."
        print('Execution LLM time:', execution_llm_time)


        llm_labels = [re.sub(r"[^\w\s\"«»']", "", label.strip()) for label in llm_labels if label.strip()]

        gc.collect()
        torch.cuda.empty_cache()

        # Обработка эмбеддингов
        embedding_model = SentenceTransformer("/home/dev/fastapi/analytics_app/data/embed_files/DeepPavlov/rubert-base-cased-sentence") # 768-hidden
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
            async with AsyncSession(engine) as session:
                # Пример использования функции сохранения
                await save_embedding_to_pgvector(session, user_id=int(task_data["user_id"]), filename=new_filename, 
                                                folder_name=task_data["folder_name"], vectors=embeddings)
                    
            # Преобразование списка эмбеддингов в массив NumPy
            embeddings = np.array(embeddings)

            umap_model = UMAP(n_neighbors=2, n_components=min(len(embeddings), 5), min_dist=0.0, metric="cosine", random_state=42)
            embeddings_umap = umap_model.fit_transform(embeddings)

            # Обновляем статус после завершения обработки эмбеддингов
            await redis_db.hset(f"task:{task_data['task_id']}", mapping={"embedding_status": "done", "embedding_completed": num_embeddings, "embedding_progress": 100})
        else:
            print("Нет доступных эмбеддингов для обработки.")

        hdbscan_model = HDBSCAN(min_cluster_size=15, metric="euclidean", cluster_selection_method="eom", prediction_data=True)
        hdbscan_model.fit(embeddings_umap)

        labels = hdbscan_model.labels_
        unique_labels, counts = np.unique(labels, return_counts=True)
        num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        num_noise_points = counts[np.where(unique_labels == -1)[0][0]] if -1 in unique_labels else 0

        # Используем преобразованные эмбеддинги для topic_model
        topic_model = BERTopic(embedding_model=embedding_model, verbose=True)
        topics, probs = topic_model.fit_transform(llm_labels, embeddings)  # Теперь `embeddings` - это NumPy массив

        async def generate_topic_label(client, key_words):
            url = "http://localhost:11434/api/generate"
            payload = {
                "model": "llama3",
                "prompt": (
                    f"Используя ключевые слова: {key_words}, сгенерируй на русском языке короткий "
                    "(1 предложение) и понятный заголовок для данной темы, пиши только сам заголовок на русском языке."
                ),
                "stream": False
            }
            with torch.no_grad():
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=payload) as response:
                        if response.status == 200:
                            response_json = await response.json()
                            return response_json.get("response", "")
                        else:
                            print(f"Ошибка при запросе к Ollama: {response.status}")
                            return None

        topic_labels_llama3 = []
        for i, topic in enumerate(topic_model.get_topics().values()):
            key_words = " | ".join(token[0] for token in topic[:10])
            label = await generate_topic_label(client, key_words)
            if label:
                topic_labels_llama3.append(label)

        for i, label in enumerate(topic_labels_llama3):
            print(f"Тема {i}: {label}")

        def shorten_by_words(text, max_words):
            words = text.split()
            if len(words) > max_words:
                return ' '.join(words[:max_words]) + '...'
            return text

        topic_labels_llama3 = [shorten_by_words(topic, 7) for topic in topic_labels_llama3]
        topic_model.set_topic_labels(topic_labels_llama3)

        fig = topic_model.visualize_documents(llm_labels, reduced_embeddings=embeddings_umap, hide_annotations=True, 
                                        hide_document_hover=False, custom_labels=True)

        file_location = f'/home/dev/fastapi/analytics_app/data/{task_data["user_id"]}/bertopic_files_directory/{task_data["folder_name"]}/'
        os.makedirs(os.path.dirname(file_location), exist_ok=True)
        os.chdir(file_location)
        fig.write_html(file_location + new_filename)

        # print(f'Сохранение эмбеддингов: {len(embeddings_umap)}')
        # Сохранение эмбеддингов
        # async with AsyncSession(engine) as session:
        #     await save_embeddings(user_id=task_data["user_id"], filename=new_filename, folder_name=task_data['folder_name'], 
        #                           embeddings=embeddings, session=session)

        filename = 'topic_model_' + new_filename.split('.html')[0]

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


        user_data = await redis_db.execute_command('HGETALL', task_data['user_id'])
        user_data = {key.decode('utf-8'): value.decode('utf-8') for key, value in user_data.items()}

        creation_date = datetime.strptime(current_time, "%Y%m%d_%H%M%S")

        file_info = {
            "html-file": f"{indexes[int(task_data['index'])]}_{current_time}.html",
            "model-file": filename,
            "creation_date": str(creation_date.strftime("%Y-%m-%d %H:%M:%S")),
            "execution_llm_time": execution_llm_time,
            "execution_all_time": execution_all_time, 
            "min_date": task_data['min_date'],
            "max_date": task_data['max_date'],
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
        await reset_gpu_status()
        logging.info(f"GPU статус сброшен. Задача {task_data['task_id']} завершена.")
        print(f"GPU статус сброшен. Задача {task_data['task_id']} завершена.")

        async def reset_all_gpu_processes():
            import subprocess
            subprocess.call("nvidia-smi | awk '/[0-9]+/ {print $5}' | xargs -r kill -9", shell=True)

        if not torch.cuda.is_available():  
            await reset_all_gpu_processes()