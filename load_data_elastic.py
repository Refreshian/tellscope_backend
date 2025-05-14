import time
import json
import uuid
import numpy as np
import tiktoken
from elasticsearch.helpers import bulk
from elasticsearch import Elasticsearch
import os
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models
from tqdm import tqdm
import threading
import logging
import concurrent.futures

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("qdrant_loader")

# Подключение к Elasticsearch
es = Elasticsearch(
    ['194.146.113.123'],
    port=9200
)
background_tasks = {}

client_qdrant = QdrantClient("localhost", port=6333)

client = OpenAI(
    api_key="sk-aitunnel-PrKMg8fNFewHciI2DvmAHGaD8g7cSyjD",
    base_url="https://api.aitunnel.ru/v1/",
)

embed_model = 'text-embedding-3-small'
MAX_TOKENS = 7500
OVERLAP = 200
PARALLEL_WORKERS = 8
EMBED_BATCH_SIZE = 50

encoding = tiktoken.get_encoding("cl100k_base")

def split_text_into_chunks(text, max_tokens=MAX_TOKENS, overlap=OVERLAP):
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return [text]
    chunks = []
    for i in range(0, len(tokens), max_tokens - overlap):
        chunk_tokens = tokens[i:i + max_tokens]
        chunks.append(encoding.decode(chunk_tokens))
    return chunks

def get_embeddings(texts, model=embed_model):
    """Получает эмбеддинги для списка текстов батчами."""
    all_vectors = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i:i + EMBED_BATCH_SIZE]
        try:
            response = client.embeddings.create(
                input=batch,
                model=model
            )
            vectors = [embedding.embedding for embedding in response.data]
            all_vectors.extend(vectors)
        except Exception as e:
            logger.error(f"Ошибка получения эмбеддингов: {str(e)} для фрагментов: {batch[:3]}...")
            all_vectors.extend([None]*len(batch))
    return all_vectors

def process_document_embeddings(document):
    """Обрабатывает документ, разбивает текст и возвращает кортеж (id, chunks, метаданные)."""
    # Приоритетный поиск текстовых полей
    text = None
    # Поля, в которых может содержаться основной текст, в порядке приоритета
    text_fields = ["text", "Текст сообщения", "title"]
    for field in text_fields:
        if field in document and document[field] and isinstance(document[field], str):
            text = document[field].strip()
            if text:  # Если нашли непустое поле с текстом
                break
    if not text:
        return None
    chunks = split_text_into_chunks(text)
    if not chunks:
        return None

    metadata = document.copy()
    metadata["used_text_field"] = next(
        (field for field in text_fields if field in document and document[field] == text), None
    )
    return (document["_id"], text, chunks, metadata)

def batch_process_documents_with_embeddings(documents):
    """Параллельно готовит документы, быстро получает ембеддинги батчами и возвращает для Qdrant."""
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
        futures = [executor.submit(process_document_embeddings, doc) for doc in documents]
        for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Подготовка документов"):
            res = f.result()
            if res: results.append(res)
    logger.info(f"Найдено документов с текстом: {len(results)} из {len(documents)}")
    if not results:
        return []

    global_chunks, index_info = [], []
    for doc_id, text, chunks, metadata in results:
        start = len(global_chunks)
        global_chunks.extend(chunks)
        end = len(global_chunks)
        index_info.append((doc_id, text, (start, end), metadata))

    logger.info(f"Получение эмбеддингов для {len(global_chunks)} текстовых фрагментов")
    all_vectors = get_embeddings(global_chunks)
    if not all_vectors or any([v is None for v in all_vectors]):
        logger.error("Ошибка получения эмбеддингов для некоторых фрагментов.")

    processed_docs = []
    for doc_id, text, (start, end), metadata in index_info:
        chunk_vectors = [v for v in all_vectors[start:end] if v is not None]
        if not chunk_vectors:
            continue
        avg_vector = np.mean(chunk_vectors, axis=0).tolist()
        processed_docs.append({
            "id": doc_id,
            "vector": avg_vector,
            "payload": {
                "text": text,
                "chunks": global_chunks[start:end] if (end-start) > 1 else None,
                "metadata": metadata
            }
        })

    logger.info(f"Подготовлено {len(processed_docs)} документов с векторами")
    return processed_docs

def load_file_to_elstic(filename, path=None):
    logger.info("Запуск загрузки файла в Elasticsearch")

    mapping = {
        "mappings": {
            "properties": {
                "title": {"type": "text", "analyzer": "russian"},
                "text": {"type": "text", "analyzer": "russian"},
                "Текст сообщения": {"type": "text", "analyzer": "russian"},
                "timeCreate": {"type": "long"},
                "hub": {"type": "keyword"},
                "city": {"type": "keyword"},
                "audienceCount": {"type": "integer"},
                "url": {"type": "text", "index": False}
            }
        },
        "settings": {
            "index": {
                "mapping.total_fields.limit": 2000,
                "mapping.ignore_malformed": True
            }
        }
    }

    if path:
        os.chdir(path)

    file_name = filename.filename if hasattr(filename, 'filename') else filename
    new_index = file_name.replace('.json', '').lower()
    logger.info(f"Создание индекса: {new_index}")

    if es.indices.exists(index=new_index):
        es.indices.delete(index=new_index, ignore=[400, 404])
        logger.info(f"Удален существующий индекс: {new_index}")

    response = es.indices.create(
        index=new_index,
        body=mapping,
        ignore=400
    )

    if 'acknowledged' in response and response['acknowledged']:
        logger.info(f"Индекс успешно создан: {new_index}")

    # Дополняем allowed_fields обязательными текстовыми полями и _id
    allowed_fields = [
        'timeCreate', 'title', 'text', 'Текст сообщения', 'hub', 'url', 'hubtype', 'type',
        'commentsCount', 'audienceCount', 'citeIndex', 'repostsCount',
        'likesCount', 'er', 'viewsCount', 'review_rating', 'duplicateCount',
        'massMediaAudience', 'toneMark', 'role', 'aggression', 'country',
        'region', 'city', 'language', 'wom', 'processed', 'authorObject', '_id'  # <--- обязательно _id!
    ]

    documents = []

    # Загрузка данных из файла
    if hasattr(filename, 'filename'):  # если это объект FileStorage
        with open(filename.filename.lower(), "r", encoding="utf-8") as json_file:
            json_data = json.load(json_file)
    else:  # если это просто имя файла
        with open(filename.lower(), "r", encoding="utf-8") as json_file:
            json_data = json.load(json_file)

    for i, doc in enumerate(json_data):
        # Всегда делаем копию под оригинальный ключ, вдруг там еще специфичные поля нужны
        filtered_doc = {k: v for k, v in doc.items() if k in allowed_fields or k in ["Текст сообщения", "title", "text"]}
        # Сохраняем оригинальный номер документа в metadata
        filtered_doc["original_id"] = doc.get("№", i)
        # Всегда делаем строковый id для elastic и qdrant (перезапишем если он уже есть, иначе uuid)
        filtered_doc["_id"] = str(doc.get("_id", str(uuid.uuid4())))

        # Гарантированно формируем поле "text" для всех документов
        if not filtered_doc.get("text"):
            if filtered_doc.get("Текст сообщения"):
                filtered_doc["text"] = filtered_doc["Текст сообщения"]
            elif filtered_doc.get("title"):
                filtered_doc["text"] = filtered_doc["title"]

        documents.append(filtered_doc)

    # Индексация в Elasticsearch
    actions = [
        {
            "_op_type": "index",
            "_index": new_index,
            "_id": doc["_id"],
            **doc
        }
        for doc in documents
    ]
    indexing = bulk(es, actions, chunk_size=100)
    logger.info(f"Индексация завершена. Успешно: {indexing[0]}, Ошибок: {len(indexing[1])}")

    # Подготовка данных для Qdrant
    processed_docs = batch_process_documents_with_embeddings(documents)

    print('processed_docs: {processed_docs}')

    skipped_docs = len(documents) - len(processed_docs)
    logger.info(f"Пропущено документов без текста или с ошибками обработки: {skipped_docs}")

    if len(processed_docs) == 0:
        logger.error("Не удалось извлечь текст ни из одного документа!")
        return {"error": "Не удалось извлечь текст для эмбеддинга из документов"}

    task_id = str(uuid.uuid4())
    background_tasks[task_id] = {
        'status': 'processing',
        'progress': 0,
        'total': len(processed_docs),
        'completed': 0,
        'start_time': time.time(),
        'skipped_docs': skipped_docs,
        'total_docs': len(documents)
    }

    def background_processing():
        try:
            logger.info(f"Начало загрузки в Qdrant для индекса {new_index}")
            client_qdrant.create_collection(
                collection_name=new_index,
                vectors_config=models.VectorParams(
                    size=1536,
                    distance=models.Distance.COSINE
                )
            )
            logger.info(f"Коллекция {new_index} создана в Qdrant")

            batch_size = 50
            total_batches = (len(processed_docs) + batch_size - 1) // batch_size

            for batch_num in range(total_batches):
                start_idx = batch_num * batch_size
                end_idx = min((batch_num + 1) * batch_size, len(processed_docs))

                batch_docs = processed_docs[start_idx:end_idx]

                points = []
                for doc in batch_docs:
                    # Используем original_id как числовой ID для Qdrant, иначе fallback на hash
                    qdrant_id = int(doc["payload"]["metadata"].get("original_id", hash(doc["id"]) % (2**63 - 1)))
                    points.append(
                        models.PointStruct(
                            id=qdrant_id,
                            vector=doc["vector"],
                            payload=doc["payload"]
                        )
                    )

                client_qdrant.upsert(
                    collection_name=new_index,
                    points=points,
                    wait=True
                )

                completed = end_idx
                progress = (completed / len(processed_docs)) * 100
                background_tasks[task_id].update({
                    'progress': round(progress, 2),
                    'completed': completed,
                    'current_batch': batch_num + 1,
                    'total_batches': total_batches
                })
                logger.info(f"Загружен батч {batch_num + 1}/{total_batches} в Qdrant")

            background_tasks[task_id]['status'] = 'completed'
            background_tasks[task_id]['end_time'] = time.time()
            logger.info(f"Загрузка в Qdrant завершена для индекса {new_index}")

        except Exception as e:
            logger.error(f"Ошибка при загрузке в Qdrant: {str(e)}")
            background_tasks[task_id]['status'] = 'failed'
            background_tasks[task_id]['error'] = str(e)

    thread = threading.Thread(target=background_processing)
    thread.daemon = True
    thread.start()

    skipped_docs = len(documents) - len(processed_docs)
    logger.info(f"Пропущено документов без текста или с ошибками обработки - вот: {skipped_docs}")

    return {
        "task_id": task_id,
        "status": "processing",
        "indexed_to_elastic": len(documents),
        "prepared_for_qdrant": len(processed_docs),
        "skipped_docs": skipped_docs
    }