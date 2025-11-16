from dotenv import load_dotenv
load_dotenv()

import time
import json
import uuid
import numpy as np
import tiktoken
from elasticsearch.helpers import bulk, parallel_bulk
from elasticsearch import Elasticsearch
import os
from qdrant_client import QdrantClient
from qdrant_client.http import models
from tqdm import tqdm
import logging
import redis
import torch
import gc
from embedding_model_manager import model_manager
from elasticsearch import helpers
from datetime import datetime
import threading
from progress_utils import safe_update_progress
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("qdrant_loader")

# Подключения
redis_client = redis.Redis(host='localhost', port=6379, db=0)

es = Elasticsearch(
    hosts=["http://localhost:9200"],
    basic_auth=("elastic", "biz8z5i1w0nLPmEweKgP"),
    verify_certs=False,
    headers={"Accept": "application/vnd.elasticsearch+json; compatible-with=9"}
)

client_qdrant = QdrantClient("localhost", port=6333)

# Оптимизированные константы
MAX_TOKENS = 6000  # Уменьшено для ускорения
OVERLAP = 150      # Уменьшено
EMBED_BATCH_SIZE = 256  # Увеличено значительно
QDRANT_BATCH_SIZE = 200  # Увеличено для Qdrant
ES_BATCH_SIZE = 2000     # Увеличено для Elasticsearch

encoding = tiktoken.get_encoding("cl100k_base")

def split_text_into_chunks_optimized(text, max_tokens=MAX_TOKENS, overlap=OVERLAP):
    """Оптимизированная разбивка текста"""
    if not text or not text.strip():
        return []
        
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return [text]
    
    chunks = []
    step = max_tokens - overlap
    
    for i in range(0, len(tokens), step):
        chunk_tokens = tokens[i:i + max_tokens]
        if len(chunk_tokens) > 50:  # Минимальный размер чанка
            chunks.append(encoding.decode(chunk_tokens))
    
    return chunks

def process_documents_batch(documents_batch):
    """Обработка батча документов (может быть распараллелена)"""
    results = []
    text_fields = ["text", "Текст сообщения", "title", "content", "message", "description"]
    
    for document in documents_batch:
        if not isinstance(document, dict):
            continue
            
        text = None
        for field in text_fields:
            if field in document and isinstance(document[field], str) and document[field].strip():
                text = document[field].strip()
                break
                
        if not text:
            continue
            
        chunks = split_text_into_chunks_optimized(text)
        if not chunks:
            continue
            
        metadata = document.copy()
        metadata["used_text_field"] = next(
            (field for field in text_fields if field in document and document[field] == text), None
        )
        
        doc_id = document.get('id', str(uuid.uuid4()))
        results.append((doc_id, text, chunks, metadata))
    
    return results

from concurrent.futures import ThreadPoolExecutor  # leave this

def batch_process_documents_with_embeddings_optimized(documents, task_id=None):
    """Оптимизированная обработка документов с векторизацией"""
    if task_id:
        safe_update_progress(task_id, 30, stage="chunking", 
                           stage_details=f"Обработка {len(documents)} документов")
    
    try:
        logger.info(f"Начало обработки {len(documents)} документов")
        
        # Параллельная обработка документов батчами
        cpu_count = min(os.cpu_count() or 1, 4)
        batch_size = max(len(documents) // cpu_count, 100)
        document_batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]

        if len(document_batches) > 1:
            with ThreadPoolExecutor(max_workers=cpu_count) as executor:
                batch_results = list(executor.map(process_documents_batch, document_batches))
            results = [item for br in batch_results for item in br]
        else:
            results = process_documents_batch(documents)
        
        if not results:
            logger.warning("Нет документов для обработки")
            return []
        
        logger.info(f"Обработано {len(results)} документов, получено текстовых фрагментов")
        
        # Подготовка данных для векторизации
        global_chunks = []
        index_info = []
        
        for doc_id, text, chunks, metadata in results:
            start = len(global_chunks)
            global_chunks.extend(chunks)
            end = len(global_chunks)
            index_info.append((doc_id, text, (start, end), metadata))
        
        logger.info(f"Подготовлено {len(global_chunks)} фрагментов для векторизации")
        
        if task_id:
            safe_update_progress(task_id, 40, stage="embedding", 
                               stage_details=f"Векторизация {len(global_chunks)} фрагментов")
        
        # Оптимизированная векторизация большими батчами
        all_vectors = []
        chunk_batch_size = EMBED_BATCH_SIZE
        total_batches = (len(global_chunks) + chunk_batch_size - 1) // chunk_batch_size
        
        for batch_idx in range(0, len(global_chunks), chunk_batch_size):
            batch_chunks = global_chunks[batch_idx:batch_idx + chunk_batch_size]
            
            try:
                batch_vectors = model_manager.encode_texts(
                    batch_chunks,
                    batch_size=chunk_batch_size,
                    normalize_embeddings=True
                )
                
                if isinstance(batch_vectors, np.ndarray):
                    batch_vectors = batch_vectors.tolist()
                elif not isinstance(batch_vectors, list):
                    batch_vectors = [batch_vectors] if batch_vectors is not None else []
                
                all_vectors.extend(batch_vectors)
                
                # Очистка памяти после каждого батча
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Обновление прогресса
                if task_id:
                    progress = 40 + int(((batch_idx + chunk_batch_size) / len(global_chunks)) * 30)
                    safe_update_progress(task_id, progress, stage="embedding",
                                       stage_details=f"Обработано {min(batch_idx + chunk_batch_size, len(global_chunks))}/{len(global_chunks)} фрагментов")
                
                logger.info(f"Обработан батч {batch_idx//chunk_batch_size + 1}/{total_batches}")
                
            except Exception as e:
                logger.error(f"Ошибка векторизации батча: {e}")
                # Добавляем пустые векторы для пропущенного батча
                all_vectors.extend([None] * len(batch_chunks))
        
        if task_id:
            safe_update_progress(task_id, 75, stage="preparing", 
                               stage_details="Подготовка документов для загрузки")
        
        # Быстрая сборка финальных документов
        processed_docs = []
        
        for doc_id, text, (start, end), metadata in index_info:
            chunk_vectors = [v for v in all_vectors[start:end] if v is not None and len(v) > 0]
            
            if not chunk_vectors:
                continue
            
            try:
                # Оптимизированное вычисление среднего
                avg_vector = np.mean(chunk_vectors, axis=0).tolist()
            except Exception as e:
                logger.error(f"Ошибка среднего вектора для {doc_id}: {e}")
                continue
            
            doc_payload = {
                "content": text,
                "chunks": global_chunks[start:end] if (end - start) > 1 else None,
                "metadata": metadata
            }
            
            processed_doc = {
                "id": doc_id,
                "vector": avg_vector,
                "payload": doc_payload
            }
            processed_docs.append(processed_doc)
        
        logger.info(f"Подготовлено {len(processed_docs)} документов для загрузки в Qdrant")
        
        if task_id:
            safe_update_progress(task_id, 80, stage="preparing",
                               stage_details=f"Подготовлено {len(processed_docs)} документов")
        
        return processed_docs
        
    except Exception as e:
        logger.error(f"Критическая ошибка в batch_process_documents_with_embeddings_optimized: {e}")
        return []
    finally:
        # Очистка памяти
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

def load_to_qdrant_optimized(collection_name, documents, task_id):
    """Оптимизированная загрузка в Qdrant"""
    if not documents:
        raise ValueError("Список документов пуст!")
    
    try:
        logger.info(f"Начало загрузки {len(documents)} документов в Qdrant")
        
        # Получение блокировки
        if not acquire_qdrant_lock(collection_name, task_id):
            raise Exception("Не удалось получить блокировку коллекции")
        
        safe_update_progress(task_id, 80, stage="qdrant_preparation", 
                           stage_details="Подготовка к загрузке в Qdrant")
        
        # Создание коллекции если не существует
        if not client_qdrant.collection_exists(collection_name):
            vector_size = len(documents[0]["vector"])
            logger.info(f"Создание коллекции {collection_name} с размерностью {vector_size}")
            
            client_qdrant.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                ),
                # Оптимизация для быстрой загрузки
                optimizers_config=models.OptimizersConfigDiff(
                    indexing_threshold=0,  # Отключаем индексацию во время загрузки
                ),
                hnsw_config=models.HnswConfigDiff(
                    payload_m=16,
                    m=0  # Временно отключаем HNSW
                )
            )
        
        # Оптимизированная загрузка большими батчами
        batch_size = QDRANT_BATCH_SIZE
        total_docs = len(documents)
        
        # Подготавливаем все точки сразу
        points = []
        for i, doc in enumerate(documents):
            if isinstance(doc["id"], str) and doc["id"].isdigit():
                point_id = int(doc["id"])
            else:
                point_id = hash(str(doc["id"])) % (2**31)  # Положительное число
            
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=doc["vector"],
                    payload=doc["payload"]
                )
            )
        
        # Загрузка батчами с параллелизмом
        uploaded = 0
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            
            try:
                client_qdrant.upsert(
                    collection_name=collection_name,
                    points=batch,
                    wait=False  # Асинхронная загрузка для ускорения
                )
                
                uploaded += len(batch)
                progress = 85 + int((uploaded / total_docs) * 15)
                
                safe_update_progress(task_id, progress, stage="qdrant_upload",
                                   stage_details=f"Загружено {uploaded}/{total_docs} документов")
                
                # logger.info(f"Загружен батч {i//batch_size + 1}, всего: {uploaded}/{total_docs}")
                
            except Exception as e:
                logger.error(f"Ошибка загрузки батча: {e}")
                # Продолжаем с меньшим батчем
                if batch_size > 50:
                    batch_size = batch_size // 2
                    continue
                raise e
        
        # Ждем завершения всех операций
        time.sleep(1)
        
        # Включаем индексацию обратно
        client_qdrant.update_collection(
            collection_name=collection_name,
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=20000,
            )
        )
        
        logger.info(f"✅ Загрузка в Qdrant завершена: {total_docs} документов")
        safe_update_progress(task_id, 100, status="completed", stage="completed",
                           stage_details="Индексация завершена успешно")
        
    except Exception as e:
        logger.error(f"Ошибка загрузки в Qdrant: {e}")
        safe_update_progress(task_id, 0, status="failed", error=str(e))
        raise e
    finally:
        release_qdrant_lock(collection_name, task_id)

def load_file_to_elstic(filename, path=None, task_id=None):
    """Оптимизированная загрузка файла"""
    # logger.info("🚀 Запуск оптимизированной загрузки файла")
    
    if task_id is None:
        task_id = str(uuid.uuid4())
    
    try:
        # Оптимизированный mapping для Elasticsearch
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
                    "mapping.total_fields.limit": 3000,
                    "mapping.ignore_malformed": True,
                    "number_of_shards": 1,
                    "number_of_replicas": 0,  # Отключаем реплики для ускорения
                    "refresh_interval": "30s",  # Увеличиваем интервал обновления
                    "translog": {
                        "flush_threshold_size": "1gb"
                    }
                }
            }
        }
        
        if path:
            os.chdir(path)
        
        file_name = filename.filename if hasattr(filename, 'filename') else filename
        new_index = file_name.replace('.json', '').lower()
        
        logger.info(f"Создание оптимизированного индекса: {new_index}")
        
        # Удаляем существующий индекс
        if es.indices.exists(index=new_index):
            es.indices.delete(index=new_index, ignore=[400, 404])
        
        # Создаем новый индекс
        response = es.indices.create(index=new_index, body=mapping, ignore=400)
        
        if not ('acknowledged' in response and response['acknowledged']):
            logger.error(f"Ошибка создания индекса: {response}")
            return {"status": "failed", "error": "Ошибка создания индекса"}
        
        # Быстрая загрузка JSON
        logger.info(f"Загрузка данных из {file_name}")
        with open(file_name, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        if not isinstance(data, list) or not data:
            return {"status": "failed", "error": "Некорректный формат JSON"}
        
        logger.info(f"Загружено {len(data)} документов из JSON")
        
        # Оптимизированная подготовка для bulk индексации
        actions = []
        for i, doc in enumerate(data):
            if not isinstance(doc, dict):
                continue
            
            doc_id = str(doc.get('id', doc.get('idExternal', str(uuid.uuid4()))))
            
            if not any(field in doc for field in ["text", "Текст сообщения", "title", "content"]):
                continue
            
            actions.append({
                "_index": new_index,
                "_id": doc_id,
                "_source": doc
            })
        
        if not actions:
            return {"status": "failed", "error": "Нет данных для индексации"}
        
        logger.info(f"Подготовлено {len(actions)} документов для Elasticsearch")
        
        # Оптимизированная bulk индексация
        success_count = 0
        error_count = 0
        
        for success, info in parallel_bulk(
            es,
            actions,
            chunk_size=ES_BATCH_SIZE,
            max_chunk_bytes=50*1024*1024,  # 50MB батчи
            thread_count=4,  # Параллельные потоки
            queue_size=8
        ):
            if not success:
                error_count += 1
                logger.error(f"Ошибка индексации: {info}")
            else:
                success_count += 1
        
        # Принудительное обновление индекса
        es.indices.refresh(index=new_index)
        total_docs = es.count(index=new_index)['count']
        
        logger.info(f"✅ Elasticsearch индексация завершена:")
        logger.info(f"   Успешно: {success_count}, Ошибок: {error_count}, Всего в индексе: {total_docs}")
        
        if total_docs == 0:
            return {"status": "failed", "error": "Индекс пуст после загрузки"}
        
        # Оптимизированная обработка для Qdrant
        logger.info("🔄 Начало обработки для Qdrant")
        processed_docs = batch_process_documents_with_embeddings_optimized(data, task_id)
        
        if not processed_docs:
            return {"status": "failed", "error": "Нет документов для Qdrant"}
        
        logger.info(f"📊 Статистика обработки:")
        logger.info(f"   Исходных документов: {len(data)}")
        logger.info(f"   Обработано для Qdrant: {len(processed_docs)}")
        logger.info(f"   Пропущено: {len(data) - len(processed_docs)}")
        
        # Создаем задачу в Redis
        redis_client.hset(
            f"task:{task_id}",
            mapping={
                "status": "processing",
                "progress": "80",
                "total": str(len(processed_docs)),
                "start_time": datetime.now().isoformat(),
                "total_docs": str(len(data))
            }
        )
        
        # Оптимизированная загрузка в Qdrant
        try:
            load_to_qdrant_optimized(new_index, processed_docs, task_id)
        except Exception as e:
            logger.error(f"Ошибка Qdrant: {e}")
            return {"status": "failed", "error": str(e)}
        
        logger.info("🎉 Обработка файла полностью завершена")
        
        return {
            "status": "completed",
            "task_id": task_id,
            "index_name": new_index,
            "processed_docs": len(processed_docs),
            "elasticsearch_docs": total_docs
        }
        
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        return {"status": "failed", "error": str(e)}
    finally:
        try:
            model_manager.cleanup()
            # logger.info("✅ Ресурсы очищены")
        except Exception as e:
            logger.warning(f"Ошибка очистки: {e}")

# Вспомогательные функции (без изменений)
def acquire_qdrant_lock(collection_name, task_id, timeout=30):
    lock_key = f"qdrant_lock:{collection_name}"
    deadline = time.time() + timeout
    
    while time.time() < deadline:
        if redis_client.set(lock_key, task_id, nx=True, ex=120):  # 2 минуты
            # logger.info(f"🔒 Получена блокировка {collection_name}")
            return True
        time.sleep(1)
    
    return False

def release_qdrant_lock(collection_name, task_id):
    lock_key = f"qdrant_lock:{collection_name}"
    owner = redis_client.get(lock_key)
    
    if owner and owner.decode('utf-8') == task_id:
        redis_client.delete(lock_key)
        # logger.info(f"🔓 Освобождена блокировка {collection_name}")
        return True
    return False