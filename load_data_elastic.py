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
    headers={"Accept": "application/vnd.elasticsearch+json; compatible-with=9"},
    # Отключаем sniffing - это важно!
    sniff_on_start=False,
    sniff_on_node_failure=False,
    sniff_before_requests=False,
    # Настройки таймаутов
    request_timeout=30,
    max_retries=3,
    retry_on_timeout=True
)

# Добавьте в начало load_data_elastic.py для проверки
try:
    info = es.info()
    logger.info(f"✅ Подключение к Elasticsearch: {info['version']['number']}")
except Exception as e:
    logger.error(f"❌ Ошибка подключения к Elasticsearch: {e}")

client_qdrant = QdrantClient("localhost", port=6333)

# Оптимизированные константы
MAX_TOKENS = 6000
OVERLAP = 150
EMBED_BATCH_SIZE = 256
QDRANT_BATCH_SIZE = 200
ES_BATCH_SIZE = 2000

encoding = tiktoken.get_encoding("cl100k_base")

def safe_bulk_index(actions, index_name, max_retries=3):
    for attempt in range(max_retries):
        try:
            success, errors = helpers.bulk(
                es,
                actions,
                chunk_size=100,
                request_timeout=120,
                raise_on_error=False
            )
            return success, errors
        except ConnectionError as e:
            if attempt == max_retries - 1:
                raise e
            logger.warning(f"Попытка {attempt + 1} не удалась, переподключаемся...")
            time.sleep(2 ** attempt)
            
def split_text_into_chunks_optimized(text, max_tokens=MAX_TOKENS, overlap=OVERLAP):
    """Оптимизированная разбивка текста"""
    if not text or not isinstance(text, str) or not text.strip():
        return []
    
    try:
        tokens = encoding.encode(text)
        
        if not tokens or len(tokens) == 0:
            return []
            
        if len(tokens) <= max_tokens:
            return [text]
        
        chunks = []
        step = max_tokens - overlap
        
        for i in range(0, len(tokens), step):
            chunk_tokens = tokens[i:i + max_tokens]
            
            if chunk_tokens and isinstance(chunk_tokens, list) and len(chunk_tokens) > 50:
                try:
                    decoded_chunk = encoding.decode(chunk_tokens)
                    if decoded_chunk and decoded_chunk.strip():
                        chunks.append(decoded_chunk)
                except Exception as e:
                    logger.warning(f"Ошибка декодирования чанка: {e}")
                    continue
        
        return chunks
        
    except Exception as e:
        logger.error(f"Ошибка разбивки текста: {e}")
        return []

def validate_document_numeric_fields(doc):
    """Валидация числовых полей документа с детальным логированием"""
    if not isinstance(doc, dict):
        return doc
    
    numeric_fields = ["timeCreate", "audienceCount"]
    
    for field in numeric_fields:
        if field in doc:
            original_value = doc[field]
            
            # 🔍 ДЕТАЛЬНОЕ ЛОГИРОВАНИЕ
            logger.debug(f"Проверка поля {field}: value={original_value}, type={type(original_value)}")
            
            # Проверка и конвертация
            if original_value is None or original_value == "" or original_value == "null":
                logger.warning(f"Поле {field} содержит None/пустое значение, устанавливаем 0")
                doc[field] = 0
            else:
                try:
                    if isinstance(original_value, (int, float)):
                        if np.isnan(original_value) or np.isinf(original_value):
                            logger.warning(f"Поле {field} содержит NaN/Inf, устанавливаем 0")
                            doc[field] = 0
                        else:
                            doc[field] = float(original_value)
                    elif isinstance(original_value, str):
                        cleaned = original_value.strip()
                        if cleaned and cleaned.lower() not in ['none', 'null', 'nan']:
                            cleaned = cleaned.replace(',', '.').replace(' ', '')
                            if cleaned.replace('.', '').replace('-', '').replace('+', '').isdigit():
                                converted = float(cleaned)
                                if np.isnan(converted) or np.isinf(converted):
                                    logger.warning(f"Поле {field} конвертировано в NaN/Inf, устанавливаем 0")
                                    doc[field] = 0
                                else:
                                    doc[field] = converted
                            else:
                                logger.warning(f"Поле {field} не является числом: '{cleaned}', устанавливаем 0")
                                doc[field] = 0
                        else:
                            logger.warning(f"Поле {field} пустое после очистки, устанавливаем 0")
                            doc[field] = 0
                    else:
                        logger.warning(f"Поле {field} неизвестного типа: {type(original_value)}, устанавливаем 0")
                        doc[field] = 0
                        
                except (ValueError, TypeError, AttributeError) as e:
                    logger.warning(f"Ошибка конвертации {field}: {original_value} -> {e}, устанавливаем 0")
                    doc[field] = 0
            
            # 🔍 ФИНАЛЬНАЯ ПРОВЕРКА
            final_value = doc[field]
            logger.debug(f"Поле {field} после валидации: {final_value}, type={type(final_value)}")
            
            # Критическая проверка
            if final_value is None:
                logger.error(f"❌ КРИТИЧЕСКАЯ ОШИБКА: Поле {field} все еще None после валидации!")
                doc[field] = 0
    
    return doc

def process_documents_batch(documents_batch):
    """Обработка батча документов с детальным логированием"""
    results = []
    text_fields = ["text", "Текст сообщения", "title", "content", "message", "description"]
    
    for idx, document in enumerate(documents_batch):
        try:
            if not isinstance(document, dict):
                logger.warning(f"Документ {idx} не является словарем: {type(document)}")
                continue
            
            # 🔍 ЛОГИРОВАНИЕ ДО ВАЛИДАЦИИ
            logger.debug(f"Документ {idx} ДО валидации: timeCreate={document.get('timeCreate')}, audienceCount={document.get('audienceCount')}")
            
            # Валидация
            document = validate_document_numeric_fields(document)
            
            # 🔍 ЛОГИРОВАНИЕ ПОСЛЕ ВАЛИДАЦИИ
            logger.debug(f"Документ {idx} ПОСЛЕ валидации: timeCreate={document.get('timeCreate')}, audienceCount={document.get('audienceCount')}")
            
            # Критическая проверка
            for field in ["timeCreate", "audienceCount"]:
                if field in document:
                    value = document[field]
                    if value is None:
                        logger.error(f"❌ НАЙДЕН None в документе {idx} поле {field} ПОСЛЕ валидации!")
                        document[field] = 0
                    # 🔍 ПРОВЕРКА НА ВОЗМОЖНОСТЬ СРАВНЕНИЯ
                    try:
                        _ = value < 0  # Пробуем сравнение
                    except TypeError as te:
                        logger.error(f"❌ Ошибка сравнения в документе {idx} поле {field}: {te}")
                        logger.error(f"   Значение: {value}, тип: {type(value)}")
                        document[field] = 0
            
            # Поиск текстового поля
            text = None
            for field in text_fields:
                if field in document:
                    field_value = document[field]
                    if isinstance(field_value, str) and field_value.strip():
                        text = field_value.strip()
                        break
            
            if not text:
                continue
            
            # Разбивка на чанки
            chunks = split_text_into_chunks_optimized(text)
            
            if not chunks or len(chunks) == 0:
                logger.warning(f"Не удалось создать чанки для документа {document.get('id', 'unknown')}")
                continue
            
            # Подготовка метаданных
            metadata = document.copy()
            
            # 🔍 ФИНАЛЬНАЯ ПРОВЕРКА МЕТАДАННЫХ
            for key in ["timeCreate", "audienceCount"]:
                if key in metadata and metadata[key] is None:
                    logger.error(f"❌ Найден None в метаданных для ключа {key}")
                    metadata[key] = 0
            
            metadata["used_text_field"] = next(
                (field for field in text_fields if field in document and document.get(field) == text), 
                None
            )
            
            doc_id = document.get('id') or document.get('idExternal') or str(uuid.uuid4())
            results.append((doc_id, text, chunks, metadata))
            
        except Exception as e:
            logger.error(f"Ошибка обработки документа {idx}: {e}", exc_info=True)
            continue
    
    return results

def batch_process_documents_with_embeddings_optimized(documents, task_id=None):
    """Оптимизированная обработка с детальным логированием"""
    if task_id:
        safe_update_progress(task_id, 30, stage="chunking", 
                           stage_details=f"Обработка {len(documents)} документов")
    
    try:
        logger.info(f"Начало обработки {len(documents)} документов")
        
        # 🔍 ПРОВЕРКА ВХОДНЫХ ДАННЫХ
        logger.info(f"Проверка первых 3 документов на None значения...")
        for i, doc in enumerate(documents[:3]):
            if isinstance(doc, dict):
                for field in ["timeCreate", "audienceCount"]:
                    if field in doc:
                        value = doc[field]
                        logger.info(f"  Документ {i}, поле {field}: {value} (type: {type(value)})")
                        if value is None:
                            logger.error(f"  ❌ НАЙДЕН None во входных данных!")
        
        # Параллельная обработка
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
        
        logger.info(f"Обработано {len(results)} документов")
        
        # Подготовка для векторизации
        global_chunks = []
        index_info = []
        
        for doc_id, text, chunks, metadata in results:
            # 🔍 ПРОВЕРКА МЕТАДАННЫХ
            for field in ["timeCreate", "audienceCount"]:
                if field in metadata:
                    value = metadata[field]
                    if value is None:
                        logger.error(f"❌ None в метаданных документа {doc_id} поле {field}")
                        metadata[field] = 0
            
            start = len(global_chunks)
            global_chunks.extend(chunks)
            end = len(global_chunks)
            index_info.append((doc_id, text, (start, end), metadata))
        
        logger.info(f"Подготовлено {len(global_chunks)} фрагментов для векторизации")
        
        if task_id:
            safe_update_progress(task_id, 40, stage="embedding", 
                               stage_details=f"Векторизация {len(global_chunks)} фрагментов")
        
        # Векторизация
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
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                if task_id:
                    progress = 40 + int(((batch_idx + chunk_batch_size) / len(global_chunks)) * 30) if global_chunks else 40
                    safe_update_progress(task_id, progress, stage="embedding",
                                       stage_details=f"Обработано {min(batch_idx + chunk_batch_size, len(global_chunks))}/{len(global_chunks)} фрагментов")
                
                logger.info(f"Обработан батч {batch_idx//chunk_batch_size + 1}/{total_batches}")
                
            except Exception as e:
                logger.error(f"Ошибка векторизации батча: {e}")
                all_vectors.extend([None] * len(batch_chunks))
        
        if task_id:
            safe_update_progress(task_id, 75, stage="preparing", 
                               stage_details="Подготовка документов для загрузки")
        
        # Сборка финальных документов
        processed_docs = []
        
        for doc_id, text, (start, end), metadata in index_info:
            # 🔍 ФИНАЛЬНАЯ ПРОВЕРКА ПЕРЕД ДОБАВЛЕНИЕМ
            for field in ["timeCreate", "audienceCount"]:
                if field in metadata:
                    value = metadata[field]
                    if value is None:
                        logger.error(f"❌ None в финальных метаданных {doc_id} поле {field}")
                        metadata[field] = 0
                    # Проверка на возможность сравнения
                    try:
                        _ = value < 0
                    except TypeError as te:
                        logger.error(f"❌ Ошибка сравнения в финальных метаданных: {te}")
                        logger.error(f"   doc_id={doc_id}, field={field}, value={value}, type={type(value)}")
                        metadata[field] = 0
            
            chunk_vectors = [v for v in all_vectors[start:end] if v is not None and len(v) > 0]
            
            if not chunk_vectors:
                continue
            
            try:
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
        logger.error(f"Критическая ошибка в batch_process_documents_with_embeddings_optimized: {e}", exc_info=True)
        return []
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

def load_to_qdrant_optimized(collection_name, documents, task_id):
    """Оптимизированная загрузка в Qdrant"""
    if not documents:
        raise ValueError("Список документов пуст!")
    
    try:
        logger.info(f"Начало загрузки {len(documents)} документов в Qdrant")
        
        if not acquire_qdrant_lock(collection_name, task_id):
            raise Exception("Не удалось получить блокировку коллекции")
        
        safe_update_progress(task_id, 80, stage="qdrant_preparation", 
                           stage_details="Подготовка к загрузке в Qdrant")
        
        if not client_qdrant.collection_exists(collection_name):
            vector_size = len(documents[0]["vector"])
            logger.info(f"Создание коллекции {collection_name} с размерностью {vector_size}")
            
            client_qdrant.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                ),
                optimizers_config=models.OptimizersConfigDiff(
                    indexing_threshold=0,
                ),
                hnsw_config=models.HnswConfigDiff(
                    payload_m=16,
                    m=0
                )
            )
        
        batch_size = QDRANT_BATCH_SIZE
        total_docs = len(documents)
        
        points = []
        for i, doc in enumerate(documents):
            if isinstance(doc["id"], str) and doc["id"].isdigit():
                point_id = int(doc["id"])
            else:
                point_id = hash(str(doc["id"])) % (2**31)
            
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=doc["vector"],
                    payload=doc["payload"]
                )
            )
        
        uploaded = 0
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            
            try:
                client_qdrant.upsert(
                    collection_name=collection_name,
                    points=batch,
                    wait=False
                )
                
                uploaded += len(batch)
                progress = 85 + int((uploaded / total_docs) * 15) if total_docs > 0 else 85
                
                safe_update_progress(task_id, progress, stage="qdrant_upload",
                                   stage_details=f"Загружено {uploaded}/{total_docs} документов")
                
            except Exception as e:
                logger.error(f"Ошибка загрузки батча: {e}")
                if batch_size > 50:
                    batch_size = batch_size // 2
                    continue
                raise e
        
        time.sleep(1)
        
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
        logger.error(f"Ошибка загрузки в Qdrant: {e}", exc_info=True)
        safe_update_progress(task_id, 0, status="failed", error=str(e))
        raise e
    finally:
        release_qdrant_lock(collection_name, task_id)

def load_file_to_elstic(filename, path=None, task_id=None):
    """Загрузка файла с детальным логированием"""
    
    if task_id is None:
        task_id = str(uuid.uuid4())
    
    try:
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
                    "number_of_replicas": 0,
                    "refresh_interval": "30s",
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
        
        if es.indices.exists(index=new_index):
            es.indices.delete(index=new_index, ignore=[400, 404])
        
        response = es.indices.create(index=new_index, body=mapping, ignore=400)
        
        if not ('acknowledged' in response and response['acknowledged']):
            logger.error(f"Ошибка создания индекса: {response}")
            return {"status": "failed", "error": "Ошибка создания индекса"}
        
        logger.info(f"Загрузка данных из {file_name}")
        with open(file_name, 'r', encoding='utf-8') as file:
            data = json.load(file)

        if not isinstance(data, list) or not data:
            return {"status": "failed", "error": "Некорректный формат JSON"}

        # 🔍 ДЕТАЛЬНАЯ ПРОВЕРКА ИСХОДНЫХ ДАННЫХ
        logger.info("=" * 50)
        logger.info("ПРОВЕРКА ИСХОДНЫХ ДАННЫХ ИЗ JSON")
        logger.info("=" * 50)
        
        for i, doc in enumerate(data[:5]):  # Проверяем первые 5 документов
            if isinstance(doc, dict):
                logger.info(f"\nДокумент {i}:")
                for field in ["timeCreate", "audienceCount"]:
                    if field in doc:
                        value = doc[field]
                        logger.info(f"  {field}: {value} (type: {type(value).__name__})")
                        
                        # Проверка на None
                        if value is None:
                            logger.error(f"  ❌ НАЙДЕН None В ИСХОДНОМ JSON!")
                        
                        # Проверка на возможность сравнения
                        try:
                            _ = value < 0
                            logger.info(f"  ✅ Сравнение возможно")
                        except TypeError as te:
                            logger.error(f"  ❌ Ошибка сравнения: {te}")

        # Предварительная очистка
        cleaned_data = []
        for idx, doc in enumerate(data):
            if isinstance(doc, dict):
                # Логируем ДО валидации
                if idx < 3:
                    logger.info(f"\nОчистка документа {idx} ДО валидации:")
                    logger.info(f"  timeCreate: {doc.get('timeCreate')} (type: {type(doc.get('timeCreate')).__name__})")
                    logger.info(f"  audienceCount: {doc.get('audienceCount')} (type: {type(doc.get('audienceCount')).__name__})")
                
                doc = validate_document_numeric_fields(doc)
                
                # Логируем ПОСЛЕ валидации
                if idx < 3:
                    logger.info(f"  ПОСЛЕ валидации:")
                    logger.info(f"  timeCreate: {doc.get('timeCreate')} (type: {type(doc.get('timeCreate')).__name__})")
                    logger.info(f"  audienceCount: {doc.get('audienceCount')} (type: {type(doc.get('audienceCount')).__name__})")
                
                cleaned_data.append(doc)
            else:
                logger.warning(f"Пропущен документ {idx} неверного типа: {type(doc)}")

        data = cleaned_data
        logger.info(f"После очистки осталось {len(data)} валидных документов")
        
        # Загрузка в Elasticsearch
        try:
            from elasticsearch.helpers import streaming_bulk
            
            def actions_generator():
                for doc in data:
                    if not isinstance(doc, dict):
                        continue
                    
                    doc_id = str(doc.get('id', doc.get('idExternal', str(uuid.uuid4()))))
                    
                    if not any(field in doc for field in ["text", "Текст сообщения", "title", "content"]):
                        continue
                    
                    yield {
                        "_index": new_index,
                        "_id": doc_id,
                        "_source": doc
                    }
            
            success_count = 0
            for ok, response in streaming_bulk(
                es,
                actions_generator(),
                chunk_size=200,
                max_retries=3,
                initial_backoff=2,
                yield_ok=False,
                raise_on_error=False
            ):
                if ok:
                    success_count += 1
                else:
                    logger.warning(f"Ошибка индексации: {response}")
                    
        except Exception as bulk_error:
            logger.error(f"Ошибка bulk индексации: {bulk_error}", exc_info=True)
        
        es.indices.refresh(index=new_index)
        total_docs = es.count(index=new_index)['count']
        
        logger.info(f"✅ Elasticsearch индексация завершена:")
        logger.info(f"   Успешно: {success_count}, Всего в индексе: {total_docs}")
        
        if total_docs == 0:
            return {"status": "failed", "error": "Индекс пуст после загрузки"}
        
        # Обработка для Qdrant
        logger.info("🔄 Начало обработки для Qdrant")
        processed_docs = batch_process_documents_with_embeddings_optimized(data, task_id)
        
        if not processed_docs:
            return {"status": "failed", "error": "Нет документов для Qdrant"}
        
        logger.info(f"📊 Статистика обработки:")
        logger.info(f"   Исходных документов: {len(data)}")
        logger.info(f"   Обработано для Qdrant: {len(processed_docs)}")
        logger.info(f"   Пропущено: {len(data) - len(processed_docs)}")
        
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
        
        try:
            load_to_qdrant_optimized(new_index, processed_docs, task_id)
        except Exception as e:
            logger.error(f"Ошибка Qdrant: {e}", exc_info=True)
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
        logger.error(f"Критическая ошибка: {e}", exc_info=True)
        return {"status": "failed", "error": str(e)}
    finally:
        try:
            model_manager.cleanup()
        except Exception as e:
            logger.warning(f"Ошибка очистки: {e}")

def acquire_qdrant_lock(collection_name, task_id, timeout=30):
    lock_key = f"qdrant_lock:{collection_name}"
    deadline = time.time() + timeout
    
    while time.time() < deadline:
        if redis_client.set(lock_key, task_id, nx=True, ex=120):
            return True
        time.sleep(1)
    
    return False

def release_qdrant_lock(collection_name, task_id):
    lock_key = f"qdrant_lock:{collection_name}"
    owner = redis_client.get(lock_key)
    
    if owner and owner.decode('utf-8') == task_id:
        redis_client.delete(lock_key)
        return True
    return False