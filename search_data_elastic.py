from elasticsearch import Elasticsearch
from typing import Optional, List, Dict
import re

es = Elasticsearch(
    "http://localhost:9200",
    basic_auth=("elastic", "biz8z5i1w0nLPmEweKgP")
)

def update_max_result_window(index_name: str, max_window: int = 5000000):
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