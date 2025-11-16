import asyncio
import aiohttp
import os
import json
import time
from search_data_elastic import elastic_query

THEME_INDEX = "smirnov_stroim_dom_01.05.2024-28.07.2025"
OUTPUT_PATH = "/home/dev/tellscope_app/cennosti_upd.jsonl"
SYSTEM_PROMPT = "Ты дружелюбный ассистент для разметки текстов"

QUESTION = """Я анализирую тему человеческих ценностей, вот перечень заданных ценностей - ['Любовь и взаимоотношения', 'Здоровье и безопасность', 'Уважение и признание', 'Развитие и самосовершенствование', 'Поддержка и сотрудничество', 'Достижения и успех', 'Справедливость и равенство', 'Духовность и этика', 'Стойкость и смелость', 'Качество и эффективность', 'Доброта и благотворительность', 'Память и традиции', 'Комфорт и стабильность', 'Открытость и добрососедство', 'Служение и лидерство', 'Гибкость и адаптивность', 'Природа и экология', 'Красота и эстетика', 'Вдохновение и оптимизм', 'Наставничество и обучение', 'Уверенность и независимость', 'Надежда и вера', 'Свобода и правомочия', 'Культура и наследие', 'Энергия и спорт'], есть ли в тексте какая-либо из этих ценностей и как она связана с Патриотизмом? Отвечай по возможности кратко в несколько слов, отделяй знаком & если ты отвечаешь на второй вопрос про патриотизм, если как-то связано с патриотизмом, то кратко поясни как"""

LLM_URL = "http://localhost:8000/v1/chat/completions"

BATCH_SIZE = 10
SAVE_EVERY = 1000

# Подробнее логируем ошибки серверных дисконнектов
def log_error(msg):
    print(f"[ERROR] {msg}")

async def generate_answer(text, question, session, system_prompt=None, retries=5, batch_number=0):
    url = LLM_URL
    headers = {"Content-Type": "application/json"}
    system_line = system_prompt.strip() if system_prompt else SYSTEM_PROMPT

    user_content = (
        f"Текст: {text[:7500].strip()}\n\nВопрос: {question.strip()}\n\n"
        "Ответ (строго кратко, только факт, без разъяснений):"
    )

    payload = {
        "model": "Qwen/Qwen3-32B-FP8",
        "messages": [
            {"role": "system", "content": system_line},
            {"role": "user", "content": user_content}
        ],
        "temperature": 0.7,
        "top_p": 0.8,
        "chat_template_kwargs": {"enable_thinking": False}
    }

    error_cnt = 0
    for attempt in range(retries):
        try:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    generated = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    return generated.strip()
                else:
                    error_txt = await response.text()
                    log_error(f"LLM API error {response.status}: {error_txt}")
                    if response.status in (502, 503, 504):
                        # Признак, что server disconn/disrupted
                        error_cnt += 1
                        await asyncio.sleep(1.5 * (error_cnt + 1))
                        continue
                    return f"LLM error: {response.status} " + error_txt[:200]
        except Exception as e:
            # Именно такие обрывы ловим по тексту
            msg = str(e)
            if "Server disconnected" in msg:
                error_cnt += 1
                log_error(f"Server disconnected (batch {batch_number}), try {attempt+1}")
                await asyncio.sleep(1.5 * (error_cnt + 1))  # увеличиваем паузу при обрывах
                continue
            # другие ошибки — аналогично
            log_error(f"Exception in generate_answer: {e}")
            if attempt < retries-1:
                await asyncio.sleep(1.2 * (attempt + 1))
                continue
            return f"Exception: {e}"
    return f"Failed after {retries} attempts"

def load_processed_hashes(path):
    """Позволяет дозапускать обработку, не повторяя то, что уже посчитано"""
    if not os.path.exists(path):
        return set()
    hashes = set()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                j = json.loads(line)
                if 'hash' in j:
                    hashes.add(j['hash'])
            except Exception:
                continue
    return hashes

async def process_batch(batch_entries, system_prompt, batch_number):
    # !!! Каждый раз новая сессия !!!
    timeout = aiohttp.ClientTimeout(total=180)
    connector = aiohttp.TCPConnector(limit=20, limit_per_host=10, force_close=True)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = []
        for entry in batch_entries:
            tasks.append(generate_answer(entry['text'], QUESTION, session, system_prompt=system_prompt, batch_number=batch_number))
        start = time.time()
        llm_labels = await asyncio.gather(*tasks)
        end = time.time()
        print(f"Батч #{batch_number} ({len(batch_entries)} шт) обработан за {end-start:.2f} сек")
        results = []
        for i, entry in enumerate(batch_entries):
            results.append({
                "hash": entry['hash'],
                "text": entry['text'],
                "llm_label": llm_labels[i]
            })
            print(f"Processed {entry['hash']}: {llm_labels[i]}")
        return results

async def main():
    docs = elastic_query(
        theme_index=THEME_INDEX,
        query_str="all",
        min_date=1714581787,
        max_date=1753687182,
        scroll_time="10m",
    )
    print(f'Документов: {len(docs)}')

    entries = []
    for doc in docs:
        text = doc.get('text') or doc.get('Текст сообщения', '')
        _id_hash = doc.get('hash')
        if not text or not _id_hash:
            continue
        entries.append({'hash': _id_hash, 'text': text})

    already = load_processed_hashes(OUTPUT_PATH)
    to_process = [e for e in entries if e['hash'] not in already]
    total = len(to_process)
    print(f'К обработке (неповторно): {total}')

    saved = 0
    batch_number = 0

    for i in range(0, total, BATCH_SIZE):
        batch = to_process[i:i+BATCH_SIZE]
        batch_number += 1

        retry_count = 0
        MAX_BATCH_RETRIES = 4
        while True:
            try:
                results = await process_batch(batch, SYSTEM_PROMPT, batch_number=batch_number)
                break
            except Exception as e:
                log_error(f"Ошибка обработки батча #{batch_number}: {e}")
                retry_count += 1
                if retry_count > MAX_BATCH_RETRIES:
                    log_error(f"Фатальный сбой батча #{batch_number}, пропускаем его!")
                    results = []
                    break
                await asyncio.sleep(4.0 * (retry_count + 1))
                continue

        # Запись батча в файл
        with open(OUTPUT_PATH, 'a', encoding='utf-8') as f:
            for record in results:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                saved += 1

        print(f"Суммарно обработано и сохранено {saved}/{total}")

        if saved % SAVE_EVERY < BATCH_SIZE and saved != 0:
            print(f"[SAVE] Промежуточное сохранение на {saved}")

    print(f"Готово. Все результаты дозаписаны в {OUTPUT_PATH}")

if __name__ == "__main__":
    asyncio.run(main())