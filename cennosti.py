import asyncio
import aiohttp
import os
import json
import time
from search_data_elastic import elastic_query

THEME_INDEX = "smirnov_stroim_dom_01.05.2024-28.07.2025"
OUTPUT_PATH = "/home/dev/tellscope_app/cennosti.jsonl"  # jsonlines формат!
SYSTEM_PROMPT = "Ты дружелюбный ассистент для разметки текстов"

QUESTION = """Я анализирую тему человеческих ценностей, вот перечень заданных ценностей - ['Любовь и взаимоотношения', 'Здоровье и безопасность', 'Уважение и признание', 'Развитие и самосовершенствование', 'Поддержка и сотрудничество', 'Достижения и успех', 'Справедливость и равенство', 'Духовность и этика', 'Стойкость и смелость', 'Качество и эффективность', 'Доброта и благотворительность', 'Память и традиции', 'Комфорт и стабильность', 'Открытость и добрососедство', 'Служение и лидерство', 'Гибкость и адаптивность', 'Природа и экология', 'Красота и эстетика', 'Вдохновение и оптимизм', 'Наставничество и обучение', 'Уверенность и независимость', 'Надежда и вера', 'Свобода и правомочия', 'Культура и наследие', 'Энергия и спорт'], есть ли в тексте какая-либо из этих ценностей и как она связана с Патриотизмом? Отвечай по возможности кратко в несколько слов, отделяй знаком & если ты отвечаешь на второй вопрос про патриотизм, если как-то связано с патриотизмом, то кратко поясни как"""

LLM_URL = "http://localhost:8000/v1/chat/completions"

BATCH_SIZE = 10
SAVE_EVERY = 1000  # Сохранять каждые 1000 обработанных

async def generate_answer(text, question, system_prompt=None, session=None, retries=3):
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

    for attempt in range(retries):
        try:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    try:
                        generated = data["choices"][0]["message"]["content"]
                    except Exception:
                        generated = ""
                    return generated.strip()
                else:
                    return f"LLM error: {response.status}"
        except Exception as e:
            if attempt < retries - 1:
                await asyncio.sleep(2 * (attempt + 1))  # увеличивайте промежутки между попытками
                continue
            return f"Exception: {e}"

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

async def process_batch(batch_entries, session):
    """Запуск батча на обработку и возвращение готовых результатов"""
    tasks = []
    for entry in batch_entries:
        # Можно делать замер START тут
        tasks.append(generate_answer(entry['text'], QUESTION, SYSTEM_PROMPT, session))
    start = time.time()
    llm_labels = await asyncio.gather(*tasks)
    end = time.time()
    # ЗАМЕР: сколько длилась обработка батча?
    print(f"Батч из {len(batch_entries)} обработан за {end-start:.2f} секунд")
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
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
        for i in range(0, total, BATCH_SIZE):
            batch = to_process[i:i+BATCH_SIZE]
            results = await process_batch(batch, session)
            # Запись батча в файл (append по одной строке)
            with open(OUTPUT_PATH, 'a', encoding='utf-8') as f:
                for record in results:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
                    saved += 1

            print(f"Суммарно обработано и сохранено {saved}/{total}")

            # Периодический save
            if saved % SAVE_EVERY < BATCH_SIZE and saved != 0:
                print(f"[SAVE] Промежуточное сохранение на {saved}")

            # ЗАМЕР: здесь выводите ваши нужные статистики, ETA, etc.

    print(f"Готово. Все результаты дозаписаны в {OUTPUT_PATH}")

if __name__ == "__main__":
    asyncio.run(main())