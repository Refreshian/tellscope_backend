import asyncio
import json
import os
import re
import sys
import time
from typing import List, Dict, Any

import pandas as pd
import aiohttp
import matplotlib.pyplot as plt
from io import BytesIO

from docx import Document
from docx.shared import Inches
from openai import OpenAI

# ==========================================
# КОНФИГУРАЦИЯ (без изменений)
# ==========================================
EXTERNAL_API_KEY = "sk-aitunnel-PrKMg8fNFewHciI2DvmAHGaD8g7cSyjD"
EXTERNAL_BASE_URL = "https://api.aitunnel.ru/v1/"
EXTERNAL_MODEL = "claude-sonnet-4.5"
LOCAL_MODEL_NAME = "Qwen/Qwen3-32B-FP8"
LOCAL_LLM_URL = "http://localhost:8000/v1/chat/completions"
BATCH_SIZE = 32
MAX_CONCURRENCY = 64
CONNECT_TIMEOUT = 15
TOTAL_TIMEOUT = 180
MAX_RETRIES = 2

# ==========================================
# 1. МОДУЛЬ ВНЕШНЕЙ LLM (Оркестратор) - ЗНАЧИТЕЛЬНЫЕ УЛУЧШЕНИЯ
# ==========================================

class ExternalLLMbrain:
    def __init__(self):
        self.client = OpenAI(
            api_key=EXTERNAL_API_KEY,
            base_url=EXTERNAL_BASE_URL,
        )

    def _is_english_requested(self, text: str) -> bool:
        lowered = text.lower()
        return "english" in lowered or "англий" in lowered

    def plan_task(self, user_query: str, data_sample: str, available_columns: List[str] = None) -> Dict[str, Any]:
        columns_info = f"\n\nДоступные колонки в данных: {', '.join(available_columns)}" if available_columns else ""
        needs_english = self._is_english_requested(user_query)
        target_language = "английском" if needs_english else "русском"

        system_prompt = f"""
        Ты - гениальный архитектор анализа данных. Твоя задача - превратить сложный запрос пользователя в детализированный, структурированный JSON-план для Python-скрипта.
        На вход подается:
        1. Запрос пользователя.
        2. Образец данных (JSON структура).{columns_info}

        Твоя цель - создать JSON-план, который агент сможет выполнить шаг за шагом. Особое внимание удели запросам, где требуется анализ "каждой тематики". Для этого используй блок `"type": "thematic_breakdown"`.

        Язык ответа в отчете: {target_language}.

        ТЫ ДОЛЖЕН ВЕРНУТЬ ТОЛЬКО ВАЛИДНЫЙ JSON СТРОГО ПО ФОРМАТУ НИЖЕ. БЕЗ ПОЯСНЕНИЙ.

        ФОРМАТ JSON-ПЛАНА:
        {{
            "filters": {{...}},
            "analysis_needed": true,
            "local_llm_system_prompt": "...",
            "local_llm_user_question": "...",
            "report_title": "Название отчета",
            "report_structure": [
                {{ "type": "title" }},
                {{ "type": "section", "title": "1. Введение", "content_source": "user_query" }},
                {{ "type": "overall_stats" }},
                {{
                    "type": "thematic_breakdown",
                    "title": "2. Детальный анализ тематик",
                    "breakdown_by_column": "llm_result_clean",
                    "max_themes": 7,
                    "num_examples": 5,
                    "thematic_section_structure": ["introduction", "demographics_chart", "examples"]
                }},
                {{ "type": "section", "title": "3. Итоговые выводы", "content_source": "final_conclusions" }}
            ]
        }}

        ПРИМЕР ЗАПРОСА ПОЛЬЗОВАТЕЛЯ: "Проанализируй все сообщения темы 'Платон', создай отчет: какие тематики есть и сделай отчет по каждой тематике: 1. Во введении каждой тематики расскажи про что она. 2. Нарисуй график распределения авторов по полу и возрасту. 3. Приведи 5 примеров."

        ПРИМЕР ИДЕАЛЬНОГО JSON-ПЛАНА ДЛЯ ЭТОГО ЗАПРОСА:
        {{
            "filters": {{}},
            "analysis_needed": true,
            "local_llm_system_prompt": "Ты — классификатор текста. Определи основную тему сообщения. Ответ - ТОЛЬКО название темы (1-3 слова) на русском. Без рассуждений.",
            "local_llm_user_question": "Определи тему этого сообщения.",
            "report_title": "Анализ обсуждений системы 'Платон' в соцмедиа",
            "report_structure": [
                {{ "type": "title" }},
                {{ "type": "section", "title": "1. Введение", "content_source": "user_query" }},
                {{ "type": "overall_stats", "title": "Общая статистика по теме 'Платон'" }},
                {{
                    "type": "thematic_breakdown",
                    "title": "2. Детальный анализ тематик обсуждений",
                    "breakdown_by_column": "llm_result_clean",
                    "max_themes": 7,
                    "num_examples": 5,
                    "thematic_section_structure": ["introduction", "demographics_chart", "examples"]
                }},
                {{ "type": "section", "title": "3. Итоговые выводы", "content_source": "final_conclusions" }}
            ]
        }}
        """
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Запрос: {user_query}\n\nСэмпл данных:\n{data_sample}"}
                ],
                model=EXTERNAL_MODEL,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            print(f"DEBUG: Получен план от API:\n{content}\n")
            plan = json.loads(content)
            if "report_structure" not in plan or not plan["report_structure"]:
                 raise ValueError("План не содержит report_structure")
            return plan
        except Exception as e:
            print(f"ОШИБКА при создании плана: {e}. Используется план по умолчанию.")
            return self._get_default_plan(user_query, target_language)

    def generate_thematic_intro(self, topic_name: str, topic_examples: List[str], user_query: str) -> str:
        examples_str = "\n".join([f"- {ex[:250]}..." for ex in topic_examples])
        prompt = f"""
        Ты аналитик соцмедиа. Напиши короткое (3-4 предложения) введение для раздела отчета, посвященного тематике "{topic_name}".
        Общий контекст анализа: "{user_query}".

        Вот примеры сообщений из этой тематики:
        {examples_str}

        В своем тексте кратко опиши:
        1. О чем эта тематика.
        2. В каком ключе (позитивном, негативном, нейтральном) идут обсуждения, судя по примерам.

        Отвечай только готовым текстом на русском языке, без заголовков, Markdown и рассуждений.
        """
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=EXTERNAL_MODEL,
                temperature=0.4
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"ОШИБКА при генерации введения для темы '{topic_name}': {e}")
            return f"Не удалось автоматически сгенерировать описание для тематики '{topic_name}'. Эта тема выделена на основе анализа сообщений."

    def generate_narrative(self, stats_data: str, user_query: str) -> str:
        prompt = f"""
        Ты профессиональный аналитик соцмедиа. Пользователь просил: "{user_query}".
        Вот ОБЩАЯ итоговая статистика после обработки всех данных:
        {stats_data}

        Напиши раздел "Итоговые выводы" для Word-отчета.
        Обобщи основные тренды, дай интерпретацию цифрам и результатам анализа тематик. Текст должен быть связным, профессиональным, на русском языке.
        Не используй Markdown, не используй жирный/курсив. Не добавляй рассуждения. Дай чистый готовый текст.
        """
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=EXTERNAL_MODEL,
                temperature=0.5
            )
            content = response.choices[0].message.content
            return content.strip()
        except Exception as e:
            print(f"ОШИБКА при генерации выводов: {e}")
            return f"Автоматическая генерация выводов недоступна. Статистика:\n{stats_data}"

    def _get_default_plan(self, user_query: str, target_language: str = "русском") -> Dict[str, Any]:
        return {
            "filters": {},
            "analysis_needed": True,
            "local_llm_system_prompt": f"Ты — классификатор текста. Определи основную тему сообщения. Ответ - ТОЛЬКО название темы (1-3 слова) на {target_language}. Без рассуждений.",
            "local_llm_user_question": "Определи тему этого сообщения.",
            "report_title": f"Анализ по запросу: {user_query[:50]}",
            "report_structure": [
                {"type": "title"},
                {"type": "section", "title": "1. Введение", "content_source": "user_query"},
                {"type": "overall_stats", "title": "Общая статистика"},
                {
                    "type": "thematic_breakdown",
                    "title": "2. Анализ ключевых тематик",
                    "breakdown_by_column": "llm_result_clean",
                    "max_themes": 5, "num_examples": 5,
                    "thematic_section_structure": ["introduction", "demographics_chart", "examples"]
                },
                {"type": "section", "title": "3. Итоговые выводы", "content_source": "final_conclusions"}
            ]
        }
# ... Классы LocalLLMWorker и ReportBuilder остаются без изменений ...
class LocalLLMWorker:
    def __init__(self): self.session = None
    async def _create_session(self): return aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=512, limit_per_host=256), timeout=aiohttp.ClientTimeout(total=TOTAL_TIMEOUT, connect=CONNECT_TIMEOUT))
    async def _generate_single(self, text: str, system_prompt: str, user_question: str) -> str:
        if not text: return "Пустой текст"
        cleaned_text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)[:4000]
        payload = {"model": LOCAL_MODEL_NAME, "messages": [{"role": "system", "content": system_prompt},{"role": "user", "content": f"{user_question}\n\nТекст:\n{cleaned_text}"}],"temperature": 0.1, "max_tokens": 200 }
        for attempt in range(MAX_RETRIES + 1):
            try:
                async with self.session.post(LOCAL_LLM_URL, json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json(); answer = data['choices'][0]['message']['content'].strip(); return re.sub(r'(?i)\bthink\b[:：]?', '', answer).split("Ответ:")[-1].strip()
            except Exception: await asyncio.sleep(0.5 * (2 ** attempt))
        return "Ошибка анализа"
    async def process_batch(self, texts: List[str], system_prompt: str, user_question: str) -> List[str]:
        self.session = await self._create_session()
        semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
        async def worker(txt):
            async with semaphore: return await self._generate_single(txt, system_prompt, user_question)
        tasks = [asyncio.create_task(worker(t)) for t in texts]; total = len(tasks)
        for i, f in enumerate(asyncio.as_completed(tasks)):
            if i % 10 == 0:
                await self.progress_callback(f"Анализ текстов: {i}/{total}...")
            await f
        results = await asyncio.gather(*tasks)
        await self.session.close()
        return results

class ReportBuilder:
    def __init__(self, filename="Analytics_Report.docx"): self.doc = Document(); self.filename = filename
    def add_title(self, text): self.doc.add_heading(text, 0)
    def add_heading(self, text, level=1): self.doc.add_heading(text, level=level)
    def add_paragraph(self, text): self.doc.add_paragraph(text)
    def add_section(self, title, content, level=1): self.add_heading(title, level); self.add_paragraph(content)
    def add_list_stats(self, title, stats_dict: Dict, level=2):
        self.add_heading(title, level)
        for key, value in stats_dict.items(): self.doc.add_paragraph(f"{key}: {value}", style='List Bullet')
    def add_examples(self, examples: List[Dict], title="Примеры сообщений", level=3):
        self.add_heading(title, level)
        for ex in examples:
            p = self.doc.add_paragraph(); p.add_run(f"Автор: {ex.get('authorObject', {}).get('fullname', 'N/A')}, Источник: {ex.get('hub', 'N/A')}\n").bold = True
            p.add_run(ex.get('text', '')[:300] + "...\n"); p.add_run(f"Ссылка: {ex.get('url', '#')}\n").italic = True
    def add_chart(self, title: str, data: pd.Series, chart_type: str = "bar", level=2):
        # ИСПРАВЛЕНИЕ 1: Проверяем, есть ли вообще данные
        if data.empty or data.sum() == 0:
            self.add_heading(title, level)
            self.add_paragraph(f"На графике '{title}' нет данных для отображения.")
            return

        self.add_heading(title, level)
        plt.figure(figsize=(6.5, 4))

        # Отрисовка графика (без изменений)
        if chart_type == "bar":
            data.plot(kind='bar', color='skyblue')
        elif chart_type == 'pie':
            # Для pie чартов лучше отфильтровать очень маленькие значения
            data_for_pie = data[data > 0]
            data_for_pie.plot(kind='pie', autopct='%1.1f%%', startangle=90)
            plt.ylabel('')
        else:
            data.plot(kind='bar')

        plt.title(title)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        img_stream = BytesIO()
        plt.savefig(img_stream, format='png')
        plt.close()
        img_stream.seek(0)
        self.doc.add_picture(img_stream, width=Inches(6))

        # ИСПРАВЛЕНИЕ 2: Полностью переработанная логика объяснения
        explanation = f"На графике '{title}' представлено распределение авторов."
        
        # Проверяем, есть ли реальные данные для анализа
        if not data.empty and data.max() > 0:
            max_value = data.max()
            max_category = data.idxmax()
            # Заменяем пустую категорию на более понятное описание
            if str(max_category).strip() == '':
                max_category = 'Не указан'

            if chart_type == 'pie':
                total = data.sum()
                percentage = (max_value / total) if total > 0 else 0
                explanation += f" Наибольшую долю ({percentage:.1%}) составляет категория '{max_category}' ({int(max_value)} авторов)."
            else:
                explanation += f" Самой многочисленной категорией является '{max_category}' с {int(max_value)} авторами."
        else:
            explanation += " Нет данных для определения доминирующей категории."
            
        self.add_paragraph(explanation)

    def save(self): self.doc.save(self.filename); return self.filename

# ==========================================
# 4. ГЛАВНЫЙ КЛАСС АГЕНТА - ВСЯ ЛОГИКА ЗДЕСЬ
# ==========================================

class SocialMediaAgent:
    def __init__(self, progress_callback=None):
        self.brain = ExternalLLMbrain()
        self.worker = LocalLLMWorker()
        self.worker.progress_callback = progress_callback # Передаем callback в воркер
        self.data_df = None
        self.progress_callback = progress_callback

    async def _log_progress(self, message: str):
        if self.progress_callback: await self.progress_callback(message)
        print(message)

    def load_data(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f: data = json.load(f)
        self.data_df = pd.DataFrame(data['items'] if isinstance(data, dict) and 'items' in data else data)
        # Очистка и обогащение данных
        if not self.data_df.empty:
            self.data_df['sex'] = self.data_df['authorObject'].apply(lambda x: x.get('sex') if isinstance(x, dict) else None)
            self.data_df['age'] = self.data_df['authorObject'].apply(lambda x: x.get('age') if isinstance(x, dict) else None)
            self.data_df['age'] = pd.to_numeric(self.data_df['age'], errors='coerce')
            self.data_df['age'] = self.data_df['age'].where((self.data_df['age'] >= 0) & (self.data_df['age'] <= 120))
            bins = [0, 18, 25, 35, 45, 60, 120]; labels = ['<18', '18-24', '25-34', '35-44', '45-59', '60+']
            self.data_df['age_group'] = pd.cut(self.data_df['age'], bins=bins, labels=labels, right=False)

    def filter_data(self, filters: Dict):
        df = self.data_df.copy()
        # Тут может быть сложная логика фильтрации
        self.data_df = df
        
    async def run_task(self, user_query: str, input_file: str, report_save_path: str):
        # 1. ЗАГРУЗКА И ПЛАНИРОВАНИЕ
        await self._log_progress("Загрузка и подготовка данных...")
        self.load_data(input_file)
        sample = self.data_df.head(1).to_json(orient='records', force_ascii=False) if not self.data_df.empty else "{}"
        columns = self.data_df.columns.tolist() if not self.data_df.empty else []
        
        await self._log_progress("Планирование анализа (мозг агента)...")
        plan = await asyncio.to_thread(self.brain.plan_task, user_query, sample, columns)
        await self._log_progress(f"План утвержден: {json.dumps(plan, ensure_ascii=False, indent=2)}")

        # 2. ФИЛЬТРАЦИЯ
        await self._log_progress("Применение фильтров...")
        self.filter_data(plan.get('filters', {}))
        if self.data_df.empty:
            raise Exception("Нет данных после фильтрации. Отчет не может быть создан.")

        # 3. АНАЛИЗ ТЕМ (КЛАССИФИКАЦИЯ)
        if plan.get('analysis_needed'):
            await self._log_progress("Этап 1: Классификация сообщений...")
            texts = self.data_df['text'].fillna("").tolist()
            labels = await self.worker.process_batch(texts, plan['local_llm_system_prompt'], plan['local_llm_user_question'])
            self.data_df['llm_result_clean'] = [re.sub(r'[^\w\s-]', '', str(x)).strip().capitalize() for x in labels]
            self.data_df = self.data_df[self.data_df['llm_result_clean'].str.len() > 1]
            self.data_df = self.data_df[~self.data_df['llm_result_clean'].str.contains("Ошибка анализа", case=False)]
        
        # 4. СБОРКА ОТЧЕТА ПО ПЛАНУ
        await self._log_progress("Этап 2: Сборка итогового отчета...")
        report = ReportBuilder(filename=report_save_path)
        
        report_context = { "user_query": user_query, "report_title": plan.get("report_title", "Аналитический отчет") }

        for section_plan in plan.get("report_structure", []):
            section_type = section_plan.get("type")
            
            if section_type == "title": report.add_title(report_context["report_title"])
            
            elif section_type == "section" and section_plan.get("content_source") == "user_query":
                report.add_section(section_plan.get("title", "Введение"), f'Отчет подготовлен по запросу: "{user_query}"')
                
            elif section_type == "overall_stats":
                await self._log_progress("Подсчет общей статистики...")
                total_messages = len(self.data_df); unique_authors = self.data_df['authorObject'].apply(lambda x: x.get('id') if isinstance(x, dict) else None).nunique()
                top_hubs = self.data_df['hub'].value_counts().head(5)
                report.add_heading(section_plan.get("title", "Общая статистика"), level=1)
                report.add_paragraph(f"Всего проанализировано сообщений: {total_messages}\nУникальных авторов: {unique_authors}")
                report.add_chart("Топ-5 площадок по сообщениям", top_hubs, chart_type='bar', level=2)
                
            elif section_type == "thematic_breakdown" and 'llm_result_clean' in self.data_df.columns:
                await self._log_progress("Начинаю детальный анализ по тематикам...")
                report.add_heading(section_plan.get("title", "Детальный анализ тематик"), level=1)
                
                breakdown_col = section_plan.get("breakdown_by_column", "llm_result_clean")
                max_themes = section_plan.get("max_themes", 7)
                topics_to_analyze = self.data_df[breakdown_col].value_counts().head(max_themes)
                
                for i, (topic, count) in enumerate(topics_to_analyze.items()):
                    await self._log_progress(f"Анализ темы {i+1}/{len(topics_to_analyze)}: '{topic}'")
                    topic_df = self.data_df[self.data_df[breakdown_col] == topic]
                    report.add_heading(f"{i+1}. Тематика: {topic} ({count} сообщ.)", level=2)
                    
                    for sub_task in section_plan.get("thematic_section_structure", []):
                        if sub_task == "introduction":
                            examples = topic_df['text'].head(5).tolist()
                            intro_text = await asyncio.to_thread(self.brain.generate_thematic_intro, topic, examples, user_query)
                            report.add_paragraph(intro_text)
                            
                        if sub_task == "demographics_chart":
                            sex_counts = topic_df['sex'].astype(str).replace(['nan', 'None', ''], 'Не указан').value_counts()
                            # Убираем "Не указан" из pie-чарта, если он неинформативен, или оставляем, если это основная масса
                            if 'Не указан' in sex_counts and len(sex_counts) > 1:
                                # Можно решить, показывать ли "Не указан"
                                pass # Пока оставляем для полноты картины
                            sex_dist = sex_counts

                            age_dist = topic_df['age_group'].dropna().value_counts().sort_index()
                            
                            report.add_chart(f"Распределение по полу (тема: {topic})", sex_dist, chart_type='pie', level=3)
                            report.add_chart(f"Распределение по возрасту (тема: {topic})", age_dist, chart_type='bar', level=3)

                        if sub_task == "examples":
                            num_examples = section_plan.get("num_examples", 7)
                            examples_data = topic_df.head(num_examples).to_dict('records')
                            report.add_examples(examples_data, title="Примеры сообщений по теме", level=3)

        # 5. ГЕНЕРАЦИЯ ИТОГОВЫХ ВЫВОДОВ
        final_conclusion_section = next((s for s in plan["report_structure"] if s.get("content_source") == "final_conclusions"), None)
        if final_conclusion_section:
            await self._log_progress("Этап 3: Генерация итоговых выводов...")
            stats_summary = {"total_messages": len(self.data_df), "unique_authors": self.data_df['authorObject'].apply(lambda x: x.get('id') if isinstance(x, dict) else None).nunique(), "analysis_top_results": self.data_df.get('llm_result_clean', pd.Series(dtype=str)).value_counts().head(5).to_dict()}
            stats_str = json.dumps(stats_summary, ensure_ascii=False)
            conclusions = await asyncio.to_thread(self.brain.generate_narrative, stats_str, user_query)
            report.add_section(final_conclusion_section.get("title", "Итоговые выводы"), conclusions, level=1)

        # 6. СОХРАНЕНИЕ
        saved_name = report.save()
        await self._log_progress(f"✅ Готово! Отчет сохранен: {saved_name}")
        return saved_name

async def main():
    if not os.path.exists("data.json"): print("Файл data.json не найден."); return
    user_request = 'Проанализируй все сообщения темы "Платон - система взимания оплаты проезда", создай итоговый отчет Word на основе обсуждений в соцмедиа: какие тематики есть и сделай отчет по каждой тематике: 1. Во введении каждой тематики расскажи про что эта тематика и как ее обсуждают, 2. Нарисуй и объясни график распределения авторов по тематике: по полу, возрасту. 3. Приведи (5-7) примеров сообщений по каждой тематике. 4. Сделай выводы по отчету обсуждений'
    agent = SocialMediaAgent(progress_callback=lambda msg: print(f"[PROGRESS] {msg}"))
    try: await agent.run_task(user_request, "data.json", f"Detailed_Report_{int(time.time())}.docx")
    except Exception as e: print(f"Критическая ошибка: {e}"); import traceback; traceback.print_exc()

if __name__ == "__main__":
    if sys.platform == 'win32': asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())