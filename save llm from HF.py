# # https://modelscope.cn/models/LLM-Research/Meta-Llama-3-8B-Instruct

# from modelscope import snapshot_download
# # from huggingface_hub import snapshot_download
# import shutil
# import os

# # Укажите путь для сохранения модели
# target_dir = "/home/dev/fastapi/analytics_app/data/LLM_models/"

# # Укажите ID модели, которую вы хотите скачать
# model_id = 'LLM-Research/Meta-Llama-3-8B-Instruct'  # Замените 'model-id' на ID модели, которую вы хотите скачать

# # Скачиваем модель
# model_path = snapshot_download(model_id)

# # Проверьте, существует ли директория, если нет, создайте её
# if not os.path.exists(target_dir):
#     os.makedirs(target_dir)

# # Перемещаем или копируем скачанную модель в целевую директорию
# shutil.move(model_path, target_dir)  # Убедитесь, что model_path — это корректный путь к директории или файлу

# https://modelscope.cn/models/LLM-Research/Meta-Llama-3-8B-Instruct

from huggingface_hub import snapshot_download
import shutil
import os

# Укажите путь для сохранения модели
target_dir = "/home/dev/tellscope_app/tellscope_backend/data/embed_models/"

# Укажите ID модели, которую вы хотите скачать
model_id = 'deepvk/USER-base'  # Исправленное название модели

try:
    # Скачиваем модель из Hugging Face Hub
    model_path = snapshot_download(repo_id=model_id, cache_dir=target_dir)
    print(f"Модель успешно скачана в: {model_path}")
    
except Exception as e:
    print(f"Ошибка при скачивании модели: {str(e)}")
    
    # Попробуем альтернативные названия модели
    alternative_models = [
        'deepvk/USER-base',
        'ai-forever/ru-clip-tiny',
        'ai-forever/rubert-base-cased',
        'cointegrated/rubert-tiny2'
    ]
    
    for alt_model in alternative_models:
        try:
            print(f"Пробуем скачать модель: {alt_model}")
            model_path = snapshot_download(repo_id=alt_model, cache_dir=target_dir)
            print(f"Модель {alt_model} успешно скачана в: {model_path}")
            break
        except Exception as alt_e:
            print(f"Не удалось скачать {alt_model}: {str(alt_e)}")
            continue