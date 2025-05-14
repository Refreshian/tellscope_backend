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

from modelscope import snapshot_download
# from huggingface_hub import snapshot_download
import shutil
import os

# Укажите путь для сохранения модели
target_dir = "/home/dev/fastapi/analytics_app/data/LLM_models/"

# Укажите ID модели, которую вы хотите скачать
model_id = 'LLM-Research/Vikhr-Llama3.1-8B-Instruct-R-21-09-24'  # Замените 'model-id' на ID модели, которую вы хотите скачать

# Скачиваем модель
model_path = snapshot_download(model_id)

# Проверьте, существует ли директория, если нет, создайте её
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# Перемещаем или копируем скачанную модель в целевую директорию
shutil.move(model_path, target_dir)  # Убедитесь, что model_path — это корректный путь к директории или файлу