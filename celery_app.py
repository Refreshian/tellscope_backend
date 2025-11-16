from celery import Celery
import os
from dotenv import load_dotenv
import multiprocessing
import atexit
import psutil
import signal
import subprocess
import logging
import time

logger = logging.getLogger(__name__)

# Загружаем переменные окружения
load_dotenv()

# КРИТИЧЕСКИ ВАЖНО для CUDA
multiprocessing.set_start_method('spawn', force=True)

def kill_gpu_processes():
    """Убиваем все процессы Python, использующие GPU 3"""
    try:
        # Получаем список процессов на GPU 3
        cmd = "nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader,nounits"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpu_3_pids = []
            
            for line in lines:
                if line.strip():
                    parts = line.split(', ')
                    if len(parts) >= 2:
                        pid = parts[0].strip()
                        # Проверяем, что это процесс на GPU 3 (по индексу или UUID)
                        # Здесь нужно адаптировать под ваш конкретный GPU 3
                        gpu_3_pids.append(pid)
            
            # Убиваем процессы
            for pid in gpu_3_pids:
                try:
                    subprocess.run(f"kill -9 {pid}", shell=True)
                    logger.info(f"Убит процесс GPU: {pid}")
                except:
                    pass
        
        # Дополнительная очистка через nvidia-smi
        subprocess.run("nvidia-smi --gpu-reset -i 3", shell=True, capture_output=True)
        
    except Exception as e:
        logger.error(f"Ошибка при очистке GPU процессов: {e}")

def cleanup_orphan_processes():
    """Очистка зависших процессов при завершении"""
    try:
        kill_gpu_processes()
        
        current_pid = os.getpid()
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if 'python' in proc.info['name'].lower() and proc.info['pid'] != current_pid:
                    cmdline = proc.info['cmdline'] or []
                    if any('tellscope' in str(cmd).lower() for cmd in cmdline):
                        proc.terminate()
                        time.sleep(0.5)
                        if proc.is_running():
                            proc.kill()
                        logger.info(f"Завершен процесс: {proc.info['pid']}")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception as e:
        logger.error(f"Ошибка при очистке процессов: {e}")

# Регистрируем очистку при завершении
atexit.register(cleanup_orphan_processes)

# Обработчик сигналов для корректного завершения
def signal_handler(signum, frame):
    logger.info(f"Получен сигнал {signum}, очищаем ресурсы...")
    cleanup_orphan_processes()
    os._exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# Создаем приложение Celery
celery_app = Celery('tellscope_backend')

# Настройки Celery с принудительным перезапуском воркеров
celery_app.conf.update(
    broker_url=os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0'),
    result_backend=os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0'),
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    
    # КРИТИЧЕСКИ ВАЖНЫЕ НАСТРОЙКИ для очистки памяти
    worker_preload_app=False,
    worker_pool='processes',
    worker_concurrency=2,  # Увеличьте concurrency
    worker_max_tasks_per_child=10,  # Увеличьте лимит задач
    worker_proc_alive_timeout=10,  # Быстрое завершение
    task_soft_time_limit=7200,     # 2 час на задачу
    task_time_limit=3900,          # 65 минут максимум
    
    # Настройки для быстрого освобождения ресурсов
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    
    include=['tasks'],

    # КРИТИЧЕСКИ ВАЖНО для стабильности Elasticsearch соединений
    broker_connection_retry_on_startup=True,
    broker_connection_retry=True,
    broker_connection_max_retries=10,
    
    # Настройки для предотвращения memory leaks
    worker_disable_rate_limits=True,
    worker_send_task_events=False,
    task_send_sent_event=False,
    
    # Более агрессивная очистка
    worker_cancel_long_running_tasks_on_connection_loss=True
)

# Автоматическое обнаружение задач
celery_app.autodiscover_tasks(['tasks'])



if __name__ == '__main__':
    celery_app.start()