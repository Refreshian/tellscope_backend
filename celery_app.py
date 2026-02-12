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

# 🔥 КРИТИЧЕСКИ ВАЖНО: Устанавливаем ПЕРЕД любыми импортами
os.environ['CUDA_VISIBLE_DEVICES'] = '3'  # Только GPU 3
os.environ['TORCH_NVML_BASED_CUDA_CHECK'] = '0'  # Отключаем NVML
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# Загружаем переменные окружения
load_dotenv()

# КРИТИЧЕСКИ ВАЖНО для CUDA
multiprocessing.set_start_method('spawn', force=True)

def kill_gpu_processes():
    """Убиваем все процессы, использующие GPU 3"""
    try:
        cmd = "nvidia-smi --id=3 --query-compute-apps=pid --format=csv,noheader"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            
            for pid in pids:
                pid = pid.strip()
                if pid:
                    try:
                        subprocess.run(f"kill -9 {pid}", shell=True)
                        logger.info(f"✅ Убит процесс GPU: {pid}")
                    except:
                        pass
        
        # Сброс GPU
        subprocess.run("nvidia-smi --gpu-reset -i 3", shell=True, capture_output=True)
        logger.info("✅ GPU 3 сброшен")
        
    except Exception as e:
        logger.error(f"❌ Ошибка при очистке GPU: {e}")

def cleanup_orphan_processes():
    """Очистка зависших процессов"""
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
                        logger.info(f"✅ Завершен процесс: {proc.info['pid']}")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception as e:
        logger.error(f"❌ Ошибка при очистке процессов: {e}")

atexit.register(cleanup_orphan_processes)

def signal_handler(signum, frame):
    logger.info(f"📡 Получен сигнал {signum}, очищаем ресурсы...")
    cleanup_orphan_processes()
    os._exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# Создаем приложение Celery
celery_app = Celery('tellscope_backend')

# Настройки Celery
celery_app.conf.update(
    broker_url=os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0'),
    result_backend=os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0'),
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    
    # 🔥 ВАЖНО: spawn для корректной работы CUDA
    worker_pool='processes',
    worker_preload_app=False,  # Не загружаем модель в главном процессе
    worker_concurrency=2,
    worker_max_tasks_per_child=5,  # Перезапуск после 5 задач
    worker_proc_alive_timeout=10,
    
    task_soft_time_limit=7200,
    task_time_limit=3900,
    
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    
    imports=['tasks'],
    
    broker_connection_retry_on_startup=True,
    broker_connection_retry=True,
    broker_connection_max_retries=10,
    
    worker_disable_rate_limits=True,
    worker_send_task_events=False,
    task_send_sent_event=False,
    
    worker_cancel_long_running_tasks_on_connection_loss=True
)

celery_app.autodiscover_tasks(['tasks'])

if __name__ == '__main__':
    celery_app.start()