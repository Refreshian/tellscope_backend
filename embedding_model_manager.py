import torch
import gc
import logging
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import time
import threading
import subprocess

logger = logging.getLogger(__name__)

async def force_clear_gpu():
    """Принудительная очистка всех процессов на GPU"""
    try:
        # 1. Очистка через PyTorch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        
        # 2. Убиваем все процессы Python, использующие GPU
        result = subprocess.run(
            "nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9",
            shell=True,
            capture_output=True,
            text=True
        )
        logging.info(f"Очистка GPU: {result.stdout or 'Успешно'}")
        
        # 3. Дополнительная очистка через NVIDIA-SMI
        os.system("nvidia-smi --gpu-reset -i 3")
        return {"status": "GPU memory cleared"}
    except Exception as e:
        logging.error(f"Ошибка очистки GPU: {e}")
        raise

class ModelManager:
    _instance = None
    _model = None
    _initialized = False
    _lock = threading.Lock()
    _model_lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, device_id=None):  # 🔥 Изменено: device_id по умолчанию None
        with self._lock:
            if not self._initialized:
                os.chdir('/home/dev/tellscope_app/tellscope_backend/data/embed_models')
                self.model_path = "deepvk/USER2-base"
                self.device = None  # 🔥 Инициализация откладывается
                self.preferred_device_id = device_id  # 🔥 Сохраняем предпочтительный ID
                self._initialized = True
    
    def encode_texts(self, texts, batch_size=32, **kwargs):
        with self._model_lock:
            logger.info(f"🔒 Получена блокировка модели для {len(texts)} текстов")
            
            model = self.get_model()
            self.clear_cuda_memory()
            
            if isinstance(texts, str):
                texts = [texts]
            if len(texts) == 0:
                return np.array([])

            orig_batch_size = batch_size
            while batch_size >= 1:
                try:
                    logger.info(f"Кодирование {len(texts)} текстов с batch_size={batch_size}")
                    embeddings = model.encode(
                        texts,
                        convert_to_tensor=False,
                        normalize_embeddings=kwargs.get('normalize_embeddings', True),
                        show_progress_bar=False,
                        batch_size=batch_size
                    )
                    logger.info(f"✅ Получено {len(embeddings)} эмбеддингов")
                    return embeddings
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        logger.warning(f"CUDA out of memory при batch_size={batch_size}. Уменьшаем batch...")
                        self.clear_cuda_memory()
                        batch_size = batch_size // 2
                        if batch_size < 1:
                            logger.error("Невозможно подобрать подходящий batch_size (<1).")
                            raise
                        time.sleep(0.5)
                    else:
                        logger.error(f"Ошибка при кодировании: {e}")
                        if self.device != 'cpu':
                            logger.info("Переключение на CPU из-за ошибки")
                            self.device = 'cpu'
                            self._model = None
                            return self.encode_texts(texts, batch_size=orig_batch_size, **kwargs)
                        raise e
                except Exception as e:
                    logger.error(f"Ошибка при кодировании: {e}")
                    raise e
    
    def initialize_model(self):
        """Безопасная инициализация GPU с учетом видимости устройств"""
        try:
            import multiprocessing
            current_process = multiprocessing.current_process()
            
            if not torch.cuda.is_available():
                logger.info("CUDA недоступна, используем CPU")
                return 'cpu'
            
            # Устанавливаем переменные окружения для стабильности
            os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
            os.environ['TORCH_USE_CUDA_DSA'] = '1'
            
            available_gpus = torch.cuda.device_count()
            if available_gpus == 0:
                logger.info("Нет доступных GPU, используем CPU")
                return 'cpu'
            
            logger.info(f"Доступно GPU: {available_gpus}")
            
            # 🔥 ВАЖНО: Определяем доступное устройство
            if self.preferred_device_id is not None:
                # Проверяем, что запрашиваемый device_id доступен в PyTorch
                if self.preferred_device_id >= available_gpus:
                    logger.warning(f"GPU {self.preferred_device_id} недоступен (доступно {available_gpus}). Используем последний доступный.")
                    device_id = available_gpus - 1
                else:
                    device_id = self.preferred_device_id
            else:
                # Используем последний доступный GPU (обычно это нужный нам GPU 3)
                device_id = available_gpus - 1
            
            # Проверяем реальную доступность
            try:
                torch.cuda.set_device(device_id)
                device = f'cuda:{device_id}'
                
                # Тестовая операция
                test_tensor = torch.randn(10, 10, device=device)
                _ = torch.mm(test_tensor, test_tensor.T)
                del test_tensor
                torch.cuda.empty_cache()
                
                logger.info(f"✅ GPU {device} успешно инициализирован в процессе {current_process.name}")
                return device
                
            except RuntimeError as e:
                if "invalid device ordinal" in str(e):
                    logger.warning(f"GPU {device_id} недоступен: {e}")
                    # Пробуем первый доступный GPU
                    if device_id != 0:
                        device_id = 0
                        torch.cuda.set_device(device_id)
                        device = f'cuda:{device_id}'
                        
                        test_tensor = torch.randn(10, 10, device=device)
                        _ = torch.mm(test_tensor, test_tensor.T)
                        del test_tensor
                        torch.cuda.empty_cache()
                        
                        logger.info(f"✅ Переключились на GPU {device}")
                        return device
                    else:
                        raise e
                else:
                    raise e
            
        except Exception as e:
            logger.warning(f"GPU инициализация не удалась: {e}, используем CPU")
            return 'cpu'
    
    def clear_cuda_memory(self):
        """Улучшенная очистка CUDA памяти"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                
                # Дополнительная очистка для всех устройств
                for i in range(torch.cuda.device_count()):
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
            
            gc.collect()
            
        except Exception as e:
            logger.debug(f"Ошибка при очистке CUDA памяти: {e}")
    
    def get_model(self):
        """Ленивая инициализация модели в воркер процессе"""
        with self._lock:
            if self._model is None:
                # Инициализируем устройство только при первом обращении
                if self.device is None:
                    self.device = self.initialize_model()
                
                logger.info(f"Загрузка модели на устройство: {self.device}")
                
                try:
                    # Загружаем модель
                    os.chdir('/home/dev/tellscope_app/tellscope_backend/data/embed_models')
                    self._model = SentenceTransformer(
                        self.model_path,
                        device=self.device,
                        cache_folder='/tmp/sentence_transformers',
                        # local_files_only=True
                    )
                    
                    # Тестовое кодирование
                    test_embedding = self._model.encode(
                        ["тест"], 
                        show_progress_bar=False,
                        batch_size=1
                    )
                    del test_embedding
                    
                    logger.info(f"✅ Модель загружена на {self.device}")
                    
                except Exception as e:
                    logger.error(f"Ошибка загрузки модели: {e}")
                    if self.device != 'cpu':
                        logger.info("Переключаемся на CPU")
                        self.device = 'cpu'
                        self._model = None
                        return self.get_model()
                    raise e
                    
            return self._model

    def cleanup(self):
        """Удаляет модель из памяти, очищает CUDA"""
        with self._lock:
            if self._model is not None:
                try:
                    del self._model
                except Exception:
                    pass
                self._model = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                gc.collect()
        self.device = None

# Глобальный экземпляр - без передачи device_id
model_manager = ModelManager()