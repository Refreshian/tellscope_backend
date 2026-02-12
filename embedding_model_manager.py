import os
import torch
import gc
import logging
from sentence_transformers import SentenceTransformer
import numpy as np
import time
import threading

logger = logging.getLogger(__name__)

# 🔥 КРИТИЧЕСКИ ВАЖНО: Устанавливаем переменные ДО импорта torch
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# 🔥 НОВОЕ: Отключаем NVML в форк-процессах
os.environ['CUDA_VISIBLE_DEVICES'] = '3'  # Используем только GPU 3
os.environ['TORCH_NVML_BASED_CUDA_CHECK'] = '0'  # Отключаем проверку через NVML

class ModelManager:
    _instance = None
    _model = None
    _initialized = False
    _lock = threading.Lock()
    _model_lock = threading.Lock()
    _process_id = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, device_id=None):
        with self._lock:
            if not self._initialized:
                self.model_path = "deepvk/USER2-base"
                self.cache_folder = '/home/dev/tellscope_app/tellscope_backend/data/embed_models'
                self.device = None
                self.preferred_device_id = device_id if device_id is not None else 0  # Используем GPU 0 (видимый как единственный)
                self._initialized = True
                
                os.makedirs(self.cache_folder, exist_ok=True)
    
    def _check_process_changed(self):
        """Проверяет, сменился ли процесс (fork)"""
        import multiprocessing
        current_pid = multiprocessing.current_process().pid
        
        if self._process_id is None:
            self._process_id = current_pid
            return False
        
        if self._process_id != current_pid:
            logger.info(f"🔄 Обнаружена смена процесса: {self._process_id} -> {current_pid}")
            self._process_id = current_pid
            return True
        
        return False
    
    def encode_texts(self, texts, batch_size=32, **kwargs):
        with self._model_lock:
            logger.info(f"🔒 Получена блокировка модели для {len(texts)} текстов")
            
            # Проверяем смену процесса
            if self._check_process_changed():
                logger.info("🔄 Сброс модели из-за смены процесса")
                self._model = None
                self.device = None
                self.clear_cuda_memory()
            
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
                    error_msg = str(e)
                    
                    if "CUDA out of memory" in error_msg:
                        logger.warning(f"CUDA out of memory при batch_size={batch_size}. Уменьшаем batch...")
                        self.clear_cuda_memory()
                        batch_size = batch_size // 2
                        if batch_size < 1:
                            logger.error("Невозможно подобрать подходящий batch_size (<1).")
                            raise
                        time.sleep(0.5)
                    
                    elif "NVML" in error_msg or "CUDA initialization" in error_msg:
                        logger.error(f"❌ Ошибка инициализации CUDA: {e}")
                        logger.info("🔄 Переключение на CPU")
                        self.device = 'cpu'
                        self._model = None
                        return self.encode_texts(texts, batch_size=orig_batch_size, **kwargs)
                    
                    else:
                        logger.error(f"Неизвестная ошибка при кодировании: {e}")
                        if self.device != 'cpu':
                            logger.info("🔄 Переключение на CPU из-за ошибки")
                            self.device = 'cpu'
                            self._model = None
                            return self.encode_texts(texts, batch_size=orig_batch_size, **kwargs)
                        raise e
                        
                except Exception as e:
                    logger.error(f"Ошибка при кодировании: {e}")
                    raise e
    
    def initialize_model(self):
        """Безопасная инициализация GPU с защитой от форков"""
        try:
            import multiprocessing
            current_process = multiprocessing.current_process()
            logger.info(f"🔧 Инициализация модели в процессе: {current_process.name} (PID: {current_process.pid})")
            
            # Проверяем доступность CUDA
            if not torch.cuda.is_available():
                logger.warning("⚠️ CUDA недоступна, используем CPU")
                return 'cpu'
            
            # 🔥 НОВОЕ: Безопасная инициализация CUDA
            try:
                # Принудительно инициализируем CUDA без NVML
                torch.cuda.init()
                
                available_gpus = torch.cuda.device_count()
                if available_gpus == 0:
                    logger.warning("⚠️ Нет доступных GPU, используем CPU")
                    return 'cpu'
                
                logger.info(f"✅ Доступно GPU: {available_gpus}")
                
                # Используем первый доступный GPU (это GPU 3 из-за CUDA_VISIBLE_DEVICES)
                device_id = 0
                torch.cuda.set_device(device_id)
                device = f'cuda:{device_id}'
                
                # Тестовая операция
                test_tensor = torch.randn(10, 10, device=device)
                _ = torch.mm(test_tensor, test_tensor.T)
                del test_tensor
                torch.cuda.empty_cache()
                
                logger.info(f"✅ GPU {device} успешно инициализирован")
                return device
                
            except RuntimeError as cuda_error:
                error_msg = str(cuda_error)
                
                if "NVML" in error_msg or "initialization" in error_msg:
                    logger.warning(f"⚠️ Ошибка инициализации CUDA: {cuda_error}")
                    logger.info("🔄 Переключение на CPU")
                    return 'cpu'
                else:
                    raise cuda_error
            
        except Exception as e:
            logger.warning(f"⚠️ GPU инициализация не удалась: {e}, используем CPU")
            return 'cpu'
    
    def clear_cuda_memory(self):
        """Улучшенная очистка CUDA памяти"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                
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
            # Проверяем смену процесса
            if self._check_process_changed():
                logger.info("🔄 Сброс модели из-за смены процесса в get_model")
                self._model = None
                self.device = None
            
            if self._model is None:
                if self.device is None:
                    self.device = self.initialize_model()
                
                logger.info(f"📦 Загрузка модели на устройство: {self.device}")
                
                try:
                    self._model = SentenceTransformer(
                        self.model_path,
                        device=self.device,
                        cache_folder=self.cache_folder,
                        trust_remote_code=True
                    )
                    
                    # Тестовое кодирование
                    test_embedding = self._model.encode(
                        ["тест"], 
                        show_progress_bar=False,
                        batch_size=1
                    )
                    del test_embedding
                    
                    logger.info(f"✅ Модель загружена на {self.device}")
                    logger.info(f"✅ Файлы модели в: {self.cache_folder}")
                    
                except Exception as e:
                    logger.error(f"❌ Ошибка загрузки модели: {e}")
                    
                    if self.device != 'cpu':
                        logger.info("🔄 Переключаемся на CPU")
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
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                except:
                    pass
                    
            gc.collect()
            
        self.device = None
        self._process_id = None

model_manager = ModelManager()