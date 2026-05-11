"""Конфигурация агента"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API
    MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-Coder-Next")
    MODEL_URL = os.getenv("MODEL_URL", "https://foundation-models.api.cloud.ru/v1")
    API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY", "")
    
    # Модель
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.5"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS_RESPONSE", "4096"))
    TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "120"))
    
    # Контекст
    MAX_CONTEXT = int(os.getenv("MAX_CONTEXT_TOKENS", "262144"))
    SUMMARIZE_AT = 80000
    CRITICAL_AT = 200000
    
    # RAG
    RAG_TOP_K = int(os.getenv("RAG_TOP_K", "10"))
    CHUNK_SIZE = 2000
    CHUNK_OVERLAP = 400
    
    # Граф
    RECURSION_LIMIT = int(os.getenv("RECURSION_LIMIT", "30"))
    
    # Директории
    WORK_DIR = Path("./workdir")
    CHECKPOINT_DIR = Path("./checkpoints")
    CHROMA_DIR = Path("./chroma_db")
    LOGS_DIR = Path("./logs")
    
    def __init__(self):
        for d in [self.WORK_DIR, self.CHECKPOINT_DIR, self.CHROMA_DIR, self.LOGS_DIR]:
            d.mkdir(parents=True, exist_ok=True)

# Создаём ОДИН экземпляр
config = Config()