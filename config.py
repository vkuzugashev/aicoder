"""
Централизованная конфигурация агента
"""
import os
from dataclasses import dataclass, field
from typing import Set
from pathlib import Path

import dotenv

dotenv.load_dotenv()

@dataclass
class AgentConfig:
    """Продвинутая конфигурация с управлением памятью"""
    
    # ===== Лимиты контекста =====
    MAX_CONTEXT_TOKENS: int = int(os.environ.get("MAX_CONTEXT_TOKENS", "200000"))
    TARGET_CONTEXT_USAGE: float = 0.6  # Целевое заполнение (60%)
    CRITICAL_CONTEXT_USAGE: float = 0.85  # Критический порог
    MIN_CONTEXT_FOR_TRIM: int = 50000  # Минимум токенов для обрезки
    
    # ===== Параметры памяти =====
    WORKING_MEMORY_SIZE: int = 20  # Размер рабочей памяти
    EPISODIC_MEMORY_SIZE: int = 50  # Размер эпизодической памяти
    CONTEXT_WINDOW_SIZE: int = 10  # Окно контекста для LLM
    
    # ===== Параметры суммаризации =====
    SUMMARIZATION_TRIGGER: float = float(os.environ.get("SUMMARIZATION_TRIGGER", "0.5"))
    SUMMARIZATION_KEEP: int = int(os.environ.get("SUMMARIZATION_KEEP", "3"))
    SUMMARIZATION_CHUNK_SIZE: int = 5000
    SUMMARIZATION_OVERLAP: int = 500
    MAX_SUMMARY_LENGTH: int = 2000
    
    # ===== Приоритеты суммаризации =====
    PRIORITY_PATTERNS: dict = field(default_factory=lambda: {
        'code_creation': ['создай', 'напиши', 'create', 'write', 'implement'],
        'bug_fix': ['ошибка', 'исправь', 'fix', 'bug', 'error'],
        'search': ['найди', 'поиск', 'find', 'search', 'locate'],
        'modification': ['измени', 'обнови', 'update', 'modify', 'change']
    })
    
    # ===== Параметры RAG =====
    RAG_TOP_K: int = int(os.environ.get("RAG_TOP_K", "10"))
    RAG_FETCH_K: int = int(os.environ.get("RAG_FETCH_K", "20"))
    RAG_SIMILARITY_THRESHOLD: float = 0.6
    RAG_LAMBDA_MULT: float = 0.7  # Баланс relevance vs diversity
    
    # ===== Параметры модели =====
    MODEL_NAME: str = "Qwen/Qwen3-Coder-Next"
    MODEL_URL: str = os.environ.get("MODEL_URL", "https://foundation-models.api.cloud.ru/v1")
    API_KEY: str = os.environ.get("API_KEY", "")
    TEMPERATURE: float = float(os.environ.get("TEMPERATURE", "0.5"))
    MAX_TOKENS_RESPONSE: int = 2500
    REQUEST_TIMEOUT: int = 120
    MAX_RETRIES: int = 3
    
    # ===== Параметры агента =====
    DEFAULT_RECURSION_LIMIT: int = int(os.environ.get("DEFAULT_RECURSION_LIMIT", "30"))
    STREAM_TIMEOUT: int = 180
    RETRY_DELAY_MIN: int = 2
    RETRY_DELAY_MAX: int = 15
    
    # ===== Директории =====
    WORK_DIR: Path = Path("./workdir")
    CHECKPOINT_DIR: Path = Path("./checkpoints")
    CHROMA_DIR: Path = Path("./chroma_db")
    LOGS_DIR: Path = Path("./logs")
    CACHE_DIR: Path = Path("./cache")
    SUMMARIES_DIR: Path = Path("./summaries")
    
    # ===== Поддерживаемые типы файлов =====
    SUPPORTED_EXTENSIONS: Set[str] = field(default_factory=lambda: {
        '.py', '.js', '.ts', '.jsx', '.tsx', '.vue', 
        '.html', '.css', '.scss', '.json', '.md',
        '.vb', '.frm', '.bas', '.cls', '.frx'
    })
    
    # ===== Исключаемые директории =====
    EXCLUDED_DIRS: Set[str] = field(default_factory=lambda: {
        '.git', '__pycache__', 'node_modules', 
        '.vscode', 'dist', 'build', 'backup', 'archive'
    })
    
    # ===== Runtime файлы =====
    PROGRESS_FILE: str = "runtime/progress.md"
    MODULES_FILE: str = "runtime/app_modules.md"
    DATABASE_FILE: str = "runtime/database.md"
    DESCRIPTION_FILE: str = "runtime/description.md"
    
    def __post_init__(self):
        """Создание директорий при инициализации"""
        for dir_path in [self.WORK_DIR, self.CHECKPOINT_DIR, 
                        self.CHROMA_DIR, self.LOGS_DIR, 
                        self.CACHE_DIR, self.SUMMARIES_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)

# Глобальный экземпляр конфигурации
config = AgentConfig()