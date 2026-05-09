#!/usr/bin/env python3
"""
Модернизированный AI агент для переписывания VB6 → NestJS + React
Версия 2.0 с улучшенной архитектурой и современными подходами
"""

import sys
import os
import asyncio
import traceback
import logging
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import aiosqlite
import httpx
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_core.tools import tool
from langchain_core.messages import AIMessageChunk, HumanMessage, SystemMessage, AIMessage
from langchain_core.messages.utils import trim_messages, count_tokens_approximately
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import tools
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

# os.environ["SSL_CERT_FILE"] = certifi.where()
# os.environ['CURL_CA_BUNDLE'] = certifi.where()
# os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

# ===== НАСТРОЙКА ОКРУЖЕНИЯ =====
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

MODEL_URL = os.getenv('MODEL_URL')

# ===== КОНФИГУРАЦИЯ АГЕНТА =====
class AgentConfig:
    """Конфигурация агента с современными настройками"""
    MAX_CONTEXT_TOKENS = 200000
    DEFAULT_RECURSION_LIMIT = 200
    MAX_CONTEXT_MESSAGES = 10
    PERSIST_DIRECTORY = './chroma_db'
    CHECKPOINT_DIRECTORY = './checkpoints'
    LOG_DIRECTORY = './logs'
    METRICS_DIRECTORY = './metrics'
    
    # Новые параметры
    ENABLE_STREAMING = True
    ENABLE_METRICS = True
    ENABLE_LOGGING = True
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    MAX_RETRIES = 3
    TIMEOUT_SECONDS = 180
    RETRY_DELAY_MIN = 2
    RETRY_DELAY_MAX = 15
    
    # Пути к runtime файлам
    RUNTIME_DIR = './runtime'
    PROGRESS_FILE = f'{RUNTIME_DIR}/progress.md'
    MODULES_FILE = f'{RUNTIME_DIR}/app_modules.md'
    DATABASE_FILE = f'{RUNTIME_DIR}/database.md'
    DESCRIPTION_FILE = f'{RUNTIME_DIR}/description.md'
    
    # Поддерживаемые расширения файлов
    SUPPORTED_EXTENSIONS = {'.py', '.js', '.ts', '.jsx', '.tsx', '.vue', '.html', '.css', '.scss', '.json', '.md'}
    
    # Исключаемые директории
    EXCLUDED_DIRS = {'.git', '__pycache__', 'node_modules', '.vscode', 'dist', 'build'}

config = AgentConfig()

# ===== НАСТРОЙКА ЛОГИРОВАНИЯ =====
def setup_logging():
    """Настраивает логирование с современными подходами"""
    if not config.ENABLE_LOGGING:
        return
    
    os.makedirs(config.LOG_DIRECTORY, exist_ok=True)
    
    log_filename = f"agent_{datetime.now().strftime('%Y%m%d')}.log"
    log_path = os.path.join(config.LOG_DIRECTORY, log_filename)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

logger = setup_logging()

# ===== СИСТЕМА МЕТРИК =====
class MetricsCollector:
    """Сбор метрик производительности агента"""
    
    def __init__(self):
        self.metrics = {
            'requests_count': 0,
            'total_tokens_used': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0,
            'tool_usage': {},
            'session_start': datetime.now()
        }
        os.makedirs(config.METRICS_DIRECTORY, exist_ok=True)
    
    def record_request(self, tokens: int, response_time: float, success: bool, tools_used: List[str]):
        """Записывает метрики запроса"""
        self.metrics['requests_count'] += 1
        self.metrics['total_tokens_used'] += tokens
        
        if success:
            self.metrics['successful_requests'] += 1
        else:
            self.metrics['failed_requests'] += 1
        
        # Обновляем среднее время ответа
        total_requests = self.metrics['requests_count']
        current_avg = self.metrics['average_response_time']
        self.metrics['average_response_time'] = (
            (current_avg * (total_requests - 1) + response_time) / total_requests
        )
        
        # Учитываем использование инструментов
        for tool in tools_used:
            self.metrics['tool_usage'][tool] = self.metrics['tool_usage'].get(tool, 0) + 1
    
    def save_metrics(self):
        """Сохраняет метрики в файл"""
        if not config.ENABLE_METRICS:
            return
        
        metrics_file = os.path.join(config.METRICS_DIRECTORY, f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        metrics_data = self.metrics.copy()
        metrics_data['session_start'] = metrics_data['session_start'].isoformat()
        metrics_data['session_duration'] = str(datetime.now() - metrics_data['session_start'])
        
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, indent=2, ensure_ascii=False)
        
        if logger:
            logger.info(f"Метрики сохранены в {metrics_file}")

metrics = MetricsCollector()

# ===== ПОДГОТОВКА ДИРЕКТОРИЙ =====
def prepare_directories():
    """Создает необходимые директории"""
    directories = [
        config.CHECKPOINT_DIRECTORY,
        config.LOG_DIRECTORY,
        config.METRICS_DIRECTORY,
        config.RUNTIME_DIR
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        if logger:
            logger.debug(f"Директория подготовлена: {directory}")

prepare_directories()

# ===== МОДЕРНИЗИРОВАННАЯ СИСТЕМА ИНСТРУМЕНТОВ =====

def log_tool_usage(tool_name: str, args: tuple, success: bool, result: str = None):
    """Логирует использование инструментов"""
    if logger:
        status = "✅" if success else "❌"
        logger.info(f"{status} Инструмент: {tool_name}, args: {args}")
        if not success and result:
            logger.warning(f"Ошибка инструмента {tool_name}: {result}")

@tool("list_directory", description="Получить полный список файлов и папок в указанной директории")
def list_directory_tool(path: str):
    """Возвращает полный список файлов без обрезаний с улучшенной обработкой ошибок"""
    try:
        result = tools.list_dir(path)
        print(f"📁 Прочитана папка: {path}")
        log_tool_usage("list_directory", (path,), True)
        return result
    except FileNotFoundError as e:
        error_msg = f"Директория не найдена: {path}"
        print(f"❌ {error_msg}")
        log_tool_usage("list_directory", (path,), False, error_msg)
        return f"Ошибка: {error_msg}"
    except PermissionError as e:
        error_msg = f"Нет доступа к директории: {path}"
        print(f"❌ {error_msg}")
        log_tool_usage("list_directory", (path,), False, error_msg)
        return f"Ошибка: {error_msg}"
    except Exception as e:
        error_msg = f"Неожиданная ошибка при чтении директории {path}: {str(e)}"
        print(f"❌ {error_msg}")
        log_tool_usage("list_directory", (path,), False, error_msg)
        return f"Ошибка: {error_msg}"

@tool("read_file", description="Прочитать полное содержимое файла. encoding: utf-8, cp1251")
def read_file_tool(path: str, encoding: str = "utf-8"):
    """Читает файл полностью с улучшенной обработкой ошибок и метаданными"""
    try:
        result = tools.read_file(path, encoding)
        file_size = len(str(result))
        print(f"📖 Прочитан файл: {path} (полный размер: {file_size:,} символов)")
        
        # Логируем статистику
        if logger:
            logger.info(f"📖 Файл прочитан: {path}, размер: {file_size:,}, кодировка: {encoding}")
        
        log_tool_usage("read_file", (path, encoding), True)
        
        # Добавляем метаданные о файле
        return f"=== ФАЙЛ: {path} (размер: {file_size:,} символов, кодировка: {encoding}) ===\n\n{result}"
        
    except FileNotFoundError as e:
        error_msg = f"Файл не найден: {path}"
        print(f"❌ {error_msg}")
        log_tool_usage("read_file", (path, encoding), False, error_msg)
        return f"Ошибка: {error_msg}"
    except UnicodeDecodeError as e:
        error_msg = f"Ошибка кодировки при чтении файла {path}: {str(e)}. Попробуйте другую кодировку (cp1251, latin1)"
        print(f"❌ {error_msg}")
        log_tool_usage("read_file", (path, encoding), False, error_msg)
        return f"Ошибка: {error_msg}"
    except PermissionError as e:
        error_msg = f"Нет доступа к файлу: {path}"
        print(f"❌ {error_msg}")
        log_tool_usage("read_file", (path, encoding), False, error_msg)
        return f"Ошибка: {error_msg}"
    except Exception as e:
        error_msg = f"Неожиданная ошибка при чтении файла {path}: {str(e)}"
        print(f"❌ {error_msg}")
        log_tool_usage("read_file", (path, encoding), False, error_msg)
        return f"Ошибка: {error_msg}"

@tool("write_file", description="Записать или создать файл. overwrite=True для перезаписи существующего")
def write_file_tool(path: str, content: str, overwrite: bool = False):
    """Записывает файл с улучшенной безопасностью и валидацией"""
    try:
        # Проверка безопасности пути
        if '..' in path or path.startswith('/'):
            error_msg = f"Небезопасный путь: {path}"
            print(f"❌ {error_msg}")
            log_tool_usage("write_file", (path, f"<{len(content)} chars>", overwrite), False, error_msg)
            return f"Ошибка безопасности: {error_msg}"
        
        # Проверка существования файла
        exists = tools.file_exists(path)
        if exists and not overwrite:
            error_msg = f"Файл {path} существует. Используй overwrite=True чтобы перезаписать"
            print(f"⚠️ {error_msg}")
            log_tool_usage("write_file", (path, f"<{len(content)} chars>", overwrite), False, error_msg)
            return error_msg
        
        # Запись файла
        result = tools.write_file(path, content, overwrite=overwrite)
        if result:
            print(f"✍️ Записан файл: {path} ({len(content):,} символов)")
            
            # Логируем успешную запись
            if logger:
                logger.info(f"✍️ Файл записан: {path}, размер: {len(content):,}, перезапись: {overwrite}")
            
            log_tool_usage("write_file", (path, f"<{len(content)} chars>", overwrite), True)
            return f"✅ Файл {path} успешно записан ({len(content):,} символов)"
        
        error_msg = f"Не удалось записать файл {path}"
        print(f"❌ {error_msg}")
        log_tool_usage("write_file", (path, f"<{len(content)} chars>", overwrite), False, error_msg)
        return f"Ошибка: {error_msg}"
        
    except PermissionError as e:
        error_msg = f"Нет прав на запись в файл: {path}"
        print(f"❌ {error_msg}")
        log_tool_usage("write_file", (path, f"<{len(content)} chars>", overwrite), False, error_msg)
        return f"Ошибка: {error_msg}"
    except Exception as e:
        error_msg = f"Неожиданная ошибка при записи файла {path}: {str(e)}"
        print(f"❌ {error_msg}")
        log_tool_usage("write_file", (path, f"<{len(content)} chars>", overwrite), False, error_msg)
        return f"Ошибка: {error_msg}"

@tool("create_directory", description="Создать новую папку")
def create_directory_tool(path: str):
    """Создает директорию с улучшенной обработкой ошибок"""
    try:
        # Проверка безопасности пути
        if '..' in path or path.startswith('/'):
            error_msg = f"Небезопасный путь: {path}"
            print(f"❌ {error_msg}")
            log_tool_usage("create_directory", (path,), False, error_msg)
            return f"Ошибка безопасности: {error_msg}"
        
        result = tools.create_dir(path)
        if result:
            print(f"📁 Создана папка: {path}")
            log_tool_usage("create_directory", (path,), True)
            return f"✅ Папка {path} создана"
        
        error_msg = f"Не удалось создать папку {path}"
        print(f"❌ {error_msg}")
        log_tool_usage("create_directory", (path,), False, error_msg)
        return f"Ошибка: {error_msg}"
        
    except Exception as e:
        error_msg = f"Ошибка при создании папки {path}: {str(e)}"
        print(f"❌ {error_msg}")
        log_tool_usage("create_directory", (path,), False, error_msg)
        return f"Ошибка: {error_msg}"

@tool("file_exists", description="Проверить, существует ли файл")
def file_exists_tool(path: str):
    """Проверяет существование файла с логированием"""
    try:
        exists = tools.file_exists(path)
        result = f"Файл {path} {'существует' if exists else 'не существует'}"
        print(f"🔍 {result}")
        log_tool_usage("file_exists", (path,), True)
        return result
    except Exception as e:
        error_msg = f"Ошибка при проверке файла {path}: {str(e)}"
        print(f"❌ {error_msg}")
        log_tool_usage("file_exists", (path,), False, error_msg)
        return f"Ошибка: {error_msg}"

@tool("dir_exists", description="Проверить, существует ли папка")
def dir_exists_tool(path: str):
    """Проверяет существование директории с логированием"""
    try:
        exists = tools.dir_exists(path)
        result = f"Папка {path} {'существует' if exists else 'не существует'}"
        print(f"🔍 {result}")
        log_tool_usage("dir_exists", (path,), True)
        return result
    except Exception as e:
        error_msg = f"Ошибка при проверке папки {path}: {str(e)}"
        print(f"❌ {error_msg}")
        log_tool_usage("dir_exists", (path,), False, error_msg)
        return f"Ошибка: {error_msg}"

@tool("delete_dir", description="Удалить директорию (осторожно!)")
def delete_dir_tool(path: str):
    """Удаляет директорию с проверками безопасности"""
    try:
        # Предупреждение об опасной операции
        print(f"⚠️ Попытка удаления директории: {path}")
        
        # Проверка безопасности
        if path in ['.', './', 'workdir', 'src', 'api', 'client']:
            error_msg = f"Удаление системной директории {path} запрещено"
            print(f"❌ {error_msg}")
            log_tool_usage("delete_dir", (path,), False, error_msg)
            return f"Ошибка безопасности: {error_msg}"
        
        result = tools.delete_dir(path)
        if result:
            print(f"🗑️ Удалёна директория: {path}")
            log_tool_usage("delete_dir", (path,), True)
            return f"✅ Директория {path} удалёна"
        
        error_msg = f"Не удалось удалить директорию {path}"
        print(f"❌ {error_msg}")
        log_tool_usage("delete_dir", (path,), False, error_msg)
        return f"Ошибка: {error_msg}"
        
    except Exception as e:
        error_msg = f"Ошибка при удалении директории {path}: {str(e)}"
        print(f"❌ {error_msg}")
        log_tool_usage("delete_dir", (path,), False, error_msg)
        return f"Ошибка: {error_msg}"

@tool("delete_file", description="Удалить файл (осторожно!)")
def delete_file_tool(path: str):
    """Удаляет файл с проверками безопасности"""
    try:
        # Предупреждение об опасной операции
        print(f"⚠️ Попытка удаления файла: {path}")
        
        # Проверка безопасности
        dangerous_files = ['agent.py', 'agentv1.py', 'tools.py', 'requirements.txt', '.env']
        if any(path.endswith(file) for file in dangerous_files):
            error_msg = f"Удаление системного файла {path} запрещено"
            print(f"❌ {error_msg}")
            log_tool_usage("delete_file", (path,), False, error_msg)
            return f"Ошибка безопасности: {error_msg}"
        
        result = tools.delete_file(path)
        if result:
            print(f"🗑️ Удалён файл: {path}")
            log_tool_usage("delete_file", (path,), True)
            return f"✅ Файл {path} удалён"
        
        error_msg = f"Не удалось удалить файл {path}"
        print(f"❌ {error_msg}")
        log_tool_usage("delete_file", (path,), False, error_msg)
        return f"Ошибка: {error_msg}"
        
    except Exception as e:
        error_msg = f"Ошибка при удалении файла {path}: {str(e)}"
        print(f"❌ {error_msg}")
        log_tool_usage("delete_file", (path,), False, error_msg)
        return f"Ошибка: {error_msg}"

@tool("npm_install", description="Выполнить команду установки пакетов в терминале для указанной папки, передаётся в параметре path: npm install")
def npm_install_tool(path: str, options: str):
    """Выполняет npm install с улучшенным логированием"""
    try:
        print(f"🖥️ Выполняю npm install в {path} с опциями: {options}")
        result = tools.npm_install(path, options)
        log_tool_usage("npm_install", (path, options), True)
        return result
    except Exception as e:
        error_msg = f"Ошибка при выполнении npm install: {str(e)}"
        print(f"❌ {error_msg}")
        log_tool_usage("npm_install", (path, options), False, error_msg)
        return f"Ошибка: {error_msg}"

@tool("npm_build", description="Выполнить команду build проекта в терминале для указанной папки, передаётся в параметре path: npm build")
def npm_build_tool(path: str, options: str):
    """Выполняет npm build с улучшенным логированием"""
    try:
        print(f"🖥️ Выполняю npm build в {path} с опциями: {options}")
        result = tools.npm_build(path, options)
        log_tool_usage("npm_build", (path, options), True)
        return result
    except Exception as e:
        error_msg = f"Ошибка при выполнении npm build: {str(e)}"
        print(f"❌ {error_msg}")
        log_tool_usage("npm_build", (path, options), False, error_msg)
        return f"Ошибка: {error_msg}"

@tool("pwd", description="Получить имя текущей директории. аналог команды pwd")
def pwd_tool():
    """Получает текущую директорию с логированием"""
    try:
        result = tools.pwd()
        print(f"📍 Текущая директория: {result}")
        log_tool_usage("pwd", (), True)
        return result
    except Exception as e:
        error_msg = f"Ошибка при получении текущей директории: {str(e)}"
        print(f"❌ {error_msg}")
        log_tool_usage("pwd", (), False, error_msg)
        return f"Ошибка: {error_msg}"

# Глобальные переменные для векторного хранилища
embeddings = None
vectorstore = None
retriever = None

def initialize_vector_store():
    """Инициализирует векторное хранилище с обработкой ошибок"""
    global embeddings, vectorstore, retriever
    
    try:
        print("🔄 Инициализация векторного хранилища...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma(persist_directory=config.PERSIST_DIRECTORY, embedding_function=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        print("✅ Векторное хранилище инициализировано")
        if logger:
            logger.info("Векторное хранилище успешно инициализировано")
    except Exception as e:
        error_msg = f"Ошибка при инициализации векторного хранилища: {str(e)}"
        print(f"❌ {error_msg}")
        if logger:
            logger.error(error_msg)

@tool
def search_codebase_tool(query: str) -> str:
    """
    ИЩЕТ РЕЛЕВАНТНЫЕ ФРАГМЕНТЫ КОДА по смыслу с улучшенной обработкой.
    Используй этот инструмент, когда тебе нужно найти, где в проекте реализована какая-то функция,
    найти все места, использующие определенную библиотеку, или вспомнить логику работы какого-то модуля.
    """
    try:
        if not retriever:
            error_msg = "Векторное хранилище не инициализировано"
            print(f"❌ {error_msg}")
            log_tool_usage("search_codebase", (query,), False, error_msg)
            return f"Ошибка: {error_msg}"
        
        docs = retriever.invoke(query)
        if not docs:
            result = "Ничего не найдено по вашему запросу."
            print(f"🔍 {result}")
            log_tool_usage("search_codebase", (query,), True, result)
            return result

        # Форматируем результат, показывая содержимое чанков и их источник
        context = "\n\n---\n\n".join([f"Файл: {doc.metadata['source']}\nСодержание:\n{doc.page_content}" for doc in docs])
        result = f"Вот наиболее релевантные фрагменты из кодовой базы (найдено {len(docs)} чанков):\n\n{context}"
        print(f"🔍 Найдено {len(docs)} релевантных фрагментов по запросу: {query}")
        log_tool_usage("search_codebase", (query,), True)
        return result
        
    except Exception as e:
        error_msg = f"Ошибка при поиске в кодовой базе: {str(e)}"
        print(f"❌ {error_msg}")
        log_tool_usage("search_codebase", (query,), False, error_msg)
        return f"Ошибка: {error_msg}"

# ===== СПИСОК ИНСТРУМЕНТОВ =====
tools_list = [
    list_directory_tool,
    read_file_tool,
    write_file_tool,
    create_directory_tool,
    file_exists_tool,
    dir_exists_tool,
    delete_dir_tool,
    delete_file_tool,
    npm_install_tool,
    npm_build_tool,
    pwd_tool,
    search_codebase_tool
]

# ===== ОПТИМИЗИРОВАННАЯ РАБОТА С ВЕКТОРНЫМ ХРАНИЛИЩЕМ =====

def load_codebase(directory: str) -> List:
    """
    Загружает все файлы проекта с улучшенной обработкой и фильтрацией.
    Использует современную конфигурацию для расширений и исключений.
    """
    docs = []
    indexed_files = 0
    skipped_files = 0
    
    print(f"🔍 Индексация проекта: {directory}")
    
    for root, dirs, files in os.walk(directory):
        # Фильтруем исключенные директории
        dirs[:] = [d for d in dirs if d not in config.EXCLUDED_DIRS]
        
        for file in files:
            # Проверяем расширение файла
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext not in config.SUPPORTED_EXTENSIONS:
                skipped_files += 1
                continue
            
            file_path = os.path.join(root, file)
            try:
                # Определяем кодировку
                encoding = 'utf-8'
                if 'cp1251' in file_path or file_ext in ['.vb', '.frm']:
                    encoding = 'cp1251'
                
                loader = TextLoader(file_path, encoding=encoding)
                file_docs = loader.load()
                
                # Добавляем метаданные
                for doc in file_docs:
                    doc.metadata.update({
                        'file_path': file_path,
                        'relative_path': os.path.relpath(file_path, directory),
                        'file_extension': file_ext,
                        'file_size': os.path.getsize(file_path),
                        'indexed_at': datetime.now().isoformat()
                    })
                
                docs.extend(file_docs)
                indexed_files += 1
                
                if indexed_files % 10 == 0:
                    print(f"📄 Проиндексировано: {indexed_files} файлов")
                    
            except UnicodeDecodeError as e:
                print(f"⚠️ Пропуск файла из-за ошибки кодировки {file_path}: {str(e)}")
                skipped_files += 1
            except Exception as e:
                print(f"❌ Ошибка загрузки {file_path}: {str(e)}")
                skipped_files += 1
    
    print(f"✅ Загрузка завершена: {indexed_files} файлов проиндексировано, {skipped_files} пропущено")
    if logger:
        logger.info(f"Индексация завершена: {indexed_files} файлов успешно, {skipped_files} с ошибками")
    
    return docs

def chunk_documents(docs: List) -> List:
    """
    Разбивает документы на чанки с улучшенными настройками и метаданными.
    """
    print(f"🔄 Разбиение {len(docs)} документов на чанки...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
        add_start_index=True,
        strip_whitespace=True
    )
    
    chunks = text_splitter.split_documents(docs)
    
    # Обогащаем чанки метаданными
    for i, chunk in enumerate(chunks):
        chunk.metadata.update({
            'chunk_id': i,
            'chunk_size': len(chunk.page_content),
            'chunk_created_at': datetime.now().isoformat()
        })
    
    print(f"✅ Создано {len(chunks)} чанков")
    if logger:
        logger.info(f"Документы разбиты на {len(chunks)} чанков")
    
    return chunks

def create_vectorstore(chunks: List, persist_directory: str = None) -> Optional[Chroma]:
    """
    Создает и сохраняет векторное хранилище с обработкой ошибок.
    """
    if persist_directory is None:
        persist_directory = config.PERSIST_DIRECTORY
    
    try:
        print("🔧 Создание векторного хранилища...")
        
        # Создаем директорию если нет
        os.makedirs(persist_directory, exist_ok=True)
        
        # Используем локальные эмбеддинги
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_folder="./embeddings_cache"
        )
        
        # Создаем векторное хранилище
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        
        print(f"✅ Векторное хранилище создано в {persist_directory}")
        print(f"📊 Статистика: {len(chunks)} чанков проиндексировано")
        
        if logger:
            logger.info(f"Векторное хранилище создано: {len(chunks)} документов в {persist_directory}")
        
        return vectorstore
        
    except Exception as e:
        error_msg = f"Ошибка при создании векторного хранилища: {str(e)}"
        print(f"❌ {error_msg}")
        if logger:
            logger.error(error_msg)
        return None

def initialize_or_load_vectorstore(project_path: str) -> bool:
    """
    Инициализирует или загружает существующее векторное хранилище.
    """
    global embeddings, vectorstore, retriever
    
    try:
        # Проверяем существование хранилища
        if os.path.exists(config.PERSIST_DIRECTORY):
            print("📁 Найдено существующее векторное хранилище, загрузка...")
            try:
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    cache_folder="./embeddings_cache"
                )
                vectorstore = Chroma(
                    persist_directory=config.PERSIST_DIRECTORY,
                    embedding_function=embeddings
                )
                retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
                print("✅ Векторное хранилище загружено")
                return True
            except Exception as e:
                print(f"⚠️ Ошибка загрузки хранилища, создаю новое: {str(e)}")
        
        # Создаем новое хранилище
        print("🆕 Создание нового векторного хранилища...")
        documents = load_codebase(project_path)
        if not documents:
            print("⚠️ Не найдено документов для индексации")
            return False
        
        chunks = chunk_documents(documents)
        vectorstore = create_vectorstore(chunks)
        
        if vectorstore:
            retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            return True
        
        return False
        
    except Exception as e:
        error_msg = f"Ошибка при инициализации векторного хранилища: {str(e)}"
        print(f"❌ {error_msg}")
        if logger:
            logger.error(error_msg)
        return False


# ===== 2. МОДЕЛЬ =====
model = init_chat_model(
    model="Qwen/Qwen3-Coder-Next",
    base_url=MODEL_URL,
    model_provider='openai',
    temperature=0.5,
    timeout=120,
    max_tokens=2500,  # Увеличил для больших ответов
    max_retries=3,
    streaming=True
)

# # ===== 3. ОБРЕЗКА ИСТОРИИ (ТОЛЬКО ДИАЛОГА, НЕ ФАЙЛОВ!) =====
# def trim_history(state: Dict[str, Any]) -> Dict[str, Any]:
#     """Обрезает только историю диалога, но не трогает содержимое файлов"""
#     messages = state.get("messages", [])
    
#     # Оставляем ВСЕ human-сообщения (вопросы пользователя)
#     human_msgs = [m for m in messages if isinstance(m, HumanMessage)]
    
#     # Из AI-ответов оставляем только последние 5
#     ai_msgs = [m for m in messages if not isinstance(m, HumanMessage)]
#     ai_msgs = ai_msgs[-5:]
    
#     # Объединяем
#     trimmed = sorted(human_msgs + ai_msgs, key=lambda x: x.timestamp if hasattr(x, 'timestamp') else 0)
    
#     # Возвращаем только изменённое поле
#     return {"messages": trimmed}

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.messages.utils import count_tokens_approximately

# Константы для суммаризации
SUMMARIZATION_TRIGGER = 0.7  # При 70% заполнения контекста
SUMMARIZATION_KEEP = 10       # Оставляем последних 10 сообщений
SUMMARIZATION_MODEL = model   # Используем ту же модель (или можно дешевле)

def summarize_old_messages_sync(messages, keep_last: int = 10):
    """Синхронная версия суммаризации"""
    if len(messages) <= keep_last + 1:
        return messages
    
    system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
    other_msgs = [m for m in messages if not isinstance(m, SystemMessage)]
    
    to_summarize = other_msgs[:-keep_last]
    to_keep = other_msgs[-keep_last:]
    
    # Формируем краткую выжимку (без вызова LLM)
    # Это fallback, если LLM недоступна
    summary_parts = []
    for msg in to_summarize[-10:]:  # Берём только последние 10 из старых
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        content = msg.content[:100] if len(msg.content) > 100 else msg.content
        summary_parts.append(f"{role}: {content}")
    
    summary = f"[Сжатая история из {len(to_summarize)} сообщений]: " + " | ".join(summary_parts)
    
    if len(summary) > 1000:
        summary = summary[:1000] + "..."
    
    summary_message = SystemMessage(content=f"КРАТКАЯ ИСТОРИЯ:\n{summary}")
    
    new_messages = system_msgs + [summary_message] + to_keep
    
    print(f"📝 Суммаризировано {len(to_summarize)} сообщений (упрощённо)")
    return new_messages

def trim_history_by_tokens(state: Dict[str, Any]) -> Dict[str, Any]:
    """Обрезает или суммаризирует историю при превышении лимита"""
    messages = state.get("messages", [])
    if not messages:
        return {}
    
    # Подсчитываем текущие токены
    total_tokens = sum(count_tokens_approximately(msg.content) for msg in messages if hasattr(msg, 'content'))
    usage_percent = total_tokens / MAX_CONTEXT_TOKENS * 100
    
    print(f"\n📊 Токенов ДО обработки: {total_tokens:,} / {MAX_CONTEXT_TOKENS:,} ({usage_percent:.1f}%)")
    print(f"   Сообщений: {len(messages)}")
    
    # Если в лимите - ничего не делаем
    if total_tokens <= MAX_CONTEXT_TOKENS * 0.8:  # 80% лимита
        print(f"✅ Контекст в пределах лимита")
        return {}
    
    # Определяем стратегию
    if usage_percent > 95:
        # Критическое переполнение - экстренная обрезка
        print(f"⚠️ КРИТИЧЕСКОЕ ПЕРЕПОЛНЕНИЕ! Применяю экстренную обрезку...")
        return emergency_trim(messages)
    
    if usage_percent > SUMMARIZATION_TRIGGER * 100:
        # Превышен порог - пробуем суммаризацию
        print(f"📝 Превышен порог ({usage_percent:.1f}%), пробую суммаризацию...")
        
        # Запускаем суммаризацию (нужен async, поэтому хитрость)
        import asyncio
        try:
            new_messages = summarize_old_messages_sync(messages)            
            if new_messages:
                new_tokens = sum(count_tokens_approximately(msg.content) for msg in new_messages if hasattr(msg, 'content'))
                print(f"📊 Токенов ПОСЛЕ суммаризации: {new_tokens:,} / {MAX_CONTEXT_TOKENS:,} ({new_tokens/MAX_CONTEXT_TOKENS*100:.1f}%)")
                print(f"🧹 Сообщений: {len(messages)} → {len(new_messages)}")
                return {"messages": new_messages}
        except Exception as e:
            print(f"⚠️ Ошибка при суммаризации: {e}, применяю обычную обрезку")
    
    # Если суммаризация не помогла или не сработала - обрезаем
    return aggressive_trim(messages)

def emergency_trim(messages):
    """Экстренная обрезка при критическом переполнении"""
    # Оставляем только последние 5 сообщений + системное
    system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
    other_msgs = [m for m in messages if not isinstance(m, SystemMessage)]
    
    # Берём последние 5 сообщений
    kept = other_msgs[-5:] if len(other_msgs) > 5 else other_msgs
    
    new_messages = system_msgs + kept
    
    print(f"🚨 ЭКСТРЕННАЯ ОБРЕЗКА: {len(messages)} → {len(new_messages)} сообщений")
    
    return {"messages": new_messages}

def aggressive_trim(messages):
    """Агрессивная обрезка через trim_messages"""
    trimmed = trim_messages(
        messages,
        max_tokens=MAX_CONTEXT_TOKENS - 10000,  # Оставляем запас
        strategy="last",
        token_counter=count_tokens_approximately,
        include_system=True,
        allow_partial=True,  # Важно для больших сообщений
        start_on="human",
    )
    
    new_tokens = sum(count_tokens_approximately(msg.content) for msg in trimmed if hasattr(msg, 'content'))
    print(f"📊 Токенов ПОСЛЕ обрезки: {new_tokens:,} / {MAX_CONTEXT_TOKENS:,} ({new_tokens/MAX_CONTEXT_TOKENS*100:.1f}%)")
    print(f"🧹 Сообщений: {len(messages)} → {len(trimmed)}")
    
    return {"messages": trimmed}


# ===== 4. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ =====
def print_separator(char: str = "=", length: int = 60):
    print(char * length)

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

@retry(
    retry=retry_if_exception_type((ConnectionError, httpx.RemoteProtocolError, asyncio.TimeoutError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    reraise=True
)
async def stable_stream_response(agent, user_input: str, config: Dict):
    """Стабильный стриминг с повторными попытками"""
    full_response = ""
    
    try:
        async for event in agent.astream_events(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
            version="v2"
        ):
            if event["event"] == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if hasattr(chunk, 'content') and chunk.content:
                    if full_response == "":
                        print("💭 ", end="", flush=True)
                    print(chunk.content, end="", flush=True)
                    full_response += chunk.content
            # Игнорируем другие события
        
        if full_response:
            print("\n")
        return full_response if full_response else "Ответ получен"
        
    except Exception as e:
        error_msg = str(e)
        if "peer closed connection" in error_msg or "RemoteProtocolError" in error_msg:
            print(f"\n⚠️ Соединение разорвано. Повторная попытка...")
            raise asyncio.TimeoutError("Connection closed")  # Вызываем retry
        else:
            print(f"\n❌ Ошибка: {error_msg}")
            return None

# ===== МОДЕРНИЗИРОВАННЫЙ ОСНОВНОЙ ЦИКЛ АГЕНТА =====

class EnhancedAgent:
    """Улучшенный класс агента с метриками и логированием"""
    
    def __init__(self):
        self.agent = None
        self.checkpointer = None
        self.session_config = {
            "configurable": {"thread_id": "memory-session"},
            "recursion_limit": config.DEFAULT_RECURSION_LIMIT
        }
        self.start_time = datetime.now()
    
    async def initialize(self) -> bool:
        """Инициализирует агент с обработкой ошибок"""
        try:
            checkpoint_path = os.path.join(config.CHECKPOINT_DIRECTORY, "agent_memory.sqlite")
            
            async with aiosqlite.connect(checkpoint_path) as db:
                self.checkpointer = AsyncSqliteSaver(db)
                await self.checkpointer.setup()
                
                # Создаем улучшенный промпт
                system_prompt = self._create_enhanced_prompt()
                
                self.agent = create_react_agent(
                    model=model,
                    tools=tools_list,
                    checkpointer=self.checkpointer,
                    pre_model_hook=trim_history_by_tokens,
                    prompt=system_prompt
                )
                
                if logger:
                    logger.info("Агент успешно инициализирован")
                return True
                
        except Exception as e:
            error_msg = f"Ошибка инициализации агента: {str(e)}"
            print(f"❌ {error_msg}")
            if logger:
                logger.error(error_msg)
            return False
    
    def _create_enhanced_prompt(self) -> str:
        """Создает улучшенный системный промпт"""
        return f"""
Ты — профессиональный ассистент по переписыванию VB6-приложения на современный стек NestJS + React.

**Контекст сессии:**
- Сессия начата: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}
- Версия агента: 2.0 (модернизированная)
- Максимальный контекст: {config.MAX_CONTEXT_TOKENS:,} токенов

**ОСНОВНЫЕ ПРАВИЛА:**
1. **Безопасность файла:** Всегда проверяй существование файла перед записью через file_exists
2. **Безопасность пути:** Используй только относительные пути внутри workdir
3. **Кодировка:** Все файлы записывай в UTF-8, для чтения пробуй utf-8, затем cp1251
4. **Структура проекта:**
   - `src/` - только для чтения (оригинальный VB6 код)
   - `api/` - новый NestJS backend
   - `client/` - новый React frontend
   - `runtime/` - файлы прогресса и документации

**Органзация работы:**
- Работай по модулям: сначала api, потом client
- Веди учет прогресса в {config.PROGRESS_FILE}
- Описывай модули в {config.MODULES_FILE}
- Документируй базу данных в {config.DATABASE_FILE}
- Создай общее описание в {config.DESCRIPTION_FILE}

**Технический стек:**
- Backend: NestJS + TypeScript + Oracle (oracledb, connection pooling)
- Frontend: React + Redux + Bootstrap + dhtmlx
- Без TypeORM - прямые SQL запросы
- Структура таблиц Oracle остается неизменной

**Твои возможности:**
- 📖 Чтение файлов любых размеров без ограничений
- 📝 Создание и редактирование файлов с безопасностью
- 🔍 Поиск по кодовой базе с векторными индексами
- 🛠️ Выполнение npm команд
- 📊 Анализ и рефакторинг кода

**ВАЖНО:**
- Отвечай на русском языке
- Будь проактивным и предлагай улучшения
- Помни контекст всей сессии
- При ошибках предлагай альтернативные решения
- Следи за чистотой и структурой кода

Будь полезным, точным и современным!
"""
    
    async def process_request(self, user_input: str) -> Optional[str]:
        """Обрабатывает запрос с метриками и логированием"""
        request_start = datetime.now()
        tools_used = []
        
        try:
            # Логируем запрос
            if logger:
                logger.info(f"Новый запрос: {user_input[:100]}...")
            
            # Выполняем запрос
            response = await asyncio.wait_for(
                stable_stream_response(self.agent, user_input, self.session_config),
                timeout=config.TIMEOUT_SECONDS
            )
            
            # Собираем метрики
            request_time = (datetime.now() - request_start).total_seconds()
            tokens_estimate = len(user_input.split()) + len(response.split()) if response else len(user_input.split())
            
            metrics.record_request(tokens_estimate, request_time, True, tools_used)
            
            if logger:
                logger.info(f"Запрос обработан за {request_time:.2f}с")
            
            return response
            
        except asyncio.TimeoutError:
            error_msg = f"Превышено время ожидания ({config.TIMEOUT_SECONDS} сек)"
            print(f"\n⏰ {error_msg}")
            if logger:
                logger.warning(error_msg)
            
            # Записываем метрики ошибки
            request_time = (datetime.now() - request_start).total_seconds()
            metrics.record_request(0, request_time, False, tools_used)
            
            return None
            
        except KeyboardInterrupt:
            print("\n\n👋 Запрос прерван пользователем")
            if logger:
                logger.info("Запрос прерван пользователем")
            return None
            
        except Exception as e:
            error_msg = f"Ошибка при обработке запроса: {str(e)}"
            print(f"\n❌ {error_msg}")
            if logger:
                logger.error(error_msg)
            
            # Записываем метрики ошибки
            request_time = (datetime.now() - request_start).total_seconds()
            metrics.record_request(0, request_time, False, tools_used)
            
            return None
    
    def print_stats(self):
        """Выводит статистику сессии"""
        session_duration = datetime.now() - self.start_time
        
        print_separator("📊", 70)
        print("📈 СТАТИСТИКА СЕССИИ")
        print(f"⏱️ Длительность: {session_duration}")
        print(f"📨 Запросов всего: {metrics.metrics['requests_count']}")
        print(f"✅ Успешных: {metrics.metrics['successful_requests']}")
        print(f"❌ Неудачных: {metrics.metrics['failed_requests']}")
        print(f"🔤 Токенов использовано: {metrics.metrics['total_tokens_used']:,}")
        print(f"⚡ Среднее время ответа: {metrics.metrics['average_response_time']:.2f}с")
        
        if metrics.metrics['tool_usage']:
            print("\n🛠️ Использование инструментов:")
            for tool, count in sorted(metrics.metrics['tool_usage'].items(), key=lambda x: x[1], reverse=True):
                print(f"   {tool}: {count} раз")
        
        print_separator("📊", 70)

async def run_agent():
    """Основная функция с улучшенной архитектурой"""
    print_separator("🚀", 70)
    print("🤖 МОДЕРНИЗИРОВАННЫЙ АГЕНТ ПЕРЕПИСЫВАНИЯ VB6 → NestJS + React")
    print(f"📁 Версия: 2.0 с улучшенной архитектурой")
    print(f"💾 Контекст: {config.MAX_CONTEXT_TOKENS:,} токенов")
    print(f"📊 Метрики: {'включены' if config.ENABLE_METRICS else 'выключены'}")
    print(f"📝 Логирование: {'включено' if config.ENABLE_LOGGING else 'выключено'}")
    print_separator("🚀", 70)
    
    # Инициализируем агент
    agent = EnhancedAgent()
    
    if not await agent.initialize():
        print("❌ Не удалось инициализировать агента")
        return 1
    
    print("✅ Агент готов к работе")
    print("⌨️ Команды: 'exit' - выход, 'stats' - статистика")
    print_separator("-", 70)
    
    try:
        while True:
            try:
                user_input = input("\n👤 User> ").strip()
                
                if not user_input:
                    continue
                
                # Обработка служебных команд
                if user_input.lower() == "exit":
                    print("\n👋 Сохраняю метрики и завершаю работу...")
                    metrics.save_metrics()
                    agent.print_stats()
                    break
                
                elif user_input.lower() == "stats":
                    agent.print_stats()
                    continue
                
                elif user_input.lower() == "help":
                    print("\n📚 Доступные команды:")
                    print("   exit  - выход из программы")
                    print("   stats - показать статистику сессии")
                    print("   help  - показать эту справку")
                    continue
                
                # Обработка обычного запроса
                response = await agent.process_request(user_input)
                
                if not response:
                    print("⚠️ Не удалось получить ответ. Попробуйте переформулировать запрос.")
                
            except KeyboardInterrupt:
                print("\n\n👋 Работа прервана. Сохраняю метрики...")
                metrics.save_metrics()
                agent.print_stats()
                break
                
    except Exception as e:
        error_msg = f"Критическая ошибка в главном цикле: {str(e)}"
        print(f"\n💥 {error_msg}")
        if logger:
            logger.critical(error_msg)
        traceback.print_exc()
        
        # Сохраняем метрики даже при ошибке
        try:
            metrics.save_metrics()
        except:
            pass
        
        return 1
    
    return 0

if __name__ == "__main__":
    # Проверяем переменные окружения
    if not MODEL_URL:
        print("❌ Ошибка: переменная окружения MODEL_URL не установлена")
        sys.exit(1)
    
    # Инициализируем векторное хранилище
    PROJECT_PATH = "./workdir"
    if not initialize_or_load_vectorstore(PROJECT_PATH):
        print("⚠️ Векторное хранилище не инициализировано, поиск по кодовой базе будет недоступен")
    
    # Устанавливаем политику asyncio для Windows
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Запускаем агент
    try:
        exit_code = asyncio.run(run_agent())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 Программа завершена пользователем")
        sys.exit(0)
    except Exception as e:
        print(f"💥 Критическая ошибка при запуске: {str(e)}")
        sys.exit(1)
