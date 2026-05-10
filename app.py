#!/usr/bin/env python3
"""
Главный запускаемый файл продвинутого агента
"""
import sys
import os
import asyncio
from pathlib import Path

# Добавляем корень проекта в путь
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from stores.rag_store import create_vectorstore, load_dir_to_vectorstore, get_retriever

async def initialize_system():
    """Инициализация всех компонентов системы"""
    print("=" * 70)
    print("🔧 ИНИЦИАЛИЗАЦИЯ СИСТЕМЫ")
    print("=" * 70)
    
    # 1. Создаем необходимые директории
    for dir_path in [config.WORK_DIR, config.CHECKPOINT_DIR, 
                    config.CHROMA_DIR, config.LOGS_DIR, 
                    config.CACHE_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    print("✅ Директории созданы")
    
    # 2. Инициализируем RAG хранилище
    print("📚 Инициализация RAG хранилища...")
    try:
        # Создаем или загружаем векторное хранилище
        vectorstore = create_vectorstore(str(config.CHROMA_DIR))
        
        # Проверяем, нужно ли индексировать
        from langchain_chroma import Chroma
        collection_count = vectorstore._collection.count() if hasattr(vectorstore, '_collection') else 0
        
        if collection_count == 0:
            print(f"📂 Индексация проекта: {config.WORK_DIR / 'src'}")
            if (config.WORK_DIR / 'src').exists():
                load_dir_to_vectorstore(str(config.WORK_DIR / 'src'))
            else:
                print(f"⚠️ Директория src не найдена в {config.WORK_DIR}")
        else:
            print(f"✅ Векторное хранилище содержит {collection_count} документов")
        
        # Проверяем retriever
        retriever = get_retriever()
        print("✅ RAG система готова")
        
    except Exception as e:
        print(f"❌ Ошибка инициализации RAG: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 3. Проверяем наличие необходимых файлов
    required_files = [
        "prompts/instruction.txt",
        "prompts/summary_prompt.txt"
    ]
    
    for file_path in required_files:
        if not Path(file_path).exists():
            print(f"❌ Отсутствует файл: {file_path}")
            return False
    
    print("✅ Все компоненты инициализированы")
    print("=" * 70)
    print()
    
    return True

async def main():
    """Главная функция"""
    print("🚀 ЗАПУСК ПРОДВИНУТОГО АГЕНТА КОДЕРА v3.0")
    print(f"📁 Рабочая директория: {config.WORK_DIR}")
    print(f"💾 Лимит контекста: {config.MAX_CONTEXT_TOKENS:,} токенов")
    print()
    
    # Инициализируем систему
    if not await initialize_system():
        print("❌ Ошибка инициализации системы")
        return 1
    
    # Запускаем агента
    from agents.agent import run_advanced_agent
    return await run_advanced_agent()

if __name__ == "__main__":
    # Настройка для Windows
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except Exception as e:
        print(f"💥 Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)