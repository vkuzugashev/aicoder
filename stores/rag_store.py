"""
ЕДИНОЕ RAG ХРАНИЛИЩЕ
ChromaDB для кода + истории диалога + решений
"""
import os
import sys
import time
import asyncio
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import config

class RAGStore:
    """Единое хранилище: код VB6 + история диалога + решения"""
    
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_folder="./embeddings_cache"
        )
        
        # Отдельные коллекции для кода и истории
        self.code_store: Optional[Chroma] = None
        self.memory_store: Optional[Chroma] = None
        
        self._init_stores()
    
    def _init_stores(self):
        """Инициализация ChromaDB коллекций"""
        code_dir = config.CHROMA_DIR / "code"
        memory_dir = config.CHROMA_DIR / "memory"
        
        code_dir.mkdir(parents=True, exist_ok=True)
        memory_dir.mkdir(parents=True, exist_ok=True)
        
        self.code_store = Chroma(
            embedding_function=self.embeddings,
            persist_directory=str(code_dir)
        )
        
        self.memory_store = Chroma(
            embedding_function=self.embeddings,
            persist_directory=str(memory_dir)
        )
        
        # Статистика
        try:
            code_count = self.code_store._collection.count()
            mem_count = self.memory_store._collection.count()
            print(f"📚 RAG: {code_count} документов кода, {mem_count} записей истории")
        except:
            pass
    
    # ===== ДЛЯ КОДА =====
    def index_directory(self, directory: str):
        """Индексация с проверкой на повтор"""
    
        # ✅ Проверяем, не проиндексирована ли уже эта директория
        if self._is_indexed(directory):
            print(f"📚 Директория уже проиндексирована: {directory}")
            return
        
        docs = []
        path = Path(directory)
        
        if not path.exists():
            print(f"❌ Директория не найдена: {directory}")
            return
        
        # Этап 1: Сканирование
        print(f"\n📂 Сканирование: {directory}")
        files_to_index = []
        
        for file_path in path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in {
                '.py', '.js', '.ts', '.jsx', '.tsx', '.vue',
                '.html', '.css', '.json', '.md',
                '.vb', '.frm', '.bas', '.cls'
            }:
                if not any(d in file_path.parts for d in {'.git', 'node_modules', '__pycache__', 'dist', 'build'}):
                    files_to_index.append(file_path)
        
        if not files_to_index:
            print("⚠️ Нет файлов для индексации")
            return
        
        print(f"   Найдено файлов: {len(files_to_index)}")
        
        # Этап 2: Загрузка с прогресс-баром
        print(f"\n📖 Загрузка файлов:")
        errors = 0
        
        for i, file_path in enumerate(files_to_index):
            # Прогресс-бар
            percent = (i + 1) / len(files_to_index) * 100
            bar_len = 30
            filled = int(bar_len * (i + 1) / len(files_to_index))
            bar = '█' * filled + '░' * (bar_len - filled)
            
            print(f"\r   [{bar}] {percent:5.1f}%  {i+1}/{len(files_to_index)}  "
                f"✅{i+1-errors} ❌{errors}", end="", flush=True)
            
            try:
                encoding = 'cp1251' if file_path.suffix.lower() in {'.frm', '.bas', '.cls', '.vb'} else 'utf-8'
                loader = TextLoader(str(file_path), encoding=encoding)
                file_docs = loader.load()
                for doc in file_docs:
                    doc.metadata['source'] = str(file_path.relative_to(path))
                docs.extend(file_docs)
            except Exception:
                errors += 1
        
        print()  # Новая строка после прогресс-бара
        
        if not docs:
            print("⚠️ Нет документов для индексации")
            return
        
        # Этап 3: Разбиение на чанки с прогресс-баром
        print(f"\n🧩 Разбиение на чанки...")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        
        chunks = splitter.split_documents(docs)
        
        # Прогресс-бар для чанков
        bar_filled = 30
        bar = '█' * bar_filled
        print(f"   [{bar}] 100.0%  {len(chunks)} чанков создано")
        
        # Этап 4: Сохранение с прогресс-баром
        print(f"\n💾 Сохранение в хранилище:")
        
        batch_size = 500
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(chunks) + batch_size - 1) // batch_size
            
            percent = batch_num / total_batches * 100
            filled = int(bar_len * batch_num / total_batches)
            bar = '█' * filled + '░' * (bar_len - filled)
            
            print(f"\r   [{bar}] {percent:5.1f}%  пакет {batch_num}/{total_batches}", end="", flush=True)
            
            self.code_store.add_documents(batch)
        
        print()  # Новая строка
        
        # Итог
        print(f"\n{'='*50}")
        print(f"✅ ИНДЕКСАЦИЯ ЗАВЕРШЕНА")
        print(f"{'='*50}")
        print(f"   📁 Файлов:      {len(files_to_index)}")
        print(f"   📄 Документов:   {len(docs)}")
        print(f"   🧩 Чанков:       {len(chunks)}")
        print(f"   ❌ Ошибок:       {errors}")
        print(f"{'='*50}\n")
    
    def _is_indexed(self, directory: str) -> bool:
        """Проверяет, есть ли файлы из директории в RAG"""
        if not self.code_store:
            return False
        
        try:
            # Ищем любой файл из этой директории
            results = self.code_store.similarity_search(
                directory, k=1
            )
            
            if results:
                source = results[0].metadata.get('source', '')
                # Проверяем что файл из той же директории
                if directory in source or any(
                    source.endswith(ext) for ext in ['.frm', '.bas', '.cls']
                ):
                    return True
            
            return False
        except:
            return False

    def _mark_indexed(self, directory: str):
        """Сохраняет отметку об индексации (опционально)"""
        # Можно сохранить в отдельную коллекцию или файл
        pass


    def search_code(self, query: str, k: int = 5) -> str:
        """Поиск по коду"""
        docs = self.code_store.similarity_search(query, k=k)
        if not docs:
            return "Ничего не найдено в коде"
        
        return "\n\n".join([
            f"📁 {d.metadata.get('source', '?')}:\n{d.page_content[:500]}"
            for d in docs
        ])    

    # ===== ДЛЯ ИСТОРИИ ДИАЛОГА =====
    
    def add_message(self, msg: BaseMessage, importance: float = 0.5):
        """Добавляет сообщение в историю"""
        content = msg.content if hasattr(msg, 'content') else str(msg)
        if not content or len(content) < 10:
            return
        
        role = 'user' if isinstance(msg, HumanMessage) else 'assistant'
        
        doc = Document(
            page_content=content[:2000],
            metadata={
                'type': 'message',
                'role': role,
                'importance': importance,
                'timestamp': time.time(),
                'is_decision': importance > 0.7
            }
        )
        
        self.memory_store.add_documents([doc])
    
    def add_decision(self, text: str):
        """Добавляет важное решение"""
        doc = Document(
            page_content=f"✅ РЕШЕНИЕ: {text[:1000]}",
            metadata={
                'type': 'decision',
                'importance': 1.0,
                'timestamp': time.time(),
                'is_decision': True
            }
        )
        self.memory_store.add_documents([doc])
    
    def search_memory(self, query: str, k: int = 10) -> str:
        """Поиск по истории диалога"""
        docs = self.memory_store.similarity_search(query, k=k)
        if not docs:
            return "История пуста"
        
        results = []
        for doc in docs:
            role = doc.metadata.get('role', '?')
            prefix = '👤' if role == 'user' else '🤖'
            results.append(f"{prefix} {doc.page_content[:300]}")
        
        return "\n\n".join(results)
    
    def get_decisions(self, limit: int = 15) -> str:
        """Получает последние важные решения"""
        # Ищем документы с is_decision=True
        docs = self.memory_store.similarity_search(
            "решение создание файла прогресс",  # Общий запрос для решений
            k=limit,
            filter={"is_decision": True}
        )
        
        if not docs:
            # Если нет с фильтром — ищем все
            docs = self.memory_store.similarity_search(
                "важное решение создан файл модуль",
                k=limit
            )
        
        if not docs:
            return "Нет записей о решениях"
        
        # Сортируем по времени
        docs.sort(key=lambda d: d.metadata.get('timestamp', 0), reverse=True)
        
        return "\n".join([
            f"{doc.page_content[:300]}"
            for doc in docs[:limit]
        ])
    
    def get_recent_context(self, query: str, limit: int = 10) -> str:
        """Получает релевантный контекст: код + история"""
        code_docs = self.code_store.similarity_search(query, k=3)
        mem_docs = self.memory_store.similarity_search(query, k=5)
        
        parts = []
        
        if code_docs:
            parts.append("📁 КОД:\n" + "\n".join([
                f"  {d.metadata.get('source', '?')}: {d.page_content[:200]}"
                for d in code_docs
            ]))
        
        if mem_docs:
            parts.append("🧠 ИСТОРИЯ:\n" + "\n".join([
                f"  {d.page_content[:200]}"
                for d in mem_docs
            ]))
        
        return "\n\n".join(parts) if parts else "Ничего не найдено"
    
    def get_stats(self) -> dict:
        try:
            return {
                'code_docs': self.code_store._collection.count(),
                'memory_docs': self.memory_store._collection.count()
            }
        except:
            return {'code_docs': 0, 'memory_docs': 0}

# Глобальное хранилище
rag = RAGStore()