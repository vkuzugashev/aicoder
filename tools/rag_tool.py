"""RAG хранилище с визуальной индексацией"""
import os
import sys
from pathlib import Path
from typing import List, Optional
from datetime import datetime
from collections import defaultdict

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

# Фиксированный импорт
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import config

class RAGStore:
    def __init__(self):
        print(f"📚 Инициализация RAG...")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_folder="./embeddings_cache"
        )
        
        self.vectorstore: Optional[Chroma] = None
        self.stats = {
            'total_files': 0,
            'total_chunks': 0,
            'file_types': defaultdict(int)
        }
        
        self._init_store()
        print(f"✅ RAG готов\n")
    
    def _init_store(self):
        if config.CHROMA_DIR.exists() and list(config.CHROMA_DIR.iterdir()):
            try:
                self.vectorstore = Chroma(
                    embedding_function=self.embeddings,
                    persist_directory=str(config.CHROMA_DIR)
                )
                count = self.vectorstore._collection.count()
                print(f"   Загружено: {count:,} документов")
            except:
                self.vectorstore = Chroma(
                    embedding_function=self.embeddings,
                    persist_directory=str(config.CHROMA_DIR)
                )
        else:
            self.vectorstore = Chroma(
                embedding_function=self.embeddings,
                persist_directory=str(config.CHROMA_DIR)
            )
            print(f"   Создано новое хранилище")
    
    def index_directory(self, directory: str):
        """Индексация с визуализацией"""
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
                '.vb', '.frm', '.bas', '.cls', '.frx'
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
            percent = (i + 1) / len(files_to_index) * 100
            bar_len = 30
            filled = int(bar_len * (i + 1) / len(files_to_index))
            bar = '█' * filled + '░' * (bar_len - filled)
            
            print(f"\r   [{bar}] {percent:5.1f}%  {i+1}/{len(files_to_index)}  ✅{i+1-errors} ❌{errors}", end="", flush=True)
            
            try:
                encoding = 'cp1251' if file_path.suffix.lower() in {'.frm', '.bas', '.cls', '.vb'} else 'utf-8'
                loader = TextLoader(str(file_path), encoding=encoding)
                file_docs = loader.load()
                for doc in file_docs:
                    doc.metadata['source'] = str(file_path.relative_to(path))
                docs.extend(file_docs)
            except Exception:
                errors += 1
        
        print()
        
        if not docs:
            print("⚠️ Нет документов для индексации")
            return
        
        # Этап 3: Разбиение на чанки
        print(f"\n🧩 Разбиение на чанки...")
        
        # ИСПОЛЬЗУЕМ config.CHUNK_SIZE и config.CHUNK_OVERLAP
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        
        chunks = splitter.split_documents(docs)
        print(f"   Создано: {len(chunks)} чанков")
        
        # Этап 4: Сохранение
        print(f"\n💾 Сохранение в хранилище:")
        
        batch_size = 500
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            batch_num = i // batch_size + 1
            
            percent = batch_num / total_batches * 100
            filled = int(30 * batch_num / total_batches)
            bar = '█' * filled + '░' * (30 - filled)
            
            print(f"\r   [{bar}] {percent:5.1f}%  пакет {batch_num}/{total_batches}", end="", flush=True)
            
            self.vectorstore.add_documents(batch)
        
        print()
        
        # Итог
        print(f"\n{'='*50}")
        print(f"✅ ИНДЕКСАЦИЯ ЗАВЕРШЕНА")
        print(f"{'='*50}")
        print(f"   📁 Файлов:      {len(files_to_index)}")
        print(f"   📄 Документов:   {len(docs)}")
        print(f"   🧩 Чанков:       {len(chunks)}")
        print(f"   ❌ Ошибок:       {errors}")
        print(f"{'='*50}\n")
        
        self.stats['total_files'] = len(files_to_index)
        self.stats['total_chunks'] = len(chunks)
    
    def search(self, query: str, k: int = None) -> List[Document]:
        """Поиск"""
        if k is None:
            k = config.RAG_TOP_K
        if not self.vectorstore:
            return []
        
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        return retriever.invoke(query)
    
    def search_formatted(self, query: str) -> str:
        """Форматированный поиск"""
        docs = self.search(query)
        if not docs:
            return "Ничего не найдено"
        
        parts = []
        for doc in docs:
            src = doc.metadata.get('source', 'unknown')
            parts.append(f"📁 {src}:\n{doc.page_content[:500]}")
        
        return "\n\n---\n\n".join(parts)
    
    def has_source(self, directory: str) -> bool:
        """Проверка наличия файлов из директории"""
        if not self.vectorstore:
            return False
        
        try:
            count = self.vectorstore._collection.count()
            if count == 0:
                return False
            
            test_docs = self.search("frm bas cls vb6", k=5)
            for doc in test_docs:
                source = doc.metadata.get('source', '')
                if '.frm' in source or '.bas' in source or '.cls' in source:
                    return True
            return False
        except:
            return False

# Глобальный экземпляр
rag_store = RAGStore()