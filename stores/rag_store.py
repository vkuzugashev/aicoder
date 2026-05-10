"""
Улучшенное RAG хранилище с поддержкой VB6 кода
"""
import os
import re
from typing import List, Optional, Dict, Any
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config

# Глобальные переменные
vectorstore: Optional[Chroma] = None
retriever: Optional[VectorStoreRetriever] = None
file_metadata_cache: Dict[str, Dict] = {}

# ===== СПЕЦИАЛЬНЫЕ ОБРАБОТЧИКИ ДЛЯ VB6 =====

class VB6CodeParser:
    """Парсер VB6 кода для улучшенной индексации"""
    
    @staticmethod
    def extract_functions(content: str, file_path: str) -> List[Document]:
        """Извлекает функции и процедуры из VB6 кода"""
        documents = []
        
        # Паттерны для VB6 функций и процедур
        patterns = [
            # Function ... End Function
            r'(Public\s+|Private\s+|Friend\s+)?(Static\s+)?Function\s+(\w+)\s*\([^)]*\)(.*?)End\s+Function',
            # Sub ... End Sub
            r'(Public\s+|Private\s+|Friend\s+)?(Static\s+)?Sub\s+(\w+)\s*\([^)]*\)(.*?)End\s+Sub',
            # Property Get/Let/Set
            r'(Public\s+|Private\s+|Friend\s+)?Property\s+(Get|Let|Set)\s+(\w+)\s*\([^)]*\)(.*?)End\s+Property'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE | re.DOTALL)
            for match in matches:
                func_name = match.group(3) if match.lastindex >= 3 else "Unknown"
                func_code = match.group(0)
                
                doc = Document(
                    page_content=f"VB6 Функция: {func_name}\n\n{func_code}",
                    metadata={
                        'source': file_path,
                        'type': 'vb6_function',
                        'function_name': func_name,
                        'language': 'vb6'
                    }
                )
                documents.append(doc)
        
        return documents
    
    @staticmethod
    def extract_controls(content: str, file_path: str) -> List[Document]:
        """Извлекает описания контролов из VB6 форм"""
        documents = []
        
        # Паттерн для контролов в .frm файлах
        control_pattern = r'Begin\s+(VB\.\w+)\s+(\w+)\s*\n(.*?)End\s*\n'
        matches = re.finditer(control_pattern, content, re.IGNORECASE | re.DOTALL)
        
        for match in matches:
            control_type = match.group(1)
            control_name = match.group(2)
            control_props = match.group(3)
            
            doc = Document(
                page_content=f"VB6 Контрол: {control_name} (Тип: {control_type})\n\nСвойства:\n{control_props}",
                metadata={
                    'source': file_path,
                    'type': 'vb6_control',
                    'control_name': control_name,
                    'control_type': control_type,
                    'language': 'vb6'
                }
            )
            documents.append(doc)
        
        return documents
    
    @staticmethod
    def extract_sql_queries(content: str, file_path: str) -> List[Document]:
        """Извлекает SQL запросы из VB6 кода"""
        documents = []
        
        # Паттерны для SQL запросов
        sql_patterns = [
            r'(?:SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP)\s+.*?(?:;|"|\n\n)',
            r'strSQL\s*=\s*"(.*?)"',
            r'\.Execute\s*\(\s*"(.*?)"',
        ]
        
        for pattern in sql_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE | re.DOTALL)
            for match in matches:
                sql = match.group(1) if match.lastindex else match.group(0)
                
                doc = Document(
                    page_content=f"SQL запрос из VB6:\n{sql}",
                    metadata={
                        'source': file_path,
                        'type': 'sql_query',
                        'language': 'sql'
                    }
                )
                documents.append(doc)
        
        return documents

def detect_vb6_encoding(file_path: str) -> str:
    """Определяет кодировку VB6 файла"""
    # VB6 файлы обычно в cp1251 (кириллица)
    if any(file_path.lower().endswith(ext) for ext in ['.frm', '.bas', '.cls', '.vbp', '.vbw']):
        return 'cp1251'
    return 'utf-8'

def load_codebase(directory: str) -> List[Document]:
    """
    Загружает код из указанной директории с улучшенной обработкой VB6
    """
    docs = []
    vb6_parser = VB6CodeParser()
    
    print(f"📂 Сканирование директории: {directory}")
    
    for root, dirs, files in os.walk(directory):
        # Игнорируем исключенные папки
        dirs[:] = [d for d in dirs if d not in config.EXCLUDED_DIRS]
        
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()
            
            # Проверяем расширение
            if file_ext not in config.SUPPORTED_EXTENSIONS:
                continue
            
            try:
                # Определяем кодировку
                encoding = detect_vb6_encoding(file_path)
                
                # Загружаем файл
                loader = TextLoader(file_path, encoding=encoding)
                file_docs = loader.load()
                
                if not file_docs:
                    continue
                
                content = file_docs[0].page_content
                rel_path = os.path.relpath(file_path, directory)
                
                # Сохраняем метаданные файла
                file_metadata_cache[rel_path] = {
                    'size': os.path.getsize(file_path),
                    'encoding': encoding,
                    'extension': file_ext,
                    'lines': content.count('\n')
                }
                
                # Для VB6 файлов извлекаем функции и контролы
                if file_ext in ['.frm', '.bas', '.cls']:
                    # Извлекаем функции
                    func_docs = vb6_parser.extract_functions(content, rel_path)
                    docs.extend(func_docs)
                    
                    # Извлекаем контролы (для .frm)
                    if file_ext == '.frm':
                        control_docs = vb6_parser.extract_controls(content, rel_path)
                        docs.extend(control_docs)
                    
                    # Извлекаем SQL запросы
                    sql_docs = vb6_parser.extract_sql_queries(content, rel_path)
                    docs.extend(sql_docs)
                    
                    print(f"  ✅ {rel_path}: найдено {len(func_docs)} функций, "
                          f"{len(control_docs) if file_ext == '.frm' else 0} контролов, "
                          f"{len(sql_docs)} SQL запросов")
                else:
                    # Для обычных файлов добавляем как есть
                    for doc in file_docs:
                        doc.metadata.update({
                            'source': rel_path,
                            'type': 'code',
                            'language': 'typescript' if file_ext in ['.ts', '.tsx'] else 'javascript'
                        })
                    docs.extend(file_docs)
                    print(f"  ✅ {rel_path}: загружен полностью")
                
            except UnicodeDecodeError as e:
                print(f"  ⚠️ Ошибка кодировки {file_path}: {e}")
                # Пробуем альтернативные кодировки
                for alt_encoding in ['cp1251', 'latin1', 'utf-8-sig']:
                    try:
                        loader = TextLoader(file_path, encoding=alt_encoding)
                        file_docs = loader.load()
                        if file_docs:
                            for doc in file_docs:
                                doc.metadata['source'] = os.path.relpath(file_path, directory)
                            docs.extend(file_docs)
                            print(f"  ✅ {os.path.relpath(file_path, directory)}: загружен с кодировкой {alt_encoding}")
                            break
                    except:
                        continue
                    
            except Exception as e:
                print(f"  ❌ Ошибка загрузки {file_path}: {e}")
    
    print(f"\n📊 Всего загружено: {len(docs)} документов из {len(file_metadata_cache)} файлов")
    return docs

def chunk_documents(docs: List[Document]) -> List[Document]:
    """
    Разбивает документы на чанки с оптимизацией для кода
    """
    # Разные размеры чанков для разных типов
    code_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CODE_CHUNK_SIZE,
        chunk_overlap=config.CODE_CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = []
    for doc in docs:
        doc_type = doc.metadata.get('type', 'code')
        
        # Для VB6 функций - не разбиваем
        if doc_type == 'vb6_function':
            chunks.append(doc)
        else:
            # Для остального - разбиваем
            doc_chunks = code_splitter.split_documents([doc])
            
            # Обогащаем метаданными
            for i, chunk in enumerate(doc_chunks):
                chunk.metadata.update({
                    'chunk_id': i,
                    'chunk_size': len(chunk.page_content),
                    'total_chunks': len(doc_chunks)
                })
            
            chunks.extend(doc_chunks)
    
    return chunks

def create_vectorstore(persist_directory: str = None) -> Chroma:
    """
    Создает новое или загружает существующее векторное хранилище
    """
    global vectorstore
    
    if persist_directory is None:
        persist_directory = str(config.CHROMA_DIR)
    
    # Инициализируем эмбеддинги
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder="./embeddings_cache",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Проверяем существование хранилища
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        print(f"📚 Загружено существующее векторное хранилище из {persist_directory}")
        vectorstore = Chroma(
            embedding_function=embeddings,
            persist_directory=persist_directory
        )
        
        try:
            count = vectorstore._collection.count()
            print(f"   Документов в хранилище: {count}")
        except:
            pass
    else:
        print(f"✅ Создано новое векторное хранилище в {persist_directory}")
        vectorstore = Chroma.from_documents(
            documents=[],
            embedding=embeddings,
            persist_directory=persist_directory
        )
    
    return vectorstore

def load_dir_to_vectorstore(directory: str):
    """
    Индексирует папку с кодом в векторное хранилище
    """
    if not vectorstore:
        raise ValueError('Векторное хранилище не инициализировано!')
    
    print(f"🔍 Индексация папки: {directory}")
    
    # Загружаем документы
    docs = load_codebase(directory)
    print(f'📄 Прочитано файлов: {len(file_metadata_cache)}')
    print(f'📝 Создано документов: {len(docs)}')
    
    # Разбиваем на чанки
    chunks = chunk_documents(docs)
    print(f'🧩 Создано чанков: {len(chunks)}')
    
    if chunks:
        print(f'💾 Добавляем в хранилище {len(chunks)} чанков...')
        
        # Добавляем чанки батчами
        batch_size = 500
        added_count = 0
        
        with tqdm(total=len(chunks), desc="Индексация", unit="чанк",
                 ncols=80, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i+batch_size]
                try:
                    vectorstore.add_documents(batch)
                    added_count += len(batch)
                    pbar.update(len(batch))
                    
                    # Обновляем прогресс
                    if i > 0 and i % (batch_size * 5) == 0:
                        progress = (added_count / len(chunks)) * 100
                        pbar.set_postfix({
                            "прогресс": f"{progress:.1f}%",
                            "файлов": f"{len(file_metadata_cache)}"
                        })
                        
                except Exception as e:
                    print(f'\n❌ Ошибка на пакете {i//batch_size + 1}: {e}')
        
        print(f'✅ Успешно добавлено {added_count} чанков')
    else:
        print('⚠️ Нет чанков для индексации')

def load_file_to_vectorstore(file_path: str, encoding: str = None):
    """
    Индексирует отдельный файл
    
    Args:
        file_path: Путь к файлу
        encoding: Кодировка (если None - определяется автоматически)
    """
    if not vectorstore:
        raise ValueError('Векторное хранилище не инициализировано!')
    
    print(f"📄 Индексация файла: {file_path}")
    
    if encoding is None:
        encoding = detect_vb6_encoding(file_path)
    
    docs = load_file_codebase(file_path, encoding)
    if docs:
        chunks = chunk_documents(docs)
        if chunks:
            vectorstore.add_documents(chunks)
            print(f"✅ Добавлено {len(chunks)} чанков")
        else:
            print("⚠️ Нет чанков для добавления")
    else:
        print("⚠️ Файл не загружен")

def load_file_codebase(file_path: str, encoding: str = 'utf-8') -> List[Document]:
    """
    Загружает код из указанного файла с улучшенной обработкой
    
    Args:
        file_path: Путь к файлу
        encoding: Кодировка
    
    Returns:
        Список документов
    """
    try:
        loader = TextLoader(file_path, encoding=encoding)
        docs = loader.load()
        
        # Добавляем метаданные
        for doc in docs:
            doc.metadata.update({
                'source': file_path,
                'encoding': encoding
            })
        
        return docs
        
    except Exception as e:
        print(f"❌ Ошибка загрузки {file_path} (encoding={encoding}): {e}")
        return []

def get_retriever(k: int = None) -> VectorStoreRetriever:
    """
    Получает retriever с настройками
    
    Args:
        k: Количество возвращаемых документов
    
    Returns:
        Настроенный retriever
    """
    global retriever, vectorstore
    
    if not vectorstore:
        raise ValueError('Векторное хранилище не инициализировано!')
    
    if k is None:
        k = config.RAG_TOP_K
    
    if not retriever or retriever.search_kwargs.get('k') != k:
        retriever = vectorstore.as_retriever(
            search_type="mmr",  # Maximum Marginal Relevance
            search_kwargs={
                "k": k,
                "fetch_k": k * 2,
                "lambda_mult": 0.7  # Баланс relevance vs diversity
            }
        )
    
    return retriever

def search_codebase_smart(query: str) -> str:
    """
    Умный поиск по кодовой базе с контекстом
    
    Args:
        query: Поисковый запрос
    
    Returns:
        Отформатированный результат поиска
    """
    ret = get_retriever()
    
    # Выполняем поиск
    docs = ret.invoke(query)
    
    if not docs:
        return "🔍 Ничего не найдено по вашему запросу."
    
    # Форматируем результат
    result_parts = []
    
    # Группируем по типам
    functions = []
    controls = []
    sql_queries = []
    code_snippets = []
    
    for doc in docs:
        doc_type = doc.metadata.get('type', 'code')
        source = doc.metadata.get('source', 'Unknown')
        
        if doc_type == 'vb6_function':
            functions.append(f"📁 {source}\n{doc.page_content}")
        elif doc_type == 'vb6_control':
            controls.append(f"📁 {source}\n{doc.page_content}")
        elif doc_type == 'sql_query':
            sql_queries.append(f"📁 {source}\n{doc.page_content}")
        else:
            code_snippets.append(f"📁 {source}\n{doc.page_content[:500]}...")
    
    # Собираем результат
    if functions:
        result_parts.append(f"🔧 ФУНКЦИИ ({len(functions)}):\n" + "\n\n".join(functions[:5]))
    if controls:
        result_parts.append(f"🎛️ КОНТРОЛЫ ({len(controls)}):\n" + "\n\n".join(controls[:3]))
    if sql_queries:
        result_parts.append(f"💾 SQL ЗАПРОСЫ ({len(sql_queries)}):\n" + "\n\n".join(sql_queries[:5]))
    if code_snippets:
        result_parts.append(f"📝 КОД ({len(code_snippets)}):\n" + "\n\n".join(code_snippets[:7]))
    
    result = f"🔍 Найдено {len(docs)} релевантных фрагментов:\n\n" + "\n\n---\n\n".join(result_parts)
    
    return result

def get_file_info(relative_path: str) -> Optional[Dict]:
    """
    Получает информацию о файле
    
    Args:
        relative_path: Относительный путь к файлу
    
    Returns:
        Информация о файле или None
    """
    return file_metadata_cache.get(relative_path)

def get_statistics() -> Dict[str, Any]:
    """
    Получает статистику хранилища
    
    Returns:
        Словарь со статистикой
    """
    stats = {
        'files_indexed': len(file_metadata_cache),
        'files_by_type': {},
        'total_documents': 0
    }
    
    # Считаем типы файлов
    for file_path, info in file_metadata_cache.items():
        ext = info['extension']
        stats['files_by_type'][ext] = stats['files_by_type'].get(ext, 0) + 1
    
    # Количество документов в хранилище
    if vectorstore:
        try:
            stats['total_documents'] = vectorstore._collection.count()
        except:
            pass
    
    return stats