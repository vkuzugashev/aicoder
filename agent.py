import os
import asyncio
import sys
import traceback
from typing import Any, Dict

import aiosqlite
import httpx
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage
from langchain_core.messages.utils import trim_messages, count_tokens_approximately

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import tools

os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

# ===== КОНСТАНТЫ =====
MAX_CONTEXT_TOKENS = 200000
DEFAULT_RECURSION_LIMIT = 20
MAX_CONTEXT_MESSAGES = 10
PERSIST_DIRECTORY = './chroma_db'


# ===== 1. ИНСТРУМЕНТЫ =====
@tool("list_directory", description="Получить полный список файлов и папок в указанной директории")
def list_directory_tool(path: str):
    """Возвращает полный список файлов без обрезаний"""
    try:
        result = tools.list_dir(path)
        print(f"📁 Прочитана папка: {path}")
        return result
    except Exception as e:
        return f"Ошибка при чтении директории {path}: {str(e)}"


@tool("read_file", description="Прочитать полное содержимое файла. encoding: utf-8, cp1251")
def read_file_tool(path: str, encoding: str = "utf-8"):
    """Читает файл полностью - без каких-либо обрезаний"""
    try:
        result = tools.read_file(path, encoding)
        print(f"📖 Прочитан файл: {path} (полный размер: {len(str(result))} символов)")
        return result
    except Exception as e:
        return f"Ошибка при чтении файла {path}: {str(e)}"


@tool("write_file", description="Записать или создать файл. overwrite=True для перезаписи существующего")
def write_file_tool(path: str, content: str, overwrite: bool = False):
    try:
        exists = tools.file_exists(path)
        if exists and not overwrite:
            return f"Файл {path} существует. Используй overwrite=True чтобы перезаписать"

        result = tools.write_file(path, content, overwrite=overwrite)
        if result:
            print(f"✍️ Записан файл: {path} ({len(content)} символов)")
            return f"✅ Файл {path} успешно записан"
        return f"❌ Ошибка при записи {path}"
    except Exception as e:
        return f"Ошибка: {str(e)}"


@tool("create_directory", description="Создать новую папку")
def create_directory_tool(path: str):
    try:
        result = tools.create_dir(path)
        if result:
            print(f"📁 Создана папка: {path}")
            return f"✅ Папка {path} создана"
        return f"❌ Не удалось создать {path}"
    except Exception as e:
        return f"Ошибка: {str(e)}"


@tool("file_exists", description="Проверить, существует ли файл")
def file_exists_tool(path: str):
    try:
        exists = tools.file_exists(path)
        return f"Файл {path} {'существует' if exists else 'не существует'}"
    except Exception as e:
        return f"Ошибка: {str(e)}"


@tool("dir_exists", description="Проверить, существует ли папка")
def dir_exists_tool(path: str):
    try:
        exists = tools.dir_exists(path)
        return f"Папка {path} {'существует' if exists else 'не существует'}"
    except Exception as e:
        return f"Ошибка: {str(e)}"


@tool("delete_dir", description="Удалить директорию (осторожно!)")
def delete_dir_tool(path: str):
    try:
        result = tools.delete_dir(path)
        if result:
            print(f"🗑️ Удалёна директория: {path}")
            return f"✅ Директория {path} удалёна"
        return f"❌ Не удалось удалить директорию {path}"
    except Exception as e:
        return f"Ошибка: {str(e)}"


@tool("delete_file", description="Удалить файл (осторожно!)")
def delete_file_tool(path: str):
    try:
        result = tools.delete_file(path)
        if result:
            print(f"🗑️ Удалён файл: {path}")
            return f"✅ Файл {path} удалён"
        return f"❌ Не удалось удалить {path}"
    except Exception as e:
        return f"Ошибка: {str(e)}"


@tool("npm_install", description="Выполнить команду установки пакетов в терминале для указанной папки: npm install")
def npm_install_tool(path: str, options: str = ""):
    return tools.npm_install(path, options)


@tool("npm_build", description="Выполнить команду build проекта в терминале для указанной папки: npm run build")
def npm_build_tool(path: str, options: str = ""):
    return tools.npm_build(path, options)


@tool("pwd", description="Получить имя текущей директории. аналог команды pwd")
def pwd_tool():
    return tools.pwd()


# ===== 2. ВЕКТОРНАЯ БАЗА ДАННЫХ =====
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})


@tool
def search_codebase_tool(query: str) -> str:
    """
    ИЩЕТ РЕЛЕВАНТНЫЕ ФРАГМЕНТЫ КОДА по смыслу.
    Используй этот инструмент, когда тебе нужно найти, где в проекте реализована какая-то функция,
    найти все места, использующие определенную библиотеку, или вспомнить логику работы какого-то модуля.
    """
    docs = retriever.invoke(query)
    if not docs:
        return "Ничего не найдено по вашему запросу."

    context = "\n\n---\n\n".join([f"Файл: {doc.metadata['source']}\nСодержание:\n{doc.page_content}" for doc in docs])
    return f"Вот наиболее релевантные фрагменты из кодовой базы:\n\n{context}"


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
    search_codebase_tool,
]


# ===== 3. ФУНКЦИИ ДЛЯ ЗАГРУЗКИ И ИНДЕКСАЦИИ =====
def load_codebase(directory):
    docs = []
    for root, dirs, files in os.walk(directory):
        # Игнорируем папки
        dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'node_modules']]
        
        for file in files:
            if file.endswith(('.py', '.js', '.ts', '.jsx', '.tsx', '.frm', '.frx', '.bas')):
                file_path = os.path.join(root, file)
                
                # Определяем кодировку по пути
                rel_path = os.path.relpath(file_path, directory)
                if rel_path.startswith('workdir/src'):
                    encoding = 'cp1251'
                else:
                    encoding = 'utf-8'
                
                try:
                    loader = TextLoader(file_path, encoding=encoding)
                    docs.extend(loader.load())
                except Exception as e:
                    print(f"Ошибка загрузки {file}: {e}")
    return docs


def chunk_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(docs)


def create_vectorstore(chunks, persist_directory=PERSIST_DIRECTORY):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    print(f"Векторное хранилище создано в {persist_directory}")
    return vectorstore


# ===== 4. МОДЕЛЬ =====
model = init_chat_model(
    model="Qwen/Qwen3-Coder-Next",
    base_url='https://foundation-models.api.cloud.ru/v1',
    model_provider='openai',
    temperature=0.5,
    timeout=120,
    max_tokens=2500,
    max_retries=3,
    streaming=True
)


# ===== 5. СУММАРИЗАЦИЯ =====
SUMMARIZATION_TRIGGER = 0.7
SUMMARIZATION_KEEP = MAX_CONTEXT_MESSAGES - 5
SUMMARIZATION_MODEL = model


def summarize_old_messages_sync(messages, keep_last: int = None):
    """
    Синхронная версия суммаризации с вызовом LLM.
    Возвращает НОВЫЙ список сообщений.
    """
    if keep_last is None:
        keep_last = SUMMARIZATION_KEEP

    if len(messages) <= keep_last + 1:
        return messages

    # Разделяем сообщения
    system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
    other_msgs = [m for m in messages if not isinstance(m, SystemMessage)]

    to_summarize = other_msgs[:-keep_last]
    to_keep = other_msgs[-keep_last:]

    if not to_summarize:
        return messages

    # Формируем промпт для суммаризации
    conversation_parts = []
    for msg in to_summarize:
        if isinstance(msg, HumanMessage):
            role = "User"
        elif isinstance(msg, AIMessage):
            role = "Assistant"
        else:
            role = "System"

        # Ограничиваем длину каждого сообщения для резюме
        # content = msg.content[:500] if len(msg.content) > 500 else msg.content
        content = msg.content
        conversation_parts.append(f"{role}: {content}")

    conversation_text = "\n".join(conversation_parts)

    summary_prompt = f"""
Ты должен сжать следующий диалог в краткое резюме (не более 500 слов).
Сохрани ключевую информацию:
- Какие файлы были созданы/изменены
- Какие команды были выполнены
- Какие решения были приняты
- Какой прогресс достигнут

Диалог для сжатия:
{conversation_text}

Краткое резюме:
"""

    try:
        response = SUMMARIZATION_MODEL.invoke([HumanMessage(content=summary_prompt)])
        summary = response.content

        if len(summary) > 2000:
            summary = summary[:2000] + "..."

        summary_message = SystemMessage(
            content=f"КРАТКАЯ ИСТОРИЯ:\n{summary}"
        )

        new_messages = system_msgs + [summary_message] + to_keep

        old_tokens = sum(count_tokens_approximately(msg.content) for msg in to_summarize if hasattr(msg, 'content'))
        new_tokens = sum(count_tokens_approximately(msg.content) for msg in [summary_message] if hasattr(msg, 'content'))

        print(f"📝 Суммаризировано {len(to_summarize)} сообщений")
        print(f"   Сэкономлено токенов: ~{old_tokens - new_tokens:,} (было {old_tokens:,}, стало {new_tokens:,})")

        return new_messages

    except Exception as e:
        print(f"⚠️ Ошибка суммаризации: {e}, применяю упрощённую обрезку")
        summary = f"[Сжатая история из {len(to_summarize)} сообщений]"
        for msg in to_summarize[-5:]:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            content = msg.content[:100] if len(msg.content) > 100 else msg.content
            summary += f"\n{role}: {content}"

        summary_message = SystemMessage(content=f"КРАТКАЯ ИСТОРИЯ:\n{summary[:1500]}")
        return system_msgs + [summary_message] + to_keep


def emergency_trim(messages):
    """Экстренная обрезка при критическом переполнении"""
    system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
    other_msgs = [m for m in messages if not isinstance(m, SystemMessage)]

    kept = other_msgs[-5:] if len(other_msgs) > 5 else other_msgs
    new_messages = system_msgs + kept

    print(f"🚨 ЭКСТРЕННАЯ ОБРЕЗКА: {len(messages)} → {len(new_messages)} сообщений")
    return {"messages": new_messages}


def aggressive_trim(messages):
    """Агрессивная обрезка через trim_messages"""
    trimmed = trim_messages(
        messages,
        max_tokens=MAX_CONTEXT_TOKENS - 10000,
        strategy="last",
        token_counter=count_tokens_approximately,
        include_system=True,
        allow_partial=True,
        start_on="human",
    )

    new_tokens = sum(count_tokens_approximately(msg.content) for msg in trimmed if hasattr(msg, 'content'))
    print(f"📊 Токенов ПОСЛЕ обрезки: {new_tokens:,} / {MAX_CONTEXT_TOKENS:,} ({new_tokens / MAX_CONTEXT_TOKENS * 100:.1f}%)")
    print(f"🧹 Сообщений: {len(messages)} → {len(trimmed)}")

    return {"messages": trimmed}


def trim_history_by_tokens(state: Dict[str, Any]) -> Dict[str, Any]:
    """Обрезает или суммаризирует историю при превышении лимита"""
    messages = state.get("messages", [])
    if not messages:
        return {}

    total_tokens = sum(count_tokens_approximately(msg.content) for msg in messages if hasattr(msg, 'content'))
    usage_percent = total_tokens / MAX_CONTEXT_TOKENS * 100

    print(f"\n📊 Токенов ДО обработки: {total_tokens:,} / {MAX_CONTEXT_TOKENS:,} ({usage_percent:.1f}%)")
    print(f"   Сообщений: {len(messages)}")

    if usage_percent <= SUMMARIZATION_TRIGGER * 100:
        # print(f"✅ Контекст в пределах лимита")
        return {}

    if usage_percent > SUMMARIZATION_TRIGGER * 100:
        print(f"📝 Превышен порог ({usage_percent:.1f}%), пробую суммаризацию...")
        try:
            new_messages = summarize_old_messages_sync(messages)
            new_tokens = sum(count_tokens_approximately(msg.content) for msg in new_messages if hasattr(msg, 'content'))
            if new_tokens < total_tokens:
                print(f"📊 Токенов ПОСЛЕ суммаризации: {new_tokens:,} / {MAX_CONTEXT_TOKENS:,} ({new_tokens / MAX_CONTEXT_TOKENS * 100:.1f}%)")
                print(f"🧹 Сообщений: {len(messages)} → {len(new_messages)}")
                return {"messages": new_messages}
            else:
                print(f"⚠️ Суммаризация не уменьшила токены!")
        except Exception as e:
            print(f"⚠️ Ошибка при суммаризации: {e}, применяю обычную обрезку")

    if usage_percent > 95:
        print(f"⚠️ КРИТИЧЕСКОЕ ПЕРЕПОЛНЕНИЕ! Применяю экстренную обрезку...")
        return emergency_trim(messages)

    return aggressive_trim(messages)

async def cleanup_history(agent, config: Dict, checkpointer):
    """
    Периодическая очистка истории в checkpoint.
    Вызывается после каждого запроса или раз в N шагов.
    """
    # Получаем текущее состояние
    state = await agent.aget_state(config)
    if not state or not state.values:
        return
    
    messages = state.values.get("messages", [])
    
    # Проверяем необходимость очистки
    total_tokens = sum(count_tokens_approximately(msg.content) for msg in messages if hasattr(msg, 'content'))
    usage_percent = total_tokens / MAX_CONTEXT_TOKENS * 100
    
    if usage_percent <= SUMMARIZATION_TRIGGER * 100:
        return
    
    # Выполняем суммаризацию
    new_messages = summarize_old_messages_sync(messages)
    
    if len(new_messages) < len(messages):
        # Обновляем состояние в checkpoint
        await agent.aupdate_state(
            config,
            {"messages": new_messages},
            as_node="__start__"
        )
        print(f"💾 Checkpoint обновлён: {len(messages)} -> {len(new_messages)} сообщений")

# ===== 6. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ =====
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

        if full_response:
            print("\n")
        return full_response if full_response else "Ответ получен"

    except Exception as e:
        error_msg = str(e)
        if "peer closed connection" in error_msg or "RemoteProtocolError" in error_msg:
            print(f"\n⚠️ Соединение разорвано. Повторная попытка...")
            raise asyncio.TimeoutError("Connection closed")
        else:
            print(f"\n❌ Ошибка: {error_msg}")
            return None


# ===== 7. ОСНОВНАЯ ФУНКЦИЯ =====
async def run_agent():
    print_separator("=", 70)
    print("🚀 АГЕНТ ПЕРЕПИСЫВАНИЯ VB6 → NestJS + React")
    print(f"💾 Помнит последние {MAX_CONTEXT_MESSAGES} сообщений диалога")
    print_separator("=", 70)

    checkpoint_path = "checkpoints/agent_memory.sqlite"
    os.makedirs("checkpoints", exist_ok=True)

    session_config = {
        "configurable": {"thread_id": "memory-session"},
        "recursion_limit": DEFAULT_RECURSION_LIMIT
    }

    try:
        async with aiosqlite.connect(checkpoint_path) as db:
            checkpointer = AsyncSqliteSaver(db)
            await checkpointer.setup()

            agent = create_react_agent(
                model=model,
                tools=tools_list,
                checkpointer=checkpointer,
                pre_model_hook=trim_history_by_tokens,
                prompt="""
Ты — ассистент по переписыванию VB6-приложения на NestJS + React.
Ты ПОМНИШЬ историю диалога и видишь ПОЛНОЕ содержимое файлов.

ОСНОВНЫЕ ПРАВИЛА:
- Используй всегда инструмент кодовой базы search_codebase_tool(query) для поиска
- Перед записью файла проверь его существование через file_exists
- Перед записью файла проверь его существование и если есть прочитай его содержимое, если нужно запиши с override=true
- Рабочая папка workdir в ней все другие, все делается внутри её.
- Папка `src` — только для чтения, НЕ ИЗМЕНЯЙ её
- Новый код пиши в `api/` и `client/`
- Пиши в файл runtime/progress.md что сделано и что планируется
- В progress.md бери один модуль и с ним работай, после окончания сохрани туда статус.
- Выполняй перекодирование по модулям сначала api, потом client и сохраняй статус в progress.md

Твои возможности:
- Искать в кодовой базе search_codebase_tool(query) для поиска
- Читать полное содержимое любых файлов
- Просматривать структуру папок
- Создавать/редактировать файлы
- Выполнять npm команды

ВАЖНО:
- Отвечай на русском
- Отвечай кратко только на поставленный вопрос не фантазируй
- Помни предыдущие просьбы пользователя
- Ищи контекст только в кодовой базе search_codebase_tool(query)
- TypeORM не используется — прямые запросы к Oracle через `oracledb` с использованием pool соединений
- Структура таблиц не меняется — только перепись API
- Client использует стек: bootstrap, dhtmlx, react, redux
- Api использует стек: NestJS, oracledb, typescript 
- Все файлы записываешь в UTF-8

Будь полезным и конкретным!
"""
            )

            print("✅ Агент готов")
            print("⌨️ Введите 'exit' для выхода")
            print_separator("-", 70)

            while True:
                try:
                    user_input = input("\n👤 User> ").strip()

                    if not user_input:
                        continue

                    if user_input.lower() in ["exit", "quit"]:
                        print("👋 До свидания!")
                        break

                    response = await asyncio.wait_for(
                        stable_stream_response(agent, user_input, session_config),
                        timeout=180
                    )

                    if not response:
                        print("⚠️ Не удалось получить ответ")
                    else:
                        # После ответа проверяем и чистим историю
                        await cleanup_history(agent, session_config, checkpointer)

                except asyncio.TimeoutError:
                    print("\n⚠️ Превышено время ожидания (180 сек)")
                except KeyboardInterrupt:
                    print("\n\n👋 Прервано пользователем")
                    break
                except Exception as e:
                    print(f"\n❌ Ошибка: {str(e)}")

    except Exception as e:
        print(f"❌ Критическая ошибка: {str(e)}")
        traceback.print_exc()
        return 1

    return 0


# ===== 8. ТОЧКА ВХОДА =====
if __name__ == "__main__":
    PROJECT_PATH = "./workdir"
    print(f"Индексация проекта: {PROJECT_PATH}")

    if not os.path.exists(PERSIST_DIRECTORY) or not os.listdir(PERSIST_DIRECTORY):
        documents = load_codebase(PROJECT_PATH)
        print(f"Загружено документов: {len(documents)}")
        if documents:
            chunks = chunk_documents(documents)
            print(f"Создано чанков: {len(chunks)}")
            vectorstore = create_vectorstore(chunks)
    else:
        print(f"Векторное хранилище уже существует в {PERSIST_DIRECTORY}")

    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    exit_code = asyncio.run(run_agent())
    sys.exit(exit_code)