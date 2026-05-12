"""
PRODUCTION AI АГЕНТ С СЕМАНТИЧЕСКОЙ ПАМЯТЬЮ
StateGraph + RAG + SemanticMemory + Checkpoints
Вместо обрезки — поиск релевантной истории
"""
import os
import sys
import asyncio
import time
import signal
from typing import Annotated, List, TypedDict, Optional, Dict, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.prebuilt import ToolNode
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
)
from langchain_core.messages.utils import count_tokens_approximately

import aiosqlite
import httpx

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

from config import config
from stores.rag_store import rag_store
from utils.metrics import metrics

from tools.file_tools import (
    list_dir, read_file, write_file, create_dir,
    file_exists, dir_exists, delete_dir, delete_file, pwd
)
from tools.build_tools import npm_install, npm_build
from tools.rag_tool import search_codebase

import logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(f"logs/agent_{datetime.now():%Y%m%d}.log", encoding='utf-8')]
)
for lib in ["httpx", "httpcore", "langchain", "langgraph"]:
    logging.getLogger(lib).setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

# ===== ИНСТРУМЕНТЫ =====
tools = [
    list_dir, read_file, write_file, create_dir,
    file_exists, dir_exists, delete_dir, delete_file,
    npm_install, npm_build, pwd, search_codebase
]

# ===== ЗАГРУЗКА ПРОМПТА =====
try:
    with open("prompts/instruction.txt", "r", encoding="utf-8") as f:
        SYSTEM_PROMPT = f.read()
except:
    SYSTEM_PROMPT = "Ты AI ассистент для работы с кодом."

# ===== МОДЕЛЬ =====
model = init_chat_model(
    model=config.MODEL_NAME,
    base_url=config.MODEL_URL,
    api_key=config.API_KEY or "not-needed",
    model_provider='openai',
    temperature=config.TEMPERATURE,
    timeout=config.TIMEOUT,
    max_tokens=config.MAX_TOKENS,
).bind_tools(tools)

# ===== ThreadPool =====
executor = ThreadPoolExecutor(max_workers=2)

# ===== RAG КЭШ =====
rag_cache: Dict[str, tuple] = {}
MAX_CACHE_SIZE = 200
CACHE_TTL = 300

def get_cached_rag(query: str) -> Optional[str]:
    cache_key = query[:100]
    if cache_key in rag_cache:
        result, timestamp = rag_cache[cache_key]
        if time.time() - timestamp < CACHE_TTL:
            return result
        else:
            del rag_cache[cache_key]
    return None

def set_cached_rag(query: str, result: str):
    cache_key = query[:100]
    if len(rag_cache) >= MAX_CACHE_SIZE:
        sorted_cache = sorted(rag_cache.items(), key=lambda x: x[1][1])
        for old_key, _ in sorted_cache[:MAX_CACHE_SIZE // 2]:
            del rag_cache[old_key]
    rag_cache[cache_key] = (result, time.time())

# ===== СЕМАНТИЧЕСКАЯ ПАМЯТЬ =====
class SemanticMemory:
    """Хранит историю в векторах и ищет релевантные сообщения"""
    
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_folder="./embeddings_cache"
        )
        self.memory: List[dict] = []
        self.important_ids: set = set()
        self.max_memory_size = 1000  # Максимальный размер памяти
    
    def add_message(self, msg: BaseMessage, importance: float = 0.5):
        """Добавляет сообщение в векторную память"""
        content = msg.content if hasattr(msg, 'content') else str(msg)
        
        if not content:
            return
        
        # Создаём эмбеддинг
        try:
            embedding = self.embeddings.embed_query(content[:1000])
        except:
            return
        
        entry = {
            'msg': msg,
            'embedding': embedding,
            'importance': importance,
            'timestamp': time.time(),
            'type': type(msg).__name__,
            'content_preview': content[:200]
        }
        
        self.memory.append(entry)
        
        # Важные сообщения
        if importance > 0.8:
            self.important_ids.add(len(self.memory) - 1)
        
        # Ограничиваем размер памяти
        if len(self.memory) > self.max_memory_size:
            # Удаляем старые неважные
            candidates = [
                (i, item) for i, item in enumerate(self.memory)
                if i not in self.important_ids
            ]
            candidates.sort(key=lambda x: x[1]['timestamp'])
            
            for i, _ in candidates[:100]:  # Удаляем 100 самых старых
                if i in self.memory:
                    self.memory[i] = None
            
            self.memory = [m for m in self.memory if m is not None]
            # Обновляем индексы важных
            new_important = set()
            for i, item in enumerate(self.memory):
                if item['importance'] > 0.8:
                    new_important.add(i)
            self.important_ids = new_important
    
    def search_relevant(self, query: str, top_k: int = 15) -> List[BaseMessage]:
        """Ищет сообщения, семантически похожие на запрос"""
        if not self.memory:
            return []
        
        # Эмбеддинг запроса
        try:
            query_embedding = self.embeddings.embed_query(query[:1000])
        except:
            return [m['msg'] for m in self.memory[-5:]]
        
        # Считаем схожесть с каждым сообщением
        scored = []
        for i, item in enumerate(self.memory):
            try:
                sim = cosine_similarity(
                    [query_embedding],
                    [item['embedding']]
                )[0][0]
            except:
                sim = 0.0
            
            # Бонус за важность
            importance_bonus = 0.3 if i in self.important_ids else 0
            importance_score = item['importance'] * 0.2
            
            # Бонус за свежесть (затухание за 24 часа)
            age_hours = (time.time() - item['timestamp']) / 3600
            time_bonus = max(0, 0.15 * (1 - age_hours / 24))
            
            final_score = sim + importance_bonus + importance_score + time_bonus
            
            scored.append((final_score, i, item))
        
        # Сортируем по релевантности
        scored.sort(key=lambda x: x[0], reverse=True)
        
        # Собираем результат: top_k + все важные
        result = []
        added_important = set()
        
        for score, idx, item in scored:
            if len(result) >= top_k * 2:
                break
            
            if idx in self.important_ids:
                if idx not in added_important:
                    result.append(item['msg'])
                    added_important.add(idx)
            elif len(result) - len(added_important) < top_k:
                result.append(item['msg'])
        
        return result
    
    def get_context_for_query(self, query: str, max_tokens: int = 50000) -> List[BaseMessage]:
        """Формирует оптимальный контекст"""
        relevant = self.search_relevant(query, top_k=15)
        
        selected = []
        total_tokens = 0
        
        for msg in relevant:
            content = msg.content if hasattr(msg, 'content') else ''
            tokens = count_tokens_approximately(content)
            
            if total_tokens + tokens <= max_tokens:
                selected.append(msg)
                total_tokens += tokens
            else:
                break
        
        return selected
    
    def get_stats(self) -> dict:
        """Статистика памяти"""
        return {
            'total_messages': len(self.memory),
            'important_messages': len(self.important_ids),
            'memory_size': len(self.memory)
        }

# ===== ГЛОБАЛЬНАЯ ПАМЯТЬ =====
semantic_memory = SemanticMemory()

# ===== СОСТОЯНИЕ =====
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    summary: Optional[str]
    errors: int

# ===== RAG поиск =====
async def background_rag_search(query: str) -> str:
    """RAG поиск с кэшированием"""
    cached = get_cached_rag(query)
    if cached is not None:
        return cached
    
    loop = asyncio.get_event_loop()
    docs = await loop.run_in_executor(executor, rag_store.search, query, 5)
    
    if docs:
        by_file = {}
        for doc in docs:
            source = doc.metadata.get('source', 'unknown')
            if source not in by_file:
                by_file[source] = []
            by_file[source].append(doc.page_content[:300])
        
        result = "\n\n".join([
            f"📁 {src}:\n" + "\n---\n".join(contents[:2])
            for src, contents in list(by_file.items())[:3]
        ])
        
        set_cached_rag(query, result)
        return result
    
    set_cached_rag(query, "")
    return ""

# ===== УЗЛЫ ГРАФА =====

async def check_context(state: AgentState) -> dict:
    """Проверка контекста (без суммаризации, только крит. очистка)"""
    messages = state["messages"]
    tokens = sum(
        count_tokens_approximately(m.content)
        for m in messages if hasattr(m, 'content') and m.content
    )
    
    # Только критическое переполнение
    if tokens > config.CRITICAL_AT:
        logger.warning(f"КРИТИЧЕСКИЙ КОНТЕКСТ: {tokens:,}")
        
        # Сохраняем важные решения в summary
        important = []
        for m in messages:
            content = m.content if hasattr(m, 'content') else ''
            if any(w in content.lower() for w in ['создан', 'записан', '✅', 'решение', 'модуль', 'progress']):
                important.append(f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {content[:200]}")
        
        summary = "\n".join(important[-15:])
        
        # Оставляем только систему + последнее сообщение
        system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        other = [m for m in messages if not isinstance(m, SystemMessage)]
        last = other[-2:] if len(other) > 2 else other
        
        return {
            "messages": system_msgs + last,
            "summary": summary[:2000] if summary else "",
            "errors": state.get("errors", 0) + 1
        }
    
    return {}

# ===== ДОБАВИТЬ ПЕРЕД call_model =====
async def _invoke_model(messages: list):
    """Вызов модели с повторными попытками при ошибках сети"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return await model.ainvoke(messages)
        except (httpx.RemoteProtocolError, ConnectionError, asyncio.TimeoutError) as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.warning(f"Попытка {attempt+1} не удалась: {e}. Повтор через {wait_time}с...")
                await asyncio.sleep(wait_time)
            else:
                raise


# ===== ИСПРАВЛЕННЫЙ call_model =====
async def call_model(state: AgentState) -> dict:
    """Вызов модели с семантической памятью"""
    messages = list(state["messages"])
    
    # Системный промпт
    full_messages = [SystemMessage(content=SYSTEM_PROMPT)]
    
    # Summary важных решений
    summary = state.get("summary", "")
    if summary:
        full_messages.append(SystemMessage(
            content=f"📝 Ключевые решения:\n{summary}"
        ))
    
    # Находим последний запрос
    last_human = None
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            last_human = m.content
            break
    
    if last_human and len(last_human) > 15:
        # 🔍 СЕМАНТИЧЕСКИЙ ПОИСК по истории
        relevant_history = semantic_memory.get_context_for_query(
            last_human,
            max_tokens=40000
        )
        
        if relevant_history:
            history_text = "\n".join([
                f"{'👤' if isinstance(m, HumanMessage) else '🤖'}: {m.content[:300]}"
                for m in relevant_history[-12:]
            ])
            
            full_messages.append(SystemMessage(
                content=f"🔍 Релевантная история диалога:\n{history_text}\n\nИспользуй эту историю для контекста."
            ))
        
        # RAG поиск по коду
        rag_context = await background_rag_search(last_human)
        if rag_context:
            full_messages.append(SystemMessage(
                content=f"📁 Релевантный код из проекта:\n{rag_context}"
            ))
    
    # Добавляем последние сообщения
    full_messages.extend(messages[-2:])
    
    # Сохраняем сообщения в семантическую память
    for msg in messages[-5:]:
        importance = 0.5
        content = msg.content if hasattr(msg, 'content') else ''
        
        if isinstance(msg, HumanMessage):
            if any(w in content.lower() for w in ['создай', 'напиши', 'важно', 'решение', 'progress', 'модуль']):
                importance = 0.9
            elif len(content) > 100:
                importance = 0.7
        elif isinstance(msg, AIMessage):
            if '```' in content:
                importance = 0.8
            elif '✅' in content:
                importance = 0.7
        elif isinstance(msg, ToolMessage):
            if '✅' in str(content):
                importance = 0.6
        
        semantic_memory.add_message(msg, importance)
    
    # Вызов модели с повторными попытками
    try:
        response = await _invoke_model(full_messages)
    except Exception as e:
        logger.error(f"Ошибка вызова модели: {e}")
        return {"messages": [AIMessage(content=f"❌ Ошибка модели. Попробуйте позже.")]}
    
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """Маршрутизация"""
    messages = state["messages"]
    last = messages[-1] if messages else None
    
    if last and hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END

# ===== ПОСТРОЕНИЕ ГРАФА =====
def build_graph():
    workflow = StateGraph(AgentState)
    
    tool_node = ToolNode(tools)
    
    workflow.add_node("check_context", check_context)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    
    workflow.add_edge(START, "check_context")
    workflow.add_edge("check_context", "agent")
    workflow.add_conditional_edges("agent", should_continue, {
        "tools": "tools",
        END: END
    })
    workflow.add_edge("tools", "agent")
    
    return workflow.compile()

# ===== АГЕНТ =====
class ModernAgent:
    def __init__(self):
        self.graph = None
        self.checkpointer = None
        self.db = None
        self.thread_id = "main-session"
        self.total_requests = 0
        self.start_time = datetime.now()
        self._shutting_down = False
    
    async def initialize(self):
        path = config.CHECKPOINT_DIR / "memory.sqlite"
        path.parent.mkdir(parents=True, exist_ok=True)
        
        self.db = await aiosqlite.connect(str(path))
        self.checkpointer = AsyncSqliteSaver(self.db)
        await self.checkpointer.setup()
        
        self.graph = build_graph()
        self.graph.checkpointer = self.checkpointer
        
        # Проверяем RAG
        src_path = config.WORK_DIR / "src"
        if src_path.exists():
            existing_docs = rag_store.search("VB6 form module function", k=1)
            
            has_src = any(
                '.frm' in doc.metadata.get('source', '') or
                '.bas' in doc.metadata.get('source', '')
                for doc in existing_docs
            )
            
            if has_src:
                print(f"📚 RAG уже содержит файлы из src/")
            else:
                print(f"📂 Индексация src/ в фоне...")
                loop = asyncio.get_event_loop()
                loop.run_in_executor(executor, rag_store.index_directory, str(src_path))
        
        return True
    
    async def force_shutdown(self):
        """Принудительное завершение"""
        self._shutting_down = True
    
    async def process(self, user_input: str) -> Optional[str]:
        """Обработка запроса"""
        if self._shutting_down:
            return None
        
        self.total_requests += 1
        start_time = time.time()
        
        cfg = {
            "configurable": {"thread_id": self.thread_id},
            "recursion_limit": config.RECURSION_LIMIT
        }
        
        print()
        full_response = ""
        tools_called = []
        step_count = 0
        
        try:
            async for chunk in self.graph.astream(
                {"messages": [HumanMessage(content=user_input)]},
                cfg,
                stream_mode="values"
            ):
                if self._shutting_down:
                    print("\n⏹️ Прервано")
                    return None
                
                if "messages" not in chunk:
                    continue
                
                step_count += 1
                msg = chunk["messages"][-1]
                
                # Ответ модели
                if isinstance(msg, AIMessage) and msg.content:
                    new_part = msg.content[len(full_response):]
                    if new_part:
                        if not full_response:
                            print("🤖 ", end="", flush=True)
                        print(new_part, end="", flush=True)
                        full_response = msg.content
                
                # Вызовы инструментов
                elif hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tc in msg.tool_calls:
                        if self._shutting_down:
                            break
                        name = tc.get('name', '')
                        if name:
                            tools_called.append(name)
                            short = name.replace('_tool', '').replace('_', ' ')
                            print(f"\n🔧 {short}...", end="", flush=True)
                
                # Результаты инструментов
                elif isinstance(msg, ToolMessage):
                    content = str(msg.content)
                    if any(x in content for x in ["✅", "📁", "📖", "✍️", "🔍"]):
                        print(" ✅", end="", flush=True)
                    elif any(x in content for x in ["❌", "Ошибка"]):
                        print(" ❌", end="", flush=True)
                    else:
                        print(" ✓", end="", flush=True)
            
            if full_response and not self._shutting_down:
                print(f"\n   ⏱️ {time.time()-start_time:.1f}с | {step_count} шагов")
            
            elapsed = time.time() - start_time
            if not self._shutting_down:
                metrics.record(
                    success=bool(full_response),
                    tokens=len(full_response.split()) if full_response else 0,
                    time=elapsed,
                    tools=tools_called
                )
            
            return full_response if not self._shutting_down else None
            
        except asyncio.CancelledError:
            print("\n⏹️ Отменено")
            return None
        except Exception as e:
            if self._shutting_down:
                return None
            
            elapsed = time.time() - start_time
            metrics.record(False, 0, elapsed)
            
            error_msg = str(e)
            if "maximum context length" in error_msg:
                print(f"\n⚠️ Контекст переполнен! Очищаю...")
                await self._emergency_clean()
                print("🔄 Повторите запрос")
                return None
            
            logger.error(f"Ошибка: {error_msg[:200]}")
            print(f"\n❌ {error_msg[:200]}")
            return None
    
    async def _emergency_clean(self):
        """Экстренная очистка"""
        try:
            state = await self.graph.aget_state(
                {"configurable": {"thread_id": self.thread_id}}
            )
            if state and state.values and "messages" in state.values:
                messages = state.values["messages"]
                system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
                kept = [m for m in messages if not isinstance(m, SystemMessage)][-2:]
                
                await self.graph.aupdate_state(
                    {"configurable": {"thread_id": self.thread_id}},
                    {"messages": system_msgs + kept, "summary": ""}
                )
                print("✅ Контекст очищен")
        except Exception as e:
            logger.error(f"Ошибка очистки: {e}")
    
    def get_stats(self) -> dict:
        """Расширенная статистика"""
        s = metrics.stats()
        mem = semantic_memory.get_stats()
        
        return {
            **s,
            "uptime": str(datetime.now() - self.start_time),
            "total_requests": self.total_requests,
            "rag_cache_size": len(rag_cache),
            "memory_messages": mem['total_messages'],
            "memory_important": mem['important_messages']
        }
    
    async def close(self):
        """Graceful shutdown"""
        logger.info("Закрытие агента...")
        metrics.save()
        rag_cache.clear()
        
        if self.db:
            await self.db.close()
        
        executor.shutdown(wait=True)
        logger.info("Агент остановлен")

# ===== ОБРАБОТЧИКИ СИГНАЛОВ =====
_shutdown_requested = False
_current_agent = None

def signal_handler(sig, frame):
    global _shutdown_requested
    _shutdown_requested = True
    print(f"\n\n👋 Завершение...")
    if _current_agent:
        asyncio.create_task(_current_agent.force_shutdown())

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ===== ЗАПУСК =====
async def main():
    global _current_agent, _shutdown_requested  # ← ДОБАВИТЬ _shutdown_requested
    
    os.system('clear' if os.name != 'nt' else 'cls')
    
    print("=" * 70)
    print("🤖 AI АГЕНТ С СЕМАНТИЧЕСКОЙ ПАМЯТЬЮ")
    print("=" * 70)
    print(f"🧠 Память: векторный поиск релевантной истории")
    print(f"🔗 Модель: {config.MODEL_NAME}")
    print(f"📚 RAG: ChromaDB (кэш {len(rag_cache)} запросов)")
    print(f"💾 Чекпоинты: {config.CHECKPOINT_DIR}")
    print("=" * 70)
    
    agent = ModernAgent()
    _current_agent = agent
    
    try:
        if not await agent.initialize():
            return 1
        
        print("✅ Агент готов")
        print("📝 Команды: exit | stats | clear | memory | help")
        print("-" * 70)
        
        while not _shutdown_requested and not agent._shutting_down:
            try:
                # Простой неблокирующий ввод БЕЗ таймаута
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: input("\n👤 Вы: ").strip()
                )
                
                if _shutdown_requested or agent._shutting_down:
                    break
                
                if not user_input:
                    continue
                
                cmd = user_input.lower()
                
                # Обработка команд
                if cmd in ["exit", "quit"]:
                    s = agent.get_stats()
                    print(f"\n👋 Сессия завершена")
                    print(f"📊 Запросов: {s['requests']} ({s['success_rate']})")
                    print(f"🧠 Память: {s['memory_messages']} сообщений")
                    print(f"⏱️  Аптайм: {s['uptime']}")
                    break
                
                elif cmd == "stats":
                    s = agent.get_stats()
                    print(f"\n{'='*50}")
                    print(f"📊 СТАТИСТИКА")
                    print(f"{'='*50}")
                    print(f"⏱️  Аптайм:     {s['uptime']}")
                    print(f"📨 Запросов:   {s['requests']} ({s['success_rate']})")
                    print(f"🧠 Память:     {s['memory_messages']} сообщений ({s['memory_important']} важных)")
                    print(f"📚 RAG кэш:    {s['rag_cache_size']} запросов")
                    print(f"💾 Токенов:    {s['total_tokens']}")
                    print(f"⚡ Среднее t:  {s['avg_time']}")
                    if s.get('tools'):
                        print(f"\n🔧 Инструменты:")
                        for tool, count in sorted(s['tools'].items(), key=lambda x: x[1], reverse=True)[:5]:
                            print(f"   {tool}: {count}")
                    print(f"{'='*50}")
                    continue
                
                elif cmd == "clear":
                    rag_cache.clear()
                    print("🧹 RAG кэш очищен")
                    continue
                
                elif cmd == "memory":
                    mem = semantic_memory.get_stats()
                    print(f"\n🧠 СЕМАНТИЧЕСКАЯ ПАМЯТЬ:")
                    print(f"   Сообщений: {mem['total_messages']}")
                    print(f"   Важных: {mem['important_messages']}")
                    print(f"   Размер: {mem['memory_size']}")
                    continue
                
                elif cmd == "summarize":
                    await agent._emergency_clean()
                    continue
                
                elif cmd == "help":
                    print("""
📚 КОМАНДЫ:
  exit      - выход с сохранением
  stats     - статистика сессии
  clear     - очистить RAG кэш
  memory    - состояние семантической памяти
  summarize - очистить контекст
  help      - справка

🧠 СЕМАНТИЧЕСКАЯ ПАМЯТЬ:
  История не обрезается, а ищется по смыслу.
  Модель получает только релевантные сообщения.
                    """)
                    continue
                
                # Запускаем обработку запроса
                task = asyncio.create_task(agent.process(user_input))
                
                # Ждём завершения с проверкой shutdown
                while not task.done():
                    if _shutdown_requested or agent._shutting_down:
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                        break
                    await asyncio.sleep(0.1)
                
            except KeyboardInterrupt:
                print("\n\n👋 Завершение...")
                _shutdown_requested = True
                agent._shutting_down = True
                break
            except EOFError:
                print("\n\n👋 EOF получен")
                break
            except Exception as e:
                if _shutdown_requested:
                    break
                logger.error(f"Ошибка в главном цикле: {e}")
                print(f"\n❌ {str(e)[:200]}")
    
    finally:
        print("\n💾 Сохранение...")
        await agent.close()
        print("✅ Готово")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Завершено")
