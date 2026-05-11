"""
PRODUCTION AI АГЕНТ НА LANGGRAPH
StateGraph + RAG + Суммаризация + Метрики + Checkpoints
"""
import os
import sys
import asyncio
import time
import signal
import json
from typing import Annotated, List, TypedDict, Optional, Dict, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.prebuilt import ToolNode

from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
)
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.runnables import RunnableConfig

import aiosqlite

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

from config import config
from stores.rag_store import rag_store
from utils.summarizer import Summarizer
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

# ===== СУММАРИЗАТОР =====
summarizer = Summarizer(model)

# ===== ThreadPool =====
executor = ThreadPoolExecutor(max_workers=2)

# ===== RAG КЭШ =====
@lru_cache(maxsize=100)
def _cached_rag_search(query_hash: str) -> str:
    """Кэшированный RAG поиск (не используется напрямую)"""
    return ""

rag_cache: Dict[str, tuple] = {}  # query -> (result, timestamp)

# ===== СОСТОЯНИЕ =====
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    summary: Optional[str]
    summarizations: int
    tokens_saved: int
    rag_queries: int
    tools_used: int
    errors: int

# ===== RAG поиск с кэшем =====
async def background_rag_search(query: str) -> str:
    """Фоновый RAG поиск с кэшированием"""
    # Проверяем кэш
    cache_key = query[:100]  # Первые 100 символов как ключ
    if cache_key in rag_cache:
        result, timestamp = rag_cache[cache_key]
        if time.time() - timestamp < 300:  # 5 минут кэш
            return result
    
    loop = asyncio.get_event_loop()
    docs = await loop.run_in_executor(executor, rag_store.search, query, 5)
    
    if docs:
        # Группируем по файлам
        by_file = {}
        for doc in docs:
            source = doc.metadata.get('source', 'unknown')
            if source not in by_file:
                by_file[source] = []
            by_file[source].append(doc.page_content[:300])
        
        result = "\n\n".join([
            f"📁 {src}:\n" + "\n---\n".join(contents[:2])
            for src, contents in by_file.items()
        ])
        
        # Кэшируем
        rag_cache[cache_key] = (result, time.time())
        return result
    
    return ""

# ===== УЗЛЫ ГРАФА =====

async def check_context(state: AgentState) -> dict:
    """Проверка и очистка контекста"""
    messages = state["messages"]
    tokens = sum(
        count_tokens_approximately(m.content)
        for m in messages if hasattr(m, 'content') and m.content
    )
    
    # Критическое переполнение — экстренная обрезка
    if tokens > config.CRITICAL_AT:
        logger.warning(f"КРИТИЧЕСКИЙ КОНТЕКСТ: {tokens:,} токенов")
        system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        other = [m for m in messages if not isinstance(m, SystemMessage)]
        kept = other[-3:] if len(other) > 3 else other
        return {
            "messages": system_msgs + kept,
            "errors": state.get("errors", 0) + 1
        }
    
    # Нужна суммаризация
    if tokens > config.SUMMARIZE_AT and len(messages) > 10:
        logger.info(f"Суммаризация: {tokens:,} токенов")
        summary, new_messages, saved = await summarizer.summarize(messages)
        if summary:
            return {
                "summary": summary,
                "messages": new_messages,
                "summarizations": state.get("summarizations", 0) + 1,
                "tokens_saved": state.get("tokens_saved", 0) + saved
            }
    
    return {}

async def call_model(state: AgentState) -> dict:
    """Вызов модели с RAG контекстом"""
    messages = list(state["messages"])
    summary = state.get("summary", "")
    
    # Системный промпт + история
    full_messages = [SystemMessage(content=SYSTEM_PROMPT)]
    
    if summary:
        full_messages.append(SystemMessage(
            content=f"📝 Краткая история диалога:\n{summary}\n\nОтвечай, учитывая эту историю."
        ))
    
    # RAG: ищем релевантный код
    last_human = None
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            last_human = m.content
            break
    
    if last_human and len(last_human) > 20:
        rag_context = await background_rag_search(last_human)
        if rag_context:
            full_messages.append(SystemMessage(
                content=f"🔍 Релевантный код из VB6 проекта:\n{rag_context}\n\nИспользуй этот код для ответа."
            ))
    
    full_messages.extend(messages)
    
    response = await model.ainvoke(full_messages)
    return {"messages": [response]}

def should_continue(state: AgentState) -> str:
    """Маршрутизация: продолжать или завершить"""
    messages = state["messages"]
    last = messages[-1] if messages else None
    
    if last and hasattr(last, "tool_calls") and last.tool_calls:
        # Считаем использованные инструменты
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
    
    async def process(self, user_input: str) -> Optional[str]:
        self.total_requests += 1
        start_time = time.time()
        
        cfg = {
            "configurable": {"thread_id": self.thread_id},
            "recursion_limit": config.RECURSION_LIMIT
        }
        
        print()
        full_response = ""
        tools_called = []
        
        try:
            async for chunk in self.graph.astream(
                {"messages": [HumanMessage(content=user_input)]},
                cfg,
                stream_mode="values"
            ):
                if "messages" not in chunk:
                    continue
                
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
                        name = tc.get('name', '')
                        if name:
                            tools_called.append(name)
                            # Красивое сокращение имён
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
            
            if full_response:
                print("\n")
            
            elapsed = time.time() - start_time
            metrics.record(
                success=bool(full_response),
                tokens=len(full_response.split()) if full_response else 0,
                time=elapsed,
                tools=tools_called
            )
            
            return full_response
            
        except Exception as e:
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
        """Экстренная очистка контекста"""
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
        return {
            **s,
            "uptime": str(datetime.now() - self.start_time),
            "total_requests": self.total_requests,
            "rag_cache_size": len(rag_cache)
        }
    
    async def close(self):
        """Graceful shutdown"""
        logger.info("Закрытие агента...")
        
        # Сохраняем метрики
        metrics.save()
        
        # Очищаем кэш
        rag_cache.clear()
        
        # Закрываем БД
        if self.db:
            await self.db.close()
        
        # Oстанавливаем executor
        executor.shutdown(wait=True)
        
        logger.info("Агент остановлен")

# ===== ОБРАБОТЧИКИ СИГНАЛОВ =====
_shutdown_requested = False

def signal_handler(sig, frame):
    global _shutdown_requested
    _shutdown_requested = True
    print(f"\n👋 Получен сигнал завершения...")

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ===== ЗАПУСК =====
async def main():
    os.system('clear' if os.name != 'nt' else 'cls')
    
    print("=" * 70)
    print("🤖 PRODUCTION AI АГЕНТ")
    print("=" * 70)
    print(f"🏗️  StateGraph + Pregel + Checkpoints")
    print(f"🔗 Модель: {config.MODEL_NAME}")
    print(f"📚 RAG: ChromaDB (кэш {len(rag_cache)} запросов)")
    print(f"🗜️  Суммаризация: при {config.SUMMARIZE_AT:,} токенов")
    print(f"🛡️  Критический порог: {config.CRITICAL_AT:,} токенов")
    print(f"💾 Чекпоинты: {config.CHECKPOINT_DIR}")
    print("=" * 70)
    
    agent = ModernAgent()
    
    try:
        if not await agent.initialize():
            return 1
        
        print("✅ Агент готов к работе")
        print("📝 Команды: exit | stats | summarize | clear | help")
        print("-" * 70)
        
        while not _shutdown_requested:
            try:
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: input("\n👤 Вы: ").strip()
                )
                
                if not user_input:
                    continue
                
                cmd = user_input.lower()
                
                if cmd in ["exit", "quit"]:
                    s = agent.get_stats()
                    print(f"\n👋 Сессия завершена")
                    print(f"📊 Запросов: {s['requests']} ({s['success_rate']})")
                    print(f"⏱️  Аптайм: {s['uptime']}")
                    break
                
                elif cmd == "stats":
                    s = agent.get_stats()
                    print(f"\n📊 СТАТИСТИКА:")
                    print(f"   Запросов: {s['requests']} ({s['success_rate']})")
                    print(f"   Токенов: {s['total_tokens']}")
                    print(f"   Среднее время: {s['avg_time']}")
                    print(f"   Аптайм: {s['uptime']}")
                    print(f"   RAG кэш: {s['rag_cache_size']} запросов")
                    if s.get('tools'):
                        print(f"   🔧 Инструменты: {dict(list(s['tools'].items())[:5])}")
                    continue
                
                elif cmd == "summarize":
                    await agent._emergency_clean()
                    continue
                
                elif cmd == "clear":
                    rag_cache.clear()
                    print("🧹 RAG кэш очищен")
                    continue
                
                elif cmd == "help":
                    print("""
📚 КОМАНДЫ:
  exit      - выход с сохранением
  stats     - расширенная статистика
  summarize - очистить контекст
  clear     - очистить RAG кэш
  help      - эта справка

💡 ГОРЯЧИЕ КЛАВИШИ:
  Ctrl+C    - завершение (graceful shutdown)
                    """)
                    continue
                
                await agent.process(user_input)
                
            except KeyboardInterrupt:
                print("\n\n👋 Завершение...")
                break
            except Exception as e:
                print(f"\n❌ {str(e)[:200]}")
    
    finally:
        print("💾 Сохранение метрик...")
        await agent.close()
        print("✅ Готово")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(main())