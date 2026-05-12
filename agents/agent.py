"""
AI АГЕНТ С RAG-ПАМЯТЬЮ
Единое ChromaDB для кода и истории
"""
import os
import sys
import asyncio
import time
import signal
from typing import Annotated, List, TypedDict, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool

from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
)
from langchain_core.messages.utils import count_tokens_approximately

import aiosqlite
import httpx

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

from config import config
from stores.rag_store import rag  # Единое RAG хранилище
from utils.metrics import metrics

from tools.file_tools import (
    list_dir, read_file, write_file, create_dir,
    file_exists, dir_exists, delete_dir, delete_file, pwd
)
from tools.build_tools import npm_install, npm_build

import logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(f"logs/agent_{datetime.now():%Y%m%d}.log", encoding='utf-8')]
)
for lib in ["httpx", "httpcore", "langchain", "langgraph"]:
    logging.getLogger(lib).setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

# ===== ИНСТРУМЕНТЫ ДЛЯ RAG-ПАМЯТИ =====
@tool
def search_memory(query: str) -> str:
    """Поиск по истории диалога. Что обсуждали, какие решения приняли."""
    return rag.search_memory(query)

@tool
def get_decisions() -> str:
    """ПОЛУЧИТЬ ЧТО УЖЕ СДЕЛАНО. Вызывай В НАЧАЛЕ работы чтобы не повторяться."""
    return rag.get_decisions()

@tool
def search_codebase(query: str) -> str:
    """Поиск по коду VB6. Ищи функции, модули, SQL, контролы."""
    return rag.search_code(query)

# ===== ВСЕ ИНСТРУМЕНТЫ =====
tools = [
    # Файлы
    list_dir, read_file, write_file, create_dir,
    file_exists, dir_exists, delete_dir, delete_file,
    # Сборка
    npm_install, npm_build,
    # Навигация
    pwd,
    # RAG-память
    search_memory,
    get_decisions,
    search_codebase,
]

# ===== ПРОМПТ =====
SYSTEM_PROMPT = """Ты AI ассистент для переписывания VB6 → NestJS + React.

🧠 ТВОЯ ПАМЯТЬ:
- get_decisions() — узнай что уже сделано (вызывай ПЕРВЫМ)
- search_memory(запрос) — найди что обсуждали
- search_codebase(запрос) — найди код в VB6 проекте

📋 ПРАВИЛА:
- ВСЕГДА начинай с get_decisions()
- НЕ повторяй уже сделанное
- Рабочая папка: workdir/
- src/ только для чтения
- Новый код в api/ и client/
- Записывай прогресс в runtime/progress.md
- Отвечай на русском, кратко
"""

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

# ===== СОСТОЯНИЕ =====
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# ===== УЗЛЫ ГРАФА =====

def _get_importance(msg: BaseMessage) -> float:
    """Важность сообщения для сохранения в RAG"""
    content = msg.content if hasattr(msg, 'content') else str(msg)
    
    if isinstance(msg, HumanMessage):
        if any(w in content.lower() for w in ['создай', 'напиши', 'сделай', 'progress']):
            return 0.9
        return 0.5
    
    if isinstance(msg, AIMessage):
        if '```' in content and len(content) > 300:
            return 0.9
        if '✅' in content:
            return 0.8
        return 0.4
    
    if isinstance(msg, ToolMessage):
        if '✅' in str(content):
            return 0.7
        return 0.3
    
    return 0.3

async def call_model(state: AgentState) -> dict:
    """Вызов модели с минимальным контекстом"""
    messages = list(state["messages"])
    
    # Подсчёт токенов для лога
    total_tokens = sum(count_tokens_approximately(m.content) for m in messages if hasattr(m, 'content'))
    
    # ✅ МИНИМАЛЬНЫЙ КОНТЕКСТ: системный промпт + последние 3 сообщения
    minimal = [SystemMessage(content=SYSTEM_PROMPT)]
    minimal.extend(messages[-3:])
    
    context_tokens = sum(count_tokens_approximately(m.content) for m in minimal if hasattr(m, 'content'))
    logger.info(f"Контекст: {len(messages)}→{len(minimal)} сообщений, {total_tokens:,}→{context_tokens:,} токенов")
    
    # Сохраняем ВАЖНЫЕ сообщения в RAG (решения, код)
    for msg in messages[-5:]:
        imp = _get_importance(msg)
        if imp > 0.6:
            rag.add_message(msg, imp)
        
        # Решения сохраняем отдельно
        content = msg.content if hasattr(msg, 'content') else ''
        if isinstance(msg, AIMessage) and '✅' in content:
            rag.add_decision(content[:500])
        elif isinstance(msg, ToolMessage) and '✅' in str(msg.content):
            rag.add_decision(str(msg.content)[:500])
    
    # Вызов модели
    try:
        response = await _invoke_model(minimal)
    except Exception as e:
        logger.error(f"Ошибка: {e}")
        return {"messages": [AIMessage(content=f"❌ Ошибка")]}
    
    return {"messages": [response]}

async def _invoke_model(messages: list):
    """Повторные попытки"""
    for attempt in range(3):
        try:
            return await model.ainvoke(messages)
        except (httpx.RemoteProtocolError, ConnectionError, asyncio.TimeoutError) as e:
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
            else:
                raise

def should_continue(state: AgentState) -> str:
    last = state["messages"][-1] if state["messages"] else None
    if last and hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END

# ===== ГРАФ =====
def build_graph():
    workflow = StateGraph(AgentState)
    tool_node = ToolNode(tools)
    
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    workflow.add_edge("tools", "agent")
    
    return workflow.compile()

# ===== АГЕНТ =====
class ModernAgent:
    def __init__(self):
        self.graph = None
        self.checkpointer = None
        self.db = None
        self.thread_id = "main"
        self.total_requests = 0
        self.start_time = datetime.now()
        self._shutting_down = False
    
    async def initialize(self):
        path = config.CHECKPOINT_DIR / "checkpoint.sqlite"
        path.parent.mkdir(parents=True, exist_ok=True)
        
        self.db = await aiosqlite.connect(str(path))
        self.checkpointer = AsyncSqliteSaver(self.db)
        await self.checkpointer.setup()
        
        self.graph = build_graph()
        self.graph.checkpointer = self.checkpointer
        
        # ✅ Проверяем, есть ли уже файлы из src в RAG
        src = config.WORK_DIR / "src"
        if src.exists():
            stats = rag.get_stats()
            
            if stats['code_docs'] > 0:
                # Проверяем, есть ли в RAG файлы с расширениями VB6
                test_result = rag.search_code("VB6 form module function", k=3)
                has_vb6 = any(
                    ext in test_result for ext in ['.frm', '.bas', '.cls', '.vb']
                )
                
                if has_vb6:
                    print(f"📚 RAG уже содержит VB6 код ({stats['code_docs']} документов)")
                    print(f"   Индексация не требуется")
                else:
                    print(f"📂 RAG есть, но без VB6 кода. Индексация src/...")
                    loop = asyncio.get_event_loop()
                    loop.run_in_executor(executor, rag.index_directory, str(src))
            else:
                print(f"📂 RAG пуст. Индексация src/...")
                loop = asyncio.get_event_loop()
                loop.run_in_executor(executor, rag.index_directory, str(src))
        else:
            print(f"⚠️ Директория src/ не найдена")
        
        return True
    
    async def process(self, user_input: str) -> Optional[str]:
        if self._shutting_down:
            return None
        
        self.total_requests += 1
        start = time.time()
        
        cfg = {
            "configurable": {"thread_id": self.thread_id},
            "recursion_limit": config.RECURSION_LIMIT
        }
        
        print()
        full = ""
        tools_used = []
        
        try:
            async for chunk in self.graph.astream(
                {"messages": [HumanMessage(content=user_input)]},
                cfg, stream_mode="values"
            ):
                if self._shutting_down:
                    break
                
                if "messages" not in chunk:
                    continue
                
                msg = chunk["messages"][-1]
                
                if isinstance(msg, AIMessage) and msg.content:
                    new = msg.content[len(full):]
                    if new:
                        if not full:
                            print("🤖 ", end="", flush=True)
                        print(new, end="", flush=True)
                        full = msg.content
                
                elif hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tc in msg.tool_calls:
                        name = tc.get('name', '')
                        if name:
                            tools_used.append(name)
                            print(f"\n🔧 {name}...", end="", flush=True)
                
                elif isinstance(msg, ToolMessage):
                    c = str(msg.content)[:50]
                    print(" ✅" if "✅" in c else " ✓", end="", flush=True)
            
            if full:
                print(f"\n   ⏱️ {time.time()-start:.1f}с")
            
            metrics.record(bool(full), len(full.split()) if full else 0, time.time()-start, tools_used)
            return full
            
        except Exception as e:
            if "maximum context length" in str(e):
                print("\n⚠️ Контекст! Очистка...")
                await self._clean()
            else:
                print(f"\n❌ {str(e)[:200]}")
            return None
    
    async def _clean(self):
        try:
            state = await self.graph.aget_state({"configurable": {"thread_id": self.thread_id}})
            if state and state.values:
                msgs = state.values.get("messages", [])
                await self.graph.aupdate_state(
                    {"configurable": {"thread_id": self.thread_id}},
                    {"messages": msgs[-2:]}
                )
        except:
            pass
    
    def get_stats(self):
        s = metrics.stats()
        r = rag.get_stats()
        return {**s, "uptime": str(datetime.now() - self.start_time), **r}
    
    async def close(self):
        metrics.save()
        if self.db:
            await self.db.close()
        executor.shutdown(wait=True)

# ===== СИГНАЛЫ =====
_shutdown = False
_agent = None

def handler(sig, frame):
    global _shutdown
    _shutdown = True
    if _agent:
        _agent._shutting_down = True

signal.signal(signal.SIGINT, handler)
signal.signal(signal.SIGTERM, handler)

# ===== ЗАПУСК =====
async def main():
    global _agent, _shutdown
    
    os.system('clear' if os.name != 'nt' else 'cls')
    
    stats = rag.get_stats()
    print("=" * 60)
    print("🤖 AI АГЕНТ С RAG-ПАМЯТЬЮ")
    print("=" * 60)
    print(f"📚 Код: {stats['code_docs']} док. | 🧠 История: {stats['memory_docs']} зап.")
    print(f"🔗 Модель: {config.MODEL_NAME}")
    print("=" * 60)
    
    agent = ModernAgent()
    _agent = agent
    
    if not await agent.initialize():
        return 1
    
    print("✅ Готов. Команды: exit | stats | help")
    print("-" * 60)
    
    while not _shutdown:
        try:
            user_input = await asyncio.get_event_loop().run_in_executor(
                None, lambda: input("\n👤 Вы: ").strip()
            )
            
            if not user_input:
                continue
            
            cmd = user_input.lower()
            
            if cmd in ["exit", "quit"]:
                break
            elif cmd == "stats":
                s = agent.get_stats()
                print(f"\n📊 Запросов: {s['requests']} | 📚 Код: {s['code_docs']} | 🧠 Ист: {s['memory_docs']}")
                continue
            elif cmd == "help":
                print("\n📚 exit | stats | help")
                continue
            
            await agent.process(user_input)
            
        except KeyboardInterrupt:
            break
        except EOFError:
            break
    
    await agent.close()
    print("✅ Завершено")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())