"""
Продвинутый AI агент с суммаризацией (исправленный)
"""
import os
import sys
import asyncio
import time
import traceback
from typing import Any, Dict, Optional, List
from datetime import datetime

import aiosqlite
import httpx
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from langchain_core.messages import (
    AIMessage, HumanMessage, SystemMessage, ToolMessage
)
from langchain_core.messages.utils import count_tokens_approximately

from utils.token_counter import count_tokens_for_qwen

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from agents.memory_manager import memory_manager
from utils.summarizer import AdvancedSummarizer, SummarizationStrategy
from utils.metrics import metrics

from tools.code_base import search_codebase
from tools.file_tools import (
    list_dir, read_file, write_file, create_dir, 
    file_exists, dir_exists, delete_dir, delete_file, pwd
)
from tools.build_tools import npm_install, npm_build

os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

import logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(
            config.LOGS_DIR / f"agent_{datetime.now().strftime('%Y%m%d')}.log",
            encoding='utf-8'
        )
    ]
)

for lib in ["httpx", "httpcore", "langchain", "langgraph", "aiosqlite"]:
    logging.getLogger(lib).setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

tools_list = [
    list_dir, read_file, write_file, create_dir,
    file_exists, dir_exists, delete_dir, delete_file,
    npm_install, npm_build, pwd, search_codebase
]

# Модель
try:
    api_key = config.API_KEY
    model = init_chat_model(
        model=config.MODEL_NAME,
        base_url=config.MODEL_URL,
        api_key=api_key if api_key else "not-needed",
        model_provider='openai',
        temperature=config.TEMPERATURE,
        timeout=config.REQUEST_TIMEOUT,
        max_tokens=config.MAX_TOKENS_RESPONSE,
        max_retries=2,
        streaming=True,
    )
    logger.info(f"Модель готова: {config.MODEL_NAME}")
except Exception as e:
    print(f"❌ Ошибка модели: {e}")
    sys.exit(1)

summarizer = AdvancedSummarizer(model)

class AdvancedAgent:
    """Агент с суммаризацией (без ensure_future)"""
    
    def __init__(self):
        self.agent = None
        self.checkpointer = None
        self.db = None
        self.session_config = {
            "configurable": {"thread_id": "session-1"},
            "recursion_limit": config.DEFAULT_RECURSION_LIMIT
        }
        self.is_initialized = False
        self.total_requests = 0
        self.max_context_seen = 0
        self.summarization_count = 0
        self.tokens_saved = 0
        self.need_summarize = False  # Флаг для отложенной суммаризации
    
    async def initialize(self) -> bool:
        """Инициализация"""
        try:
            checkpoint_path = config.CHECKPOINT_DIR / "agent_memory.sqlite"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.db = await aiosqlite.connect(str(checkpoint_path))
            self.checkpointer = AsyncSqliteSaver(self.db)
            await self.checkpointer.setup()
            
            try:
                with open("./prompts/instruction.txt", "r", encoding="utf-8") as f:
                    system_prompt = f.read()
            except:
                system_prompt = "Ты - AI ассистент для работы с кодом."
            
            # СОЗДАЕМ АГЕНТА БЕЗ pre_model_hook
            self.agent = create_react_agent(
                model=model,
                tools=tools_list,
                checkpointer=self.checkpointer,
                prompt=system_prompt
            )
            
            self.is_initialized = True
            return True
                
        except Exception as e:
            logger.error(f"Ошибка инициализации: {e}")
            print(f"❌ Ошибка: {e}")
            return False
    
    def _sync_check_context(self, messages: list) -> tuple:
        """
        СИНХРОННАЯ проверка контекста.
        Возвращает (нужна_суммаризация, обрезанные_сообщения)
        """
        if not messages:
            return False, messages
        
        # total_tokens = sum(
        #     count_tokens_approximately(m.content) 
        #     for m in messages 
        #     if hasattr(m, 'content') and m.content
        # )
        
        total_tokens = sum(
            count_tokens_for_qwen(m.content) 
            for m in messages 
            if hasattr(m, 'content') and m.content
        )

        self.max_context_seen = max(self.max_context_seen, total_tokens)
        
        threshold = int(config.MAX_CONTEXT_TOKENS * config.SUMMARIZATION_TRIGGER)
        critical = int(config.MAX_CONTEXT_TOKENS * config.CRITICAL_CONTEXT_USAGE)
        
        if total_tokens > critical:
            # Критическое - экстренная обрезка
            return "critical", self._emergency_trim(messages, total_tokens)
        
        elif total_tokens > threshold:
            # Нужна суммаризация (НО не здесь, а в асинхронном коде)
            return "summarize", messages
        
        return None, messages
    
    def _emergency_trim(self, messages: list, total_tokens: int) -> list:
        """Синхронная экстренная обрезка"""
        system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        protected = self._extract_protected(messages)
        other = [m for m in messages if m not in protected and not isinstance(m, SystemMessage)]
        kept = other[-5:] if len(other) > 5 else other
        
        result = system_msgs + protected + kept
        
        # new_tokens = sum(
        #     count_tokens_approximately(m.content) 
        #     for m in result 
        #     if hasattr(m, 'content') and m.content
        # )
        
        new_tokens = sum(
            count_tokens_for_qwen(m.content) 
            for m in result 
            if hasattr(m, 'content') and m.content
        )

        print(f"\n🚨 Экстренная обрезка: {total_tokens:,} → {new_tokens:,} токенов")
        print(f"   Сообщений: {len(messages)} → {len(result)}")
        
        return result
    
    def _extract_protected(self, messages: list) -> list:
        """Защищенные tool chains"""
        protected = []
        indices = set()
        
        for i, msg in enumerate(messages):
            if i in indices:
                continue
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                protected.append(msg)
                indices.add(i)
                for tc in msg.tool_calls:
                    tc_id = tc.get('id') if isinstance(tc, dict) else getattr(tc, 'id', None)
                    if tc_id:
                        for j in range(i + 1, min(i + 10, len(messages))):
                            if j in indices:
                                continue
                            m = messages[j]
                            if isinstance(m, ToolMessage) and hasattr(m, 'tool_call_id') and m.tool_call_id == tc_id:
                                protected.append(m)
                                indices.add(j)
        return protected
    
    async def _do_summarization(self):
        """Асинхронная суммаризация (вызывается в правильном event loop)"""
        try:
            state = await self.agent.aget_state(self.session_config)
            if not state or not state.values:
                return
            
            messages = state.values.get("messages", [])
            if not messages:
                return
            
            total_tokens = sum(
                count_tokens_approximately(m.content) 
                for m in messages 
                if hasattr(m, 'content') and m.content
            )
            
            print(f"\n📝 Суммаризация: {total_tokens:,} токенов → ", end="", flush=True)
            
            summarized, saved = await summarizer.summarize(
                messages,
                preserve_tools=True
            )
            
            new_tokens = sum(
                count_tokens_approximately(m.content) 
                for m in summarized 
                if hasattr(m, 'content') and m.content
            )
            
            print(f"{new_tokens:,} (экономия {saved:,})")
            
            await self.agent.aupdate_state(
                self.session_config,
                {"messages": summarized},
                as_node="__start__"
            )
            
            self.summarization_count += 1
            self.tokens_saved += saved
            self.need_summarize = False
            
        except Exception as e:
            logger.error(f"Ошибка суммаризации: {e}")
    
    async def process_request(self, user_input: str) -> Optional[str]:
        """Обработка запроса"""
        
        if not self.is_initialized or not self.agent:
            print("❌ Агент не инициализирован")
            return None
        
        self.total_requests += 1
        
        # 1. Проверяем контекст ДО запроса
        action = None
        try:
            state = await self.agent.aget_state(self.session_config)
            if state and state.values:
                messages = state.values.get("messages", [])
                action, new_messages = self._sync_check_context(messages)
                
                if action == "critical":
                    # Сохраняем обрезанные сообщения
                    await self.agent.aupdate_state(
                        self.session_config,
                        {"messages": new_messages},
                        as_node="__start__"
                    )
                elif action == "summarize":
                    # Запускаем суммаризацию
                    await self._do_summarization()
        except Exception as e:
            logger.error(f"Ошибка проверки контекста: {e}")
        
        # 2. Периодическая суммаризация (каждые 5 запросов)
        if self.total_requests % 5 == 0:
            await self._do_summarization()
        
        # 3. Обрабатываем запрос
        try:
            full_response = ""
            
            print()
            
            async for event in self.agent.astream_events(
                {"messages": [HumanMessage(content=user_input)]},
                config=self.session_config,
                version="v2"
            ):
                kind = event["event"]
                
                if kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if hasattr(chunk, 'content') and chunk.content:
                        if not full_response:
                            print("🤖 ", end="", flush=True)
                        print(chunk.content, end="", flush=True)
                        full_response += chunk.content
                
                elif kind == "on_tool_start":
                    tool_name = event.get("name", "unknown")
                    short = tool_name.replace("_tool", "").replace("_", " ")
                    print(f"\n🔧 {short}...", end="", flush=True)
                
                elif kind == "on_tool_end":
                    output = str(event.get("data", {}).get("output", ""))
                    if any(x in output for x in ["✅", "📁", "📖", "✍️", "🔍"]):
                        print(" ✅", end="", flush=True)
                    elif any(x in output for x in ["❌", "Ошибка"]):
                        print(" ❌", end="", flush=True)
                    else:
                        print(" ✓", end="", flush=True)
            
            if full_response:
                print("\n")
                memory_manager.add_interaction(user_msg=user_input, ai_msg=full_response)
                return full_response
            else:
                print("\n⚠️ Пустой ответ")
                return None
            
        except Exception as e:
            error_str = str(e)
            
            if "maximum context length" in error_str:
                print(f"\n⚠️ Контекст переполнен!")
                await self._do_summarization()
                print("🔄 Повторите запрос")
                return None
            
            logger.error(f"Ошибка: {e}")
            print(f"\n❌ {str(e)[:200]}")
            return None
    
    async def close(self):
        if self.db:
            await self.db.close()

async def run_advanced_agent():
    os.system('clear' if os.name != 'nt' else 'cls')
    
    print("=" * 70)
    print("🤖 ПРОДВИНУТЫЙ АГЕНТ КОДЕР v3.0")
    print(f"📋 VB6 → NestJS + React")
    print(f"🔗 {config.MODEL_NAME}")
    print(f"💾 Контекст: до {config.MAX_CONTEXT_TOKENS:,} токенов")
    print(f"🗜️ Суммаризация: при {config.SUMMARIZATION_TRIGGER*100:.0f}%")
    print("=" * 70)
    
    agent = AdvancedAgent()
    
    try:
        if not await agent.initialize():
            return 1
        
        print("✅ Агент готов")
        print("📝 exit | summarize | stats | help")
        print("-" * 70)
        
        while True:
            try:
                user_input = input("\n👤 Вы: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ["exit", "quit"]:
                    print(f"\n👋 Сессия завершена")
                    print(f"📊 Суммаризаций: {agent.summarization_count}")
                    print(f"💾 Сэкономлено: {agent.tokens_saved:,} токенов")
                    break
                
                elif user_input.lower() == "summarize":
                    await agent._do_summarization()
                    continue
                
                elif user_input.lower() == "stats":
                    print(f"\n📊 СТАТИСТИКА:")
                    print(f"   Запросов: {agent.total_requests}")
                    print(f"   Суммаризаций: {agent.summarization_count}")
                    print(f"   Сэкономлено: {agent.tokens_saved:,} токенов")
                    print(f"   Пик контекста: {agent.max_context_seen:,} токенов")
                    continue
                
                elif user_input.lower() == "help":
                    print("\n📚 КОМАНДЫ:")
                    print("   exit      - выход")
                    print("   summarize - суммаризация сейчас")
                    print("   stats     - статистика")
                    print("   help      - справка")
                    continue
                
                await agent.process_request(user_input)
                
            except KeyboardInterrupt:
                print(f"\n\n👋 Прервано")
                break
            except Exception as e:
                print(f"\n❌ {e}")
    
    finally:
        await agent.close()

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    try:
        exit_code = asyncio.run(run_advanced_agent())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 Завершено")
        sys.exit(0)