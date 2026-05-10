"""
Продвинутый AI агент для переписывания VB6 → NestJS + React
Версия 3.0 с интеллектуальным управлением контекстом и суммаризацией
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

# Добавляем корень проекта в путь
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from agents.memory_manager import memory_manager
from utils.summarizer import AdvancedSummarizer, SummarizationStrategy
from utils.metrics import metrics

# Инструменты
from tools.code_base import search_codebase
from tools.file_tools import (
    list_dir, read_file, write_file, create_dir, 
    file_exists, dir_exists, delete_dir, delete_file, pwd
)
from tools.build_tools import npm_install, npm_build

os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

# Логи только в файл
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

# Инициализация модели
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

# Суммаризатор
summarizer = AdvancedSummarizer(model)

class AdvancedAgent:
    """Продвинутый агент с суммаризацией"""
    
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
    
    async def initialize(self) -> bool:
        """Инициализация"""
        try:
            checkpoint_path = config.CHECKPOINT_DIR / "agent_memory.sqlite"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.db = await aiosqlite.connect(str(checkpoint_path))
            self.checkpointer = AsyncSqliteSaver(self.db)
            await self.checkpointer.setup()
            
            # Загружаем промпт
            try:
                with open("./prompts/instruction.txt", "r", encoding="utf-8") as f:
                    system_prompt = f.read()
            except:
                system_prompt = "Ты - AI ассистент для работы с кодом."
            
            # Создаем агента с хуком суммаризации
            self.agent = create_react_agent(
                model=model,
                tools=tools_list,
                checkpointer=self.checkpointer,
                prompt=system_prompt,
                pre_model_hook=self._summarization_hook  # Хук суммаризации
            )
            
            self.is_initialized = True
            return True
                
        except Exception as e:
            logger.error(f"Ошибка инициализации: {e}")
            print(f"❌ Ошибка: {e}")
            return False
    
    def _summarization_hook(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        СИНХРОННЫЙ хук суммаризации.
        Вызывается перед каждым запросом к модели.
        """
        messages = state.get("messages", [])
        if not messages:
            return {}
        
        # Считаем токены
        total_tokens = sum(
            count_tokens_approximately(m.content) 
            for m in messages 
            if hasattr(m, 'content') and m.content
        )
        
        self.max_context_seen = max(self.max_context_seen, total_tokens)
        
        # Проверяем, нужна ли суммаризация
        SUMMARIZE_THRESHOLD = int(config.MAX_CONTEXT_TOKENS * config.SUMMARIZATION_TRIGGER)
        CRITICAL_THRESHOLD = int(config.MAX_CONTEXT_TOKENS * config.CRITICAL_CONTEXT_USAGE)
        
        if total_tokens > SUMMARIZE_THRESHOLD:
            # Нужна суммаризация или обрезка
            
            if total_tokens > CRITICAL_THRESHOLD:
                # Критическое переполнение - экстренная обрезка
                return self._emergency_sync_trim(messages, total_tokens)
            else:
                # Обычная суммаризация - запускаем асинхронно
                asyncio.ensure_future(self._async_summarize_and_save(messages, total_tokens))
                
                # Пока делаем легкую обрезку для текущего запроса
                return self._light_sync_trim(messages, total_tokens)
        
        return {}
    
    async def _async_summarize_and_save(self, messages: list, current_tokens: int):
        """Асинхронная суммаризация и сохранение"""
        try:
            print(f"\n📝 Суммаризация: {current_tokens:,} токенов → ", end="", flush=True)
            
            # Запускаем суммаризацию
            summarized, saved = await summarizer.summarize(
                messages,
                preserve_tools=True
            )
            
            new_tokens = sum(
                count_tokens_approximately(m.content) 
                for m in summarized 
                if hasattr(m, 'content') and m.content
            )
            
            print(f"{new_tokens:,} токенов (сэкономлено {saved:,})")
            
            # Сохраняем в чекпоинт
            await self.agent.aupdate_state(
                self.session_config,
                {"messages": summarized},
                as_node="__start__"
            )
            
            self.summarization_count += 1
            self.tokens_saved += saved
            
            logger.info(f"Суммаризация #{self.summarization_count}: {current_tokens:,} → {new_tokens:,} токенов")
            
        except Exception as e:
            logger.error(f"Ошибка суммаризации: {e}")
    
    def _emergency_sync_trim(self, messages: list, total_tokens: int) -> Dict:
        """Экстренная синхронная обрезка"""
        system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        
        # Защищаем tool chains
        protected = self._extract_protected(messages)
        
        # Остальные
        other = [m for m in messages if m not in protected and not isinstance(m, SystemMessage)]
        
        # Оставляем минимум
        kept = other[-3:] if len(other) > 3 else other
        
        new_messages = system_msgs + protected + kept
        
        new_tokens = sum(
            count_tokens_approximately(m.content) 
            for m in new_messages 
            if hasattr(m, 'content') and m.content
        )
        
        print(f"\n🚨 ЭКСТРЕННАЯ ОБРЕЗКА: {total_tokens:,} → {new_tokens:,} токенов")
        print(f"   Сообщений: {len(messages)} → {len(new_messages)}")
        
        return {"messages": new_messages}
    
    def _light_sync_trim(self, messages: list, total_tokens: int) -> Dict:
        """Легкая синхронная обрезка"""
        system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        protected = self._extract_protected(messages)
        other = [m for m in messages if m not in protected and not isinstance(m, SystemMessage)]
        
        kept = other[-10:] if len(other) > 10 else other
        
        new_messages = system_msgs + protected + kept
        
        new_tokens = sum(
            count_tokens_approximately(m.content) 
            for m in new_messages 
            if hasattr(m, 'content') and m.content
        )
        
        if new_tokens < total_tokens:
            print(f"\n✂️ Легкая обрезка: {total_tokens:,} → {new_tokens:,} токенов")
        
        return {"messages": new_messages}
    
    def _extract_protected(self, messages: list) -> list:
        """Извлекает защищенные tool chains"""
        protected = []
        protected_indices = set()
        
        for i, msg in enumerate(messages):
            if i in protected_indices:
                continue
            
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                protected.append(msg)
                protected_indices.add(i)
                
                # Ищем связанные ToolMessage
                for tc in msg.tool_calls:
                    tc_id = tc.get('id') if isinstance(tc, dict) else getattr(tc, 'id', None)
                    if tc_id:
                        for j in range(i + 1, min(i + 10, len(messages))):
                            if j in protected_indices:
                                continue
                            m = messages[j]
                            if isinstance(m, ToolMessage) and hasattr(m, 'tool_call_id') and m.tool_call_id == tc_id:
                                protected.append(m)
                                protected_indices.add(j)
        
        return protected
    
    async def process_request(self, user_input: str) -> Optional[str]:
        """Обработка запроса"""
        
        if not self.is_initialized or not self.agent:
            print("❌ Агент не инициализирован")
            return None
        
        self.total_requests += 1
        
        # Проверяем контекст и делаем суммаризацию если нужно
        await self._check_and_summarize()
        
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
                    short_name = tool_name.replace("_tool", "").replace("_", " ")
                    print(f"\n🔧 {short_name}...", end="", flush=True)
                
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
                
                memory_manager.add_interaction(
                    user_msg=user_input,
                    ai_msg=full_response,
                    metadata={'request_num': self.total_requests}
                )
                
                return full_response
            else:
                print("\n⚠️ Пустой ответ")
                return None
            
        except Exception as e:
            error_str = str(e)
            
            if "maximum context length" in error_str:
                print(f"\n⚠️ Контекст переполнен! Экстренная очистка...")
                await self._emergency_summarize()
                print("🔄 Повторите запрос")
                return None
            
            logger.error(f"Ошибка: {e}")
            print(f"\n❌ {str(e)[:200]}")
            return None
    
    async def _check_and_summarize(self):
        """Проверка контекста и суммаризация при необходимости"""
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
            
            threshold = int(config.MAX_CONTEXT_TOKENS * config.SUMMARIZATION_TRIGGER)
            
            if total_tokens > threshold:
                print(f"\n📝 Авто-суммаризация: {total_tokens:,} токенов...", end="", flush=True)
                
                summarized, saved = await summarizer.summarize(
                    messages,
                    preserve_tools=True
                )
                
                new_tokens = sum(
                    count_tokens_approximately(m.content) 
                    for m in summarized 
                    if hasattr(m, 'content') and m.content
                )
                
                print(f" → {new_tokens:,} токенов (экономия {saved:,})")
                
                await self.agent.aupdate_state(
                    self.session_config,
                    {"messages": summarized},
                    as_node="__start__"
                )
                
                self.summarization_count += 1
                self.tokens_saved += saved
                
        except Exception as e:
            logger.error(f"Ошибка проверки: {e}")
    
    async def _emergency_summarize(self):
        """Экстренная суммаризация"""
        try:
            state = await self.agent.aget_state(self.session_config)
            if not state or not state.values:
                return
            
            messages = state.values.get("messages", [])
            
            # Максимально сжимаем
            summarized, saved = await summarizer.summarize(
                messages,
                strategy=SummarizationStrategy.EXTRACTIVE,
                preserve_tools=True
            )
            
            await self.agent.aupdate_state(
                self.session_config,
                {"messages": summarized},
                as_node="__start__"
            )
            
            print(f"✅ Экстренная суммаризация: экономия {saved:,} токенов")
            
        except Exception as e:
            logger.error(f"Ошибка экстренной суммаризации: {e}")
            # Удаляем чекпоинт
            try:
                path = config.CHECKPOINT_DIR / "agent_memory.sqlite"
                if path.exists():
                    path.unlink()
                    print("⚠️ Чекпоинт удален")
            except:
                pass
    
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
    print(f"🗜️ Суммаризация: при {config.SUMMARIZATION_TRIGGER*100:.0f}% заполнения")
    print(f"🛡️ Крит. обрезка: при {config.CRITICAL_CONTEXT_USAGE*100:.0f}%")
    print("=" * 70)
    
    agent = AdvancedAgent()
    
    try:
        if not await agent.initialize():
            return 1
        
        print("✅ Агент готов")
        print("📝 Команды: exit | summarize | stats | help")
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
                    print("📝 Принудительная суммаризация...")
                    await agent._emergency_summarize()
                    continue
                
                elif user_input.lower() == "stats":
                    print(f"\n📊 СТАТИСТИКА:")
                    print(f"   Запросов: {agent.total_requests}")
                    print(f"   Суммаризаций: {agent.summarization_count}")
                    print(f"   Сэкономлено: {agent.tokens_saved:,} токенов")
                    print(f"   Пик контекста: {agent.max_context_seen:,} токенов")
                    print(f"   Память: {len(memory_manager.working_memory)} раб. / {len(memory_manager.episodic_memory)} эпизод.")
                    continue
                
                elif user_input.lower() == "help":
                    print("\n📚 КОМАНДЫ:")
                    print("   exit      - выход")
                    print("   summarize - принудительная суммаризация")
                    print("   stats     - статистика")
                    print("   help      - справка")
                    continue
                
                await agent.process_request(user_input)
                
            except KeyboardInterrupt:
                print(f"\n\n👋 Прервано. Суммаризаций: {agent.summarization_count}")
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