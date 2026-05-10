"""
Продвинутый AI агент с разделением логов и ответов
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

# Импорты инструментов
from tools.code_base import search_codebase
from tools.file_tools import (
    list_dir, read_file, write_file, create_dir, 
    file_exists, dir_exists, delete_dir, delete_file, pwd
)
from tools.build_tools import npm_install, npm_build

# Настройка окружения
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

# ===== НАСТРОЙКА ЛОГИРОВАНИЯ (только в файл, не в консоль) =====
import logging

# Создаем директорию для логов
config.LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Настраиваем логирование ТОЛЬКО в файл
logging.basicConfig(
    level=logging.WARNING,  # Высокий уровень - меньше логов
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(
            config.LOGS_DIR / f"agent_{datetime.now().strftime('%Y%m%d')}.log",
            encoding='utf-8'
        )
        # НЕ выводим логи в консоль!
    ]
)

# Отключаем логи от библиотек
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("langgraph").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Список инструментов
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
        max_retries=config.MAX_RETRIES,
        streaming=True,
    )
    logger.info(f"Модель инициализирована: {config.MODEL_NAME}")
except Exception as e:
    logger.error(f"Ошибка инициализации модели: {e}")
    print(f"❌ Ошибка инициализации модели: {e}")
    sys.exit(1)

# Инициализация суммаризатора
summarizer = AdvancedSummarizer(model)

class AdvancedAgent:
    """Агент с чистым выводом ответов"""
    
    def __init__(self):
        self.agent = None
        self.checkpointer = None
        self.db = None
        self.session_config = {
            "configurable": {"thread_id": "advanced-session"},
            "recursion_limit": config.DEFAULT_RECURSION_LIMIT
        }
        self.context_stats = {
            'total_summarizations': 0,
            'total_tokens_saved': 0,
            'connection_errors': 0
        }
        self.is_initialized = False
    
    async def initialize(self) -> bool:
        """Инициализация агента"""
        try:
            checkpoint_path = config.CHECKPOINT_DIR / "agent_memory.sqlite"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.db = await aiosqlite.connect(str(checkpoint_path))
            self.checkpointer = AsyncSqliteSaver(self.db)
            await self.checkpointer.setup()
            
            # Загружаем системный промпт
            try:
                with open("./prompts/instruction.txt", "r", encoding="utf-8") as f:
                    system_prompt = f.read()
            except FileNotFoundError:
                system_prompt = "Ты - AI ассистент для работы с кодом."
            
            # Создаем агента
            self.agent = create_react_agent(
                model=model,
                tools=tools_list,
                checkpointer=self.checkpointer,
                prompt=system_prompt
            )
            
            self.is_initialized = True
            logger.info("Агент инициализирован")
            return True
                
        except Exception as e:
            logger.error(f"Ошибка инициализации: {e}")
            print(f"❌ Ошибка инициализации: {e}")
            return False
    
    async def process_request(self, user_input: str) -> Optional[str]:
        """Обработка запроса с чистым выводом"""
        
        if not self.is_initialized or not self.agent:
            print("❌ Агент не инициализирован")
            return None
        
        request_start = time.time()
        
        try:
            full_response = ""
            tool_messages = []  # Для сбора сообщений от инструментов
            
            print()  # Пустая строка перед ответом
            
            # Стриминг ответа
            async for event in self.agent.astream_events(
                {"messages": [HumanMessage(content=user_input)]},
                config=self.session_config,
                version="v2"
            ):
                kind = event["event"]
                
                # Вывод ответа модели (без логов)
                if kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if hasattr(chunk, 'content') and chunk.content:
                        if not full_response:
                            print("🤖 ", end="", flush=True)
                        print(chunk.content, end="", flush=True)
                        full_response += chunk.content
                
                # Собираем информацию о tool calls (но не выводим логи)
                elif kind == "on_tool_start":
                    tool_name = event.get("name", "unknown")
                    tool_input = event.get("data", {}).get("input", "")
                    # Тихо логируем в файл
                    logger.debug(f"Tool: {tool_name}, Input: {str(tool_input)[:200]}")
                    
                    # Краткий индикатор в консоли
                    short_name = tool_name.replace("_tool", "").replace("_", " ")
                    print(f"\n🔧 {short_name}...", end="", flush=True)
                
                elif kind == "on_tool_end":
                    tool_output = event.get("data", {}).get("output", "")
                    # Тихо логируем результат
                    logger.debug(f"Tool output: {str(tool_output)[:200]}")
                    
                    # Показываем краткий результат
                    if tool_output:
                        output_str = str(tool_output)
                        if "✅" in output_str:
                            print(" ✅", end="", flush=True)
                        elif "❌" in output_str or "Ошибка" in output_str:
                            print(" ❌", end="", flush=True)
                        else:
                            print(" ✓", end="", flush=True)
            
            if full_response:
                print("\n")
                logger.info(f"Ответ: {len(full_response)} символов")
                
                # Обновляем память
                memory_manager.add_interaction(
                    user_msg=user_input,
                    ai_msg=full_response
                )
                
                return full_response
            else:
                print("\n⚠️ Пустой ответ")
                return None
            
        except Exception as e:
            logger.error(f"Ошибка: {e}")
            print(f"\n❌ Ошибка: {str(e)[:200]}")
            return None
    
    async def close(self):
        """Закрытие соединений"""
        try:
            if self.db:
                await self.db.close()
        except:
            pass

# ===== Основная функция =====
async def run_advanced_agent():
    """Запуск агента с чистым выводом"""
    
    # Очищаем экран
    os.system('clear' if os.name != 'nt' else 'cls')
    
    print("=" * 70)
    print("🤖 АГЕНТ КОДЕР v3.0")
    print(f"📋 Миссия: Переписывание VB6 → NestJS + React")
    print(f"🔗 Модель: {config.MODEL_NAME}")
    print(f"💾 Контекст: до {config.MAX_CONTEXT_TOKENS:,} токенов")
    print("=" * 70)
    
    # Создаем агента
    agent = AdvancedAgent()
    
    try:
        if not await agent.initialize():
            return 1
        
        print("✅ Агент готов")
        print("📝 Команды: exit | stats | help")
        print("-" * 70)
        
        while True:
            try:
                user_input = input("\n👤 Вы: ").strip()
                
                if not user_input:
                    continue
                
                # Команды
                if user_input.lower() in ["exit", "quit"]:
                    print("\n👋 До свидания!")
                    break
                
                elif user_input.lower() == "stats":
                    print("\n📊 Статистика:")
                    print(f"   Ошибок соединения: {agent.context_stats['connection_errors']}")
                    print(f"   Суммаризаций: {agent.context_stats['total_summarizations']}")
                    continue
                
                elif user_input.lower() == "help":
                    print("\n📚 КОМАНДЫ:")
                    print("   exit  - выход")
                    print("   stats - статистика")
                    print("   help  - справка")
                    continue
                
                # Обработка запроса
                await agent.process_request(user_input)
                
            except KeyboardInterrupt:
                print("\n\n👋 Прервано")
                break
            except Exception as e:
                logger.error(f"Ошибка цикла: {e}")
                print(f"\n❌ Ошибка: {e}")
    
    finally:
        await agent.close()
        metrics.save_metrics()
    
    return 0

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    try:
        exit_code = asyncio.run(run_advanced_agent())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 Завершено")
        sys.exit(0)