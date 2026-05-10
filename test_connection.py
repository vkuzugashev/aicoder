#!/usr/bin/env python3
"""Проверка соединения с моделью"""
import os
import asyncio
import dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

dotenv.load_dotenv()

# Проверяем разные варианты API ключа
MODEL_URL = os.environ.get("MODEL_URL", "https://foundation-models.api.cloud.ru/v1")
API_KEY = os.environ.get("OPENAI_API_KEY") or "no-key-required"

async def test_connection():
    print(f"🔗 Проверка соединения с {MODEL_URL}")
    print(f"🔑 API Key: {'установлен' if API_KEY != 'no-key-required' else 'не требуется'}")
    
    try:
        # Пробуем с API ключом
        model = init_chat_model(
            model="Qwen/Qwen3-Coder-Next",
            base_url=MODEL_URL,
            api_key=API_KEY,  # Добавляем API ключ
            model_provider='openai',
            temperature=0.5,
            timeout=30,
            max_tokens=100,
            streaming=False
        )
        
        print("📡 Отправка тестового запроса...")
        response = await model.ainvoke([HumanMessage(content="Привет! Ответь кратко: ты работаешь?")])
        
        print(f"✅ Ответ получен: {response.content[:200]}")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка соединения: {e}")
        
        # Пробуем альтернативные URL если есть
        alt_urls = [
            "http://localhost:8000/v1",
            "http://localhost:11434/v1",  # Ollama
            "http://localhost:1234/v1",   # LM Studio
        ]
        
        print("\n🔍 Пробую альтернативные URL...")
        for url in alt_urls:
            try:
                print(f"   Проверяю {url}...")
                model = init_chat_model(
                    model="Qwen/Qwen3-Coder-Next",
                    base_url=url,
                    api_key="not-needed",
                    model_provider='openai',
                    temperature=0.5,
                    timeout=10,
                    max_tokens=50,
                    streaming=False
                )
                
                response = await model.ainvoke([HumanMessage(content="test")])
                print(f"   ✅ {url} работает!")
                return True
                
            except Exception as e2:
                print(f"   ❌ {url}: {str(e2)[:100]}")
        
        return False

if __name__ == "__main__":
    result = asyncio.run(test_connection())
    
    if result:
        print("\n✅ Соединение работает корректно")
    else:
        print("\n❌ Проблемы с соединением")
        print("\n📋 РЕКОМЕНДАЦИИ:")
        print("1. Установите переменную окружения:")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        print("   или")
        print("   export API_KEY='your-api-key-here'")
        print("\n2. Для локальной модели укажите:")
        print("   export MODEL_URL='http://localhost:8000/v1'")
        print("\n3. Создайте .env файл в корне проекта:")
        print("   OPENAI_API_KEY=your-api-key-here")
        print("   MODEL_URL=https://foundation-models.api.cloud.ru/v1")