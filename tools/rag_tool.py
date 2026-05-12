"""Инструмент для поиска по кодовой базе"""
from langchain_core.tools import tool
from stores.rag_store import rag

@tool
def search_memory(query: str) -> str:
    """
    Поиск по истории диалога в RAG.
    Используй чтобы вспомнить: что обсуждали, какие решения приняли.
    """
    return rag.search_memory(query)

@tool
def get_decisions() -> str:
    """
    Получить последние важные решения и прогресс.
    Используй чтобы понять ЧТО УЖЕ СДЕЛАНО и НЕ ПОВТОРЯТЬСЯ.
    """
    return rag.get_decisions()

@tool
def search_codebase(query: str) -> str:
    """
    Поиск по коду VB6 проекта.
    Ищи функции, модули, формы, SQL запросы.
    """
    return rag.search_code(query)

@tool
def search_context(query: str) -> str:
    """
    Комбинированный поиск: код + история.
    Используй когда нужно понять и код, и контекст обсуждения.
    """
    return rag.get_recent_context(query)