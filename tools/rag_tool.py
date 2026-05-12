"""Инструмент для поиска по кодовой базе"""
from langchain_core.tools import tool
from stores.rag_store import rag_store

@tool
def search_codebase(query: str) -> str:
    """
    Поиск по кодовой базе VB6 проекта.
    Используй для поиска функций, модулей, SQL запросов, контролов.
    """
    return rag_store.search_formatted(query)