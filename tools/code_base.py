from langchain_core.tools import tool
from stores.rag_store import get_retriever


@tool
def search_codebase(query: str) -> str:
    """
    ИЩЕТ РЕЛЕВАНТНЫЕ ФРАГМЕНТЫ КОДА по смыслу.
    Используй этот инструмент, когда тебе нужно найти, где в проекте реализована какая-то функция,
    найти все места, использующие определенную библиотеку, или вспомнить логику работы какого-то модуля.
    """
    retriever = get_retriever()

    docs = retriever.invoke(query)
    if not docs:
        return "Ничего не найдено по вашему запросу."

    context = "\n\n---\n\n".join([f"Файл: {doc.metadata['source']}\nСодержание:\n{doc.page_content}" for doc in docs])
    return f"Вот наиболее релевантные фрагменты из кодовой базы:\n\n{context}"