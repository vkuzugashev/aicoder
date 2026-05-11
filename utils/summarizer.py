"""Умная суммаризация диалогов"""
from typing import List, Tuple
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately

class Summarizer:
    def __init__(self, model):
        self.model = model
        self.count = 0
        self.tokens_saved = 0
        self.SUMMARY_PROMPT = None
        try:
            with open("prompts/summary_prompt.txt", "r", encoding="utf-8") as f:
                self.SUMMARY_PROMPT = f.read()
        except:
            self.SUMMARY_PROMPT = """
            Сожми диалог в резюме (до 300 слов). Сохрани:
            - Созданные/изменённые файлы
            - Выполненные команды
            - Принятые решения
            - Достигнутый прогресс
            """
    
    async def summarize(self, messages: List[BaseMessage]) -> Tuple[str, List[BaseMessage], int]:
        """Суммаризация сообщений"""
        if len(messages) < 10:
            return "", messages, 0
        
        old_tokens = sum(
            count_tokens_approximately(m.content)
            for m in messages if hasattr(m, 'content')
        )
        
        # Отбираем старые сообщения
        to_summarize = messages[:-5]
        to_keep = messages[-5:]
        
        conversation = "\n".join([
            f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content[:300]}"
            for m in to_summarize[-15:]
        ])
        
        try:
            response = await self.model.ainvoke([
                SystemMessage(content=self.SUMMARY_PROMPT),
                HumanMessage(content=f"Диалог:\n{conversation}\n\nРезюме:")
            ])
            summary = response.content
            
            new_messages = [
                SystemMessage(content=f"📝 История диалога:\n{summary}")
            ] + list(to_keep)
            
            new_tokens = sum(
                count_tokens_approximately(m.content)
                for m in new_messages if hasattr(m, 'content')
            )
            
            saved = old_tokens - new_tokens
            self.count += 1
            self.tokens_saved += saved
            
            return summary, new_messages, saved
            
        except Exception:
            return "", messages, 0