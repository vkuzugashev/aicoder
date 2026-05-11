"""
Продвинутая система суммаризации с адаптивными стратегиями
"""
import hashlib
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage, 
    SystemMessage, ToolMessage
)
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.language_models import BaseChatModel

import sys

from utils.token_counter import count_tokens_for_qwen
sys.path.append('..')
from config import config

class SummarizationStrategy(Enum):
    """Стратегии суммаризации"""
    HIERARCHICAL = "hierarchical"  # Иерархическая для длинных диалогов
    EXTRACTIVE = "extractive"      # Экстрактивная для коротких
    PROGRESSIVE = "progressive"    # Прогрессивная для средних
    HYBRID = "hybrid"             # Гибридная для смешанного контента

class AdvancedSummarizer:
    """Продвинутый суммаризатор с выбором стратегии"""
    
    def __init__(self, model: BaseChatModel):
        self.model = model
        self.summary_cache = {}
        self.strategy_stats = {s: 0 for s in SummarizationStrategy}
    
    def select_strategy(self, messages: List[BaseMessage]) -> SummarizationStrategy:
        """Выбор оптимальной стратегии суммаризации"""
        if not messages:
            return SummarizationStrategy.EXTRACTIVE
        
        # Анализируем характеристики
        # total_tokens = sum(count_tokens_approximately(m.content) for m in messages if hasattr(m, 'content'))
        total_tokens = sum(count_tokens_for_qwen(m.content) for m in messages if hasattr(m, 'content'))
        avg_tokens_per_msg = total_tokens / len(messages) if messages else 0
        has_tool_calls = any(hasattr(m, 'tool_calls') and m.tool_calls for m in messages)
        has_code = any('```' in (m.content or '') for m in messages)
        
        # Выбираем стратегию
        if total_tokens > 50000 or avg_tokens_per_msg > 2000:
            return SummarizationStrategy.HIERARCHICAL
        elif has_tool_calls and has_code:
            return SummarizationStrategy.HYBRID
        elif len(messages) > 30:
            return SummarizationStrategy.PROGRESSIVE
        else:
            return SummarizationStrategy.EXTRACTIVE
    
    async def summarize(self, 
                       messages: List[BaseMessage], 
                       strategy: SummarizationStrategy = None,
                       preserve_tools: bool = True) -> Tuple[List[BaseMessage], int]:
        """
        Основной метод суммаризации
        
        Returns:
            Tuple[List[BaseMessage], int]: (сжатые сообщения, сэкономленные токены)
        """
        if not messages:
            return messages, 0
        
        if strategy is None:
            strategy = self.select_strategy(messages)
        
        self.strategy_stats[strategy] += 1
        
        # Вычисляем начальные токены
        initial_tokens = sum(count_tokens_approximately(m.content) for m in messages if hasattr(m, 'content'))
        
        # Применяем выбранную стратегию
        if strategy == SummarizationStrategy.HIERARCHICAL:
            result = await self._hierarchical_summarize(messages, preserve_tools)
        elif strategy == SummarizationStrategy.PROGRESSIVE:
            result = await self._progressive_summarize(messages, preserve_tools)
        elif strategy == SummarizationStrategy.HYBRID:
            result = await self._hybrid_summarize(messages, preserve_tools)
        else:
            result = await self._extractive_summarize(messages, preserve_tools)
        
        # Вычисляем сэкономленные токены
        final_tokens = sum(count_tokens_approximately(m.content) for m in result if hasattr(m, 'content'))
        saved_tokens = initial_tokens - final_tokens
        
        return result, saved_tokens
    
    async def _hierarchical_summarize(self, messages: List[BaseMessage], preserve_tools: bool) -> List[BaseMessage]:
        """Иерархическая суммаризация"""
        # Разделяем на группы
        system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        tool_chain_msgs = self._extract_tool_chains(messages) if preserve_tools else []
        other_msgs = [m for m in messages if not isinstance(m, SystemMessage) and m not in tool_chain_msgs]
        
        # Группируем по темам
        topic_groups = self._group_by_topics(other_msgs)
        
        result = system_msgs.copy()
        
        # Суммаризируем каждую тему
        for topic, group in topic_groups.items():
            if len(group) > 5:  # Только группы > 5 сообщений
                summary = await self._create_topic_summary(group, topic)
                result.append(SystemMessage(content=f"[{topic}] {summary}"))
            else:
                result.extend(group)
        
        # Добавляем защищенные tool chains
        result.extend(tool_chain_msgs)
        
        return result
    
    async def _progressive_summarize(self, messages: List[BaseMessage], preserve_tools: bool) -> List[BaseMessage]:
        """Прогрессивная суммаризация с сохранением последних сообщений"""
        # Защищаем tool chains
        protected = self._extract_tool_chains(messages) if preserve_tools else []
        unprotected = [m for m in messages if m not in protected]
        
        # Оставляем последние сообщения нетронутыми
        keep_last = min(config.SUMMARIZATION_KEEP, len(unprotected))
        to_summarize = unprotected[:-keep_last]
        to_keep = unprotected[-keep_last:]
        
        if not to_summarize:
            return messages
        
        # Суммаризируем чанками
        chunk_size = 10
        summaries = []
        
        for i in range(0, len(to_summarize), chunk_size):
            chunk = to_summarize[i:i+chunk_size]
            if len(chunk) > 5:
                summary = await self._summarize_chunk(chunk, i // chunk_size + 1)
                summaries.append(SystemMessage(content=summary))
        
        result = [m for m in messages if isinstance(m, SystemMessage)]
        result.extend(summaries)
        result.extend(to_keep)
        result.extend(protected)
        
        return result
    
    async def _extractive_summarize(self, messages: List[BaseMessage], preserve_tools: bool) -> List[BaseMessage]:
        """Экстрактивная суммаризация - выбираем ключевые сообщения"""
        # Защищаем важные сообщения
        system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        tool_chains = self._extract_tool_chains(messages) if preserve_tools else []
        
        # Оцениваем важность остальных
        other_msgs = [m for m in messages if m not in tool_chains and not isinstance(m, SystemMessage)]
        scored_messages = [(self._score_importance(m), m) for m in other_msgs]
        
        # Сортируем по важности
        scored_messages.sort(key=lambda x: x[0], reverse=True)
        
        # Выбираем топ-N важных
        keep_count = min(config.WORKING_MEMORY_SIZE, len(scored_messages))
        important_msgs = [msg for score, msg in scored_messages[:keep_count]]
        
        # Сортируем по времени для сохранения порядка
        result = system_msgs + sorted(important_msgs, key=lambda m: getattr(m, 'timestamp', 0))
        result.extend(tool_chains)
        
        return result
    
    async def _hybrid_summarize(self, messages: List[BaseMessage], preserve_tools: bool) -> List[BaseMessage]:
        """Гибридная суммаризация - комбинация подходов"""
        # Для tool calls используем экстрактивный подход
        # Для остального - прогрессивный
        
        tool_messages = self._extract_tool_chains(messages)
        non_tool_messages = [m for m in messages if m not in tool_messages]
        
        # Обрабатываем tool messages экстрактивно
        processed_tools = await self._extractive_summarize(tool_messages, False)
        
        # Обрабатываем остальное прогрессивно
        processed_other = await self._progressive_summarize(non_tool_messages, False)
        
        # Объединяем
        result = processed_other
        result.extend([m for m in processed_tools if m not in result])
        
        return result
    
    def _extract_tool_chains(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """Извлечение цепочек tool calls с защитой от разрыва"""
        protected = set()
        
        for i, msg in enumerate(messages):
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                # Защищаем вызов
                protected.add(i)
                
                # Ищем соответствующие ToolMessage
                for j in range(i + 1, min(i + 5, len(messages))):
                    if isinstance(messages[j], ToolMessage):
                        if messages[j].tool_call_id in [tc.get('id') for tc in msg.tool_calls]:
                            protected.add(j)
        
        return [messages[i] for i in sorted(protected)]
    
    def _group_by_topics(self, messages: List[BaseMessage]) -> Dict[str, List[BaseMessage]]:
        """Группировка сообщений по темам"""
        groups = {"Основное": []}
        
        for msg in messages:
            if not hasattr(msg, 'content'):
                continue
            
            content = msg.content.lower()
            
            # Определяем тему
            if any(w in content for w in ['vb6', 'visual basic', 'legacy', 'src/']):
                topic = "VB6 Legacy"
            elif any(w in content for w in ['nestjs', 'nest', 'api/', 'controller']):
                topic = "NestJS API"
            elif any(w in content for w in ['react', 'frontend', 'client/', 'component']):
                topic = "React Frontend"
            elif any(w in content for w in ['database', 'oracle', 'sql', 'query']):
                topic = "Database"
            elif any(w in content for w in ['ошибка', 'error', 'fix', 'bug']):
                topic = "Errors & Fixes"
            else:
                topic = "Основное"
            
            if topic not in groups:
                groups[topic] = []
            groups[topic].append(msg)
        
        return groups
    
    def _score_importance(self, message: BaseMessage) -> float:
        """Оценка важности сообщения"""
        if not hasattr(message, 'content'):
            return 0.0
        
        content = message.content
        score = 0.3  # Базовая важность
        
        # Тип сообщения
        if isinstance(message, HumanMessage):
            score += 0.2
        elif isinstance(message, SystemMessage):
            score += 0.4
        
        # Содержание
        if '```' in content:  # Код
            score += 0.3
        if any(w in content.lower() for w in ['ошибка', 'error', 'важно', 'important']):
            score += 0.3
        if len(content) > 200:  # Длинные сообщения
            score += 0.1
        
        return min(score, 1.0)
    
    async def _create_topic_summary(self, messages: List[BaseMessage], topic: str) -> str:
        """Создание суммаризации по теме"""
        # Проверяем кэш
        cache_key = hashlib.md5(f"{topic}_{len(messages)}".encode()).hexdigest()
        if cache_key in self.summary_cache:
            return self.summary_cache[cache_key]
        
        try:
            # Формируем контекст
            context = "\n".join([
                f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content[:200]}"
                for m in messages[-10:]  # Последние 10 сообщений
            ])
            
            with open("./prompts/summary_prompt.txt", "r", encoding="utf-8") as f:
                base_prompt = f.read()
            
            prompt = f"""{base_prompt}
            
Тема: {topic}
Сообщения для сжатия:
{context}

Создай краткое резюме (2-3 предложения):"""
            
            response = await self.model.ainvoke([HumanMessage(content=prompt)])
            summary = response.content[:500]
            
            self.summary_cache[cache_key] = summary
            return summary
            
        except Exception as e:
            print(f"⚠️ Ошибка суммаризации темы {topic}: {e}")
            return f"Обсуждение темы '{topic}' ({len(messages)} сообщений)"
    
    async def _summarize_chunk(self, messages: List[BaseMessage], chunk_num: int) -> str:
        """Суммаризация чанка сообщений"""
        context = "\n".join([
            f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content[:150]}"
            for m in messages[-5:]
        ])
        
        prompt = f"""Кратко опиши основную мысль этого фрагмента диалога (часть {chunk_num}):

{context}

Ответь одним предложением:"""
        
        try:
            response = await self.model.ainvoke([HumanMessage(content=prompt)])
            return response.content[:300]
        except Exception as e:
            print(f"⚠️ Ошибка суммаризации чанка {chunk_num}: {e}")
            return f"Фрагмент диалога #{chunk_num} ({len(messages)} сообщений)"
    
    def get_stats(self) -> Dict:
        """Получение статистики использования стратегий"""
        return {
            'strategies_used': dict(self.strategy_stats),
            'cache_size': len(self.summary_cache)
        }