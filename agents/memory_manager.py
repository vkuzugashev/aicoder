"""
Продвинутый менеджер памяти с иерархической структурой
"""
import hashlib
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict

from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage, 
    SystemMessage, ToolMessage
)
from langchain_core.messages.utils import count_tokens_approximately

import sys
sys.path.append('..')
from config import config

@dataclass
class MemoryEntry:
    """Запись в памяти агента"""
    content: str
    timestamp: datetime
    importance: float
    message_type: str
    metadata: Dict = field(default_factory=dict)
    token_count: int = 0

class PriorityCache:
    """LRU кэш с приоритетами для важных сообщений"""
    
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.priority_cache = {}  # Защищенные записи
    
    def add(self, key: str, value: Any, priority: float = 0):
        """Добавление с приоритетом"""
        if priority > 0.7:  # Высокоприоритетные защищаем
            self.priority_cache[key] = value
        
        self.cache[key] = value
        if len(self.cache) > self.max_size:
            # Удаляем старые, но не приоритетные
            for old_key in list(self.cache.keys()):
                if old_key not in self.priority_cache:
                    del self.cache[old_key]
                    break
    
    def get(self, key: str) -> Optional[Any]:
        """Получение с обновлением LRU"""
        if key in self.cache:
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        return None

class MemoryManager:
    """Иерархическая система памяти агента"""
    
    def __init__(self):
        # Уровни памяти
        self.working_memory: List[MemoryEntry] = []  # Оперативная
        self.episodic_memory: List[Dict] = []  # Эпизодическая
        self.semantic_memory: Dict[str, Any] = {}  # Семантическая
        
        # Кэши
        self.summary_cache = PriorityCache(50)  # Кэш суммаризаций
        self.code_context_cache = PriorityCache(30)  # Кэш контекста кода
        
        # Статистика
        self.stats = {
            'total_interactions': 0,
            'total_summarizations': 0,
            'memory_saved_tokens': 0
        }
    
    def add_interaction(self, 
                       user_msg: str, 
                       ai_msg: str, 
                       metadata: Dict = None,
                       is_important: bool = False):
        """Добавление взаимодействия с оценкой важности"""
        
        # Оцениваем важность
        importance = self._calculate_importance(user_msg, metadata)
        if is_important:
            importance = max(importance, 0.8)
        
        # Создаем запись
        entry = MemoryEntry(
            content=user_msg[:500],  # Храним сокращенную версию
            timestamp=datetime.now(),
            importance=importance,
            message_type='interaction',
            metadata={
                'ai_response': ai_msg[:300],
                **(metadata or {})
            },
            token_count=count_tokens_approximately(user_msg + ai_msg)
        )
        
        self.working_memory.append(entry)
        self.stats['total_interactions'] += 1
        
        # Если переполнение - архивируем
        if len(self.working_memory) > config.WORKING_MEMORY_SIZE:
            self._archive_to_episodic()
    
    def _calculate_importance(self, content: str, metadata: Dict = None) -> float:
        """Оценка важности сообщения"""
        if not content:
            return 0.3
        
        content_lower = content.lower()
        importance = 0.5  # Базовая важность
        
        # Проверяем паттерны
        for category, patterns in config.PRIORITY_PATTERNS.items():
            for pattern in patterns:
                if pattern in content_lower:
                    importance += 0.1
                    break
        
        # Учитываем длину
        if len(content) > 200:
            importance += 0.1
        
        # Учитываем наличие кода
        if re.search(r'```|\b(function|class|const|let|var)\b', content):
            importance += 0.15
        
        # Учитываем метаданные
        if metadata:
            if metadata.get('file_created'):
                importance += 0.2
            if metadata.get('error_encountered'):
                importance += 0.3
        
        return min(importance, 1.0)
    
    def _archive_to_episodic(self):
        """Архивация в эпизодическую память с группировкой"""
        
        # Берем старые записи
        old_entries = self.working_memory[:-config.WORKING_MEMORY_SIZE]
        
        # Группируем по временным окнам
        time_groups = self._group_by_time(old_entries, window_minutes=5)
        
        for time_window, entries in time_groups.items():
            # Создаем эпизод
            episode = {
                'timestamp': time_window,
                'entries': entries,
                'summary': self._quick_summarize_entries(entries),
                'importance': max(e.importance for e in entries),
                'key_files': self._extract_key_files(entries)
            }
            
            self.episodic_memory.append(episode)
        
        # Ограничиваем размер
        if len(self.episodic_memory) > config.EPISODIC_MEMORY_SIZE:
            # Оставляем самые важные
            self.episodic_memory.sort(key=lambda x: x['importance'], reverse=True)
            self.episodic_memory = self.episodic_memory[:config.EPISODIC_MEMORY_SIZE]
        
        # Очищаем рабочую память
        self.working_memory = self.working_memory[-config.WORKING_MEMORY_SIZE:]
    
    def _group_by_time(self, entries: List[MemoryEntry], window_minutes: int = 5) -> Dict:
        """Группировка записей по временным окнам"""
        if not entries:
            return {}
        
        groups = {}
        current_window = entries[0].timestamp
        current_group = []
        
        for entry in entries:
            if (entry.timestamp - current_window).seconds > window_minutes * 60:
                # Новое временное окно
                if current_group:
                    groups[current_window.isoformat()] = current_group
                current_window = entry.timestamp
                current_group = [entry]
            else:
                current_group.append(entry)
        
        if current_group:
            groups[current_window.isoformat()] = current_group
        
        return groups
    
    def _quick_summarize_entries(self, entries: List[MemoryEntry]) -> str:
        """Быстрая суммаризация без LLM"""
        if not entries:
            return ""
        
        # Собираем ключевые действия
        actions = []
        files_mentioned = set()
        
        for entry in entries:
            content = entry.content.lower()
            
            # Определяем тип действия
            if any(w in content for w in ['созда', 'create', 'write']):
                actions.append("создание")
            elif any(w in content for w in ['измен', 'update', 'modify']):
                actions.append("изменение")
            elif any(w in content for w in ['поиск', 'search', 'find']):
                actions.append("поиск")
            elif any(w in content for w in ['ошибка', 'error', 'fix']):
                actions.append("исправление")
            
            # Извлекаем файлы
            files = re.findall(r'(?:api|client|src)/[\w/]+\.[\w]+', content)
            files_mentioned.update(files)
        
        summary_parts = []
        if actions:
            summary_parts.append(f"Действия: {', '.join(set(actions))}")
        if files_mentioned:
            summary_parts.append(f"Файлы: {', '.join(list(files_mentioned)[:5])}")
        
        return " | ".join(summary_parts) if summary_parts else f"Группа из {len(entries)} взаимодействий"
    
    def _extract_key_files(self, entries: List[MemoryEntry]) -> List[str]:
        """Извлечение ключевых файлов из записей"""
        files = set()
        for entry in entries:
            # Ищем пути к файлам
            found = re.findall(r'[\w/]+\.\w{2,4}', entry.content)
            files.update(found)
            
            # Из метаданных
            if entry.metadata.get('file_path'):
                files.add(entry.metadata['file_path'])
        
        return list(files)[:10]
    
    def get_optimized_context(self, query: str, max_tokens: int = None) -> str:
        """Формирование оптимизированного контекста для LLM"""
        if max_tokens is None:
            max_tokens = int(config.MAX_CONTEXT_TOKENS * config.TARGET_CONTEXT_USAGE)
        
        context_parts = []
        current_tokens = 0
        
        # 1. Добавляем релевантные эпизоды (самые важные)
        if self.episodic_memory:
            relevant_episodes = self._find_relevant_episodes(query)
            for episode in relevant_episodes[:3]:  # Топ-3 эпизода
                summary = episode.get('summary', '')
                tokens = count_tokens_approximately(summary)
                if current_tokens + tokens <= max_tokens * 0.3:  # 30% на эпизоды
                    context_parts.append(f"[Контекст] {summary}")
                    current_tokens += tokens
        
        # 2. Добавляем последние взаимодействия
        recent = self.working_memory[-config.CONTEXT_WINDOW_SIZE:]
        for entry in reversed(recent):  # Самые новые в конец
            if entry.importance > 0.6:  # Только важные
                context_text = f"User: {entry.content}"
                tokens = count_tokens_approximately(context_text)
                if current_tokens + tokens <= max_tokens * 0.5:  # 50% на диалог
                    context_parts.append(context_text)
                    current_tokens += tokens
        
        return "\n".join(context_parts)
    
    def _find_relevant_episodes(self, query: str) -> List[Dict]:
        """Поиск релевантных эпизодов"""
        query_terms = set(query.lower().split())
        scored_episodes = []
        
        for episode in self.episodic_memory:
            # Простой scoring по пересечению терминов
            episode_text = episode.get('summary', '').lower()
            overlap = len(query_terms & set(episode_text.split()))
            importance = episode.get('importance', 0)
            
            score = overlap * 0.3 + importance * 0.7
            scored_episodes.append((score, episode))
        
        scored_episodes.sort(key=lambda x: x[0], reverse=True)
        return [ep for score, ep in scored_episodes[:5]]
    
    def get_summary_for_checkpoint(self) -> str:
        """Создание компактного представления для checkpoint"""
        summary_parts = []
        
        # Статистика
        summary_parts.append(f"Взаимодействий: {self.stats['total_interactions']}")
        
        # Последние действия
        if self.working_memory:
            recent = self.working_memory[-5:]
            actions = [self._quick_summarize_entries([e]) for e in recent]
            summary_parts.append("Последние: " + " | ".join(actions))
        
        # Ключевые файлы
        all_files = set()
        for entry in self.working_memory[-10:]:
            all_files.update(self._extract_key_files([entry]))
        if all_files:
            summary_parts.append("Файлы: " + ", ".join(list(all_files)[:5]))
        
        return "\n".join(summary_parts)

# Глобальный экземпляр
memory_manager = MemoryManager()