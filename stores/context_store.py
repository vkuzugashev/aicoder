"""
Хранилище контекста для сохранения и восстановления состояния агента
"""
import json
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import hashlib

from langchain_core.messages import BaseMessage

import sys
sys.path.append('..')
from config import config

class ContextStore:
    """Хранилище контекста с версионированием"""
    
    def __init__(self):
        self.store_dir = config.CACHE_DIR / "contexts"
        self.store_dir.mkdir(parents=True, exist_ok=True)
        
        # Индекс сохраненных контекстов
        self.index_file = self.store_dir / "index.json"
        self.index = self._load_index()
    
    def _load_index(self) -> Dict:
        """Загрузка индекса"""
        if self.index_file.exists():
            with open(self.index_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            'version': '1.0',
            'contexts': {},
            'last_updated': None
        }
    
    def _save_index(self):
        """Сохранение индекса"""
        self.index['last_updated'] = datetime.now().isoformat()
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(self.index, f, indent=2, ensure_ascii=False)
    
    def save_context(self, 
                    context_id: str,
                    messages: List[BaseMessage],
                    metadata: Dict = None) -> str:
        """
        Сохранение контекста
        
        Args:
            context_id: Идентификатор контекста
            messages: Список сообщений
            metadata: Дополнительные метаданные
        
        Returns:
            Путь к сохраненному файлу
        """
        # Создаем версионированное имя файла
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        content_hash = self._hash_messages(messages)
        filename = f"{context_id}_{timestamp}_{content_hash[:8]}.pkl"
        filepath = self.store_dir / filename
        
        # Подготавливаем данные
        data = {
            'messages': messages,
            'metadata': {
                'context_id': context_id,
                'timestamp': timestamp,
                'message_count': len(messages),
                'hash': content_hash,
                **(metadata or {})
            }
        }
        
        # Сохраняем
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        # Обновляем индекс
        self.index['contexts'][context_id] = {
            'latest_file': filename,
            'timestamp': timestamp,
            'message_count': len(messages),
            'hash': content_hash,
            'metadata': metadata
        }
        self._save_index()
        
        return str(filepath)
    
    def load_context(self, context_id: str) -> Optional[Dict]:
        """Загрузка контекста"""
        if context_id not in self.index['contexts']:
            return None
        
        context_info = self.index['contexts'][context_id]
        filepath = self.store_dir / context_info['latest_file']
        
        if not filepath.exists():
            return None
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        return data
    
    def get_context_summary(self, context_id: str) -> Optional[Dict]:
        """Получение информации о контексте без загрузки"""
        if context_id in self.index['contexts']:
            return self.index['contexts'][context_id]
        return None
    
    def list_contexts(self) -> List[Dict]:
        """Список всех сохраненных контекстов"""
        contexts = []
        for ctx_id, info in self.index['contexts'].items():
            contexts.append({
                'id': ctx_id,
                **info
            })
        
        return sorted(contexts, key=lambda x: x['timestamp'], reverse=True)
    
    def _hash_messages(self, messages: List[BaseMessage]) -> str:
        """Создание хеша сообщений для дедупликации"""
        content = ""
        for msg in messages[-10:]:  # Хешируем последние 10 сообщений
            if hasattr(msg, 'content'):
                content += msg.content[:100]
        
        return hashlib.md5(content.encode()).hexdigest()
    
    def cleanup_old_contexts(self, max_age_days: int = 7, max_count: int = 50):
        """Очистка старых контекстов"""
        import time
        from datetime import timedelta
        
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        to_remove = []
        
        for ctx_id, info in self.index['contexts'].items():
            ctx_date = datetime.fromisoformat(info['timestamp'])
            if ctx_date < cutoff_date:
                # Удаляем файл
                filepath = self.store_dir / info['latest_file']
                if filepath.exists():
                    filepath.unlink()
                to_remove.append(ctx_id)
        
        # Удаляем из индекса
        for ctx_id in to_remove:
            del self.index['contexts'][ctx_id]
        
        # Если всё ещё много, удаляем самые старые
        if len(self.index['contexts']) > max_count:
            sorted_contexts = sorted(
                self.index['contexts'].items(),
                key=lambda x: x[1]['timestamp']
            )
            
            for ctx_id, info in sorted_contexts[:len(sorted_contexts) - max_count]:
                filepath = self.store_dir / info['latest_file']
                if filepath.exists():
                    filepath.unlink()
                del self.index['contexts'][ctx_id]
        
        self._save_index()
    
    def export_context(self, context_id: str, format: str = 'json') -> Optional[str]:
        """Экспорт контекста в читаемый формат"""
        data = self.load_context(context_id)
        if not data:
            return None
        
        if format == 'json':
            export_data = {
                'metadata': data['metadata'],
                'messages': [
                    {
                        'type': type(msg).__name__,
                        'content': msg.content if hasattr(msg, 'content') else str(msg),
                        'timestamp': getattr(msg, 'timestamp', None)
                    }
                    for msg in data['messages']
                ]
            }
            
            export_file = self.store_dir / f"{context_id}_export.json"
            with open(export_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            return str(export_file)
        
        return None

# Глобальное хранилище
context_store = ContextStore()