"""
Система мониторинга и сбора метрик агента
"""
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict
from dataclasses import dataclass, field
import threading

import sys
sys.path.append('..')
from config import config

@dataclass
class RequestMetrics:
    """Метрики одного запроса"""
    timestamp: datetime
    user_input: str
    response_length: int
    tokens_used: int
    tokens_before: int
    tokens_after: int
    response_time: float
    success: bool
    error: Optional[str] = None
    tools_used: List[str] = field(default_factory=list)
    summarization_applied: bool = False
    context_reduction: float = 0.0

class MetricsCollector:
    """Сборщик метрик с аналитикой"""
    
    def __init__(self):
        self.requests: List[RequestMetrics] = []
        self.session_start = datetime.now()
        self._lock = threading.Lock()
        
        # Агрегированные метрики
        self.aggregated = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_tokens_used': 0,
            'total_context_saved': 0,
            'total_time': 0.0,
            'summarizations': 0,
            'tool_usage': defaultdict(int),
            'error_types': defaultdict(int),
            'peak_context_usage': 0,
            'average_context_usage': 0.0
        }
        
        # Временные ряды для графиков
        self.time_series = {
            'context_usage': [],  # (timestamp, tokens)
            'response_times': [],  # (timestamp, seconds)
            'success_rate': []     # (timestamp, rate)
        }
    
    def record_request(self, 
                      user_input: str,
                      response: Optional[str],
                      tokens_before: int,
                      tokens_after: int,
                      response_time: float,
                      success: bool,
                      error: str = None,
                      tools_used: List[str] = None,
                      summarization_applied: bool = False):
        """Запись метрик запроса"""
        
        with self._lock:
            # Создаем запись
            metrics = RequestMetrics(
                timestamp=datetime.now(),
                user_input=user_input[:200],
                response_length=len(response) if response else 0,
                tokens_used=tokens_before,  # Использовано до обработки
                tokens_before=tokens_before,
                tokens_after=tokens_after,
                response_time=response_time,
                success=success,
                error=error,
                tools_used=tools_used or [],
                summarization_applied=summarization_applied,
                context_reduction=self._calculate_reduction(tokens_before, tokens_after)
            )
            
            self.requests.append(metrics)
            
            # Обновляем агрегированные метрики
            self._update_aggregated(metrics)
            
            # Обновляем временные ряды
            self._update_time_series(metrics)
            
            # Ограничиваем размер истории
            if len(self.requests) > 1000:
                self.requests = self.requests[-500:]  # Оставляем последние 500
    
    def _calculate_reduction(self, before: int, after: int) -> float:
        """Расчет процента сжатия контекста"""
        if before > 0:
            return (before - after) / before * 100
        return 0.0
    
    def _update_aggregated(self, metrics: RequestMetrics):
        """Обновление агрегированных метрик"""
        agg = self.aggregated
        
        agg['total_requests'] += 1
        if metrics.success:
            agg['successful_requests'] += 1
        else:
            agg['failed_requests'] += 1
            if metrics.error:
                agg['error_types'][metrics.error[:50]] += 1
        
        agg['total_tokens_used'] += metrics.tokens_used
        agg['total_context_saved'] += (metrics.tokens_before - metrics.tokens_after)
        agg['total_time'] += metrics.response_time
        
        if metrics.summarization_applied:
            agg['summarizations'] += 1
        
        for tool in metrics.tools_used:
            agg['tool_usage'][tool] += 1
        
        # Пиковое использование контекста
        if metrics.tokens_before > agg['peak_context_usage']:
            agg['peak_context_usage'] = metrics.tokens_before
        
        # Среднее использование
        prev_avg = agg['average_context_usage']
        n = agg['total_requests']
        agg['average_context_usage'] = prev_avg + (metrics.tokens_before - prev_avg) / n
    
    def _update_time_series(self, metrics: RequestMetrics):
        """Обновление временных рядов"""
        now = metrics.timestamp.isoformat()
        
        self.time_series['context_usage'].append({
            'timestamp': now,
            'tokens': metrics.tokens_before
        })
        
        self.time_series['response_times'].append({
            'timestamp': now,
            'seconds': metrics.response_time
        })
        
        # Success rate (скользящее окно из 10 запросов)
        recent = self.requests[-10:]
        success_rate = sum(1 for r in recent if r.success) / len(recent) * 100 if recent else 100
        
        self.time_series['success_rate'].append({
            'timestamp': now,
            'rate': success_rate
        })
        
        # Ограничиваем размер временных рядов
        for key in self.time_series:
            if len(self.time_series[key]) > 200:
                self.time_series[key] = self.time_series[key][-100:]
    
    def get_current_stats(self) -> Dict:
        """Получение текущей статистики"""
        with self._lock:
            agg = self.aggregated
            
            # Вычисляем производные метрики
            total = agg['total_requests']
            success_rate = (agg['successful_requests'] / total * 100) if total > 0 else 100
            
            avg_response_time = (agg['total_time'] / total) if total > 0 else 0
            
            session_duration = (datetime.now() - self.session_start).total_seconds()
            requests_per_minute = (total / (session_duration / 60)) if session_duration > 0 else 0
            
            return {
                'session': {
                    'duration': str(datetime.now() - self.session_start),
                    'requests_per_minute': round(requests_per_minute, 2)
                },
                'requests': {
                    'total': total,
                    'successful': agg['successful_requests'],
                    'failed': agg['failed_requests'],
                    'success_rate': round(success_rate, 2)
                },
                'context': {
                    'total_tokens': agg['total_tokens_used'],
                    'saved_tokens': agg['total_context_saved'],
                    'peak_usage': agg['peak_context_usage'],
                    'average_usage': round(agg['average_context_usage']),
                    'summarizations': agg['summarizations']
                },
                'performance': {
                    'average_response_time': round(avg_response_time, 2),
                    'total_processing_time': round(agg['total_time'], 2)
                },
                'tools': dict(agg['tool_usage'].most_common(10)) if hasattr(agg['tool_usage'], 'most_common') else dict(sorted(agg['tool_usage'].items(), key=lambda x: x[1], reverse=True)[:10]),
                'errors': dict(agg['error_types'])
            }
    
    def save_metrics(self, filepath: str = None):
        """Сохранение метрик в файл"""
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = config.LOGS_DIR / f"metrics_{timestamp}.json"
        
        data = {
            'exported_at': datetime.now().isoformat(),
            'session_start': self.session_start.isoformat(),
            'session_duration': str(datetime.now() - self.session_start),
            'aggregated': self.aggregated,
            'time_series': self.time_series,
            'recent_requests': [
                {
                    'timestamp': r.timestamp.isoformat(),
                    'input': r.user_input,
                    'success': r.success,
                    'tokens_before': r.tokens_before,
                    'tokens_after': r.tokens_after,
                    'reduction': r.context_reduction,
                    'response_time': r.response_time,
                    'error': r.error
                }
                for r in self.requests[-20:]  # Последние 20 запросов
            ]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def print_summary(self):
        """Вывод сводки в консоль"""
        stats = self.get_current_stats()
        
        print("\n" + "="*70)
        print("📊 СТАТИСТИКА СЕССИИ")
        print("="*70)
        print(f"⏱️  Длительность: {stats['session']['duration']}")
        print(f"📨 Запросов: {stats['requests']['total']} "
              f"(✅ {stats['requests']['successful']} / "
              f"❌ {stats['requests']['failed']}) "
              f"[{stats['requests']['success_rate']:.1f}%]")
        print(f"📈 RPM: {stats['session']['requests_per_minute']}")
        print(f"💾 Токенов использовано: {stats['context']['total_tokens']:,}")
        print(f"🗜️  Сэкономлено токенов: {stats['context']['saved_tokens']:,}")
        print(f"📝 Суммаризаций: {stats['context']['summarizations']}")
        print(f"⚡ Среднее время ответа: {stats['performance']['average_response_time']:.2f}с")
        print(f"📊 Пиковый контекст: {stats['context']['peak_usage']:,} токенов")
        
        if stats['tools']:
            print(f"\n🛠️  Топ инструментов:")
            for tool, count in list(stats['tools'].items())[:5]:
                print(f"   {tool}: {count} раз")
        
        if stats['errors']:
            print(f"\n⚠️  Ошибки:")
            for error, count in stats['errors'].items():
                print(f"   {error}: {count} раз")
        
        print("="*70)

# Глобальный сборщик метрик
metrics = MetricsCollector()