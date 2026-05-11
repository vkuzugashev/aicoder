"""Метрики агента"""
import json
from datetime import datetime
from collections import defaultdict
from pathlib import Path

class Metrics:
    def __init__(self):
        self.requests = 0
        self.success = 0
        self.failed = 0
        self.total_tokens = 0
        self.tools_used = defaultdict(int)
        self.start_time = datetime.now()
        self.response_times = []
    
    def record(self, success: bool, tokens: int, time: float, tools: list = None):
        self.requests += 1
        if success:
            self.success += 1
        else:
            self.failed += 1
        self.total_tokens += tokens
        self.response_times.append(time)
        if tools:
            for t in tools:
                self.tools_used[t] += 1
    
    def stats(self) -> dict:
        avg_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        return {
            "requests": self.requests,
            "success_rate": f"{self.success/self.requests*100:.1f}%" if self.requests else "0%",
            "total_tokens": f"{self.total_tokens:,}",
            "avg_time": f"{avg_time:.2f}s",
            "tools": dict(self.tools_used)
        }
    
    def save(self, path: str = None):
        if path is None:
            path = f"logs/metrics_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.stats(), f, indent=2, ensure_ascii=False)

metrics = Metrics()