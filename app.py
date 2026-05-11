#!/usr/bin/env python3
"""Запуск агента"""
import sys
import asyncio

if __name__ == "__main__":
    from agents.agent import main
    
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(main())