import tiktoken

def count_tokens_for_qwen(text: str) -> int:
    # Загружаем токенизатор, совместимый с Qwen3-Coder-Next
    # Обычно это 'cl100k_base' для OpenAI-совместимых или свой из transformers
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def my_count_function(messages, **kwargs):
    # Ваша логика подсчета токенов для ВСЕХ сообщений
    # используя `count_tokens_for_qwen`
    total = 0
    for msg in messages:
        total += count_tokens_for_qwen(msg.content)
        # ... также нужно учесть токены роли, tool_calls и т.д.
    return total

