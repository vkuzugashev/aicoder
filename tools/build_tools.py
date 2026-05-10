import subprocess

from langchain_core.tools import tool

from tools.file_tools import WORK_DIR, _safe_path


@tool
def npm_install(path: str, options: str):
    """
    Выполнение установки пакетов
    """
    try:
        full_path = _safe_path(path, WORK_DIR)
        command = f'npm install {options}'
        print(f"🖥️ Выполняется: {command}")
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=120, cwd=full_path
        )
        output = result.stdout if result.stdout else result.stderr if result.stderr else "✅ Выполнено (нет вывода)"
        return output
    except subprocess.TimeoutExpired:
        return "❌ Команда выполнялась слишком долго (>120 сек)"
    except Exception as e:
        return f"Ошибка: {str(e)}"

@tool
def npm_build(path: str, options: str):
    """
    Построение проекта
    """
    try:
        full_path = _safe_path(path, WORK_DIR)
        command = f'npm build {options}'
        print(f"🖥️ Выполняется: {command}")
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=180, cwd=full_path
        )
        output = result.stdout if result.stdout else result.stderr if result.stderr else "✅ Выполнено (нет вывода)"
        return output
    except subprocess.TimeoutExpired:
        return "❌ Команда выполнялась слишком долго (>180 сек)"
    except Exception as e:
        return f"Ошибка: {str(e)}"